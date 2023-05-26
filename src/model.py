import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
import pickle

import blocks
from reproducibility import Parameters


class Model(torch.nn.Module):
    def __init__(self, p: Parameters):
        super().__init__()
        self.num_heads = p.num_heads
        self.device = p.device

        # embeddings
        self.embedding_handler = blocks.EmbeddingHandler(usage_dict=p.embedding_usage_dict,
                                                         embedding_size=p.embedding_size,
                                                         token_seq_len=p.token_seq_len,
                                                         dropout_dict=p.embedding_dropout_dict,
                                                         vit_config=p.vit_config,
                                                         use_layer_norm=p.use_embedding_layer_norm,
                                                         use_segment_embedder=p.use_segment_embedder)

        # encoders
        encoders = []
        for i in range(p.num_encoders):
            encoders.append(blocks.Encoder(num_heads=self.num_heads,
                                           embedding_size=p.embedding_size,
                                           forward_expansion=p.forward_expansion,
                                           mha_dropout=p.mha_dropout,
                                           mlp_dropout=p.mlp_dropout,
                                           encoder_dropout=p.encoder_dropout))
        self.encoders = torch.nn.ModuleList(encoders)

        # output
        # cls
        self.linear_cls = torch.nn.Linear(in_features=p.embedding_size, out_features=2)
        self.sigmoid = torch.nn.Sigmoid()
        # mmlm
        self.linear_mmlm = torch.nn.Linear(in_features=p.embedding_size, out_features=p.vocab_size)
        self.softmax = torch.nn.Softmax() #dim=p.vocab_size)
        # both
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, cls: bool, batch: torch.Tensor):
        # unpack batch
        vit_features = batch[0].to(self.device)
        token_input_ids = batch[1].to(self.device)
        token_attention_mask = batch[2]
        labels = batch[3].to(self.device).float()
        image = batch[4].to(self.device)
        detected_objects = batch[5].to(self.device)
        object_attention_mask = batch[6]
        detected_rois = batch[7].to(self.device)
        text_sentiment_features = batch[8].to(self.device)
        associations = batch[9].to(self.device)
        association_attention_mask = batch[10]
        association_positions = batch[11].to(self.device)

        # create input sequence embeddings
        sequence, attention_mask = self.embedding_handler(token_input_ids=token_input_ids,
                                                          token_attention_mask=token_attention_mask,
                                                          image=image,
                                                          detected_objects=detected_objects,
                                                          detected_rois=detected_rois,
                                                          object_attention_mask=object_attention_mask,
                                                          associations=associations,
                                                          association_attention_mask=association_attention_mask,
                                                          association_positions=association_positions,
                                                          vit_features=vit_features,
                                                          text_sentiment_features=text_sentiment_features)

        # encode sequence
        # batch_size x num_heads x seq_len x seq_len
        attention_scores = torch.zeros((sequence.size(0), self.num_heads, sequence.size(1), sequence.size(1)))
        for encoder in self.encoders:
            sequence, attn = encoder(input_sequence=sequence, attention_mask=attention_mask)
            attention_scores += attn.detach().cpu()

        # batch_size x seq_len x seq_len
        attention_scores = torch.sum(attention_scores, dim=1)
        pooled_output = torch.mean(input=sequence, dim=1)

        # loss calculation
        if cls:
            return self.cls_head(pooled_output=pooled_output, labels=labels, attention_scores=attention_scores)
        else:
            return self.mmlm_head(pooled_output=pooled_output, labels=labels, attention_scores=attention_scores)

    def cls_head(self, pooled_output: torch.Tensor, labels: torch.Tensor, attention_scores: torch.Tensor()):
        logits = self.linear_cls(pooled_output)
        probas = self.sigmoid(logits[:, 1])  # probas for positive class only
        loss = self.ce_loss(logits, labels.long())  # cross entropy loss itself performs softmax
        return loss, probas, attention_scores, logits

    def mmlm_head(self, pooled_output: torch.Tensor, labels: torch.Tensor, attention_scores: torch.Tensor()):
        logits = self.linear_mmlm(pooled_output)
        loss = self.ce_loss(logits, labels.squeeze().long())  # cross entropy loss itself performs softmax
        return loss, logits, attention_scores, None


def calc_gradient_norm(model: Model):
    """
    https://discuss.pytorch.org/t/check-the-norm-of-gradients/27961/3
    checked
    :param model:
    :return:
    """
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def train_one_epoch(model: Model, cls: bool, train_loader: DataLoader, optimizer, num_batches: int, grad_clip_val: float):  # todo optim class
    model.train()  # training mode
    losses = np.zeros(num_batches, dtype=float)
    gradient_norms = np.zeros(num_batches, dtype=float)
    for i, batch in enumerate(train_loader):
        loss, cls_probas, _, _ = model(cls=cls, batch=batch)
        loss.backward()
        # to determine "healthy gradient norm" for gradient clipping
        gradient_norms[i] = calc_gradient_norm(model=model)
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       max_norm=grad_clip_val)  # gradient clipping to fight exploding/vanishing gradients
        optimizer.step()
        optimizer.zero_grad()

        # print status
        losses[i] = float(loss.detach())
        if num_batches == 1:
            print(losses[-1])
        if (i + 1) % 20 == 0:
            avg_loss = losses[(i + 1) - 20:(i + 1)].mean()
            print("batch nr:", i + 1, "/", num_batches, "loss:", avg_loss)

        if (i + 1) == num_batches:
            break
    print("epoch avg gradient norm", gradient_norms.mean())


def eval_one_epoch(model: Model, loader_seq: DataLoader):
    model.eval()  # evaluation mode
    cls_labels_all = []
    cls_probas_all = []
    cls_preds_all = []
    for batch in loader_seq:
        cls_labels = batch[3]
        with torch.no_grad():
            _, cls_probas, _, _ = model(cls=True, batch=batch)
        for j in range(cls_probas.size(0)):
            cls_labels_all.append(float(cls_labels[j]))
            predicted_class = float(torch.round(cls_probas[j]))
            cls_probas_all.append(float(cls_probas[j]))  # probability for positive class
            cls_preds_all.append(predicted_class)
    return cls_labels_all, cls_probas_all, cls_preds_all


def eval_one_epoch_mmlm(model: Model, loader_seq: DataLoader):
    model.eval()  # evaluation mode
    mmlm_labels_all = []
    mmlm_preds_all = []
    for batch in loader_seq:
        mmlm_labels = batch[3]
        with torch.no_grad():
            _, mmlm_logits, _, _ = model(cls=False, batch=batch)
        mmlm_preds = torch.argmax(mmlm_logits, dim=1)
        for j in range(mmlm_labels.size(0)):
            mmlm_labels_all.append(float(mmlm_labels[j]))
            mmlm_preds_all.append(float(mmlm_preds[j]))
    return mmlm_labels_all, mmlm_preds_all


def train_eval_loop(train_loader: DataLoader, train_loader_seq: DataLoader, dev_loader_seq: DataLoader,
                    test_loader_seq: DataLoader, p: Parameters, model: Model, num_batches: int, checkpoints_dir: str):
    if num_batches < len(train_loader):
        train_loader = train_loader_seq  # for debugging
    num_epochs = p.num_epochs

    train_accs = np.zeros(num_epochs)
    train_roc_aucs = np.zeros(num_epochs)

    test_accs = np.zeros(num_epochs)
    test_roc_aucs = np.zeros(num_epochs)

    dev_accs = np.zeros(num_epochs)
    dev_roc_aucs = np.zeros(num_epochs)
    biggest_dev_roc_auc = 0
    biggest_dev_roc_epoch = 0

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=p.lr, weight_decay=p.adam_weight_decay)
    lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=p.lr_gamma)

    for epoch in range(1, num_epochs + 1):
        ep_start = time.time()
        print("epoch", epoch, "/", num_epochs)
        train_one_epoch(model=model,
                        cls=True,
                        train_loader=train_loader,
                        optimizer=optimizer,
                        num_batches=num_batches,
                        grad_clip_val=p.grad_clip_val)
        lr_scheduler.step()

        # eval on train data
        cls_labels, cls_probas, cls_preds = eval_one_epoch(model=model, loader_seq=train_loader_seq)
        train_accs[epoch - 1] = accuracy_score(y_true=cls_labels, y_pred=cls_preds)
        train_roc_aucs[epoch - 1] = roc_auc_score(y_true=cls_labels, y_score=cls_probas, average="weighted")
        if p.save_all:
            np.savetxt(f"{checkpoints_dir}/train_pred_probas_epoch{epoch}.txt", cls_probas)

        # eval on dev data
        cls_labels, cls_probas, cls_preds = eval_one_epoch(model=model, loader_seq=dev_loader_seq)
        dev_accs[epoch - 1] = accuracy_score(y_true=cls_labels, y_pred=cls_preds)
        dev_roc_aucs[epoch - 1] = roc_auc_score(y_true=cls_labels, y_score=cls_probas, average="weighted")
        if dev_roc_aucs[epoch - 1] > biggest_dev_roc_auc:
            biggest_dev_roc_auc = dev_roc_aucs[epoch - 1]
            biggest_dev_roc_epoch = epoch
        if p.save_all:
            np.savetxt(f"{checkpoints_dir}/dev_pred_probas_epoch{epoch}.txt", cls_probas)

        # eval on test data
        if p.save_all:
            cls_labels, cls_probas, cls_preds = eval_one_epoch(model=model, loader_seq=test_loader_seq)
            test_accs[epoch - 1] = accuracy_score(y_true=cls_labels, y_pred=cls_preds)
            test_roc_aucs[epoch - 1] = roc_auc_score(y_true=cls_labels, y_score=cls_probas, average="weighted")
            np.savetxt(f"{checkpoints_dir}/test_pred_probas_epoch{epoch}.txt", cls_probas)

        # save checkpoint
        if p.save_all:
            model_file = f"{checkpoints_dir}/model_epoch_{epoch}.pkl"
            torch.save(model.state_dict(), model_file)

        print("train roc_auc:", train_roc_aucs[epoch - 1])
        print("dev roc_auc:", dev_roc_aucs[epoch - 1])
        print("test roc_auc:", test_roc_aucs[epoch - 1])
        print("epoch time:", np.round(time.time() - ep_start), "seconds\n")

    np.savetxt(f"{checkpoints_dir}/train_accs.txt", train_accs)
    np.savetxt(f"{checkpoints_dir}/train_roc_aucs.txt", train_roc_aucs)
    np.savetxt(f"{checkpoints_dir}/dev_accs.txt", dev_accs)
    np.savetxt(f"{checkpoints_dir}/dev_roc_aucs.txt", dev_roc_aucs)
    np.savetxt(f"{checkpoints_dir}/test_accs.txt", test_accs)
    np.savetxt(f"{checkpoints_dir}/test_roc_aucs.txt", test_roc_aucs)
    with open(f"{checkpoints_dir}/parameters.pkl", "wb") as f:
        pickle.dump(p, f)
    print("biggest dev roc auc:", biggest_dev_roc_auc, "at epoch", biggest_dev_roc_epoch)
    np.savetxt(f"{checkpoints_dir}/0best.txt", np.array([biggest_dev_roc_auc, biggest_dev_roc_epoch]))


def pretrain_eval_loop(train_loader: DataLoader, train_loader_seq: DataLoader, dev_loader_seq: DataLoader,
                    test_loader_seq: DataLoader, p: Parameters, model: Model, num_batches: int):
    if num_batches < len(train_loader):
        train_loader = train_loader_seq  # for debugging
    num_epochs = p.num_epochs
    train_accs = np.zeros(num_epochs)
    dev_accs = np.zeros(num_epochs)
    test_accs = np.zeros(num_epochs)
    biggest_dev_acc = 0
    biggest_dev_acc_epoch = 0

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=p.lr, weight_decay=p.adam_weight_decay)
    lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=p.lr_gamma)

    for epoch in range(1, num_epochs + 1):
        ep_start = time.time()
        print("epoch", epoch, "/", num_epochs)
        train_one_epoch(model=model,
                        cls=False,
                        train_loader=train_loader,
                        optimizer=optimizer,
                        num_batches=num_batches,
                        grad_clip_val=p.grad_clip_val)
        lr_scheduler.step()

        # eval on train data
        mmlm_labels,  mmlm_preds = eval_one_epoch_mmlm(model=model, loader_seq=train_loader_seq)
        train_accs[epoch - 1] = accuracy_score(y_true=mmlm_labels, y_pred=mmlm_preds)
        np.savetxt(f"../checkpoints/train_preds_epoch{epoch}.txt", mmlm_preds)
        np.savetxt(f"../checkpoints/train_labels_epoch{epoch}.txt", mmlm_labels)

        # eval on dev data
        mmlm_labels, mmlm_preds = eval_one_epoch_mmlm(model=model, loader_seq=dev_loader_seq)
        dev_accs[epoch - 1] = accuracy_score(y_true=mmlm_labels, y_pred=mmlm_preds)
        if dev_accs[epoch - 1] > biggest_dev_acc:
            biggest_dev_acc = dev_accs[epoch - 1]
            biggest_dev_acc_epoch = epoch
        np.savetxt(f"../checkpoints/dev_preds_epoch{epoch}.txt", mmlm_preds)
        np.savetxt(f"../checkpoints/dev_labels_epoch{epoch}.txt", mmlm_labels)

        # eval on test data
        mmlm_labels, mmlm_preds = eval_one_epoch_mmlm(model=model, loader_seq=test_loader_seq)
        test_accs[epoch - 1] = accuracy_score(y_true=mmlm_labels, y_pred=mmlm_preds)
        np.savetxt(f"../checkpoints/test_preds_epoch{epoch}.txt", mmlm_preds)
        np.savetxt(f"../checkpoints/test_labels_epoch{epoch}.txt", mmlm_labels)

        # save checkpoint
        model_file = f"../checkpoints/model_epoch_{epoch}.pkl"
        torch.save(model.state_dict(), model_file)

        print("train acc:", train_accs[epoch - 1])
        print("dev acc:", dev_accs[epoch - 1])
        print("test acc:", test_accs[epoch - 1])
        print("epoch time:", np.round(time.time() - ep_start), "seconds\n")

    np.savetxt("../checkpoints/train_accs.txt", train_accs)
    np.savetxt("../checkpoints/dev_accs.txt", dev_accs)
    print("biggest dev roc auc:", biggest_dev_acc, "at epoch", biggest_dev_acc_epoch)

