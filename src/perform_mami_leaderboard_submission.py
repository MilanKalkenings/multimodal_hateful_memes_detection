import os
import pickle
import time

import numpy as np
import pandas as pd
import torch
import winsound
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler

from data_handling import ClsDataset
from model import Model, train_one_epoch, eval_one_epoch
from reproducibility import Parameters
from reproducibility import make_reproducible, seed_worker

# import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
print("setting cublas workspace config")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
make_reproducible()


def train_eval_loop(train_loader: DataLoader, train_loader_seq: DataLoader, dev_loader_seq: DataLoader,
                    test_loader_seq: DataLoader, p: Parameters, model: Model, num_batches: int, checkpoints_dir: str):
    if num_batches < len(train_loader):
        train_loader = train_loader_seq  # for debugging
    num_epochs = p.num_epochs

    train_accs = np.zeros(num_epochs)
    train_roc_aucs = np.zeros(num_epochs)
    train_f1s = np.zeros(num_epochs)

    dev_accs = np.zeros(num_epochs)
    dev_roc_aucs = np.zeros(num_epochs)
    dev_f1s = np.zeros(num_epochs)

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
        train_f1s[epoch -1] = f1_score(y_true=cls_labels, y_pred=cls_preds, average="weighted")
        if p.save_all:
            np.savetxt(f"{checkpoints_dir}/train_pred_probas_epoch{epoch}.txt", cls_probas)

        # eval on dev data
        cls_labels, cls_probas, cls_preds = eval_one_epoch(model=model, loader_seq=dev_loader_seq)
        dev_accs[epoch - 1] = accuracy_score(y_true=cls_labels, y_pred=cls_preds)
        dev_roc_aucs[epoch - 1] = roc_auc_score(y_true=cls_labels, y_score=cls_probas, average="weighted")
        dev_f1s[epoch - 1] = f1_score(y_true=cls_labels, y_pred=cls_preds, average="weighted")
        if dev_roc_aucs[epoch - 1] > biggest_dev_roc_auc:
            biggest_dev_roc_auc = dev_roc_aucs[epoch - 1]
            biggest_dev_roc_epoch = epoch
        if p.save_all:
            np.savetxt(f"{checkpoints_dir}/dev_pred_probas_epoch{epoch}.txt", cls_probas)

        # eval on test data
        if p.save_all:
            cls_labels, cls_probas, cls_preds = eval_one_epoch(model=model, loader_seq=test_loader_seq)
            np.savetxt(f"{checkpoints_dir}/test_pred_probas_epoch{epoch}.txt", cls_probas)

        # save checkpoint
        if p.save_all:
            model_file = f"{checkpoints_dir}/model_epoch_{epoch}.pkl"
            torch.save(model.state_dict(), model_file)

        print("train roc_auc:", train_roc_aucs[epoch - 1])
        print("train f1:", train_f1s[epoch - 1])
        print("dev roc_auc:", dev_roc_aucs[epoch - 1])
        print("dev f1:", dev_f1s[epoch - 1])

        print("epoch time:", np.round(time.time() - ep_start), "seconds\n")


    np.savetxt(f"{checkpoints_dir}/train_accs.txt", train_accs)
    np.savetxt(f"{checkpoints_dir}/train_roc_aucs.txt", train_roc_aucs)
    np.savetxt(f"{checkpoints_dir}/dev_accs.txt", dev_accs)
    np.savetxt(f"{checkpoints_dir}/dev_roc_aucs.txt", dev_roc_aucs)
    with open(f"{checkpoints_dir}/parameters.pkl", "wb") as f:
        pickle.dump(p, f)
    print("biggest dev roc auc:", biggest_dev_roc_auc, "at epoch", biggest_dev_roc_epoch)
    np.savetxt(f"{checkpoints_dir}/0best.txt", np.array([biggest_dev_roc_auc, biggest_dev_roc_epoch]))


p = Parameters(device="cuda",
               adam_weight_decay=0.1,
               num_encoders=2,  # best 2
               num_heads=4,  # best 4
               embedding_size=256,
               batch_size=90,  # best 90
               num_epochs=9,  # normal: 20
               lr_gamma=1,
               lr=0.0003,
               forward_expansion=2,
               mha_dropout=0.2,
               mlp_dropout=0.2,
               encoder_dropout=0.4,
               token_seq_len=32,  # fixed if not with preprocessing
               roi_pool_size=64,  # fixed if not with preprocessing
               objects_seq_len=8,  # fixed if not with preprocessing
               image_size_for_resnet=64,  # fixed if not with preprocessing
               use_embedding_layer_norm=True,
               use_segment_embedder=True,
               grad_clip_val=1,
               initialization_checkpoint="../final_runs/random_init.pkl",
               embedding_dropout_dict={"token": 0.2,
                                       "vit": 0.2,
                                       "roi": 0.8,
                                       "kg": 0.2,
                                       "sentiment": 0.2,
                                       "cnn": 0.8},
               embedding_usage_dict={"token": True,
                                     "roi": True,
                                     "sentiment": True,
                                     "kg": True,
                                     "vit": False,
                                     "cnn": False},
               save_all=True
               )

########################################################################################################################
# hm data with train test split
# print("using hm data")
# train_path = "../hateful_memes_data/final_preprocessed_train.pkl"
# dev_path = "../hateful_memes_data/final_preprocessed_dev.pkl"
# test_path = "../hateful_memes_data/final_preprocessed_test.pkl"

# splitting
"""
train_df = pd.read_json("../hateful_memes_data/train.jsonl", lines=True)
dev_df = pd.read_json("../hateful_memes_data/dev.jsonl", lines=True)
total_df = pd.concat([train_df, dev_df])
df_train_manual, df_dev_manual = train_test_split(total_df, test_size=0.2)
df_dev_manual, df_test_manual = train_test_split(df_dev_manual, test_size=0.5)
df_train_manual.to_csv("../hateful_memes_data/train_manual_split.jsonl", index=False)
df_dev_manual.to_csv("../hateful_memes_data/dev_manual_split.jsonl", index=False)
df_test_manual.to_csv("../hateful_memes_data/test_manual_split.jsonl", index=False)
"""
# with preprocessing
"""
preprocessor = DataPreprocessor(bert_seq_len=p.token_seq_len, faster_rcnn_seq_len=p.objects_seq_len,
                                roi_pool_size=p.roi_pool_size, image_size=p.image_size_for_resnet)
preprocessor.preprocess_and_save(data_file="../hateful_memes_data/dev_manual_split.jsonl",
                                 target_file=dev_path)
preprocessor.preprocess_and_save(data_file="../hateful_memes_data/test_manual_split.jsonl",
                                 target_file=test_path)
preprocessor.preprocess_and_save(data_file="../hateful_memes_data/train_manual_split.jsonl",
                                 target_file=train_path)
"""
########################################################################################################################
# misogyny
print("using misogyny data")
train_path = "../misogyny_data/final_preprocessed_train.csv"
dev_path = "../misogyny_data/final_preprocessed_dev.csv"
test_path = "../misogyny_data/final_preprocessed_test.csv"
leaderboard_path = "../mami/final_preprocessed_leaderboard.csv"

# splitting

# with preprocessing
"""
preprocessor = DataPreprocessor(bert_seq_len=p.token_seq_len,
                                faster_rcnn_seq_len=p.objects_seq_len,
                                roi_pool_size=p.roi_pool_size,
                                image_size=p.image_size_for_resnet,
                                on_hm_data=False)
preprocessor.preprocess_and_save(data_file="../misogyny_data/train.csv",
                                 target_file=train_path)
preprocessor.preprocess_and_save(data_file="../misogyny_data/dev.csv",
                                 target_file=dev_path)
preprocessor.preprocess_and_save(data_file="../misogyny_data/test.csv",
                                 target_file=test_path)
"""
########################################################################################################################
# was used to create random initial checkpoint
# print("creating random initiation checkpoint")
# model = Model(p=p).to(p.device)
# torch.save(model.state_dict(), "../final_runs/random_init.pkl")

# load datasets
print("loading data")
df_train = pd.read_pickle(train_path)
df_dev = pd.read_pickle(dev_path)
# combined for mami leaderboard
df_mami_train = pd.concat([df_train, df_dev], axis=1)
train_dataset = ClsDataset(df_train)

# test set is new eval set
df_dev = pd.read_pickle(test_path)
dev_dataset = ClsDataset(df_dev)

# leaderboard is new test set
df_test = pd.read_pickle(leaderboard_path)
test_dataset = ClsDataset(df_test)


print("creating loaders")

train_sampler_seq = SequentialSampler(data_source=train_dataset)
train_loader_seq = DataLoader(dataset=train_dataset, batch_size=p.batch_size, sampler=train_sampler_seq)

dev_sampler_seq = SequentialSampler(data_source=dev_dataset)
dev_loader_seq = DataLoader(dataset=dev_dataset, batch_size=p.batch_size, sampler=dev_sampler_seq)

test_sampler_seq = SequentialSampler(data_source=test_dataset)
test_loader_seq = DataLoader(dataset=test_dataset, batch_size=p.batch_size, sampler=test_sampler_seq)

usage_dicts = [
    {"token": True,
     "roi": True,
     "sentiment": False,
     "kg": True,
     "vit": False,
     "cnn": False},
]

for seed in [1]:
    for usage_dict in usage_dicts:
        make_reproducible(seed=seed)

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=p.batch_size,
                                  shuffle=True,  # according to random seed
                                  num_workers=0,
                                  worker_init_fn=seed_worker,
                                  pin_memory=True)

        p.embedding_usage_dict = usage_dict
        features = []
        for (k, v) in usage_dict.items():
            if v:
                features.append(k)
        dir_name = ""
        for feat in features:
            dir_name += f"_{feat}"
        checkpoints_dir = f"../mami/{seed}{dir_name}"
        os.mkdir(checkpoints_dir)

        # initiate model
        model = Model(p=p).to(p.device)
        state_dict = torch.load(p.initialization_checkpoint)
        model.load_state_dict(state_dict=state_dict)

        print("beginning training for", dir_name)
        train_eval_loop(train_loader=train_loader,
                        train_loader_seq=train_loader_seq,
                        dev_loader_seq=dev_loader_seq,
                        test_loader_seq=test_loader_seq,
                        p=p,
                        model=model,
                        num_batches=len(train_loader),  # = 1 for debugging
                        checkpoints_dir=checkpoints_dir)
        print("\n\n")

frequency = 2500
duration = 2_000
winsound.Beep(frequency, duration)



"""
#formatting:
probas = np.loadtxt("../mami/1_token_roi_kg/test_pred_probas_epoch9.txt")
preds = np.round(probas)
preds_ser = pd.Series(preds.astype(int))

df_leaderboard = pd.read_pickle("../mami/final_preprocessed_leaderboard.csv")

formatted_submission = []
for i, pred in enumerate(preds):
    formatted_submission.append(df_leaderboard["img"][i] + "\t" + str(pred))
df = pd.concat([df_leaderboard["img"], preds_ser], axis=1)
df.to_csv("../mami/1_token_roi_kg/submission/answer.txt", sep="\t", header=False, index=False)

"""
