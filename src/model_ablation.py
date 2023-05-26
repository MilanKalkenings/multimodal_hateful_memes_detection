import os
import time

import imagehash
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score

from model import Model


def confusion_indices(ground_truth: np.array, probas: np.array):
    """
    :param data_path:
    :param probas_path:
    :return:
    """
    preds = np.round(probas)
    df = pd.DataFrame(np.vstack([preds, ground_truth]).T, columns=["pred", "true"])

    df_0 = df.loc[df["true"] == 0, :]
    correct_0 = df_0.loc[df["pred"] == 0, :].index
    false_0 = df_0.loc[df["pred"] == 1, :].index

    df_1 = df.loc[df["true"] == 1, :]
    correct_1 = df_1.loc[df["pred"] == 1, :].index
    false_1 = df_1.loc[df["pred"] == 0, :].index

    all_correct = set(np.hstack([correct_0, correct_1]))

    return all_correct, correct_0, correct_1, false_0, false_1


def plot_confusion_matrix(correct_0: int, correct_1: int, false_0: int, false_1: int, plot_path: str,
                          roc_auc_score: float):
    plt.clf()
    matrix = np.array([[correct_1, false_1],
                       [false_0, correct_0]])
    labels = ["1", "0"]
    fig = plt.figure(figsize=(5, 3))
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, np.round(matrix[i, j], 2), ha="center", va="center", color="black", fontsize=16)
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    plt.ylabel("Vorhersage")
    plt.xlabel("Annotation")
    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    plt.title(f"Konfusionsmatrix, roc_auc_score: {np.round(roc_auc_score, 4)}")
    plt.tight_layout()
    plt.savefig(plot_path)


def embedding_indices(usage_dict: dict):
    last_index = 0
    indices_dict = {"CNN-Segment": None,
                    "Token-Segment": None,
                    "Kachel-Segment": None,
                    "Objekt-Segment": None,
                    "Stimmungs-Segment": None,
                    "Assoziations-Segment": None}

    # 2 positions
    if usage_dict["cnn"]:
        indices_dict["CNN-Segment"] = list(range(0, last_index + 2))
        last_index += 2

    # 32 positions
    if usage_dict["token"]:
        indices_dict["Token-Segment"] = list(range(last_index, last_index + 32))
        last_index += 32

    # 196 positions
    if usage_dict["vit"]:
        indices_dict["Kachel-Segment"] = list(range(last_index, last_index + 196))
        last_index += 196

    # 8 positions
    if usage_dict["roi"]:
        indices_dict["Objekt-Segment"] = list(range(last_index, last_index + 8))
        last_index += 8

    # 1 position
    if usage_dict["sentiment"]:
        indices_dict["Stimmungs-Segment"] = list(range(last_index, last_index + 1))
        last_index += 1

    # 16 positions
    if usage_dict["kg"]:
        indices_dict["Assoziations-Segment"] = list(range(last_index, last_index + 16))
    return indices_dict


"""
def plot_resnet_saliency_map(obs: torch.Tensor, model: Model, image_path: str, plot_name: str):
    plt.clf()  # clean cache
    image = obs[4]
    image.requires_grad_()  # to calculate the gradient wrt the image
    obs[4] = image
    _, _, _, logits = model(cls=True, batch=obs)
    class_logit = logits[0][1]  # for hateful memes
    class_logit.backward()  # calculate gradient
    saliency, _ = torch.max(image.grad.data.abs(), dim=1)

    saliency = saliency.reshape([saliency.size(1), saliency.size(2)]).cpu() ** (6)  # add heat by **6
    saliency -= torch.min(saliency)
    saliency /= torch.max(saliency)


    # transformations
    image_to_tensor = transforms.ToTensor()
    image_up_or_down_sampler = Upsample(size=512, mode="nearest")
    tensor_to_pil_image = transforms.ToPILImage()

    image_raw = Image.open(image_path).convert("RGB")
    image_tensor = image_to_tensor(image_raw)
    image_tensor = torch.unsqueeze(image_tensor, dim=0)

    saliency = torch.unsqueeze(torch.unsqueeze(saliency, dim=0), dim=0)

    image_upsampled = image_up_or_down_sampler(image_tensor).squeeze()

    saliency_upsampled = image_up_or_down_sampler(saliency).squeeze()
    saliency_upsampled = tensor_to_pil_image(saliency_upsampled * 255)
    saliency_upsampled = saliency_upsampled.filter(ImageFilter.GaussianBlur(radius=11))  # blur for less grid-like look
    saliency_upsampled = image_to_tensor(saliency_upsampled).squeeze()

    saliency_weighted_image = image_upsampled * saliency_upsampled
    saliency_weighted_image -= torch.min(saliency_weighted_image)
    saliency_weighted_image /= torch.max(saliency_weighted_image)
    plt.imshow(saliency_weighted_image.permute(1, 2, 0))
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(f"../ausarbeitung/figures/saliency/fusion{plot_name}")

    plt.imshow(image_upsampled.permute(1, 2, 0), cmap='Greys_r', interpolation='nearest')
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(f"../ausarbeitung/figures/saliency/image{plot_name}")
"""


def plot_pooled_attention_scores_hm(attention_scores: torch.Tensor, plot_path: str):
    token_to_token = attention_scores[0:32, 0:32].sum() / 32
    token_to_vit = attention_scores[0:32, 32:(32 + 196)].sum() / 32
    token_to_sentiment = attention_scores[0:32, (32 + 196):(32 + 196 + 1)].sum() / 32
    token_to_kg = attention_scores[0:32, (32 + 196 + 1):(32 + 196 + 1 + 16)].sum() / 32

    vit_to_token = attention_scores[32:(32 + 196), 0:32].sum() / 196
    vit_to_vit = attention_scores[32:(32 + 196), 32:(32 + 196)].sum() / 196
    vit_to_sentiment = attention_scores[32:(32 + 196), (32 + 196):(32 + 196 + 1)].sum() / 196
    vit_to_kg = attention_scores[32:(32 + 196), (32 + 196 + 1):(32 + 196 + 1 + 16)].sum() / 196

    sentiment_to_token = attention_scores[(32 + 196):(32 + 196 + 1), 0:32].sum()
    sentiment_to_vit = attention_scores[(32 + 196):(32 + 196 + 1), 32:(32 + 196)].sum()
    sentiment_to_sentiment = attention_scores[(32 + 196):(32 + 196 + 1), (32 + 196):(32 + 196 + 1)].sum()
    sentiment_to_kg = attention_scores[(32 + 196):(32 + 196 + 1), (32 + 196 + 1):(32 + 196 + 1 + 16)].sum()

    kg_to_token = attention_scores[(32 + 196 + 1):(32 + 196 + 1 + 16), 0:32].sum() / 16
    kg_to_vit = attention_scores[(32 + 196 + 1):(32 + 196 + 1 + 16), 32:(32 + 196)].sum() / 16
    kg_to_sentiment = attention_scores[(32 + 196 + 1):(32 + 196 + 1 + 16), (32 + 196):(32 + 196 + 1)].sum() / 16
    kg_to_kg = attention_scores[(32 + 196 + 1):(32 + 196 + 1 + 16), (32 + 196 + 1):(32 + 196 + 1 + 16)].sum() / 16

    pooled_attention_scores = np.array([[token_to_token, token_to_vit, token_to_sentiment, token_to_kg],
                                        [vit_to_token, vit_to_vit, vit_to_sentiment, vit_to_kg],
                                        [sentiment_to_token, sentiment_to_vit, sentiment_to_sentiment, sentiment_to_kg],
                                        [kg_to_token, kg_to_vit, kg_to_sentiment, kg_to_kg]]) * 100
    print(pooled_attention_scores)

    plt.clf()  # clean cache
    plt.xticks(range(4), ["Token-Modul", "Kachel-Modul", "Stimmungs-Modul", "Assoziations-Modul"],
               rotation="vertical")
    plt.yticks(range(4), ["Token-Modul", "Kachel-Modul", "Stimmungs-Modul", "Assoziations-Modul"])
    plt.imshow(pooled_attention_scores, cmap='Oranges', interpolation='nearest')
    # Loop over data dimensions and create text annotations.
    for i in range(pooled_attention_scores.shape[0]):
        for j in range(pooled_attention_scores.shape[1]):
            plt.text(j, i, str(np.round(pooled_attention_scores[i, j], 2)) + "%", ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(plot_path)


def plot_pooled_attention_scores_mis(attention_scores: torch.Tensor, plot_path: str):
    token_to_token = attention_scores[0:32, 0:32].sum() / 32
    token_to_roi = attention_scores[0:32, 32:(32 + 8)].sum() / 32
    token_to_kg = attention_scores[0:32, (32 + 8):(32 + 8 + 16)].sum() / 32

    roi_to_token = attention_scores[32:(32 + 8), 0:32].sum() / 8
    roi_to_roi = attention_scores[32:(32 + 8), 32:(32 + 8)].sum() / 8
    roi_to_kg = attention_scores[32:(32 + 8), (32 + 8):(32 + 8 + 16)].sum() / 8

    kg_to_token = attention_scores[(32 + 8):(32 + 8 + 16), 0:32].sum() / 16
    kg_to_roi = attention_scores[(32 + 8):(32 + 8 + 16), 32:(32 + 8)].sum() / 16
    kg_to_kg = attention_scores[(32 + 8):(32 + 8 + 16), (32 + 8):(32 + 8 + 16)].sum() / 16

    pooled_attention_scores = np.array([[token_to_token, token_to_roi, token_to_kg],
                                        [roi_to_token, roi_to_roi, roi_to_kg],
                                        [kg_to_token, kg_to_roi, kg_to_kg]]) * 100
    print(pooled_attention_scores)

    plt.clf()  # clean cache
    plt.xticks(range(3), ["Token-Modul", "Objekt-Modul", "Assoziations-Modul"], rotation="vertical")
    plt.yticks(range(3), ["Token-Modul", "Objekt-Modul", "Assoziations-Modul"])
    plt.imshow(pooled_attention_scores, cmap='Oranges', interpolation='nearest')
    # Loop over data dimensions and create text annotations.
    for i in range(pooled_attention_scores.shape[0]):
        for j in range(pooled_attention_scores.shape[1]):
            plt.text(j, i, str(np.round(pooled_attention_scores[i, j], 2)) + "%", ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(plot_path)


def obs_attention_scores(obs: torch.Tensor, model: Model):
    """

    :param obs: batch for classification of batch_size 1 (observation)
    :param model:
    :return:
    """
    with torch.no_grad():
        _, _, attention_scores, _ = model(cls=True, batch=obs)
    attention_score_matrix = attention_scores.squeeze()
    for i in range(
            attention_score_matrix.size(0)):  # was summed over all encoders, now normalize each row to sum up to 1
        attention_score_matrix[i] = attention_score_matrix[i] / attention_score_matrix[i].sum()
    return attention_score_matrix


def compare_images(test_image_paths: pd.Series, train_image_paths: pd.Series, thresh: int):
    match_candidates = []
    for i, test_image_path in enumerate(test_image_paths):
        start = time.time()
        test_img = Image.open(test_image_path).convert("RGB")
        current_candidates = []
        print(i + 1)
        for j, train_image_path in enumerate(train_image_paths):
            train_img = Image.open(train_image_path).convert("RGB")
            if test_img.size == train_img.size:
                current_candidates.append(train_image_path)

        match_candidates.append(current_candidates)
        print(time.time() - start)
    return match_candidates


def get_img_hash(img_path: str):
    image = Image.open(img_path).convert("RGB")
    return imagehash.average_hash(image)


def get_img_size(img_path: str):
    image = Image.open(img_path).convert("RGB")
    return image.size


def analyze_kg(test_df: pd.DataFrame, test_probas: np.array):
    def check_for_associations(association_attention_mask: torch.tensor):
        if association_attention_mask[0] == 1:
            return True
        return False

    test_df["has_association"] = test_df["association_attention_mask"].apply(check_for_associations)
    probas_with_association_label1 = []
    probas_no_association_label1 = []
    probas_with_association_label0 = []
    probas_no_association_label0 = []

    for i, proba in enumerate(test_probas):
        label = test_df["label"][i]
        if test_df["has_association"][i]:
            if label == 1:
                probas_with_association_label1.append(proba)
            else:
                probas_with_association_label0.append(proba)
        else:
            if label == 1:
                probas_no_association_label1.append(proba)
            else:
                probas_no_association_label0.append(proba)

    num_ass_label1 = len(probas_with_association_label1)
    acc_ass_label1 = accuracy_score(y_true=np.ones(num_ass_label1), y_pred=np.round(probas_with_association_label1))
    num_ass_label0 = len(probas_with_association_label0)
    acc_ass_label0 = accuracy_score(y_true=np.zeros(num_ass_label0), y_pred=np.round(probas_with_association_label0))
    num_no_ass_label1 = len(probas_no_association_label1)
    acc_no_ass_label1 = accuracy_score(y_true=np.ones(num_no_ass_label1), y_pred=np.round(probas_no_association_label1))
    num_no_ass_label0 = len(probas_no_association_label0)
    acc_no_ass_label0 = accuracy_score(y_true=np.zeros(num_no_ass_label0), y_pred=np.round(probas_no_association_label0))
    meme_nums = [num_ass_label1, num_ass_label0, num_no_ass_label1, num_no_ass_label0]
    accs = [acc_ass_label1, acc_ass_label0, acc_no_ass_label1, acc_no_ass_label0]
    return meme_nums, accs


def avg_roc_aucs_per_epoch(stage_dir: str, mode: str):
    """

    :param stage_dir:
    :param mode: in {"train", "test", "dev"}
    :return:
    """
    checkpoint_names = [name[1:] for name in os.listdir(stage_dir + "\seed1")]

    for name in checkpoint_names:
        scores = []
        seed1_scores = np.loadtxt(stage_dir + "\seed1\\" + "1" + name + "\\" + mode + "_roc_aucs.txt")
        seed2_scores = np.loadtxt(stage_dir + "\seed2\\" + "2" + name + "\\" + mode + "_roc_aucs.txt")
        seed3_scores = np.loadtxt(stage_dir + "\seed3\\" + "3" + name + "\\" + mode + "_roc_aucs.txt")
        for i in range(len(seed1_scores)):
            scores.append((seed1_scores[i] + seed2_scores[i] + seed3_scores[i]) / 3)
        scores = np.array(scores)
        max_value = np.round(np.max(scores), 4)
        max_expoch = np.argmax(scores) + 1
        np.savetxt(
            stage_dir + "\\" + mode + "_roc_auc" + str(max_value) + "epoch" + str(int(max_expoch)) + name + ".txt",
            scores)


def avg_roc_aucs_per_epoch_cross_training_mis(stage_dir: str, mode: str):
    """

    :param stage_dir:
    :param mode: in {"train", "test", "dev"}
    :return:
    """
    checkpoint_names = [name[1:] for name in os.listdir(stage_dir + "\seed1")]

    for name in checkpoint_names:
        scores = []
        seed1_scores = np.loadtxt(stage_dir + "\seed1\\" + "1" + name + "\\" + mode + "_roc_aucs_mis.txt")
        seed2_scores = np.loadtxt(stage_dir + "\seed2\\" + "2" + name + "\\" + mode + "_roc_aucs_mis.txt")
        seed3_scores = np.loadtxt(stage_dir + "\seed3\\" + "3" + name + "\\" + mode + "_roc_aucs_mis.txt")
        for i in range(len(seed1_scores)):
            scores.append((seed1_scores[i] + seed2_scores[i] + seed3_scores[i]) / 3)
        scores = np.array(scores)
        max_value = np.round(np.max(scores), 4)
        max_expoch = np.argmax(scores) + 1
        np.savetxt(
            stage_dir + "\\" + mode + "_roc_auc_mis" + str(max_value) + "epoch" + str(int(max_expoch)) + name + ".txt",
            scores)


def avg_roc_aucs_per_epoch_cross_training_hm(stage_dir: str, mode: str):
    """

    :param stage_dir:
    :param mode: in {"train", "test", "dev"}
    :return:
    """
    checkpoint_names = [name[1:] for name in os.listdir(stage_dir + "\seed1")]

    for name in checkpoint_names:
        scores = []
        seed1_scores = np.loadtxt(stage_dir + "\seed1\\" + "1" + name + "\\" + mode + "_roc_aucs_hm.txt")
        seed2_scores = np.loadtxt(stage_dir + "\seed2\\" + "2" + name + "\\" + mode + "_roc_aucs_hm.txt")
        seed3_scores = np.loadtxt(stage_dir + "\seed3\\" + "3" + name + "\\" + mode + "_roc_aucs_hm.txt")
        for i in range(len(seed1_scores)):
            scores.append((seed1_scores[i] + seed2_scores[i] + seed3_scores[i]) / 3)
        scores = np.array(scores)
        max_value = np.round(np.max(scores), 4)
        max_expoch = np.argmax(scores) + 1
        np.savetxt(
            stage_dir + "\\" + mode + "_roc_auc_hm" + str(max_value) + "epoch" + str(int(max_expoch)) + name + ".txt",
            scores)

########################################################################################################################

# to analyze architecture module selection stages
# avg_roc_aucs_per_epoch(stage_dir="../final_runs", mode="dev")

# only for cross (pre-)training
# avg_roc_aucs_per_epoch_cross_training_mis(stage_dir="../final_runs", mode="dev")
# avg_roc_aucs_per_epoch_cross_training_hm(stage_dir="../final_runs", mode="dev")
