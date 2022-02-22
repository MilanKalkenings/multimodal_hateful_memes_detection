import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score



########################################################################################################################
# mis
test_path = "../misogyny_data/final_preprocessed_test.csv"
ground_truth_labels_test = pd.read_pickle(test_path)["label"]

# kg
kg_epoch_probas_test_1 = []
kg_epoch_probas_test_2 = []
kg_epoch_probas_test_3 = []
for i in range(1, 21):
    seed1 = np.loadtxt(f"../final_runs/mis/mis_late_fusion/seed1/1_kg/test_pred_probas_epoch{i}.txt")
    kg_epoch_probas_test_1.append(seed1)
    seed2 = np.loadtxt(f"../final_runs/mis/mis_late_fusion/seed2/2_kg/test_pred_probas_epoch{i}.txt")
    kg_epoch_probas_test_2.append(seed2)
    seed3 = np.loadtxt(f"../final_runs/mis/mis_late_fusion/seed3/3_kg/test_pred_probas_epoch{i}.txt")
    kg_epoch_probas_test_3.append(seed3)

# token
token_epoch_probas_test_1 = []
token_epoch_probas_test_2 = []
token_epoch_probas_test_3 = []
for i in range(1, 21):
    seed1 = np.loadtxt(f"../final_runs/mis/mis_late_fusion/seed1/1_token/test_pred_probas_epoch{i}.txt")
    token_epoch_probas_test_1.append(seed1)
    seed2 = np.loadtxt(f"../final_runs/mis/mis_late_fusion/seed2/2_token/test_pred_probas_epoch{i}.txt")
    token_epoch_probas_test_2.append(seed2)
    seed3 = np.loadtxt(f"../final_runs/mis/mis_late_fusion/seed3/3_token/test_pred_probas_epoch{i}.txt")
    token_epoch_probas_test_3.append(seed3)

# roi
roi_epoch_probas_test_1 = []
roi_epoch_probas_test_2 = []
roi_epoch_probas_test_3 = []
for i in range(1, 21):
    seed1 = np.loadtxt(f"../final_runs/mis/mis_late_fusion/seed1/1_roi/test_pred_probas_epoch{i}.txt")
    roi_epoch_probas_test_1.append(seed1)
    seed2 = np.loadtxt(f"../final_runs/mis/mis_late_fusion/seed2/2_roi/test_pred_probas_epoch{i}.txt")
    roi_epoch_probas_test_2.append(seed2)
    seed3 = np.loadtxt(f"../final_runs/mis/mis_late_fusion/seed3/3_roi/test_pred_probas_epoch{i}.txt")
    roi_epoch_probas_test_3.append(seed3)


# average
average_probas_test_1 = []
average_probas_test_2 = []
average_probas_test_3 = []
for i in range(20):
    average_probas_test_1.append((kg_epoch_probas_test_1[i] + token_epoch_probas_test_1[i] + roi_epoch_probas_test_1[i]) / 3)
    average_probas_test_2.append((kg_epoch_probas_test_2[i] + token_epoch_probas_test_2[i] + roi_epoch_probas_test_2[i]) / 3)
    average_probas_test_3.append((kg_epoch_probas_test_3[i] + token_epoch_probas_test_3[i] + roi_epoch_probas_test_3[i]) / 3)
# calculate score for dev set to determine "best epoch"
avg_test_roc_aucs = np.zeros(20)
for i in range(20):
    test_roc_auc_1 = roc_auc_score(y_true=ground_truth_labels_test, y_score=average_probas_test_1[i])
    recall_score_1 = recall_score(y_true=ground_truth_labels_test, y_pred=np.round(average_probas_test_1[i]))
    precision_score_1 = precision_score(y_true=ground_truth_labels_test, y_pred=np.round(average_probas_test_1[i]))
    test_roc_auc_2 = roc_auc_score(y_true=ground_truth_labels_test, y_score=average_probas_test_2[i])
    recall_score_2 = recall_score(y_true=ground_truth_labels_test, y_pred=np.round(average_probas_test_1[i]))
    precision_score_2 = precision_score(y_true=ground_truth_labels_test, y_pred=np.round(average_probas_test_1[i]))
    test_roc_auc_3 = roc_auc_score(y_true=ground_truth_labels_test, y_score=average_probas_test_3[i])
    recall_score_3 = recall_score(y_true=ground_truth_labels_test, y_pred=np.round(average_probas_test_1[i]))
    precision_score_3 = precision_score(y_true=ground_truth_labels_test, y_pred=np.round(average_probas_test_1[i]))
    avg_roc_auc = (test_roc_auc_1 + test_roc_auc_2 + test_roc_auc_3) / 3
    avg_precison = (precision_score_1 + precision_score_2 + precision_score_3) / 3
    avg_recall = (recall_score_1 + recall_score_2 + recall_score_3) / 3
    avg_test_roc_aucs[i] = avg_roc_auc
    print("epoch", i+1, "test roc_auc:", avg_roc_auc, "test precision:", avg_precison, "test recall:", avg_recall)
print("best:", np.max(avg_test_roc_aucs), "at epoch", np.argmax(avg_test_roc_aucs)+1)


########################################################################################################################
"""
# hm
test_path = "../hateful_memes_data/final_preprocessed_test.pkl"
ground_truth_labels_test = pd.read_pickle(test_path)["label"]

# kg
kg_epoch_probas_test_1 = []
kg_epoch_probas_test_2 = []
kg_epoch_probas_test_3 = []
for i in range(1, 21):
    seed1 = np.loadtxt(f"../final_runs/hm/hm_late_fusion/seed1/1_kg/test_pred_probas_epoch{i}.txt")
    kg_epoch_probas_test_1.append(seed1)
    seed2 = np.loadtxt(f"../final_runs/hm/hm_late_fusion/seed2/2_kg/test_pred_probas_epoch{i}.txt")
    kg_epoch_probas_test_2.append(seed2)
    seed3 = np.loadtxt(f"../final_runs/hm/hm_late_fusion/seed3/3_kg/test_pred_probas_epoch{i}.txt")
    kg_epoch_probas_test_3.append(seed3)

# token
token_epoch_probas_test_1 = []
token_epoch_probas_test_2 = []
token_epoch_probas_test_3 = []
for i in range(1, 21):
    seed1 = np.loadtxt(f"../final_runs/hm/hm_late_fusion/seed1/1_token/test_pred_probas_epoch{i}.txt")
    token_epoch_probas_test_1.append(seed1)
    seed2 = np.loadtxt(f"../final_runs/hm/hm_late_fusion/seed2/2_token/test_pred_probas_epoch{i}.txt")
    token_epoch_probas_test_2.append(seed2)
    seed3 = np.loadtxt(f"../final_runs/hm/hm_late_fusion/seed3/3_token/test_pred_probas_epoch{i}.txt")
    token_epoch_probas_test_3.append(seed3)

# sentiment
sentiment_epoch_probas_test_1 = []
sentiment_epoch_probas_test_2 = []
sentiment_epoch_probas_test_3 = []
for i in range(1, 21):
    seed1 = np.loadtxt(f"../final_runs/hm/hm_late_fusion/seed1/1_sentiment/test_pred_probas_epoch{i}.txt")
    sentiment_epoch_probas_test_1.append(seed1)
    seed2 = np.loadtxt(f"../final_runs/hm/hm_late_fusion/seed2/2_sentiment/test_pred_probas_epoch{i}.txt")
    sentiment_epoch_probas_test_2.append(seed2)
    seed3 = np.loadtxt(f"../final_runs/hm/hm_late_fusion/seed3/3_sentiment/test_pred_probas_epoch{i}.txt")
    sentiment_epoch_probas_test_3.append(seed3)

# vit
vit_epoch_probas_test_1 = []
vit_epoch_probas_test_2 = []
vit_epoch_probas_test_3 = []
for i in range(1, 21):
    seed1 = np.loadtxt(f"../final_runs/hm/hm_late_fusion/seed1/1_vit/test_pred_probas_epoch{i}.txt")
    vit_epoch_probas_test_1.append(seed1)
    seed2 = np.loadtxt(f"../final_runs/hm/hm_late_fusion/seed2/2_vit/test_pred_probas_epoch{i}.txt")
    vit_epoch_probas_test_2.append(seed2)
    seed3 = np.loadtxt(f"../final_runs/hm/hm_late_fusion/seed3/3_vit/test_pred_probas_epoch{i}.txt")
    vit_epoch_probas_test_3.append(seed3)




# average
average_probas_test_1 = []
average_probas_test_2 = []
average_probas_test_3 = []
for i in range(20):
    average_probas_test_1.append((kg_epoch_probas_test_1[i] + token_epoch_probas_test_1[i] + sentiment_epoch_probas_test_1[i] + vit_epoch_probas_test_1[i]) / 4)
    average_probas_test_2.append((kg_epoch_probas_test_2[i] + token_epoch_probas_test_2[i] + sentiment_epoch_probas_test_2[i] + vit_epoch_probas_test_2[i]) / 4)
    average_probas_test_3.append((kg_epoch_probas_test_3[i] + token_epoch_probas_test_3[i] + sentiment_epoch_probas_test_3[i] + vit_epoch_probas_test_3[i]) / 4)
# calculate score for dev set to determine "best epoch"
avg_test_roc_aucs = np.zeros(20)
for i in range(20):
    test_roc_auc_1 = roc_auc_score(y_true=ground_truth_labels_test, y_score=average_probas_test_1[i])
    recall_score_1 = recall_score(y_true=ground_truth_labels_test, y_pred=np.round(average_probas_test_1[i]))
    precision_score_1 = precision_score(y_true=ground_truth_labels_test, y_pred=np.round(average_probas_test_1[i]))
    test_roc_auc_2 = roc_auc_score(y_true=ground_truth_labels_test, y_score=average_probas_test_2[i])
    recall_score_2 = recall_score(y_true=ground_truth_labels_test, y_pred=np.round(average_probas_test_1[i]))
    precision_score_2 = precision_score(y_true=ground_truth_labels_test, y_pred=np.round(average_probas_test_1[i]))
    test_roc_auc_3 = roc_auc_score(y_true=ground_truth_labels_test, y_score=average_probas_test_3[i])
    recall_score_3 = recall_score(y_true=ground_truth_labels_test, y_pred=np.round(average_probas_test_1[i]))
    precision_score_3 = precision_score(y_true=ground_truth_labels_test, y_pred=np.round(average_probas_test_1[i]))
    avg_roc_auc = (test_roc_auc_1 + test_roc_auc_2 + test_roc_auc_3) / 3
    avg_precison = (precision_score_1 + precision_score_2 + precision_score_3) / 3
    avg_recall = (recall_score_1 + recall_score_2 + recall_score_3) / 3
    avg_test_roc_aucs[i] = avg_roc_auc
    print("epoch", i+1, "test roc_auc:", avg_roc_auc, "test precision:", avg_precison, "test recall:", avg_recall)
print("best:", np.max(avg_test_roc_aucs), "at epoch", np.argmax(avg_test_roc_aucs)+1)
"""