import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score

# mis
test_path = "../misogyny_data/final_preprocessed_test.csv"
ground_truth_labels_test = pd.read_pickle(test_path)["label"]
probas_1 = np.loadtxt("../final_runs/mis/mis_early_fusion test_8421/seed1/1_token_roi_kg/test_pred_probas_epoch2.txt")
probas_2 = np.loadtxt("../final_runs/mis/mis_early_fusion test_8421/seed2/2_token_roi_kg/test_pred_probas_epoch2.txt")
probas_3 = np.loadtxt("../final_runs/mis/mis_early_fusion test_8421/seed3/3_token_roi_kg/test_pred_probas_epoch2.txt")

# hm
"""
test_path = "../hateful_memes_data/final_preprocessed_test.pkl"
ground_truth_labels_test = pd.read_pickle(test_path)["label"]
probas_1 = np.loadtxt("../final_runs/hm/hm_early_fusion test 7146/seed1/1_token_sentiment_kg_vit/test_pred_probas_epoch12.txt")
probas_2 = np.loadtxt("../final_runs/hm/hm_early_fusion test 7146/seed2/2_token_sentiment_kg_vit/test_pred_probas_epoch12.txt")
probas_3 = np.loadtxt("../final_runs/hm/hm_early_fusion test 7146/seed3/3_token_sentiment_kg_vit/test_pred_probas_epoch12.txt")
"""
roc_auc_1 = roc_auc_score(y_true=ground_truth_labels_test, y_score=probas_1)
roc_auc_2 = roc_auc_score(y_true=ground_truth_labels_test, y_score=probas_2)
roc_auc_3 = roc_auc_score(y_true=ground_truth_labels_test, y_score=probas_3)
roc_auc = (roc_auc_3 + roc_auc_2 + roc_auc_1) / 3

precision_1 = precision_score(y_true=ground_truth_labels_test, y_pred=np.round(probas_1))
precision_2 = precision_score(y_true=ground_truth_labels_test, y_pred=np.round(probas_2))
precision_3 = precision_score(y_true=ground_truth_labels_test, y_pred=np.round(probas_3))
precision = (precision_3 + precision_2 + precision_1) / 3

recall_1 = recall_score(y_true=ground_truth_labels_test, y_pred=np.round(probas_1))
recall_2 = recall_score(y_true=ground_truth_labels_test, y_pred=np.round(probas_2))
recall_3 = recall_score(y_true=ground_truth_labels_test, y_pred=np.round(probas_3))
recall = (recall_3 +recall_2 + recall_1) / 3
print(roc_auc, recall, precision)