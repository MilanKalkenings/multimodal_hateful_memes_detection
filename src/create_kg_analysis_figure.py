import model_ablation
from model import Model
import torch
import pickle
import pandas as pd
from data_handling import ClsDataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.nn.modules.upsampling import Upsample
from torchvision import transforms
from model_ablation import analyze_kg


# misogyny
"""
test_path = "../misogyny_data/final_preprocessed_test.csv"
probas_path_seed1 = "../final_runs/mis/mis_early_fusion test_8421/seed1/1_token_roi_kg/test_pred_probas_epoch2.txt"
probas_path_seed2 = "../final_runs/mis/mis_early_fusion test_8421/seed2/2_token_roi_kg/test_pred_probas_epoch2.txt"
probas_path_seed3 = "../final_runs/mis/mis_early_fusion test_8421/seed3/3_token_roi_kg/test_pred_probas_epoch2.txt"
"""

# hm
test_path = "../hateful_memes_data/final_preprocessed_test.pkl"
probas_path_seed1 = "../final_runs/hm/hm_early_fusion test 7146/seed1/1_token_sentiment_kg_vit/test_pred_probas_epoch12.txt"
probas_path_seed2 = "../final_runs/hm/hm_early_fusion test 7146/seed2/2_token_sentiment_kg_vit/test_pred_probas_epoch12.txt"
probas_path_seed3 = "../final_runs/hm/hm_early_fusion test 7146/seed3/3_token_sentiment_kg_vit/test_pred_probas_epoch12.txt"


# load the data
probas1 = np.loadtxt(probas_path_seed1)
probas2 = np.loadtxt(probas_path_seed2)
probas3 = np.loadtxt(probas_path_seed3)
df_test = pd.read_pickle(test_path)


meme_nums, accs1 = analyze_kg(test_df=df_test, test_probas=probas1)
_, accs2 = analyze_kg(test_df=df_test, test_probas=probas2)
_, accs3 = analyze_kg(test_df=df_test, test_probas=probas3)

acc_ass_label1 = (accs1[0] + accs2[0] + accs3[0]) / 3
acc_ass_label0 = (accs1[1] + accs2[1] + accs3[1]) / 3
acc_no_ass_label1 = (accs1[2] + accs2[2] + accs3[2]) / 3
acc_no_ass_label0 = (accs1[3] + accs2[3] + accs3[3]) / 3

print(meme_nums[0], "ass, hateful acc:", acc_ass_label1)
print(meme_nums[1], "ass, peace acc:", acc_ass_label0)
print(meme_nums[2], "without ass, hateful acc:", acc_no_ass_label1)
print(meme_nums[3], "without ass, peace acc:", acc_no_ass_label0)