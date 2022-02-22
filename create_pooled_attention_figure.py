import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.nn.modules.upsampling import Upsample
from torchvision import transforms

import model_ablation
from data_handling import ClsDataset
from model import Model
from reproducibility import make_reproducible

make_reproducible(seed=1)
parameters_path = "../final_runs/standard_parameters.pkl"

########################################################################################################################
#hm
# load the overall best model
with open(parameters_path, "rb") as f:
    p = pickle.load(f)
test_path = "../hateful_memes_data/final_preprocessed_test.pkl"
model_path = "../final_runs/hm/hm_early_fusion test 7146/seed1/1_token_sentiment_kg_vit/model_epoch_12.pkl"
p.embedding_usage_dict = {"token": True,
                          "roi": False,
                          "sentiment": True,
                          "kg": True,
                          "vit": True,
                          "cnn": False}
model = Model(p=p)
model_state_dict = torch.load(model_path)
model.load_state_dict(state_dict=model_state_dict)
model.to(p.device)


# load the dataset
df_test = pd.read_pickle(test_path)

# transformations
image_to_tensor = transforms.ToTensor()
image_up_or_down_sampler = Upsample(size=512, mode="nearest")

# load the observation
for obs_id in np.arange(start=10, stop=50):
    print(obs_id)
    df_test_starting_at_obs = df_test.iloc[obs_id:, :]
    df_test_starting_at_obs.index = range(len(df_test_starting_at_obs))
    image_path = df_test_starting_at_obs.loc[0, ["image_path"]][0]
    print(image_path)
    image_raw = Image.open(image_path).convert("RGB")
    image_tensor = image_to_tensor(image_raw)
    image_tensor = torch.unsqueeze(image_tensor, dim=0)
    image_upsampled = image_up_or_down_sampler(image_tensor).squeeze()

    test_dataset = ClsDataset(df=df_test_starting_at_obs)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              sampler=torch.utils.data.SequentialSampler(data_source=test_dataset))
    obs = next(iter(test_loader))

    # plot
    attention_scores = model_ablation.obs_attention_scores(obs=obs, model=model)
    model_ablation.plot_pooled_attention_scores_hm(attention_scores=attention_scores, plot_path=f"../ausarbeitung/figures/pooled_attention_hm/attention_scores{obs_id}.svg")
    plt.clf()  # clean cache
    plt.imshow(image_upsampled.permute(1, 2, 0), cmap='Greys_r', interpolation='nearest')
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(f"../ausarbeitung/figures/pooled_attention_hm//image{obs_id}")


########################################################################################################################
#mis
# load the overall best model
with open(parameters_path, "rb") as f:
    p = pickle.load(f)
test_path = "../misogyny_data/final_preprocessed_test.csv"
model_path = "../final_runs/mis/mis_early_fusion test_8421/seed1/1_token_roi_kg/model_epoch_2.pkl"
p.embedding_usage_dict = {"token": True,
                          "roi": True,
                          "sentiment": False,
                          "kg": True,
                          "vit": False,
                          "cnn": False}
model = Model(p=p)
model_state_dict = torch.load(model_path)
model.load_state_dict(state_dict=model_state_dict)
model.to(p.device)


# load the dataset
df_test = pd.read_pickle(test_path)

# transformations
image_to_tensor = transforms.ToTensor()
image_up_or_down_sampler = Upsample(size=512, mode="nearest")

# load the observation
for obs_id in np.arange(start=10, stop=50):
    print(obs_id)
    df_test_starting_at_obs = df_test.iloc[obs_id:, :]
    df_test_starting_at_obs.index = range(len(df_test_starting_at_obs))
    image_path = df_test_starting_at_obs.loc[0, ["image_path"]][0]
    print(image_path)
    image_raw = Image.open(image_path).convert("RGB")
    image_tensor = image_to_tensor(image_raw)
    image_tensor = torch.unsqueeze(image_tensor, dim=0)
    image_upsampled = image_up_or_down_sampler(image_tensor).squeeze()

    test_dataset = ClsDataset(df=df_test_starting_at_obs)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              sampler=torch.utils.data.SequentialSampler(data_source=test_dataset))
    obs = next(iter(test_loader))

    # plot
    attention_scores = model_ablation.obs_attention_scores(obs=obs, model=model)
    model_ablation.plot_pooled_attention_scores_mis(attention_scores=attention_scores, plot_path=f"../ausarbeitung/figures/pooled_attention_mis/attention_scores{obs_id}.svg")
    plt.clf()  # clean cache
    plt.imshow(image_upsampled.permute(1, 2, 0), cmap='Greys_r', interpolation='nearest')
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(f"../ausarbeitung/figures/pooled_attention_mis//image{obs_id}")
