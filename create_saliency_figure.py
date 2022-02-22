"""
import model_ablation
import pickle
from model import Model
import torch
import pandas as pd
from data_handling import ClsDataset
import numpy as np
from reproducibility import make_reproducible

make_reproducible(seed=1)

# misogyny:
test_path = "../misogyny_data/final_preprocessed_test.csv"
parameters_path = "../final_runs/standard_parameters.pkl"
model_path = "../final_runs/hm/cnn_on_hm/seed1/1_cnn/model_epoch_3.pkl"
# interesting observations. [292, 313, 317, 372, 549, 632, 635, 290, 295, 741]


# hm
#test_path = "../hateful_memes_data/preprocessed_test_manual_split_v2.pkl"
#parameters_path = "../final_runs/hm/pure_resnet_checkpoint/parameters.pkl"
#model_path = "../final_runs/hm/pure_resnet_checkpoint/model_epoch_1.pkl"
# interesting observations: 100, 611

# load the pure resnet model
with open(parameters_path, "rb") as f:
    p = pickle.load(f)
p.embedding_usage_dict = {"token": False,
                          "roi": False,
                          "sentiment": False,
                          "kg": False,
                          "vit": False,
                          "cnn": True}
model = Model(p=p)
model_state_dict = torch.load(model_path)
model.load_state_dict(state_dict=model_state_dict)
model.to(p.device)

# load the dataset
df_test = pd.read_pickle(test_path)

# plot
for obs_id in [292, 313, 372, 741]:
    df_test_starting_at_obs = df_test.iloc[obs_id:, :]
    image_path = df_test_starting_at_obs.loc[obs_id, ["image_path"]][0]
    df_test_starting_at_obs.index = range(len(df_test_starting_at_obs))

    test_dataset = ClsDataset(df=df_test_starting_at_obs)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              sampler=torch.utils.data.SequentialSampler(data_source=test_dataset))
    obs = next(iter(test_loader))
    model_ablation.plot_resnet_saliency_map(obs=obs, model=model, image_path=image_path, plot_name=str(obs_id))
"""