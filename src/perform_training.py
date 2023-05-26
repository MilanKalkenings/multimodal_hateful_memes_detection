import os

import pandas as pd
import torch
import winsound
from torch.utils.data import DataLoader, SequentialSampler

from data_handling import ClsDataset
from model import Model, train_eval_loop
from reproducibility import Parameters, make_reproducible, seed_worker
import pickle

# import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
print("setting cublas workspace config")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
make_reproducible()

# inits:
# standard: "../final_runs/random_init.pkl"

# init: cross_mm_hm: "../final_runs/pretraining/mm_pretraining/1_token_sentiment_kg_vit/model_epoch_49.pkl"


########################################################################################################################
# cross_pretraining
# cross_mis trained on both training sets, evaluated on mis_dev
# cross_hm trained on both training sets, evaluated on hm_dev
# init: cross_mis: "../final_runs/pretraining/cross_pretraining/seed1/1_token_roi_kg/model_epoch_15.pkl"
# init: cross_hm: "../final_runs/pretraining/cross_pretraining/seed1/1_token_sentiment_kg_vit/model_epoch_8.pkl"

########################################################################################################################
# sep_pretraining
# sep_mis trained on both training sets, evaluated on both dev sets
# sep_hm trained on both training sets, evaluated on both dev sets
# init: sep_mis: "../final_runs/pretraining/sep_pretraining/seed1/1_token_roi_kg/model_epoch_11.pkl"
# init: sep_hm: "../final_runs/pretraining/sep_pretraining/seed1/1_token_sentiment_kg_vit/model_epoch_18.pkl"

########################################################################################################################
# mis_hm_pretraining
# init: "../final_runs/pretraining/mis_pretraining_for_mis_hm/seed1/1_token_sentiment_kg_vit/model_epoch_1.pkl"

########################################################################################################################
# hm_mis_pretraining
# init: "../final_runs/pretraining/hm_pretraining_for_hm_mis/seed1/1_token_roi_kg/model_epoch_16.pkl"

p = Parameters(device="cuda",
               adam_weight_decay=0.1,
               num_encoders=2,  # best 2
               num_heads=4,  # best 4
               embedding_size=256,
               batch_size=90,  # best 90
               num_epochs=20,  # normal: 20
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
                                     "sentiment": False,
                                     "kg": True,
                                     "vit": False,
                                     "cnn": False},  # later overwritten during training process
               save_all=True
               )

#file = open("../final_runs/standard_parameters.pkl", "wb")
#pickle.dump(p, file)
#file.close()
########################################################################################################################
# hm data with train test split
print("using hm data")
train_path = "../hateful_memes_data/final_preprocessed_train.pkl"
dev_path = "../hateful_memes_data/final_preprocessed_dev.pkl"
test_path = "../hateful_memes_data/final_preprocessed_test.pkl"

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
# print("using misogyny data")
# train_path = "../misogyny_data/final_preprocessed_train.csv"
# dev_path = "../misogyny_data/final_preprocessed_dev.csv"
# test_path = "../misogyny_data/final_preprocessed_test.csv"

# splitting
"""
df_mis = pd.read_csv("../misogyny_data/TRAINING/training.csv", sep="\t")
df_mis["label"] = df_mis["misogynous"]
df_mis["text"] = df_mis["Text Transcription"]
df_mis["img"] = df_mis["file_name"]
df_mis_train, df_mis_dev = train_test_split(df_mis, test_size=0.2)
df_mis_dev, df_mis_test = train_test_split(df_mis_dev, test_size=0.5)
print(len(df_mis_test), len(df_mis_dev), len(df_mis_train))
df_mis_train.to_csv("../misogyny_data/train.csv")
df_mis_dev.to_csv("../misogyny_data/dev.csv")
df_mis_test.to_csv("../misogyny_data/test.csv")
"""
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
# used once to create random initial checkpoint:
# print("creating random initiation checkpoint")
# model = Model(p=p).to(p.device)
# torch.save(model.state_dict(), "../final_runs/random_init.pkl")

# load datasets
print("loading data")
df_train = pd.read_pickle(train_path)  # .head(180)
train_dataset = ClsDataset(df_train)

df_dev = pd.read_pickle(dev_path)  # .head(180)
dev_dataset = ClsDataset(df_dev)

df_test = pd.read_pickle(test_path)
test_dataset = ClsDataset(df_test)

print("train size", len(df_train), "dev size", len(df_dev), "test size", len(df_test))
# create loaders
print("creating loaders")
# counter = Counter(df_train["label"])
# class_count = np.array([counter[0], counter[1]])
# class_weight = 1. / class_count
# sample_weight = torch.from_numpy(np.array([class_weight[t] for t in df_train["label"]]))
# train_sampler = WeightedRandomSampler(weights=sample_weight, num_samples=len(sample_weight))
# train_loader = DataLoader(dataset=train_dataset, batch_size=p.batch_size, sampler=train_sampler)

train_sampler_seq = SequentialSampler(data_source=train_dataset)
train_loader_seq = DataLoader(dataset=train_dataset, batch_size=p.batch_size, sampler=train_sampler_seq)

dev_sampler_seq = SequentialSampler(data_source=dev_dataset)
dev_loader_seq = DataLoader(dataset=dev_dataset, batch_size=p.batch_size, sampler=dev_sampler_seq)

test_sampler_seq = SequentialSampler(data_source=test_dataset)
test_loader_seq = DataLoader(dataset=test_dataset, batch_size=p.batch_size, sampler=test_sampler_seq)

usage_dicts = [
    {"token": False,
     "roi": False,
     "sentiment": False,
     "kg": False,
     "vit": False,
     "cnn": True},
]

for seed in [1, 2, 3]:  # standard: 1,2,3 (if not only one version run, i.e. for model checkpoint creation only "1")
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
        checkpoints_dir = f"../final_runs/{seed}{dir_name}"
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
