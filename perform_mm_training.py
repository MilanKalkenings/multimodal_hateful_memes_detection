from data_handling import MMDataset

import os


import pandas as pd
import torch
import winsound
from torch.utils.data import DataLoader, SequentialSampler
from model import Model, train_eval_loop
from reproducibility import Parameters, make_reproducible, seed_worker

# import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
print("setting cublas workspace config")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
make_reproducible()

p = Parameters(device="cuda",
               adam_weight_decay=0.1,
               num_encoders=2,  # best 2
               num_heads=4,  # best 4
               embedding_size=256,
               batch_size=180,  # normal training: 90 but here: 200 for hm;
               num_epochs=50,  # for hm: 20 but only first 50 considered
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
               initialization_checkpoint="../final_runs/random_init.pkl",#"../final_runs/cross_pretraining/seed1/1_token_roi_kg/model_epoch_15.pkl", #"../final_runs/random_init.pkl", TODO
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
                                     "cnn": False},
               save_all=True
               )

########################################################################################################################
# hm data with train test split
print("using hm data")
#dev_path = "../hateful_memes_data/final_preprocessed_dev.pkl"
#test_path = "../hateful_memes_data/final_preprocessed_test.pkl"

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
#print("using misogyny data")
dev_path = "../misogyny_data/final_preprocessed_dev.csv"
test_path = "../misogyny_data/final_preprocessed_test.csv"

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
# load datasets
print("loading data")
train_path_mis = "../misogyny_data/final_preprocessed_train.csv"
train_path_hm = "../hateful_memes_data/final_preprocessed_train.pkl"
df_train_mis = pd.read_pickle(train_path_mis)
df_train_hm = pd.read_pickle(train_path_hm)
df_train = pd.concat([df_train_mis, df_train_hm], axis=0)
df_train.index = range(len(df_train))
df_train = df_train#.head(90)
train_dataset = MMDataset(df_train)

df_dev = pd.read_pickle(dev_path)#.head(90)
dev_dataset = MMDataset(df_dev)

df_test = pd.read_pickle(test_path).head(p.batch_size)  # not used further, so ignorable
test_dataset = MMDataset(df_test)

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
    {"token": True,
     "roi": True,
     "sentiment": False,
     "kg": True,
     "vit": False,
     "cnn": False},
]

for seed in [1]:  # normal 1,2,3; but mm_pretraining is only performed once
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