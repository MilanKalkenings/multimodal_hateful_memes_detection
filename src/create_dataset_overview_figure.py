import pandas as pd
from PIL import Image
import numpy as np


# misogyny
train_path = "../misogyny_data/final_preprocessed_train.csv"
dev_path = "../misogyny_data/final_preprocessed_dev.csv"
test_path = "../misogyny_data/final_preprocessed_test.csv"
dataset_path_body = "../misogyny_data/img/"

# hm
#train_path = "../hateful_memes_data/final_preprocessed_train.pkl"
#dev_path = "../hateful_memes_data/final_preprocessed_dev.pkl"
#test_path = "../hateful_memes_data/final_preprocessed_test.pkl"
#dataset_path_body = "../hateful_memes_data/"

train_df = pd.read_pickle(train_path)
dev_df = pd.read_pickle(dev_path)
test_df = pd.read_pickle(test_path)
total_df = pd.concat([train_df, dev_df, test_df])

labels = []
text_lens = []
num_objects = []
num_associatoins = []
image_size_1 = []
image_size_2 = []

for i, row in total_df.iterrows():

    label = row["label"]
    labels.append(label)

    text = row["text"]
    text_lens.append(len(text))

    object_attention_mask = row["rcnn_attention_mask"]
    num_objects.append(int(object_attention_mask.sum()))

    association_attention_mask = row["association_attention_mask"]
    num_associatoins.append(int(association_attention_mask.sum()))

    original_image_path = dataset_path_body + row["img"]
    image_raw = Image.open(original_image_path).convert("RGB")
    size = image_raw.size
    image_size_1.append(size[0])
    image_size_2.append(size[1])
    if (i % 1000 == 0) and (i != 0):
        print(i)

print("observations:", len(labels))
print("hateful:", np.array(labels).sum())

print("text lens:")
print(np.array(text_lens).mean(), np.array(text_lens).std())

print("number of objects:")
print(np.unique(np.array(num_objects), return_counts=True)[1][0])
print(np.array(num_objects).mean(), np.array(num_objects).std())

print("number of associations:")
print(np.unique(np.array(num_associatoins), return_counts=True)[1][0])
print(np.array(num_associatoins).mean(), np.array(num_associatoins).std())

print("sizes:")
print(np.array(image_size_1).mean(), np.array(image_size_1).std())
print(np.array(image_size_2).mean(), np.array(image_size_2).std())
