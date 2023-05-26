import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.nn.modules.upsampling import Upsample
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import ViTFeatureExtractor

import blocks
import object_detection
from sentiment_engineering import SentimentEngineer


class LeaderBoardDataset(Dataset):
    """
    is a map-style torch dataset, i.e.:
    - implements __getitem__() and __len__() form torch.utils.data.Dataset
    - maps indices to data samples/observations
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.resize = transforms.CenterCrop(size=128)
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item: int):
        # todo refactor row names, save all preprocessed dfs in one dir
        row = self.df.loc[item, :]
        vit_feature = row["vit_features"]
        bert_attention_mask = row["bert_attention_mask"]
        bert_input_ids = row["bert_input_ids"]
        rcnn_detected_objects = row["rcnn_detected_objects"]
        rcnn_attention_mask = row["rcnn_attention_mask"]
        rcnn_rois = row["rcnn_rois"]
        text_sentiment = row["sentiment"]
        associations = row["associations"]
        association_attention_mask = row["association_attention_mask"]
        association_positions = row["association_positions"]
        image_tensor = row["image_tensor"]

        return vit_feature, bert_input_ids, bert_attention_mask, None, image_tensor, rcnn_detected_objects, rcnn_attention_mask, rcnn_rois, text_sentiment, associations, association_attention_mask, association_positions


class LeaderBoardDataPreprocessor:
    def __init__(self, bert_seq_len: int, faster_rcnn_seq_len: int, roi_pool_size: int, image_size: int):
        tokenizer_handler = blocks.TokenizerHandler()
        self.bert_tokenizer = tokenizer_handler.tokenizer
        self.vit_feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.bert_seq_len = bert_seq_len
        self.faster_rcnn = object_detection.FRCNNDetector(device="cuda", num_positions=faster_rcnn_seq_len,
                                                          roi_pool_size=roi_pool_size)
        self.sentiment_engineer = SentimentEngineer()
        self.kg = blocks.KnowledgeGraph()
        self.kg_ids = tokenizer_handler.get_kg_ids()
        self.kg_association_ids = tokenizer_handler.get_kg_associaton_ids()
        self.image_up_or_down_sampler = Upsample(size=image_size, mode="nearest")
        self.image_to_tensor = transforms.ToTensor()

    @staticmethod
    def combine_image_path(relative_path: str, path_body: str):
        return path_body + relative_path

    def extract_image_tensor(self, image_path: str):
        image_raw = Image.open(image_path).convert("RGB")
        image_tensor = self.image_to_tensor(image_raw)
        image_tensor = torch.unsqueeze(image_tensor, dim=0)
        image_upsampled = self.image_up_or_down_sampler(image_tensor).squeeze()
        return image_upsampled

    def extract_kg_features(self, token_input_ids: torch.Tensor):
        """
        yet fixed to 16

        :param token_input_ids:
        :return:
        """
        association_sequence = torch.ones(16) * self.bert_tokenizer.pad_token_id  # first fill with pad tokens
        attention_mask = torch.zeros(16)
        original_positions = torch.zeros(16)
        last_filled = -1
        for i, id in enumerate(token_input_ids):
            associations = []
            for ass in self.kg_association_ids:
                if id in ass:
                    associations.append(ass[0])
                    associations.append(ass[1])
            associations = list(set(associations))
            if id in associations:
                associations.remove(id)  # remove itself from its associations
            for ass in associations:
                if last_filled != 15:
                    association_sequence[last_filled + 1] = ass
                    attention_mask[last_filled + 1] = 1
                    original_positions[last_filled + 1] = i
                    last_filled += 1
                else:
                    break
            if last_filled == 15:
                break
        return association_sequence, attention_mask, original_positions

    def extract_vit_features(self, image_path: str):
        extractor = self.vit_feature_extractor
        image = Image.open(image_path).convert("RGB")
        pixel_values = extractor(image, return_tensors="pt", padding=True)["pixel_values"].squeeze()
        return pixel_values

    def extract_bert_features(self, text: str):
        tokenizer = self.bert_tokenizer
        bert_seq_len = self.bert_seq_len
        tokenizer_out = tokenizer(text, max_length=bert_seq_len, padding="max_length", truncation=True,
                                  return_tensors="pt")
        input_ids = tokenizer_out["input_ids"].squeeze()  # includes cls, sep, pads
        attention_mask = tokenizer_out["attention_mask"].squeeze()
        return (input_ids, attention_mask)

    def preprocess_and_save(self, data_file: str, target_file: str):
        df = pd.read_csv(data_file)

        # relative image path
        path_body = "../mami/img/"
        df["image_path"] = df["img"].apply(self.combine_image_path, path_body=path_body)
        print("image paths created")

        # sentiment
        df["sentiment"] = self.sentiment_engineer.extract_text_sentiments(df=df)
        print("sentiment features created")

        # rcnn
        faster_rcnn_cols = df["image_path"].apply(lambda row: pd.Series(self.faster_rcnn.extract_location_features(row),
                                                                        index=["detected_objects", "attention_mask",
                                                                               "rois"]))
        df["rcnn_detected_objects"] = faster_rcnn_cols["detected_objects"]
        df["rcnn_attention_mask"] = faster_rcnn_cols["attention_mask"]
        df["rcnn_rois"] = faster_rcnn_cols["rois"]
        print("rcnn features created")

        df["image_tensor"] = df["image_path"].apply(self.extract_image_tensor)
        print("image tensors created")

        # bert
        bert_cols = df["text"].apply(
            lambda row: pd.Series(self.extract_bert_features(row), index=["input_ids", "attention_mask"]))
        df["bert_input_ids"] = bert_cols.loc[:, ["input_ids"]]
        df["bert_attention_mask"] = bert_cols.loc[:, ["attention_mask"]]
        print("bert features created")

        # kg
        kg_cols = df["bert_input_ids"].apply(lambda row: pd.Series(self.extract_kg_features(row),
                                                                   index=["associations", "association_attention_mask",
                                                                          "association_positions"]))
        df["associations"] = kg_cols.loc[:, ["associations"]]
        df["association_attention_mask"] = kg_cols.loc[:, ["association_attention_mask"]]
        df["association_positions"] = kg_cols.loc[:, ["association_positions"]]
        print("kg features created")

        # vit
        df["vit_features"] = df["image_path"].apply(self.extract_vit_features)
        print("vit features created")

        df.to_pickle(target_file)
