import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import pandas as pd
from transformers import ViTConfig
import blocks
import os
import random


class Parameters:
    def __init__(self,
                 device: str,
                 adam_weight_decay: float,
                 num_encoders: int,
                 num_heads: int,
                 embedding_size: int,
                 batch_size: int,
                 num_epochs: int,
                 lr_gamma: float,
                 lr: float,
                 forward_expansion: int,
                 mha_dropout: float,
                 mlp_dropout: float,
                 encoder_dropout: float,
                 token_seq_len: int,
                 roi_pool_size: int,
                 objects_seq_len: int,
                 image_size_for_resnet: int,
                 use_embedding_layer_norm: bool,
                 use_segment_embedder: bool,
                 initialization_checkpoint: str,
                 grad_clip_val: float,
                 embedding_dropout_dict: dict,
                 embedding_usage_dict: dict,
                 save_all: bool
                 ):
        self.vit_config = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vocab_size = blocks.TokenizerHandler().vocab_size
        self.initialization_checkpoint = initialization_checkpoint
        self.grad_clip_val = grad_clip_val
        self.image_size_for_resnet = image_size_for_resnet
        self.device = device
        self.adam_weight_decay = adam_weight_decay
        self.num_encoders = num_encoders
        self.num_heads = num_heads
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr_gamma = lr_gamma
        self.lr = lr
        self.forward_expansion = forward_expansion
        self.mha_dropout = mha_dropout
        self.mlp_dropout = mlp_dropout
        self.encoder_dropout = encoder_dropout
        self.use_embedding_layer_norm = use_embedding_layer_norm
        self.use_segment_embedder = use_segment_embedder
        self.embedding_dropout_dict = embedding_dropout_dict
        self.embedding_usage_dict = embedding_usage_dict
        self.token_seq_len = token_seq_len
        self.roi_pool_size = roi_pool_size
        self.objects_seq_len = objects_seq_len
        self.save_all = save_all


def make_reproducible(seed: int = 1):
    """
    ensures reproducibility over multiple script runs and after restarting the local machine
    """
    # cuda
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # prints
    torch.set_printoptions(sci_mode=False)
    torch.set_printoptions(threshold=100_000)
    np.set_printoptions(suppress=True)
    print("reproducibility with seed", seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
