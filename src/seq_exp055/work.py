import os
import gc
import re
import sys
import copy
import time
import glob
import math
import random
from turtle import forward
import psutil
import shutil
import typing
import numbers
import warnings
import argparse
import subprocess
from contextlib import contextmanager
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from abc import ABCMeta, abstractmethod

import pickle
import json
import yaml

from tqdm.auto import tqdm

import joblib

from box import Box
from omegaconf import DictConfig, OmegaConf
import hydra


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
import torchtext
import torchmetrics
import pytorch_lightning as pl

from adabelief_pytorch import AdaBelief

from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedKFold,
    StratifiedGroupKFold,
)

from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.impute import KNNImputer, SimpleImputer

from scipy.special import expit as sigmoid

import cupy
import cudf
import cuml

import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# utils
from utils import *

tqdm.pandas()


# ====================================================
# config
# ====================================================
DEBUG = False

SEED = 42
N_SPLITS = 30

INPUT_DIR = "../input/amex-default-prediction"
INPUT_PICKLE_DIR = "../input/amex-pickle"
INPUT_INTEGER_PICKLE_DIR = "../input/amex-integer-pickle"
INPUT_DATA_SEQUENTIAL_DIR = "../input/amex-data-sequeitial"


CAT_FEATURES = [
    "B_30",
    "B_38",
    "D_114",
    "D_116",
    "D_117",
    "D_120",
    "D_126",
    "D_63",
    "D_64",
    "D_66",
    "D_68",
]
D_FEATURES = [
    "D_39",
    "D_41",
    "D_42",
    "D_43",
    "D_44",
    "D_45",
    "D_46",
    "D_47",
    "D_48",
    "D_49",
    "D_50",
    "D_51",
    "D_52",
    "D_53",
    "D_54",
    "D_55",
    "D_56",
    "D_58",
    "D_59",
    "D_60",
    "D_61",
    "D_62",
    "D_63",
    "D_64",
    "D_65",
    "D_66",
    "D_68",
    "D_69",
    "D_70",
    "D_71",
    "D_72",
    "D_73",
    "D_74",
    "D_75",
    "D_76",
    "D_77",
    "D_78",
    "D_79",
    "D_80",
    "D_81",
    "D_82",
    "D_83",
    "D_84",
    "D_86",
    "D_87",
    "D_88",
    "D_89",
    "D_91",
    "D_92",
    "D_93",
    "D_94",
    "D_96",
    "D_102",
    "D_103",
    "D_104",
    "D_105",
    "D_106",
    "D_107",
    "D_108",
    "D_109",
    "D_110",
    "D_111",
    "D_112",
    "D_113",
    "D_114",
    "D_115",
    "D_116",
    "D_117",
    "D_118",
    "D_119",
    "D_120",
    "D_121",
    "D_122",
    "D_123",
    "D_124",
    "D_125",
    "D_126",
    "D_127",
    "D_128",
    "D_129",
    "D_130",
    "D_131",
    "D_132",
    "D_133",
    "D_134",
    "D_135",
    "D_136",
    "D_137",
    "D_138",
    "D_139",
    "D_140",
    "D_141",
    "D_142",
    "D_143",
    "D_144",
    "D_145",
]
S_FEATURES = [
    "S_2",
    "S_3",
    "S_5",
    "S_6",
    "S_7",
    "S_8",
    "S_9",
    "S_11",
    "S_12",
    "S_13",
    "S_15",
    "S_16",
    "S_17",
    "S_18",
    "S_19",
    "S_20",
    "S_22",
    "S_23",
    "S_24",
    "S_25",
    "S_26",
    "S_27",
]
P_FEATURES = [
    "P_2",
    "P_3",
    "P_4",
]
B_FEATURES = [
    "B_1",
    "B_2",
    "B_3",
    "B_4",
    "B_5",
    "B_6",
    "B_7",
    "B_8",
    "B_9",
    "B_10",
    "B_11",
    "B_12",
    "B_13",
    "B_14",
    "B_15",
    "B_16",
    "B_17",
    "B_18",
    "B_19",
    "B_20",
    "B_21",
    "B_22",
    "B_23",
    "B_24",
    "B_25",
    "B_26",
    "B_27",
    "B_28",
    "B_29",
    "B_30",
    "B_31",
    "B_32",
    "B_33",
    "B_36",
    "B_37",
    "B_38",
    "B_39",
    "B_40",
    "B_41",
    "B_42",
]
R_FEATURES = [
    "R_1",
    "R_2",
    "R_3",
    "R_4",
    "R_5",
    "R_6",
    "R_7",
    "R_8",
    "R_9",
    "R_10",
    "R_11",
    "R_12",
    "R_13",
    "R_14",
    "R_15",
    "R_16",
    "R_17",
    "R_18",
    "R_19",
    "R_20",
    "R_21",
    "R_22",
    "R_23",
    "R_24",
    "R_25",
    "R_26",
    "R_27",
    "R_28",
]


PARAMS = {
    "model": {
        "label_smoothing": 0.10,
        "encoder": {
            ### single encoder
            "type": "TransformerEncoder",  # {TransformerEncoder, GRUEncoder, LSTMEncoder, CNNEncoder}
            "params": {
                ##### Transformer
                "num_layers": 4,  # Transformer
                "dropout": 0.25,  # Transformer
                "d_model": 512,  # Transformer
                "nhead": 8,  # Transformer
                ##### LSTM, GRU
                # "num_blocks": 4,  # LSTM, GRU
                # "dropout_list": [0.00, 0.00, 0.00, 0.00],  # LSTM, GRU, # len == encoder_num_blocks
                # "hidden_size_list": [1024, 512, 256, 128],  # LSTM, GRU, # len == encoder_num_blocks
                # "num_layers_list": [1, 1, 1, 1],  # LSTM, GRU, len == encoder_num_blocks
                # "bidirectional": False,  # LSTM, GRU
                ##### CNN
                # "num_blocks": 4,  # CNN
                # "dropout_list": [0.10, 0.10, 0.10, 0.10],  # CNN, # len == encoder_num_blocks
                # "hidden_size_list": [1024, 512, 256, 128],  # CNN, # len == encoder_num_blocks
                # "kernel_size_list": [3, 3, 3, 3],  # CNN, # len == encoder_num_blocks
            },
            ### concat encoder
            # "type": "ConcatEncoder",
            # "params": [
            #     {
            #         "type": "TransformerEncoder",
            #         "params": {
            #             "num_layers": 8,  # Transformer
            #             "dropout": 0.25,  # Transformer
            #             "d_model": 512,  # Transformer
            #             "nhead": 8,  # Transformer
            #         },
            #     },
            #     {
            #         "type": "LSTMEncoder",
            #         "params": {
            #             "num_blocks": 1,  # LSTM, GRU
            #             "dropout_list": [0.00],  # LSTM, GRU, # len == encoder_num_blocks
            #             "hidden_size_list": [512],  # LSTM, GRU, # len == encoder_num_blocks
            #             "num_layers_list": [8],  # LSTM, GRU, len == encoder_num_blocks
            #             "bidirectional": False,  # LSTM, GRU
            #         },
            #     },
            #     {
            #         "type": "GRUEncoder",
            #         "params": {
            #             "num_blocks": 1,  # LSTM, GRU
            #             "dropout_list": [0.00],  # LSTM, GRU, # len == encoder_num_blocks
            #             "hidden_size_list": [512],  # LSTM, GRU, # len == encoder_num_blocks
            #             "num_layers_list": [8],  # LSTM, GRU, len == encoder_num_blocks
            #             "bidirectional": False,  # LSTM, GRU
            #         },
            #     },
            # ],
        },
        "head": {
            "type": "MultiSampleDropoutHead",  # {SimpleHead, MultiSampleDropoutHead, AttentionHead, MeanMaxPoolingHead, LSTMHead, GRUHead, CNNHead}
            "params": {
                "dropout": 0.50,  # SimpleHead, MultiSampleDropoutHead, LSTMHead, GRUHead
                "num_layers": 5,  # MultiSampleDropoutHead
                # "hidden_size": 256,  # CNNHead, AttentionHead
                # "kernel_size": 3,  # CNNHead
            },
        },
    },
    "trainer": {
        "max_epochs": 30,
        "benchmark": False,
        "deterministic": True,
        "num_sanity_val_steps": 0,
        "accumulate_grad_batches": 4,
        "precision": 16,
        "gpus": 1,
    },
    "mixup": {
        "use": False,
        "alpha": 0.5,
    },
    "dataloader": {
        "train": {
            "batch_size": 512,
            "shuffle": True,
            "drop_last": True,
            "pin_memory": True,
            "num_workers": 2,
        },
        "valid": {
            "batch_size": 512,
            "shuffle": False,
            "drop_last": False,
            "pin_memory": False,
            "num_workers": 0,
        },
        "test": {
            "batch_size": 512,
            "shuffle": False,
            "drop_last": False,
            "pin_memory": False,
            "num_workers": 0,
        },
    },
    "optimizer": {
        "name": "torch.optim.AdamW",
        "params": {
            "lr": 2.0e-05,
            "weight_decay": 0.00,
        },
        # "name": "AdaBelief",
        # "params": {
        #     "lr": 2.0e-05,
        #     "weight_decay": 0.00,
        #     "rectify": True,
        #     "weight_decouple": True,
        #     "eps": 1.0e-16,
        #     "print_change_log": False,
        # },
    },
    "scheduler": {
        # "name": "torch.optim.lr_scheduler.ReduceLROnPlateau",
        # "params": {
        #     "mode": "max",  # ReduceLROnPlateau
        #     "factor": 0.2,  # ReduceLROnPlateau
        #     "patience": 3,  # ReduceLROnPlateau
        #     "eps": 1.0e-06,  # ReduceLROnPlateau
        #     "verbose": True,
        # },
        "name": "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts",
        "params": {
            "T_0": 1,
            "T_mult": 2,
            "verbose": False,
        },
        # "name": "torch.optim.lr_scheduler.CosineAnnealingLR",
        # "params": {
        #     "T_max": 15,
        #     "verbose": False,
        # },
    },
    "loss": {
        "name": "nn.BCEWithLogitsLoss",
        "params": {},
        # "name": "FocalLoss",
        # "params": {
        #     "logits": True,
        #     "reduce": True,
        # },
    },
}


if DEBUG:
    PARAMS["dataloader"]["train"]["batch_size"] = 32
    PARAMS["dataloader"]["valid"]["batch_size"] = 32
    PARAMS["dataloader"]["test"]["batch_size"] = 32

# ====================================================
# plots
# ====================================================
def plot_target_distribution(ypred, ytrue, path):
    plt.figure()
    plt.hist(ytrue, alpha=0.5, bins=50)
    plt.hist(sigmoid(ypred), alpha=0.5, bins=50)
    plt.legend(["ytrue", "ypred"])
    plt.savefig(path)
    plt.close()


def plot_distribution(ypred, path):
    plt.figure()
    plt.hist(ypred, bins=50)
    plt.legend(["ypred"])
    plt.savefig(path)
    plt.close()


def plot_training_curve(train_history, valid_history, lr_history, filename):
    fig = plt.figure()
    ax1: plt.Axes = fig.add_subplot()
    ax2: plt.Axes = ax1.twinx()

    ax1.plot(range(len(train_history)), train_history, marker=".", color="skyblue", label="train")
    ax1.plot(range(len(valid_history)), valid_history, marker=".", color="orange", label="valid")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("metric")

    ax2.plot(range(len(lr_history)), lr_history, marker="x", linestyle="dashdot", color="gray", label="lr")
    ax2.set_ylabel("lr")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower right")

    plt.savefig(filename)
    plt.close()


# ====================================================
# metrics
# ====================================================
# https://www.kaggle.com/kyakovlev
# https://www.kaggle.com/competitions/amex-default-prediction/discussion/327534
def amex_metric(y_true, y_pred, return_details=False):

    labels = np.transpose(np.array([y_true, y_pred]))
    labels = labels[labels[:, 1].argsort()[::-1]]
    weights = np.where(labels[:, 0] == 0, 20, 1)
    cut_vals = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four = np.sum(cut_vals[:, 0]) / np.sum(labels[:, 0])

    gini = [0, 0]
    for i in [1, 0]:
        labels = np.transpose(np.array([y_true, y_pred]))
        labels = labels[labels[:, i].argsort()[::-1]]
        weight = np.where(labels[:, 0] == 0, 20, 1)
        weight_random = np.cumsum(weight / np.sum(weight))
        total_pos = np.sum(labels[:, 0] * weight)
        cum_pos_found = np.cumsum(labels[:, 0] * weight)
        lorentz = cum_pos_found / total_pos
        gini[i] = np.sum((lorentz - weight_random) * weight)

    if return_details:
        return 0.5 * (gini[1] / gini[0] + top_four), gini[1] / gini[0], top_four
    return 0.5 * (gini[1] / gini[0] + top_four)


class TorchAmexMetric(torchmetrics.Metric):
    full_state_update = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("preds", default=[])
        self.add_state("target", default=[])

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.preds.append(preds.detach().cpu())
        self.target.append(target.detach().cpu())

    def compute(self):
        preds = torch.cat(self.preds, dim=0).view(-1).detach().cpu().numpy()
        target = torch.cat(self.target, dim=0).view(-1).detach().cpu().numpy()

        score = amex_metric(target, preds)

        return torch.tensor(score)


def prepare_data(TYPE="train"):
    if TYPE == "train":
        train_ids = np.load(os.path.join(INPUT_DATA_SEQUENTIAL_DIR, "train_ids.npy"), allow_pickle=True)
        train = np.load(os.path.join(INPUT_DATA_SEQUENTIAL_DIR, "train.npy"), allow_pickle=True)
        train_labels = np.load(os.path.join(INPUT_DATA_SEQUENTIAL_DIR, "train_labels.npy"), allow_pickle=True)
        np.save("train_ids.npy", train_ids)
        np.save("train_labels.npy", train_labels)
        np.save("train.npy", train)

        num_features = joblib.load(os.path.join(INPUT_DATA_SEQUENTIAL_DIR, "num_features.pkl"))
        cat_features = joblib.load(os.path.join(INPUT_DATA_SEQUENTIAL_DIR, "cat_features.pkl"))
        joblib.dump(num_features, "num_features.pkl")
        joblib.dump(cat_features, "cat_features.pkl")

        if DEBUG:
            train_ids = train_ids[:1000]
            train = train[:1000]
            train_labels = train_labels[:1000]
        gc.collect()

        return train_ids, train, train_labels, num_features, cat_features

    elif TYPE == "public":
        test_ids = np.load(os.path.join(INPUT_DATA_SEQUENTIAL_DIR, "public_ids.npy"), allow_pickle=True)
        test = np.load(os.path.join(INPUT_DATA_SEQUENTIAL_DIR, "public.npy"), allow_pickle=True)
        np.save("public_ids.npy", test_ids)
        np.save("public.npy", test)

        num_features = joblib.load(os.path.join(INPUT_DATA_SEQUENTIAL_DIR, "num_features.pkl"))
        cat_features = joblib.load(os.path.join(INPUT_DATA_SEQUENTIAL_DIR, "cat_features.pkl"))
        joblib.dump(num_features, "num_features.pkl")
        joblib.dump(cat_features, "cat_features.pkl")

        if DEBUG:
            test_ids = test_ids[:1000]
            test = test[:1000]
        gc.collect()

        return test_ids, test, num_features, cat_features

    elif TYPE == "private":
        test_ids = np.load(os.path.join(INPUT_DATA_SEQUENTIAL_DIR, "private_ids.npy"), allow_pickle=True)
        test = np.load(os.path.join(INPUT_DATA_SEQUENTIAL_DIR, "private.npy"), allow_pickle=True)
        np.save("private_ids.npy", test_ids)
        np.save("private.npy", test)

        num_features = joblib.load(os.path.join(INPUT_DATA_SEQUENTIAL_DIR, "num_features.pkl"))
        cat_features = joblib.load(os.path.join(INPUT_DATA_SEQUENTIAL_DIR, "cat_features.pkl"))
        joblib.dump(num_features, "num_features.pkl")
        joblib.dump(cat_features, "cat_features.pkl")

        if DEBUG:
            test_ids = test_ids[:1000]
            test = test[:1000]
        gc.collect()

        return test_ids, test, num_features, cat_features

    raise NotImplementedError("TYPE must be one of (train, public, private)")


# ====================================================
# dataset, dataloader
# ====================================================
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, labels=None) -> None:
        super().__init__()
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float)

        if self.labels is None:
            return x, torch.tensor(0)

        y = torch.tensor(self.labels[idx], dtype=torch.float)
        return x, y


def get_dataloader(dataset, type_):
    if PARAMS["mixup"]["use"] and type_ == "train":
        print(f"[INFO] use mixup to {type_} dataloader")
        collate_fn = MixupCollate(PARAMS["mixup"]["alpha"])
        return torch.utils.data.DataLoader(dataset, collate_fn=collate_fn, **PARAMS["dataloader"][type_])

    return torch.utils.data.DataLoader(dataset, **PARAMS["dataloader"][type_])


class MixupCollate:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, batch):
        assert len(batch) % 2 == 0, "Batch size should be even when using this"
        x, y = list(zip(*batch))
        lam = torch.tensor(np.random.beta(self.alpha, self.alpha))

        x = torch.stack(x)
        x = x * lam + x.flip(0) * (1.0 - lam)

        y = torch.stack(y)
        y = y * lam + y.flip(0) * (1.0 - lam)

        return x, y


# ====================================================
# optimizer, scheduler
# ====================================================
def get_optimizer(name, model_parameters, params):
    return eval(name)(model_parameters, **params)


def get_scheduler(name, optimizer, params):
    return eval(name)(optimizer, **params)


# ====================================================
# loss
# ====================================================


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


def get_loss(name, params):
    return eval(name)(**params)


# ====================================================
# model
# ====================================================
class Model(pl.LightningModule):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = eval(PARAMS["model"]["encoder"]["type"])(
            input_dim=input_dim,
            params=PARAMS["model"]["encoder"]["params"],
        )
        self.head = eval(PARAMS["model"]["head"]["type"])(
            in_features=self.encoder.output_dim,
            out_features=1,
            **PARAMS["model"]["head"]["params"],
        )

        self.label_smoothing = PARAMS["model"]["label_smoothing"]
        self.criterion = get_loss(PARAMS["loss"]["name"], PARAMS["loss"]["params"])

        self.optimizer = get_optimizer(
            PARAMS["optimizer"]["name"],
            model_parameters=self.parameters(),
            params=PARAMS["optimizer"]["params"],
        )
        self.scheduler = get_scheduler(
            PARAMS["scheduler"]["name"],
            optimizer=self.optimizer,
            params=PARAMS["scheduler"]["params"],
        )

        # metrics
        self.train_metric = TorchAmexMetric(compute_on_step=False)
        self.valid_metric = TorchAmexMetric(compute_on_step=False)
        # init model training histories
        self.history = {
            "train_metric": [],
            "valid_metric": [],
            "lr": [],
        }

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x.view(-1)

    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "monitor": "valid_metric",
                "interval": "epoch",
            },
        }

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        yhat = self(x)
        y_smooth = y.float() * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        loss = self.criterion(yhat, y_smooth)
        self.log(
            name="train_loss",
            value=loss.item(),
            prog_bar=True,
            logger=False,
            on_step=False,
            on_epoch=True,
            rank_zero_only=True,
        )
        self.train_metric(yhat, y.long())
        self.log(
            name="train_metric",
            value=self.train_metric,
            prog_bar=True,
            logger=False,
            on_step=False,
            on_epoch=True,
            rank_zero_only=True,
        )
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        yhat = self(x)
        loss = self.criterion(yhat, y)
        self.log(
            name="valid_loss",
            value=loss.item(),
            prog_bar=True,
            logger=False,
            on_step=False,
            on_epoch=True,
            rank_zero_only=True,
        )
        self.valid_metric(yhat, y.long())
        self.log(
            name="valid_metric",
            value=self.valid_metric,
            prog_bar=True,
            logger=False,
            on_step=False,
            on_epoch=True,
            rank_zero_only=True,
        )
        return loss

    def on_train_epoch_end(self) -> None:
        self.history["train_metric"].append(self.train_metric.compute().detach().cpu().numpy())
        self.history["lr"].append(self.optimizers(False).param_groups[0]["lr"])
        return super().on_train_epoch_end()

    def on_validation_epoch_end(self) -> None:
        self.history["valid_metric"].append(self.valid_metric.compute().detach().cpu().numpy())
        return super().on_validation_epoch_end()


class SimpleHead(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        dropout,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_features)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, last_hidden_state):
        x = self.layer_norm(last_hidden_state[:, -1, :])
        x = self.dropout(x)
        output = self.linear(x)
        return output


class MultiSampleDropoutHead(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        dropout,
        num_layers,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_features)
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])
        self.linears = nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(num_layers)])

    def forward(self, last_hidden_state):
        x = self.layer_norm(last_hidden_state[:, -1, :])
        output = torch.stack([regressor(dropout(x)) for regressor, dropout in zip(self.linears, self.dropouts)]).mean(axis=0)
        return output


class AttentionHead(nn.Module):
    def __init__(self, in_features, hidden_size, out_features):
        super(AttentionHead, self).__init__()
        self.W = nn.Linear(in_features, hidden_size)
        self.V = nn.Linear(hidden_size, out_features)

    def forward(self, last_hidden_state):
        x = last_hidden_state[:, -1, :]
        attention_scores = self.V(torch.tanh(self.W(x)))
        attention_scores = torch.softmax(attention_scores, dim=1)
        attentive_x = attention_scores * x
        attentive_x = attentive_x.sum(axis=1)
        return attentive_x


class MeanMaxPoolingHead(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features * 2, out_features)

    def forward(self, last_hidden_state):
        mean_pooling_embeddings = torch.mean(last_hidden_state, 1)
        max_pooling_embeddings, _ = torch.max(last_hidden_state, 1)
        mean_max_embeddings = torch.cat(
            (
                mean_pooling_embeddings,
                max_pooling_embeddings,
            ),
            1,
        )
        logits = self.linear(mean_max_embeddings)  # twice the hidden size

        return logits


class LSTMHead(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        dropout,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=in_features,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.layer_norm = nn.LayerNorm(in_features * 2)
        self.linear = nn.Linear(in_features * 2, out_features)

    def forward(self, last_hidden_state):
        x, _ = self.lstm(last_hidden_state, None)
        x = self.layer_norm(x[:, -1, :])
        output = self.linear(x)
        return output


class GRUHead(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        dropout,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=in_features,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.layer_norm = nn.LayerNorm(in_features * 2)
        self.linear = nn.Linear(in_features * 2, out_features)

    def forward(self, last_hidden_state):
        x, _ = self.gru(last_hidden_state, None)
        x = self.layer_norm(x[:, -1, :])
        output = self.linear(x)
        return output.squeeze(-1)


class CNNHead(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        hidden_size,
        kernel_size,
    ):
        super().__init__()
        self.cnn1 = nn.Conv1d(in_features, hidden_size, kernel_size=kernel_size, padding=1)
        self.cnn2 = nn.Conv1d(hidden_size, out_features, kernel_size=kernel_size, padding=1)
        self.prelu = nn.PReLU()

    def forward(self, last_hidden_state):
        x = last_hidden_state.permute(0, 2, 1)
        x = self.cnn1(x)
        x = self.prelu(x)
        x = self.cnn2(x)
        x, _ = torch.max(x, 2)
        return x


class CNNEncoder(torch.nn.Module):
    def __init__(self, input_dim, params):
        super().__init__()
        self.input_dim = input_dim
        self.encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=input_dim if i == 0 else params["hidden_size_list"][i - 1],
                        out_channels=params["hidden_size_list"][i],
                        kernel_size=params["kernel_size_list"][i],
                        padding="same",
                    ),
                    nn.BatchNorm1d(params["hidden_size_list"][i]),
                    nn.PReLU(),
                    nn.Dropout(params["dropout_list"][i]),
                )
                for i in range(params["num_blocks"])
            ]
        )

        self.output_dim = params["hidden_size_list"][-1]

    def forward(self, x):
        x = x.permute(0, 2, 1)
        for encoder in self.encoders:
            x = encoder(x)
        x = x.permute(0, 2, 1)
        return x


class GRUEncoder(torch.nn.Module):
    def __init__(self, input_dim, params):
        super().__init__()
        self.input_dim = input_dim
        self.encoders = nn.ModuleList(
            [
                nn.GRU(
                    input_size=input_dim if i == 0 else params["hidden_size_list"][i - 1],
                    hidden_size=params["hidden_size_list"][i],
                    num_layers=params["num_layers_list"][i],
                    dropout=params["dropout_list"][i],
                    bidirectional=params["bidirectional"],
                    batch_first=True,
                )
                for i in range(params["num_blocks"])
            ]
        )

        self.output_dim = params["hidden_size_list"][-1] * (2 if params["bidirectional"] else 1)

    def forward(self, x):
        for encoder in self.encoders:
            x, _ = encoder(x)
        return x


class LSTMEncoder(torch.nn.Module):
    def __init__(self, input_dim, params):
        super().__init__()
        self.input_dim = input_dim
        self.encoders = nn.ModuleList(
            [
                nn.LSTM(
                    input_size=input_dim if i == 0 else params["hidden_size_list"][i - 1],
                    hidden_size=params["hidden_size_list"][i],
                    num_layers=params["num_layers_list"][i],
                    dropout=params["dropout_list"][i],
                    bidirectional=params["bidirectional"],
                    batch_first=True,
                )
                for i in range(params["num_blocks"])
            ]
        )

        self.output_dim = params["hidden_size_list"][-1] * (2 if params["bidirectional"] else 1)

    def forward(self, x):
        for encoder in self.encoders:
            x, _ = encoder(x)
        return x


class TransformerEncoder(torch.nn.Module):
    def __init__(self, input_dim, params):
        super().__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, params["d_model"])

        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=params["d_model"],
                nhead=params["nhead"],
                dropout=params["dropout"],
                activation=F.gelu,
                batch_first=True,
            ),
            num_layers=params["num_layers"],
        )
        self.output_dim = params["d_model"]

    def forward(self, x):
        x = self.linear(x)
        x = self.encoder(x)
        return x


def get_emb(sin_inp):
    # https://github.com/tatp22/multidim-positional-encoding/blob/master/positional_encodings/torch_encodings.py
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding1D(nn.Module):
    # https://github.com/tatp22/multidim-positional-encoding/blob/master/positional_encodings/torch_encodings.py
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc


class Summer(nn.Module):
    # https://github.com/tatp22/multidim-positional-encoding/blob/master/positional_encodings/torch_encodings.py
    def __init__(self, penc):
        """
        :param model: The type of positional encoding to run the summer on.
        """
        super(Summer, self).__init__()
        self.penc = penc

    def forward(self, tensor):
        """
        :param tensor: A 3, 4 or 5d tensor that matches the model output size
        :return: Positional Encoding Matrix summed to the original tensor
        """
        penc = self.penc(tensor)
        assert tensor.size() == penc.size(), "The original tensor size {} and the positional encoding tensor size {} must match!".format(
            tensor.size, penc.size
        )
        return tensor + penc


class ConcatEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        params,
    ) -> None:
        super().__init__()

        self.encoders = nn.ModuleList([eval(param["type"])(input_dim, param["params"]) for param in params])

        self.output_dim = sum([encoder.output_dim for encoder in self.encoders])
        print(self.output_dim)

    def forward(self, x):
        output = torch.cat([encoder(x) for encoder in self.encoders], dim=-1)
        return output


# ====================================================
# train fold
# ====================================================
def train_fold(fold, X_train, y_train, X_valid, y_valid, num_features, cat_features):
    print("#" * 30, f"fold{fold}", "#" * 30)

    train_ds = Dataset(X_train, y_train)
    valid_ds = Dataset(X_valid, y_valid)

    print(f"X_train.shape={X_train.shape}, X_valid.shape={X_valid.shape}")

    train_dl = get_dataloader(train_ds, type_="train")
    valid_dl = get_dataloader(valid_ds, type_="valid")

    CHECKPOINT_NAME = f"fold{fold}_model_" "{epoch:03d}_{valid_metric:.4f}"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename=CHECKPOINT_NAME,
        dirpath=".",
        monitor="valid_metric",
        mode="max",
        save_top_k=1,
        save_last=False,
        save_weights_only=True,
    )

    trainer = pl.Trainer(
        logger=None,
        callbacks=[checkpoint_callback],
        **PARAMS["trainer"],
    )

    model = Model(input_dim=X_train.shape[-1])

    trainer.fit(
        model,
        train_dataloaders=train_dl,
        val_dataloaders=valid_dl,
    )
    model_path = checkpoint_callback.best_model_path
    print(f"best model path: {model_path}")

    del train_dl, valid_dl, trainer
    gc.collect()
    torch.cuda.empty_cache()
    pl.utilities.memory.garbage_collection_cuda()

    plot_training_curve(
        model.history["train_metric"],
        model.history["valid_metric"],
        model.history["lr"],
        filename=f"training_curve_fold{fold}.png",
    )

    print(f"model_path={model_path}")

    return model_path


# ====================================================
# inference
# ====================================================
def inference(customer_ids, X, model_path):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = Model.load_from_checkpoint(model_path, input_dim=X.shape[-1])
    model.freeze()
    model.eval()
    model.to(device)

    ds = Dataset(X, labels=None)
    dl = get_dataloader(ds, type_="test")

    predictions_list = []

    with torch.no_grad():
        for x, _ in tqdm(dl, desc="inference", total=len(dl)):
            x = x.to(device)
            logits = model(x)
            predictions_list.append(logits.detach().cpu().numpy())

    prediction = np.concatenate(predictions_list, axis=0)

    prediction_df = pd.DataFrame()
    prediction_df["customer_ID"] = customer_ids
    prediction_df["prediction"] = prediction

    return prediction_df


# ====================================================
# main
# ====================================================
def training_main():
    train_ids, train, train_labels, num_features, cat_features = prepare_data(TYPE="train")

    oof_df = pd.DataFrame()
    model_paths = []
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    for fold, (train_idx, valid_idx) in enumerate(skf.split(train_ids, train_labels)):
        train_customer_ids = train_ids[train_idx]
        valid_customer_ids = train_ids[valid_idx]
        X_train = train.view()[train_idx]
        X_valid = train.view()[valid_idx]
        y_train = train_labels[train_idx]
        y_valid = train_labels[valid_idx]

        model_path = train_fold(fold, X_train, y_train, X_valid, y_valid, num_features, cat_features)
        model_paths.append(model_path)

        oof = inference(valid_customer_ids, X_valid, model_path)
        oof_df = oof_df.append(oof, ignore_index=True)

    oof_df = oof_df.groupby("customer_ID")["prediction"].mean().reset_index().sort_values("customer_ID")
    oof_score, g, d = amex_metric(
        train_labels,
        oof_df["prediction"].to_numpy(),
        return_details=True,
    )
    print("#" * 10)
    print(f"oof score: {oof_score:.4f}")
    print(f"gini: {g:.4f}")
    print(f"default rate(4%): {d:.4f}")
    print("#" * 10)

    plot_target_distribution(oof_df["prediction"], train_labels, path="oof_target_distribution.png")
    plot_distribution(oof_df["prediction"], path="oof_distribution.png")

    oof_df.to_csv("oof.csv", index=False)

    return oof_score, g, d, model_paths


def inference_main(model_paths):

    ## public
    print("#" * 30, "public", "#" * 30)
    public_ids, public, num_features, cat_features = prepare_data(TYPE="public")

    public_df = pd.DataFrame()
    for model_path in model_paths:
        print("#" * 5, model_path)
        preds_df = inference(public_ids, public, model_path)
        public_df = public_df.append(preds_df, ignore_index=True)

    public_df = public_df.groupby("customer_ID")["prediction"].mean().reset_index().sort_values("customer_ID")
    plot_distribution(public_df["prediction"], path="public_distribution.png")

    del public_ids, public
    gc.collect()

    ## private
    print("#" * 30, "private", "#" * 30)
    private_ids, private, num_features, cat_features = prepare_data(TYPE="private")

    private_df = pd.DataFrame()
    for model_path in model_paths:
        print("#" * 5, model_path)
        preds_df = inference(private_ids, private, model_path)
        private_df = private_df.append(preds_df, ignore_index=True)

    private_df = private_df.groupby("customer_ID")["prediction"].mean().reset_index().sort_values("customer_ID")
    plot_distribution(private_df["prediction"], path="private_distribution.png")

    del private_ids, private
    gc.collect()

    ## make submission file
    sub_df = pd.concat([public_df, private_df], axis=0, ignore_index=True)
    sub_df.sort_values("customer_ID", inplace=True)
    sub_df.to_csv("submission.csv", index=False)


def main():
    seed_everything(SEED)
    oof_score, g, d, model_paths = training_main()
    inference_main(model_paths)

    return Box(
        {
            "params": PARAMS,
            "metrics": {
                "valid_score": oof_score,
                "valid_g": g,
                "valid_d": d,
                "public_score": np.nan,
                "private_score": np.nan,
            },
        }
    )


if __name__ == "__main__":
    os.chdir("../work")
    results = main()
    joblib.dump(results, "results.pkl")
