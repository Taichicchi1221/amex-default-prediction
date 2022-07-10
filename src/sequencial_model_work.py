import os
import gc
from pkgutil import get_data
import re
import sys
import copy
import time
import glob
import math
import random
import psutil
import shutil
import typing
import numbers
import warnings
import argparse
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

tqdm.pandas()


# ====================================================
# config
# ====================================================
DEBUG = False

SEED = 42
N_SPLITS = 5


INPUT_DIR = "../input/amex-default-prediction"
INPUT_PICKLE_DIR = "../input/amex-pickle"
INPUT_INTEGER_PICKLE_DIR = "../input/amex-integer-pickle"


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
    "dataloader": {
        "train": {
            "batch_size": 32,
            "shuffle": True,
            "drop_last": True,
            "pin_memory": True,
            "num_workers": 2,
        },
        "valid": {
            "batch_size": 32,
            "shuffle": False,
            "drop_last": False,
            "pin_memory": False,
            "num_workers": 2,
        },
        "test": {
            "batch_size": 32,
            "shuffle": False,
            "drop_last": False,
            "pin_memory": False,
            "num_workers": 2,
        },
    },
    "trainer": {
        "max_epochs": 10,
        "benchmark": False,
        "deterministic": True,
        "num_sanity_val_steps": 0,
        "accumulate_grad_batches": 1,
        "precision": 16,
        "gpus": 1,
    },
    "optimizer": {
        "cls": torch.optim.Adam,
        "params": {
            "lr": 1e-03,
        },
    },
    "scheduler": {
        "cls": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "params": {
            "mode": "max",  # ReduceLROnPlateau
            "factor": 0.2,  # ReduceLROnPlateau
            "patience": 2,  # ReduceLROnPlateau
            "eps": 1.0e-06,  # ReduceLROnPlateau
            "verbose": True,
        },
    },
    "loss": {
        "cls": nn.BCEWithLogitsLoss,
        "params": {},
    },
}


# ====================================================
# utils
# ====================================================
def memory_used_to_str():
    pid = os.getpid()
    processs = psutil.Process(pid)
    memory_use = processs.memory_info()[0] / 2.0**30
    return "ram memory gb :" + str(np.round(memory_use, 2))


def seed_everything(seed: int = 42, deterministic: bool = False):
    """Set seeds"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = deterministic  # type: ignore


# ====================================================
# plots
# ====================================================
def plot_distribution(ypred, ytrue, path):
    plt.figure()
    plt.hist(ytrue, alpha=0.5, bins=50)
    plt.hit(sigmoid(ypred), alpha=0.5, bins=50)
    plt.legend(["ytrue", "ypred"])
    plt.savefig(path)
    plt.close()


def plot_training_curve(train_history, valid_history, filename):
    plt.figure()
    legends = []
    plt.plot(range(len(train_history)), train_history, marker=".", color="skyblue")
    legends.append("train")
    plt.plot(range(len(valid_history)), valid_history, marker=".", color="orange")
    legends.append("valid")
    plt.legend(legends)
    plt.savefig(filename)
    plt.close()


# ====================================================
# metrics
# ====================================================
# https://www.kaggle.com/kyakovlev
# https://www.kaggle.com/competitions/amex-default-prediction/discussion/327534
def amex_metric(y_true, y_pred):

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


# ====================================================
# data processing
# ====================================================
def preprocess(df: pd.DataFrame):
    # customer_ID
    df["customer_ID"] = pd.Categorical(df["customer_ID"], ordered=True)
    df.set_index("customer_ID", inplace=True)

    # S_2
    df["S_2"] = pd.to_datetime(df["S_2"], format="%Y-%m-%d")


def make_features(train, test):
    train_customer_ids = list(train.index.unique())
    test_customer_ids = list(test.index.unique())

    cat_features = CAT_FEATURES.copy()

    df = pd.concat([train, test], axis=0)
    del train, test
    gc.collect()

    # date features
    cols = ["month", "weekday", "day"]
    date_df = pd.concat(
        [
            df["S_2"].dt.month.astype("int16"),
            df["S_2"].dt.weekday.astype("int16"),
            df["S_2"].dt.day.astype("int16"),
        ],
        axis=1,
    )
    date_df.columns = cols
    df = pd.concat([df, date_df], axis=1)
    cat_features.extend(cols)

    # drop columns
    df.drop(columns="S_2", inplace=True)

    num_features = [col for col in df.columns if col not in cat_features]

    # fillna
    imputer = SimpleImputer(strategy="mean")
    num_df = pd.DataFrame(imputer.fit_transform(df[num_features]), columns=num_features, index=df.index)

    # dummies
    cat_dummies_df = pd.get_dummies(df[cat_features].astype("str"))
    cat_features = list(cat_dummies_df.columns)

    # concat
    df = pd.concat([num_df, cat_dummies_df], axis=1)

    return df.loc[train_customer_ids], df.loc[test_customer_ids], num_features, cat_features


def prepare_data():
    train_df = pd.read_pickle(Path(INPUT_PICKLE_DIR, "train.pkl"))
    test_df = pd.read_pickle(Path(INPUT_PICKLE_DIR, "test.pkl"))
    train_labels = pd.read_csv(Path(INPUT_DIR, "train_labels.csv"))

    if DEBUG:
        train_sample_ids = pd.Series(train_df["customer_ID"].unique()).sample(1000)
        train_df = train_df.loc[train_df["customer_ID"].isin(train_sample_ids)].reset_index(drop=True)
        train_labels = train_labels.loc[train_labels["customer_ID"].isin(train_sample_ids)].reset_index(drop=True)
        test_sample_ids = pd.Series(test_df["customer_ID"].unique()).sample(1000)
        test_df = test_df.loc[test_df["customer_ID"].isin(test_sample_ids)].reset_index(drop=True)

    # preprocessing
    preprocess(train_df)
    preprocess(test_df)
    train_labels["customer_ID"] = pd.Categorical(train_labels["customer_ID"], ordered=True)

    # features
    train, test, num_features, cat_features = make_features(train_df, test_df)

    train.sort_index(inplace=True)
    test.sort_index(inplace=True)
    train_labels = train_labels.set_index("customer_ID").sort_index()

    print(f"shape of train: {len(train)}, shape of test: {len(test)}")

    return train, test, train_labels, num_features, cat_features


# ====================================================
# dataset, dataloader
# ====================================================
class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, labels=None) -> None:
        super().__init__()
        self.ids = df.index.unique()
        self.df = df
        self.labels = labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        x = self.df.loc[[self.ids[idx]]].to_numpy()
        l = len(x)

        x = torch.tensor(np.pad(x, ((13 - l, 0), (0, 0))), dtype=torch.float)

        if self.labels is None:
            return x

        y = torch.tensor(self.labels.loc[self.ids[idx]], dtype=torch.float)
        return x, y


def get_dataloader(dataset, type_):
    return torch.utils.data.DataLoader(dataset, **PARAMS["dataloader"][type_])


# ====================================================
# dataset, dataloader
# ====================================================
def get_optimizer(cls_, model_parameters, params):
    return cls_(model_parameters, **params)


def get_scheduler(cls_, optimizer, params):
    return cls_(optimizer, **params)


# ====================================================
# loss
# ====================================================
def get_loss(cls_, params):
    return cls_(**params)


# ====================================================
# model
# ====================================================
class Model(pl.LightningModule):
    def __init__(self, input_dim):
        super().__init__()
        self.model = BaseModel(input_dim)

        self.criterion = get_loss(PARAMS["loss"]["cls"], PARAMS["loss"]["params"])

        self.optimizer = get_optimizer(
            PARAMS["optimizer"]["cls"],
            model_parameters=self.model.parameters(),
            params=PARAMS["optimizer"]["params"],
        )
        self.scheduler = get_scheduler(
            PARAMS["scheduler"]["cls"],
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
        }

    def forward(self, x):
        return self.model(x)

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
        loss = self.criterion(yhat, y)
        self.train_metric(yhat, y)
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
        self.valid_metric(yhat, y.squeeze(-1).long())
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
        return super().on_train_epoch_end()

    def on_validation_epoch_end(self) -> None:
        self.history["valid_metric"].append(self.valid_metric.compute().detach().cpu().numpy())
        return super().on_validation_epoch_end()


class BaseModel(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=256,
            batch_first=True,
            bidirectional=True,
            num_layers=8,
        )
        self.hidden = nn.Sequential(
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
        )
        self.classifier = nn.Linear(32, 1)

    def forward(self, x):
        _, h = self.encoder(x)
        h = h.mean(0)
        h = self.hidden(h)
        return self.classifier(h).view(-1)


# ====================================================
# train fold
# ====================================================
def train_fold(fold, X_train, y_train, X_valid, y_valid, num_features, cat_features):
    train_ds = Dataset(X_train, y_train)
    valid_ds = Dataset(X_valid, y_valid)

    train_dl = get_dataloader(train_ds, type_="train")
    valid_dl = get_dataloader(valid_ds, type_="valid")

    CHECKPOINT_NAME = f"fold{fold}_model_" "{epoch:02d}_{valid_metric:.3f}"
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

    model = Model(input_dim=X_train.shape[1])

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
        filename=f"training_curve_fold{fold}.png",
    )

    print(f"model_path={model_path}")

    return model_path


# ====================================================
# inference
# ====================================================
def inference(X_test, model_path):
    model = Model.load_from_checkpoint(model_path, input_dim=X_test.shape[1])
    model.freeze()
    model.eval()

    ds = Dataset(X_test, labels=None)
    dl = get_dataloader(ds, type_="test")

    predictions_list = []

    for batch in tqdm(dl, desc="inference", total=len(dl)):
        logits = model(batch)
        predictions_list.append(logits.detach().cpu().numpy())

    prediction = np.concatenate(predictions_list, axis=0)

    prediction_df = pd.DataFrame()
    prediction_df["customer_ID"] = ds.ids
    prediction_df["target"] = prediction

    return prediction_df


# ====================================================
# main
# ====================================================
def main():
    train, test, train_labels, num_features, cat_features = prepare_data()

    oof_df = pd.DataFrame()
    sub_df = pd.DataFrame()
    model_paths = []
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    for fold, (train_idx, valid_idx) in enumerate(skf.split(train_labels, train_labels["target"])):
        train_customer_ids = train_labels.index[train_idx]
        valid_customer_ids = train_labels.index[valid_idx]
        X_train = train.loc[train_customer_ids]
        X_valid = train.loc[valid_customer_ids]
        y_train = train_labels.loc[train_customer_ids]["target"]
        y_valid = train_labels.loc[valid_customer_ids]["target"]

        model_path = train_fold(fold, X_train, y_train, X_valid, y_valid, num_features, cat_features)
        model_paths.append(model_path)

        oof = inference(X_valid, model_path)
        oof_df = oof_df.append(oof, ignore_index=True)

        sub = inference(test, model_path)
        sub_df = sub_df.append(sub, ignore_index=True)

    oof_df = oof_df.groupby("customer_ID")["target"].mean().reset_index().sort_values("customer_ID")
    sub_df = sub_df.groupby("customer_ID")["target"].mean().reset_index().sort_values("customer_ID")

    oof_df.to_csv("oof.csv", index=False)
    oof_df.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    os.chdir("../work")
    main()
