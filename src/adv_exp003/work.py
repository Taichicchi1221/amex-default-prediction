import os
import gc
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
import subprocess
from contextlib import contextmanager
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from abc import ABCMeta, abstractmethod

import pickle
import json
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, scale
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
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA

from scipy.stats import skew, kurtosis
from scipy.special import expit as sigmoid

import cupy
import cudf
import cuml

import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# utils
from utils import *

from customer_aggregation_work import prepare_data

# ====================================================
# config
# ====================================================
SEED = 42
N_SPLITS = 5

DEBUG = False

INPUT_DIR = "../input/amex-default-prediction"
INPUT_PICKLE_DIR = "../input/amex-pickle"
INPUT_INTEGER_PICKLE_DIR = "../input/amex-integer-pickle"
INPUT_CUSTOMER_IDS_DIR = "../input/amex-customer-ids"


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
    "type": "XGBoost",
    "num_boost_round": 100000,
    "early_stopping_rounds": 500,
    "seed": SEED,
    "n_splits": N_SPLITS,
    "params": {
        "objective": "binary:logitraw",
        "eval_metric": ["auc"],
        "booster": "gbtree",
        "learning_rate": 0.01,
        # "max_depth": 4,
        # "subsample": 0.8,
        # "colsample_bytree": 0.6,
        "tree_method": "gpu_hist",
        "predictor": "gpu_predictor",
        "n_jobs": -1,
        "random_state": SEED,
    },
}


# ====================================================
# mains
# ====================================================
def validation_main(TYPE):
    print("#" * 30, TYPE, "#" * 30)
    df = pd.read_pickle("data.pkl")
    num_features = joblib.load("num_features.pkl")
    cat_features = joblib.load("cat_features.pkl")
    train_ids = np.load("train_ids.npy", allow_pickle=True)
    test_ids = np.load(f"{TYPE}_ids.npy", allow_pickle=True)

    df = df.loc[np.concatenate([train_ids, test_ids])]

    y = np.zeros(len(df), dtype=np.uint8)
    y[df.index.isin(test_ids)] = 1

    importance_df = pd.DataFrame()
    score_df = pd.DataFrame()

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for fold, (train_idx, test_idx) in enumerate(skf.split(df, y)):
        print("#" * 30, f"fold: {fold}", "#" * 30)
        xtr = df.iloc[train_idx]
        xte = df.iloc[test_idx]
        ytr = y[train_idx]
        yte = y[test_idx]

        # train
        evals_result = {}
        feature_names = list(df.columns)
        feature_types = [("c" if f in cat_features else "q") for f in feature_names]
        dtrain = xgb.DMatrix(
            data=xtr.to_numpy(),
            label=ytr,
            feature_names=feature_names,
            feature_types=feature_types,
            enable_categorical=True,
        )
        dtest = xgb.DMatrix(
            data=xte.to_numpy(),
            label=yte,
            feature_names=feature_names,
            feature_types=feature_types,
            enable_categorical=True,
        )
        model: xgb.Booster = xgb.train(
            params=PARAMS["params"],
            dtrain=dtrain,
            num_boost_round=PARAMS["num_boost_round"],
            evals=[(dtrain, "train"), (dtest, "test")],
            evals_result=evals_result,
            early_stopping_rounds=PARAMS["early_stopping_rounds"],
            verbose_eval=100,
        )

        importances = model.get_score(importance_type="gain")
        predictions = model.predict(dtest)

        importance_df = importance_df.append(
            pd.DataFrame(
                {
                    "feature": importances.keys(),
                    "importance": importances.values(),
                }
            )
        )
        score_df = score_df.append(pd.DataFrame({"target": yte, "score": predictions}, index=xte.index))

    # importance
    mean_gain = importance_df[["feature", "importance"]].groupby("feature").mean()
    importance_df["mean"] = importance_df["feature"].map(mean_gain["importance"])
    fig, ax = plt.subplots(figsize=(10, 25))
    plt.subplots_adjust(left=0.25, bottom=0.05, top=0.95)
    plt.title(f"{TYPE}_adversarial_importance")
    sns.barplot(
        x="importance",
        y="feature",
        data=importance_df.sort_values("mean", ascending=False)[: 50 * N_SPLITS],
    )
    plt.savefig(f"{TYPE}_adversarial_importance.png")
    plt.close()

    # inference
    score_df.sort_index(inplace=True)
    score = roc_auc_score(score_df["target"], score_df["score"])
    score_df.to_csv(f"{TYPE}_adversarial_score.csv")

    return score


def main():
    prepare_data(DEBUG)

    public_score = validation_main("public")
    private_score = validation_main("private")

    return Box(
        {
            "params": PARAMS,
            "metrics": {
                "public_score": public_score,
                "private_score": private_score,
            },
        }
    )


if __name__ == "__main__":
    os.chdir("../work")
    seed_everything(SEED)
    results = main()
    joblib.dump(results, "results.pkl")
