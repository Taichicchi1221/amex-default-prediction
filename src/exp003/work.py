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
from sklearn.impute import KNNImputer

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
    "type": "LightGBM",
    "num_boost_round": 100000,
    "early_stopping_rounds": 500,
    "params": {
        "objective": "binary",
        "metric": ["auc"],
        "boosting_type": "dart",  # {gbdt, dart}
        "learning_rate": 0.01,
        "num_leaves": 128,
        "min_data_in_leaf": 40,
        "reg_alpha": 1.0,
        "reg_lambda": 2.0,
        "feature_fraction": 0.20,
        "bagging_freq": 10,
        "bagging_fraction": 0.50,
        "seed": SEED,
        "bagging_seed": SEED,
        "feature_fraction_seed": SEED,
        "verbose": -1,
        "n_jobs": -1,
    },
}


# PARAMS = {
#     "type": "XGBoost",
#     "num_boost_round": 100000,
#     "early_stopping_rounds": 500,
#     "params": {
#         "objective": "binary:logitraw",
#         "eval_metric": "auc",
#         "booster": "gbtree",
#         "learning_rate": 0.01,
#         "max_depth": 4,
#         "subsample": 0.8,
#         "colsample_bytree": 0.6,
#         "tree_method": "gpu_hist",
#         "predictor": "gpu_predictor",
#         "n_jobs": -1,
#         "random_state": SEED,
#     },
# }


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
    plt.hist(sigmoid(ypred), alpha=0.5, bins=50)
    plt.legend(["ytrue", "ypred"])
    plt.savefig(path)
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


# ====================================================
# model
# ====================================================
class BaseModelWrapper(metaclass=ABCMeta):
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def inference(self):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(cls):
        pass

    @abstractmethod
    def plot_importances(self):
        pass

    @abstractmethod
    def plot_metrics(self):
        pass


class XGBoostModel(BaseModelWrapper):
    def __init__(self, params) -> None:
        super().__init__()
        self.params = params

    def train(self, X_train, y_train, X_valid, y_valid, num_features, cat_features):
        self.feature_names = list(X_train.columns)
        self.feature_types = [("c" if f in cat_features else "q") for f in self.feature_names]
        train_set = xgb.DMatrix(
            data=X_train.to_numpy(),
            label=y_train.to_numpy(),
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            enable_categorical=True,
        )
        valid_set = xgb.DMatrix(
            data=X_valid.to_numpy(),
            label=y_valid.to_numpy(),
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            enable_categorical=True,
        )

        self.evals_result = {}

        callbacks = [
            xgb.callback.EarlyStopping(
                rounds=self.params["early_stopping_rounds"],
                metric_name="amex",
                data_name="valid",
                maximize=True,
                save_best=True,
            ),
        ]
        self.model = xgb.train(
            params=self.params["params"],
            dtrain=train_set,
            num_boost_round=self.params["num_boost_round"],
            evals=[(train_set, "train"), (valid_set, "valid")],
            callbacks=callbacks,
            feval=self.amex_metric_xgb,
            evals_result=self.evals_result,
            verbose_eval=100,
        )

    def inference(self, X):
        d = xgb.DMatrix(
            data=X.to_numpy(),
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            enable_categorical=True,
        )
        return self.model.predict(d)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)

    @staticmethod
    def amex_metric_xgb(preds: np.ndarray, data: xgb.DMatrix):
        target = data.get_label()
        score = amex_metric(target, preds)
        return "amex", score

    def plot_importances(self, path):
        dd = self.model.get_score(importance_type="gain")
        df = pd.DataFrame({"feature": dd.keys(), "importance": dd.values()})
        df.sort_values("importance", inplace=True, ascending=False)
        df = df.iloc[:50]
        l = len(df)
        plt.tight_layout()
        plt.figure(figsize=(10, 25))
        plt.barh(np.arange(l, 0, -1), df.importance.values)
        plt.yticks(np.arange(l, 0, -1), df.feature.values)
        plt.title(f"XGB Feature Importance - Top 50")
        plt.savefig(path)
        plt.close()

    def plot_metrics(self, path):
        plt.tight_layout()
        lgb.plot_metric(
            self.evals_result,
            metric="amex",
        )
        plt.savefig(path)
        plt.close()


class LightGBMModel(BaseModelWrapper):
    def __init__(self, params) -> None:
        super().__init__()
        self.params = params

    def train(self, X_train, y_train, X_valid, y_valid, num_features, cat_features):
        train_set = lgb.Dataset(X_train, y_train, feature_name=list(X_train.columns))
        valid_set = lgb.Dataset(X_valid, y_valid, feature_name=list(X_valid.columns))

        self.evals_result = {}

        early_stopping_callback = DartEarlyStopping(
            data_name="valid",
            monitor_metric="amex",
            stopping_round=self.params["early_stopping_rounds"],
        )
        callbacks = [
            early_stopping_callback,
            lgb.callback.record_evaluation(eval_result=self.evals_result),
            lgb.callback.log_evaluation(period=100),
        ]
        self.model = lgb.train(
            params=self.params["params"],
            train_set=train_set,
            num_boost_round=self.params["num_boost_round"],
            valid_sets=[valid_set, train_set],
            valid_names=["valid", "train"],
            callbacks=callbacks,
            feval=[self.amex_metric_lgb],
            categorical_feature=cat_features,
        )

        if early_stopping_callback.best_model is not None:
            self.model = early_stopping_callback.best_model

    def inference(self, X):
        return self.model.predict(X, raw_score=True)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)

    @staticmethod
    def amex_metric_lgb(preds: np.ndarray, data: lgb.Dataset):
        target = data.get_label()
        score = amex_metric(target, preds)
        return "amex", score, True

    def plot_importances(self, path):
        df = pd.DataFrame(
            {
                "feature": self.model.feature_name(),
                "importance": self.model.feature_importance(importance_type="gain"),
            }
        )
        df.sort_values("importance", inplace=True, ascending=False)
        df = df.iloc[:50]
        l = len(df)
        plt.tight_layout()
        plt.figure(figsize=(10, 25))
        plt.barh(np.arange(l, 0, -1), df.importance.values)
        plt.yticks(np.arange(l, 0, -1), df.feature.values)
        plt.title(f"LGB Feature Importance - Top 50")
        plt.savefig(path)
        plt.close()

    def plot_metrics(self, path):
        plt.tight_layout()
        lgb.plot_metric(
            self.evals_result,
            metric="amex",
        )
        plt.savefig(path)
        plt.close()


class DartEarlyStopping(object):
    def __init__(self, data_name, monitor_metric, stopping_round):
        self.data_name = data_name
        self.monitor_metric = monitor_metric
        self.stopping_round = stopping_round
        self.best_score = None
        self.best_model = None
        self.best_score_list = []
        self.best_iter = 0

    def _is_higher_score(self, metric_score, is_higher_better):
        if self.best_score is None:
            return True
        return (self.best_score < metric_score) if is_higher_better else (self.best_score > metric_score)

    def _deepcopy(self, x):
        return pickle.loads(pickle.dumps(x))

    def __call__(self, env):
        evals = env.evaluation_result_list
        for data, metric, score, is_higher_better in evals:
            if data != self.data_name or metric != self.monitor_metric:
                continue
            if not self._is_higher_score(score, is_higher_better):
                if env.iteration - self.best_iter > self.stopping_round:
                    eval_result_str = "\t".join([lgb.callback._format_eval_result(x) for x in self.best_score_list])
                    lgb.basic._log_info(f"Early stopping, best iteration is:\n[{self.best_iter+1}]\t{eval_result_str}")
                    lgb.basic._log_info(f'You can get best model by "DartEarlyStopping.best_model"')
                    raise lgb.callback.EarlyStopException(self.best_iter, self.best_score_list)
                return

            self.best_model = self._deepcopy(env.model)
            self.best_iter = env.iteration
            self.best_score_list = evals
            self.best_score = score
            return
        raise ValueError("monitoring metric not found")


def get_model(params):
    if params["type"] == "XGBoost":
        return XGBoostModel(params)
    if params["type"] == "LightGBM":
        return LightGBMModel(params)

    raise NotImplementedError(f"improper model type: {params['type']}")


# ====================================================
# data processing
# ====================================================
def preprocess(df: pd.DataFrame):
    # customer_ID
    df["customer_ID"] = pd.Categorical(df["customer_ID"], ordered=True)

    # S_2
    df["S_2"] = pd.to_datetime(df["S_2"], format="%Y-%m-%d")

    # D_63
    df["D_63"] = df["D_63"].map({"CR": 0, "XZ": 1, "XM": 2, "CO": 3, "CL": 4, "XL": 5}).astype(pd.Int8Dtype())

    # D_64
    df["D_64"] = df["D_64"].map({"O": 0, "-1": 1, "R": 2, "U": 3}).astype(pd.Int8Dtype())

    # D_117
    df["D_117"] = (df["D_117"] + 1).astype(pd.Int8Dtype())

    # D_126
    df["D_126"] = (df["D_126"] + 1).astype(pd.Int8Dtype())


def aggregate_features(df):
    START_TIME = time.perf_counter()

    results = []

    # num
    num_columns = [c for c in df.columns if c not in CAT_FEATURES]
    agg_names = [
        np.mean,
        np.std,
        np.max,
        np.min,
        "last",
    ]
    num_agg_result = df.groupby("customer_ID")[num_columns].agg(agg_names).astype(pd.Float32Dtype())
    num_agg_result.columns = ["-".join(c) for c in num_agg_result.columns]
    results.append(num_agg_result.sort_index())
    print(f"aggregating num columns: {time.perf_counter() - START_TIME:.2f} seconds")

    # cat
    cat_columns = CAT_FEATURES
    agg_names = [
        "count",
        "nunique",
        "last",
    ]
    cat_agg_result = df.groupby("customer_ID")[cat_columns].agg(agg_names).astype(pd.Int8Dtype())
    cat_agg_result.columns = ["-".join(c) for c in cat_agg_result.columns]
    results.append(cat_agg_result.sort_index())
    print(f"aggregating cat columns: {time.perf_counter() - START_TIME:.2f} seconds")

    del df
    gc.collect()

    # concat
    agg_result = pd.concat(results, axis=1)
    del num_agg_result, cat_agg_result
    gc.collect()

    print(f"all process completed: {time.perf_counter() - START_TIME:.2f} seconds")

    # define categorical features
    cat_features = [f"{col}-last" for col in cat_columns]
    num_features = [col for col in agg_result]

    # process nan values
    for col in cat_features:
        m = agg_result[col].max()
        agg_result[col] = agg_result[col].fillna(m + 1).astype("int16")

    for col in num_features:
        agg_result[col] = agg_result[col].fillna(-1).astype("float32")

    return agg_result, num_features, cat_features


def prepare_data():
    train_df = pd.read_pickle(Path(INPUT_PICKLE_DIR, "train.pkl"))
    test_df = pd.read_pickle(Path(INPUT_PICKLE_DIR, "test.pkl"))
    train_labels = pd.read_csv(Path(INPUT_DIR, "train_labels.csv"))

    if DEBUG:
        train_sample_ids = train_df["customer_ID"].sample(1000)
        train_df = train_df.loc[train_df["customer_ID"].isin(train_sample_ids)].reset_index(drop=True)
        train_labels = train_labels.loc[train_labels["customer_ID"].isin(train_sample_ids)].reset_index(drop=True)
        test_sample_ids = test_df["customer_ID"].sample(1000)
        test_df = test_df.loc[test_df["customer_ID"].isin(test_sample_ids)].reset_index(drop=True)

    # preprocessing
    preprocess(train_df)
    preprocess(test_df)
    train_labels["customer_ID"] = pd.Categorical(
        train_labels["customer_ID"],
        ordered=True,
    )

    # aggregation
    train_df, num_features, cat_features = aggregate_features(train_df)
    test_df, _, _ = aggregate_features(test_df)

    train_df.sort_index(inplace=True)
    test_df.sort_index(inplace=True)

    return train_df, test_df, train_labels, num_features, cat_features


# ====================================================
# training
# ====================================================
def main():

    train, test, train_labels, num_features, cat_features = prepare_data()

    oof_prediction = np.zeros(len(train))
    test_predictions = []
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    for fold, (train_idx, valid_idx) in enumerate(
        skf.split(
            train,
            train_labels["target"],
        )
    ):
        print("#" * 30, f"fold{fold}", "#" * 30)
        X_train = train.iloc[train_idx]
        X_valid = train.iloc[valid_idx]
        y_train = train_labels.iloc[train_idx]["target"]
        y_valid = train_labels.iloc[valid_idx]["target"]

        model = get_model(PARAMS)

        # train
        model.train(
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            num_features=num_features,
            cat_features=cat_features,
        )

        # oof
        preds = model.inference(X_valid)
        oof_prediction[valid_idx] = preds
        score = amex_metric(y_valid.to_numpy(), preds)

        print(f"oof score of fold{fold}: {score:.4f}")

        # test
        test_predictions.append(model.inference(test))

        # plots
        model.plot_importances(f"importance_fold{fold}.png")
        model.plot_metrics(f"metric_fold{fold}.png")

        # save
        model.save(f"model_fold{fold}.pkl")

    # oof
    oof_df = pd.DataFrame(
        {
            "customer_ID": train.index,
            "prediction": oof_prediction,
        }
    )
    oof_score = amex_metric(
        train_labels["target"].to_numpy(),
        oof_df["prediction"].to_numpy(),
    )
    print(f"oof score: {oof_score:.4f}")
    oof_df.to_csv("oof.csv", index=False)

    plot_distribution(oof_df["prediction"], train_labels["target"], path="oof_distribution.png")

    # test
    test_prediction = np.mean(np.stack(test_predictions, axis=1), axis=1)
    test_df = pd.DataFrame(
        {
            "customer_ID": test.index,
            "prediction": test_prediction,
        }
    )
    test_df.to_csv("submission.csv", index=False)

    return Box(
        {
            "params": PARAMS,
            "metrics": {
                "valid_score": oof_score,
                "public_score": np.nan,
                "private_score": np.nan,
            },
        }
    )


# ====================================================
# main
# ====================================================

if __name__ == "__main__":
    os.chdir("../work")
    results = main()
    joblib.dump(results, "results.pkl")
