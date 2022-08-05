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
from sklearn.metrics import log_loss
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

import category_encoders as ce

# utils
from utils import *

tqdm.pandas()

# ====================================================
# config
# ====================================================
DEBUG = False

SEED = 42
N_SPLITS = 10


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
    "type": "LightGBM",
    "metric_name": "amex",  # {amex, binary_logloss}
    "num_boost_round": 100000,
    "early_stopping_rounds": 1500,
    "target_encoding": True,
    "seed": SEED,
    "n_splits": N_SPLITS,
    "params": {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "dart",  # {gbdt, dart}
        "learning_rate": 0.01,
        "num_leaves": 128,
        "min_data_in_leaf": 40,
        "reg_alpha": 3.0,
        "reg_lambda": 5.0,
        "feature_fraction": 0.25,
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
#     "metric_name": "amex",  # {amex, logloss}
#     "num_boost_round": 100000,
#     "early_stopping_rounds": 1500,
#     "target_encoding": False,
#     "seed": SEED,
#     "n_splits": N_SPLITS,
#     "params": {
#         "objective": "binary:logitraw",
#         "booster": "gbtree",
#         "learning_rate": 0.03,
#         "max_depth": 4,
#         "subsample": 0.6,
#         "colsample_bytree": 0.5,
#         "disable_default_eval_metric": "true",
#         "tree_method": "gpu_hist",
#         "predictor": "gpu_predictor",
#         "n_jobs": -1,
#         "random_state": SEED,
#     },
# }


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

    def train(
        self,
        X_train,
        y_train,
        X_valid,
        y_valid,
        sample_weight_train,
        sample_weight_valid,
        num_features,
        cat_features,
    ):
        self.feature_names = list(X_train.columns)
        self.feature_types = [("c" if f in cat_features else "q") for f in self.feature_names]

        ## target encoding
        if self.params["target_encoding"]:
            self.target_encoder = ce.TargetEncoder(
                cols=cat_features,
                min_samples_leaf=1,
                smoothing=1.0,
            )
            self.target_encoder.fit(X_train, y_train)
            X_train = self.target_encoder.transform(X_train)
            X_valid = self.target_encoder.transform(X_valid)
            self.feature_types = ["q" for f in self.feature_names]

        ## make dataset
        train_set = xgb.DMatrix(
            data=X_train.to_numpy(),
            label=y_train.to_numpy(),
            weight=sample_weight_train,
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            enable_categorical=True,
        )
        valid_set = xgb.DMatrix(
            data=X_valid.to_numpy(),
            label=y_valid.to_numpy(),
            weight=sample_weight_valid,
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            enable_categorical=True,
        )

        ## train
        self.evals_result = {}

        callbacks = [
            xgb.callback.EarlyStopping(
                rounds=self.params["early_stopping_rounds"],
                metric_name=self.params["metric_name"],
                data_name="valid",
                maximize={"amex": True, "logloss": False}[self.params["metric_name"]],
                save_best=True,
            ),
        ]
        self.model = xgb.train(
            params=self.params["params"],
            dtrain=train_set,
            num_boost_round=self.params["num_boost_round"],
            evals=[(train_set, "train"), (valid_set, "valid")],
            callbacks=callbacks,
            feval={"amex": self.amex_metric_xgb, "logloss": self.logloss_metric_xgb}[self.params["metric_name"]],
            evals_result=self.evals_result,
            verbose_eval=100,
        )

    def inference(self, X):
        if self.params["target_encoding"]:
            X = self.target_encoder.transform(X)
        d = xgb.DMatrix(
            data=X.to_numpy(),
            feature_names=self.model.feature_names,
            feature_types=self.model.feature_types,
            enable_categorical=True,
        )
        return self.model.predict(d)

    def save(self, path):
        if self.params["target_encoding"]:
            joblib.dump(self.target_encoder, os.path.splitext(path)[0] + "_encoder" + os.path.splitext(path)[1])
        joblib.dump(self.model, path)

    def load(self, path):
        if self.params["target_encoding"]:
            self.target_encoder = joblib.load(os.path.splitext(path)[0] + "_encoder" + os.path.splitext(path)[1])
        self.model = joblib.load(path)

    @staticmethod
    def logloss_metric_xgb(preds: np.ndarray, data: xgb.DMatrix):
        target = data.get_label()
        score = log_loss(target, sigmoid(preds))
        return "logloss", score

    @staticmethod
    def amex_metric_xgb(preds: np.ndarray, data: xgb.DMatrix):
        target = data.get_label()
        score = amex_metric(target, preds)
        return "amex", score

    def plot_importances(self, path):
        fig, ax = plt.subplots(figsize=(10, 25))
        plt.subplots_adjust(left=0.45, bottom=0.05, top=0.95)
        xgb.plot_importance(
            self.model,
            importance_type="gain",
            max_num_features=50,
            xlabel="",
            ylabel="",
            ax=ax,
        )
        plt.savefig(path)
        plt.close()

    def plot_metrics(self, path):
        plt.tight_layout()
        lgb.plot_metric(
            self.evals_result,
            metric=self.params["metric_name"],
        )
        plt.savefig(path)
        plt.close()


class LightGBMModel(BaseModelWrapper):
    def __init__(self, params) -> None:
        super().__init__()
        self.params = params

    def train(
        self,
        X_train,
        y_train,
        X_valid,
        y_valid,
        sample_weight_train,
        sample_weight_valid,
        num_features,
        cat_features,
    ):

        ## target encoding
        if self.params["target_encoding"]:
            self.target_encoder = ce.TargetEncoder(
                cols=cat_features,
                min_samples_leaf=1,
                smoothing=1.0,
            )
            self.target_encoder.fit(X_train, y_train)
            X_train = self.target_encoder.transform(X_train)
            X_valid = self.target_encoder.transform(X_valid)
            cat_features = "auto"

        train_set = lgb.Dataset(X_train, y_train, weight=sample_weight_train, feature_name=list(X_train.columns))
        valid_set = lgb.Dataset(X_valid, y_valid, weight=sample_weight_valid, feature_name=list(X_valid.columns))

        self.evals_result = {}

        early_stopping_callback = DartEarlyStopping(
            data_name="valid",
            monitor_metric=self.params["metric_name"],
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
        if self.params["target_encoding"]:
            X = self.target_encoder.transform(X)
        return self.model.predict(X, raw_score=True)

    def save(self, path):
        if self.params["target_encoding"]:
            joblib.dump(self.target_encoder, os.path.splitext(path)[0] + "_encoder" + os.path.splitext(path)[1])
        joblib.dump(self.model, path)

    def load(self, path):
        if self.params["target_encoding"]:
            self.target_encoder = joblib.load(os.path.splitext(path)[0] + "_encoder" + os.path.splitext(path)[1])
        self.model = joblib.load(path)

    @staticmethod
    def amex_metric_lgb(preds: np.ndarray, data: lgb.Dataset):
        target = data.get_label()
        score = amex_metric(target, preds)
        return "amex", score, True

    def plot_importances(self, path):
        fig, ax = plt.subplots(figsize=(10, 25))
        plt.subplots_adjust(left=0.45, bottom=0.05, top=0.95)
        lgb.plot_importance(
            self.model,
            importance_type="gain",
            max_num_features=50,
            xlabel="",
            ylabel="",
            ax=ax,
        )
        plt.savefig(path)
        plt.close()

    def plot_metrics(self, path):
        plt.tight_layout()
        lgb.plot_metric(
            self.evals_result,
            metric=self.params["metric_name"],
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

    # D_64
    df["D_64"] = df["D_64"].replace(1, pd.NA)

    # D_66
    df["D_66"] = df["D_66"].replace(0, pd.NA)

    # D_68
    df["D_68"] = df["D_68"].replace(0, pd.NA)

    # compute "after pay" features
    values = []
    for bcol in ["B_11", "B_14", "B_17"] + ["D_39", "D_131"] + ["S_16", "S_23"]:
        for pcol in ["P_2", "P_3"]:
            if bcol in df.columns:
                values.append((df[bcol] - df[pcol]).rename(f"{bcol}_{pcol}_diff").astype(pd.Float32Dtype()))
    df = pd.concat([df] + values, axis=1)
    del values
    gc.collect()

    # dropcols
    dropcols = [
        "R_1",
        "B_29",
        "S_9",
    ]
    df.drop(columns=dropcols, inplace=True)

    return df


def aggregate_features(df, filename):
    trace = Trace()

    results = []

    # date
    with trace.timer("aggregate date features"):
        date_last1 = df.groupby("customer_ID")["S_2"].nth(-1).sort_index()
        date_last2 = df.groupby("customer_ID")["S_2"].nth(-2).sort_index()
        date_first1 = df.groupby("customer_ID")["S_2"].nth(0).sort_index()
        date_agg_result = pd.concat(
            [
                df.groupby("customer_ID")["S_2"].agg("last").rename("S_2-last_weekday").sort_index().dt.weekday.astype(np.int16),
                df.groupby("customer_ID")["S_2"].agg("last").rename("S_2-last_day").sort_index().dt.day.astype(np.int16),
                (date_last1 - date_last2).rename("S_2-last_diff").dt.days.fillna(0).astype(np.int16),
                (date_last1 - date_first1).rename("S_2-last_first_diff").dt.days.fillna(0).astype(np.int16),
            ],
            axis=1,
        )

        del date_last1, date_last2, date_first1
        gc.collect()

        results.append(date_agg_result)

    # num
    with trace.timer("aggregate num features"):
        num_columns = [c for c in df.columns if c not in CAT_FEATURES + ["customer_ID", "S_2"]]
        agg_names = [
            "mean",
            "max",
            "min",
            "std",
            "last",
        ]
        num_agg_result = df.groupby("customer_ID")[num_columns].agg(agg_names).astype(pd.Float32Dtype())
        num_agg_result.columns = ["-".join(c) for c in num_agg_result.columns]
        results.append(num_agg_result.sort_index())

        # transform last num features to round2
        for col in num_agg_result.columns:
            if (col.endswith("-last") or col.endswith("-max") or col.endswith("-min")) and pd.api.types.is_float_dtype(num_agg_result[col]):
                num_agg_result[col] = num_agg_result[col].round(2)

        # na_index
        na_index_mean = df.set_index("customer_ID").isna()[num_columns].groupby(level=0)[num_columns].mean()
        na_index_mean.columns = [f"{col}-na_index_mean" for col in num_columns]
        results.append(na_index_mean.sort_index())
        del na_index_mean
        gc.collect()

        # last - shift1
        last_diff = df.groupby("customer_ID")[num_columns].nth(-1) - df.groupby("customer_ID")[num_columns].nth(-2)
        last_diff.columns = [f"{col}-last_diff" for col in num_columns]
        results.append(last_diff.sort_index())
        del last_diff
        gc.collect()

        # last / shift1
        last_frac = (
            (df.groupby("customer_ID")[num_columns].nth(-1) / df.groupby("customer_ID")[num_columns].nth(-2))
            .replace([-np.inf, np.inf], pd.NA)
            .astype(pd.Float32Dtype())
        )
        last_frac.columns = [f"{col}-last_frac" for col in num_columns]
        results.append(last_frac.sort_index())
        del last_frac
        gc.collect()

        # last - mean
        last_mean_diff = df.groupby("customer_ID")[num_columns].agg("last") - df.groupby("customer_ID")[num_columns].agg("mean")
        last_mean_diff.columns = [f"{col}-last_mean_diff" for col in num_columns]
        results.append(last_mean_diff.sort_index())
        del last_mean_diff
        gc.collect()

        # last / mean
        last_mean_frac = (
            (df.groupby("customer_ID")[num_columns].agg("last") / df.groupby("customer_ID")[num_columns].agg("mean"))
            .replace([-np.inf, np.inf], pd.NA)
            .astype(pd.Float32Dtype())
        )
        last_mean_frac.columns = [f"{col}-last_mean_frac" for col in num_columns]
        results.append(last_mean_frac.sort_index())
        del last_mean_frac
        gc.collect()

        # last - first
        last_first_diff = df.groupby("customer_ID")[num_columns].agg("last") - df.groupby("customer_ID")[num_columns].agg("first")
        last_first_diff.columns = [f"{col}-last_first_diff" for col in num_columns]
        results.append(last_first_diff.sort_index())
        del last_first_diff
        gc.collect()

        # last / first
        last_first_frac = (
            (df.groupby("customer_ID")[num_columns].agg("last") / df.groupby("customer_ID")[num_columns].agg("first"))
            .replace([-np.inf, np.inf], pd.NA)
            .astype(pd.Float32Dtype())
        )
        last_first_frac.columns = [f"{col}-last_first_frac" for col in num_columns]
        results.append(last_first_frac.sort_index())
        del last_first_frac
        gc.collect()

        # left_std, right_std
        left_std = df.groupby("customer_ID")[num_columns].agg("mean") - df.groupby("customer_ID")[num_columns].agg("std")
        right_std = df.groupby("customer_ID")[num_columns].agg("mean") + df.groupby("customer_ID")[num_columns].agg("std")
        left_std.columns = [f"{col}-left_std" for col in num_columns]
        right_std.columns = [f"{col}-right_std" for col in num_columns]
        results.append(left_std.sort_index())
        results.append(right_std.sort_index())
        del left_std, right_std
        gc.collect()

    # cat
    with trace.timer("aggregate cat features"):
        cat_columns = CAT_FEATURES
        agg_names = [
            "count",
            "nunique",
            "last",
        ]
        cat_agg_result = df.groupby("customer_ID")[cat_columns].agg(agg_names).astype(pd.Int8Dtype())
        cat_agg_result.columns = ["-".join(c) for c in cat_agg_result.columns]
        results.append(cat_agg_result.sort_index())

    del df
    gc.collect()

    # concat
    with trace.timer("concat results"):
        agg_result = pd.concat(results, axis=1)
        del date_agg_result, num_agg_result, cat_agg_result
        gc.collect()

        # define categorical features
        cat_features = [f"{col}-last" for col in cat_columns]
        num_features = [col for col in agg_result.columns if col not in cat_features]

    with trace.timer("save result"):
        # save results
        agg_result.to_pickle(filename)
        del agg_result
        gc.collect()

    return num_features, cat_features


def make_features(filenames, output_filename, num_features, cat_features):
    trace = Trace()

    USE_NUM_COLS = [col for col in num_features if col.endswith("-mean") or col.endswith("-last")]
    USE_CAT_COLS = [col for col in cat_features if col.endswith("-last")]

    USE_COLS = USE_NUM_COLS + USE_CAT_COLS

    with trace.timer("make all df"):
        df = pd.concat([pd.read_pickle(filename)[USE_COLS] for filename in filenames], axis=0)

    feature_list = []

    ############################################################
    # use GPU to make features
    ############################################################

    ### prepare features
    with trace.timer("scaling, fillna, to GPU"):
        # cat
        cat = pd.get_dummies(df[USE_CAT_COLS], columns=USE_CAT_COLS)
        USE_CAT_COLS = list(cat.columns)
        USE_COLS = USE_NUM_COLS + USE_CAT_COLS

        # num
        num = StandardScaler(copy=False).fit_transform(df[USE_NUM_COLS].astype(np.float32))

        features = pd.concat(
            [
                pd.DataFrame(cat, index=df.index, columns=USE_CAT_COLS),
                pd.DataFrame(num, index=df.index, columns=USE_NUM_COLS),
            ],
            axis=1,
        )
        features.fillna(0, inplace=True)
        features = cudf.from_pandas(features)
        del df, cat, num
        gc.collect()

    ### kmeans
    with trace.timer("kmeans"):
        N_CLUSTERS = 30
        kmeans = cuml.KMeans(n_clusters=N_CLUSTERS, random_state=SEED)
        kmeans_feature = pd.Series(
            kmeans.fit_predict(features).astype(cupy.uint8).to_numpy(),
            name=f"kmeans_{N_CLUSTERS}",
            index=features.index.to_numpy(),
        )
        feature_list.append(kmeans_feature)
        cat_features.append(f"kmeans_{N_CLUSTERS}")

    ### PCA
    with trace.timer("pca"):
        N_COMPONENTS = 50
        pca = cuml.PCA(n_components=N_COMPONENTS, random_state=SEED)
        pca_features = pd.DataFrame(
            pca.fit_transform(features).astype(cupy.float32).to_numpy(),
            columns=[f"pca_{i}" for i in range(N_COMPONENTS)],
            index=features.index.to_numpy(),
        )
        feature_list.append(pca_features)
        num_features.extend(pca_features.columns)

    ### KNN
    # with trace.timer("knn"):
    #     N_NEIGHBORS = 10
    #     METRIC = "euclidean"
    #     knn = cuml.NearestNeighbors(
    #         n_neighbors=N_NEIGHBORS,
    #         metric=METRIC,
    #         verbose=True,
    #         output_type="numpy",
    #     )
    #     knn.fit(features[USE_COLS])
    #     neighbors = knn.kneighbors(features[USE_COLS], return_distance=False).astype(np.uint16)
    #     names = []
    #     arrays = []
    #     for col in tqdm(USE_COLS, desc="nearest neighbors"):
    #         names.append(f"{col}-nn{N_NEIGHBORS}_{METRIC}_mean")
    #         arrays.append(np.nanmean(features[col].to_numpy()[neighbors], axis=1).astype(np.float32))

    #     feature_list.append(
    #         pd.DataFrame(
    #             np.stack(arrays, axis=1),
    #             columns=names,
    #             index=features.index.to_numpy(),
    #         )
    #     )
    #     num_features.extend(names)

    del features
    gc.collect()

    with trace.timer("concat results"):
        additive_features = pd.concat(feature_list, axis=1)

    with trace.timer("save result"):
        additive_features.to_pickle(output_filename)
        del additive_features
        gc.collect()

    return num_features, cat_features


def process_input(filenames, additive_features_filename, num_features, cat_features):
    trace = Trace()
    for filename in filenames:
        print("#" * 10, filename, "#" * 10)
        df = pd.read_pickle(filename)

        # additive features
        if os.path.exists(additive_features_filename):
            additive_features = pd.read_pickle(additive_features_filename)
            df = pd.concat([df, additive_features], axis=1, join="inner")

        # process nan values
        with trace.timer("process nan values"):
            # cat features
            for col in cat_features:
                m = df[col].max()
                df[col] = df[col].fillna(m + 1).astype(np.int16)

            # num features
            df.fillna(-9999, inplace=True)
            df[num_features] = df[num_features].astype(np.float32)

        df.sort_index().to_pickle(filename)
        print(f"{filename}: df.shape={df.shape}")

    return num_features, cat_features


def prepare_data(debug):
    ### train
    train_ids = np.load(Path(INPUT_CUSTOMER_IDS_DIR, "train.npy"), allow_pickle=True)
    train_df = pd.read_pickle(Path(INPUT_INTEGER_PICKLE_DIR, "train.pkl"))
    if debug:
        train_ids = np.random.choice(train_ids, size=1000, replace=False)
        train_df = train_df.loc[train_df["customer_ID"].isin(train_ids)].reset_index(drop=True)

    # preprocessing
    train_df = preprocess(train_df)

    # aggregate
    print("#" * 10, "train", "#" * 10)
    num_features, cat_features = aggregate_features(train_df, filename="train.pkl")

    ### public
    public_ids = np.load(Path(INPUT_CUSTOMER_IDS_DIR, "public.npy"), allow_pickle=True)
    test_df = pd.read_pickle(Path(INPUT_INTEGER_PICKLE_DIR, "test.pkl"))
    if debug:
        public_ids = np.random.choice(public_ids, size=1000, replace=False)

    public_df = test_df.loc[test_df["customer_ID"].isin(public_ids)]
    del test_df
    gc.collect()

    # preprocessing
    public_df = preprocess(public_df)

    # aggregate
    print("#" * 10, "public", "#" * 10)
    _, _ = aggregate_features(public_df, filename="public.pkl")

    ### ptivate
    private_ids = np.load(Path(INPUT_CUSTOMER_IDS_DIR, "private.npy"), allow_pickle=True)
    test_df = pd.read_pickle(Path(INPUT_INTEGER_PICKLE_DIR, "test.pkl"))
    if debug:
        private_ids = np.random.choice(private_ids, size=1000, replace=False)

    private_df = test_df.loc[test_df["customer_ID"].isin(private_ids)]
    del test_df
    gc.collect()

    # preprocessing
    private_df = preprocess(private_df)

    # aggregation
    print("#" * 10, "private", "#" * 10)
    _, _ = aggregate_features(private_df, filename="private.pkl")

    ### make features
    num_features, cat_features = make_features(
        filenames=[
            "train.pkl",
            "public.pkl",
            "private.pkl",
        ],
        output_filename="additive_features.pkl",
        num_features=num_features,
        cat_features=cat_features,
    )

    ### process_input
    num_features, cat_features = process_input(
        filenames=[
            "train.pkl",
            "public.pkl",
            "private.pkl",
        ],
        additive_features_filename="additive_features.pkl",
        num_features=num_features,
        cat_features=cat_features,
    )

    ### labels
    train_labels = pd.read_csv(Path(INPUT_DIR, "train_labels.csv"), dtype={"target": "uint8"})
    train_labels["customer_ID"] = pd.Categorical(train_labels["customer_ID"], ordered=True)
    train_labels.set_index("customer_ID", inplace=True)
    if debug:
        train_labels = train_labels.loc[train_ids]
    train_labels.sort_index(inplace=True)

    # save results
    train_labels.to_pickle("train_labels.pkl")
    np.save("train_ids.npy", train_ids, allow_pickle=True)
    np.save("public_ids.npy", public_ids, allow_pickle=True)
    np.save("private_ids.npy", private_ids, allow_pickle=True)
    joblib.dump(num_features, "num_features.pkl")
    joblib.dump(cat_features, "cat_features.pkl")


# ====================================================
# training
# ====================================================
def train_fold(
    fold,
    X_train,
    y_train,
    X_valid,
    y_valid,
    sample_weight_train,
    sample_weight_valid,
    num_features,
    cat_features,
):
    trace = Trace()
    model = get_model(PARAMS)

    # train
    with trace.timer(f"training fold{fold}"):
        model.train(
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            sample_weight_train=sample_weight_train,
            sample_weight_valid=sample_weight_valid,
            num_features=num_features,
            cat_features=cat_features,
        )

    # plots
    model.plot_importances(f"importance_fold{fold}.png")
    model.plot_metrics(f"metric_fold{fold}.png")

    # save
    model.save(f"model_fold{fold}.pkl")

    # oof
    oof_preds = model.inference(X_valid)
    score, g, d = amex_metric(y_valid.to_numpy(), oof_preds, return_details=True)
    print(f"oof score of fold{fold}: {score:.4f}")
    print(f"gini: {g:.4f}")
    print(f"default rate(4%): {d:.4f}")

    return oof_preds


def training_main():
    train_ids = np.load("train_ids.npy", allow_pickle=True)
    train_labels = pd.read_pickle("train_labels.pkl")
    num_features = joblib.load("num_features.pkl")
    cat_features = joblib.load("cat_features.pkl")

    train = pd.read_pickle("train.pkl")

    oof_prediction = np.zeros(len(train))
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

        sample_weight_train = None
        sample_weight_valid = None

        print(f"X_train.shape={X_train.shape}, X_valid.shape={X_valid.shape}")

        prediction = train_fold(
            fold,
            X_train,
            y_train,
            X_valid,
            y_valid,
            sample_weight_train,
            sample_weight_valid,
            num_features,
            cat_features,
        )

        oof_prediction[valid_idx] = prediction

    # feature names
    with open("feature_names.txt", "w") as f:
        f.writelines("\n".join(cat_features + num_features))

    # oof
    oof_df = pd.DataFrame(
        {
            "customer_ID": train.index,
            "prediction": oof_prediction,
        }
    )
    oof_score, g, d = amex_metric(
        train_labels["target"].to_numpy(),
        oof_df["prediction"].to_numpy(),
        return_details=True,
    )
    print("#" * 10)
    print(f"oof score: {oof_score:.4f}")
    print(f"gini: {g:.4f}")
    print(f"default rate(4%): {d:.4f}")
    print("#" * 10)

    oof_df.to_csv("oof.csv", index=False)

    plot_target_distribution(oof_df["prediction"], train_labels["target"], path="oof_target_distribution.png")
    plot_distribution(oof_df["prediction"], path="oof_distribution.png")

    return oof_score, g, d


def inference_main():
    public_ids = np.load("public_ids.npy", allow_pickle=True)
    private_ids = np.load("private_ids.npy", allow_pickle=True)
    public = pd.read_pickle("public.pkl")
    private = pd.read_pickle("private.pkl")
    num_features = joblib.load("num_features.pkl")
    cat_features = joblib.load("cat_features.pkl")

    public_predictions = []
    private_predictions = []
    for fold in range(N_SPLITS):
        model = get_model(PARAMS)
        model.load(f"model_fold{fold}.pkl")

        public_predictions.append(model.inference(public))
        private_predictions.append(model.inference(private))

    # test
    public_prediction = np.mean(np.stack(public_predictions, axis=1), axis=1)
    private_prediction = np.mean(np.stack(private_predictions, axis=1), axis=1)
    public_df = pd.DataFrame(
        {
            "customer_ID": public.index,
            "prediction": public_prediction,
        }
    )
    private_df = pd.DataFrame(
        {
            "customer_ID": private.index,
            "prediction": private_prediction,
        }
    )
    plot_distribution(public_df["prediction"], path="public_distribution.png")
    plot_distribution(private_df["prediction"], path="private_distribution.png")
    test_df = pd.concat([public_df, private_df], axis=0).reset_index(drop=True)
    test_df.to_csv("submission.csv", index=False)


def main():
    seed_everything(SEED)
    prepare_data(DEBUG)
    oof_score, g, d = training_main()
    inference_main()

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


# ====================================================
# main
# ====================================================

if __name__ == "__main__":
    os.chdir("../work")
    results = main()
    joblib.dump(results, "results.pkl")
