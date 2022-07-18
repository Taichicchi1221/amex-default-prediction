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
from sklearn.preprocessing import StandardScaler
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

# PARAMS = {
#     "type": "LightGBM",
#     "num_boost_round": 100000,
#     "early_stopping_rounds": 1500,
#     "seed": SEED,
#     "n_splits": N_SPLITS,
#     "params": {
#         "objective": "binary",
#         "metric": ["auc"],
#         "boosting_type": "dart",  # {gbdt, dart}
#         "learning_rate": 0.01,
#         "num_leaves": 128,
#         "min_data_in_leaf": 40,
#         "reg_alpha": 1.0,
#         "reg_lambda": 2.0,
#         "feature_fraction": 0.20,
#         "bagging_freq": 10,
#         "bagging_fraction": 0.50,
#         "seed": SEED,
#         "bagging_seed": SEED,
#         "feature_fraction_seed": SEED,
#         "verbose": -1,
#         "n_jobs": -1,
#     },
# }


PARAMS = {
    "type": "XGBoost",
    "num_boost_round": 100000,
    "early_stopping_rounds": 1500,
    "seed": SEED,
    "n_splits": N_SPLITS,
    "params": {
        "objective": "binary:logitraw",
        "eval_metric": "auc",
        "booster": "gbtree",
        "learning_rate": 0.03,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "tree_method": "gpu_hist",
        "predictor": "gpu_predictor",
        "n_jobs": -1,
        "random_state": SEED,
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


def get_gpu_memory(cmd_path="nvidia-smi", target_properties=("memory.total", "memory.used")):
    """
    ref: https://www.12-technology.com/2022/01/pythongpu.html
    Returns
    -------
    gpu_total : ndarray,  "memory.total"
    gpu_used: ndarray, "memory.used"
    """

    # formatオプション定義
    format_option = "--format=csv,noheader,nounits"

    # コマンド生成
    cmd = "%s --query-gpu=%s %s" % (cmd_path, ",".join(target_properties), format_option)

    # サブプロセスでコマンド実行
    cmd_res = subprocess.check_output(cmd, shell=True)

    # コマンド実行結果をオブジェクトに変換
    gpu_lines = cmd_res.decode().split("\n")[0].split(", ")

    gpu_total = int(gpu_lines[0]) / 1024
    gpu_used = int(gpu_lines[1]) / 1024

    gpu_total = np.round(gpu_used, 1)
    gpu_used = np.round(gpu_used, 1)
    return gpu_total, gpu_used


class Trace:
    cuda = torch.cuda.is_available()
    if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        is_competition_rerun = True
    else:
        is_competition_rerun = False

    @contextmanager
    def timer(self, title):
        t0 = time.time()
        p = psutil.Process(os.getpid())
        cpu_m0 = p.memory_info().rss / 2.0**30
        if self.cuda:
            gpu_m0 = get_gpu_memory()[0]
        yield
        cpu_m1 = p.memory_info().rss / 2.0**30
        if self.cuda:
            gpu_m1 = get_gpu_memory()[0]

        cpu_delta = cpu_m1 - cpu_m0
        if self.cuda:
            gpu_delta = gpu_m1 - gpu_m0

        cpu_sign = "+" if cpu_delta >= 0 else "-"
        cpu_delta = math.fabs(cpu_delta)

        if self.cuda:
            gpu_sign = "+" if gpu_delta >= 0 else "-"
        if self.cuda:
            gpu_delta = math.fabs(gpu_delta)

        cpu_message = f"{cpu_m1:.1f}GB({cpu_sign}{cpu_delta:.1f}GB)"
        if self.cuda:
            gpu_message = f"{gpu_m1:.1f}GB({gpu_sign}{gpu_delta:.1f}GB)"

        if self.cuda:
            message = f"[cpu: {cpu_message}, gpu: {gpu_message}: {time.time() - t0:.1f}sec] {title} "
        else:
            message = f"[cpu: {cpu_message}: {time.time() - t0:.1f}sec] {title} "

        print(message, file=sys.stderr)


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
        fig, ax = plt.subplots(figsize=(10, 25))
        plt.subplots_adjust(left=0.25, bottom=0.05, top=0.95)
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
        fig, ax = plt.subplots(figsize=(10, 25))
        plt.subplots_adjust(left=0.25, bottom=0.05, top=0.95)
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

    def _preprocess_categorical(x):
        assert pd.api.types.is_integer_dtype(x)
        min_value = x.min()
        return x - min_value

    for col in CAT_FEATURES:
        df[col] = _preprocess_categorical(df[col])


def aggregate_features(df):
    trace = Trace()

    results = []

    # num
    def agg_last_diff(df, num_features):
        last1 = df.groupby("customer_ID")[num_features].nth(-1).sort_index()
        last2 = df.groupby("customer_ID")[num_features].nth(-2).sort_index()
        diff = (last1 - last2).add_suffix("-last_diff")
        return diff

    with trace.timer("aggregate num features"):
        num_columns = [c for c in df.columns if c not in CAT_FEATURES + ["customer_ID", "S_2"]]
        agg_names = ["last"]
        num_agg_result = df.groupby("customer_ID")[num_columns].agg(agg_names).astype(pd.Float32Dtype())
        num_agg_result.columns = ["-".join(c) for c in num_agg_result.columns]
        results.append(num_agg_result.sort_index())

        # last - shift1
        last_diff = agg_last_diff(df, num_columns)
        results.append(last_diff.sort_index())

    # cat
    with trace.timer("aggregate cat features"):
        cat_columns = CAT_FEATURES
        agg_names = ["count", "last"]
        cat_agg_result = df.groupby("customer_ID")[cat_columns].agg(agg_names).astype(pd.Int8Dtype())
        cat_agg_result.columns = ["-".join(c) for c in cat_agg_result.columns]
        results.append(cat_agg_result.sort_index())

    del df
    gc.collect()

    # concat
    with trace.timer("concat results"):
        agg_result = pd.concat(results, axis=1)
        del num_agg_result, cat_agg_result
        gc.collect()

        # define categorical features
        cat_features = [f"{col}-last" for col in cat_columns]
        num_features = [col for col in agg_result if col not in cat_features]

    return agg_result, num_features, cat_features


def make_features(df, num_features, cat_features):
    trace = Trace()

    idx = df.index
    colnames = []
    feature_values = []

    # # round2 of last num features
    # with trace.timer("make round2 features"):
    #     for col in num_features:
    #         if col.endswith("-last") or col.endswith("-last_diff"):
    #             colnames.append(f"{col}-round2")
    #             feature_values.append(np.round(df[col], 2))

    # # the difference between last and mean
    # with trace.timer("make difference features"):
    #     for col in num_features:
    #         if col.endswith("-last"):
    #             col_base = col.split("-")[0]
    #             colnames.append(f"{col_base}-last_mean_diff")
    #             feature_values.append((df[f"{col_base}-last"] - df[f"{col_base}-mean"]).to_numpy())

    # with trace.timer("concat results"):
    #     df = pd.concat([df, pd.DataFrame(np.stack(feature_values, axis=1), index=idx, columns=colnames)], axis=1)

    # num_features = [col for col in df.columns if col not in cat_features]

    return df, num_features, cat_features


def fill_nan_values(df, num_features, cat_features):
    trace = Trace()
    # process nan values
    with trace.timer("process nan values"):
        # cat features
        for col in cat_features:
            m = df[col].max()
            df[col] = df[col].fillna(m + 1).astype(np.int16)

        # num features
        df.fillna(-1, inplace=True)
        df[num_features] = df[num_features].astype(np.float32)

    return df, num_features, cat_features


def prepare_data():
    train_df = pd.read_pickle(Path(INPUT_INTEGER_PICKLE_DIR, "train.pkl"))
    test_df = pd.read_pickle(Path(INPUT_INTEGER_PICKLE_DIR, "test.pkl"))
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
    train_labels["customer_ID"] = pd.Categorical(
        train_labels["customer_ID"],
        ordered=True,
    )

    # aggregation
    train_df, num_features, cat_features = aggregate_features(train_df)
    test_df, _, _ = aggregate_features(test_df)

    # make features
    train_df, num_features, cat_features = make_features(train_df, num_features, cat_features)
    test_df, _, _ = make_features(test_df, num_features, cat_features)

    # fill nan values
    train_df, num_features, cat_features = fill_nan_values(train_df, num_features, cat_features)
    test_df, _, _ = fill_nan_values(test_df, num_features, cat_features)

    train_df.sort_index(inplace=True)
    test_df.sort_index(inplace=True)
    train_labels.sort_values("customer_ID", inplace=True)

    print(f"shape of train: {train_df.shape}, shape of test: {test_df.shape}")

    return train_df, test_df, train_labels, num_features, cat_features


# ====================================================
# training
# ====================================================
def main():
    trace = Trace()

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
        with trace.timer(f"training fold{fold}"):
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
