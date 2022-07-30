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
import contextlib
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from abc import ABCMeta, abstractmethod

import pickle
import json
from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler
import yaml

from tqdm.auto import tqdm

import joblib


import numpy as np
import pandas as pd

import cupy
import cudf
import cuml

# utils
from utils import *

# ====================================================
# config
# ====================================================
DEBUG = False
SEED = 42

INPUT_DIR = "../input/amex-default-prediction"
INPUT_PICKLE_DIR = "../input/amex-pickle"
INPUT_INTEGER_PICKLE_DIR = "../input/amex-integer-pickle"
INPUT_DATA_SEQUENTIAL_DIR = "../input/amex-data-sequeitial"
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
# ====================================================
# data processing
# ====================================================
def preprocess(df: pd.DataFrame):
    # S_2
    df["S_2"] = pd.to_datetime(df["S_2"], format="%Y-%m-%d")

    # customer_ID
    df["customer_ID"] = pd.Categorical(df["customer_ID"], ordered=True)

    # sorting
    df.sort_values(["customer_ID", "S_2"], inplace=True)

    df.set_index("customer_ID", inplace=True)

    # D_64 convert "-1" to NA
    df["D_64"] = df["D_64"].map({"U": "U", "R": "R", "O": "O"})

    # D_66 convert 0 to NA
    df["D_66"] = df["D_66"].map({1: 1})

    # D_68 convert 0 to NA
    df["D_68"] = df["D_68"].map({1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6})

    # dropcols
    dropcols = [
        "R_1",
        "B_29",
        # "D_121",
        # "D_59",
        # "S_11",
        # "D_115",
    ]
    df.drop(columns=dropcols, inplace=True)

    return df


def make_features(df: pd.DataFrame):
    trace = Trace()

    cat_features = CAT_FEATURES.copy()
    num_features = [col for col in df.columns if col not in cat_features + ["customer_ID", "S_2"]]

    with trace.timer("make date features"):
        # date features
        cols = ["month", "weekday"]
        date_df = pd.concat(
            [
                df["S_2"].dt.month.astype("int16"),
                df["S_2"].dt.weekday.astype("int16"),
            ],
            axis=1,
        )
        date_df.columns = cols
        df = pd.concat([df, date_df], axis=1)
        cat_features.extend(cols)

        del date_df
        gc.collect()

    # compute "after pay" features
    with trace.timer("compute after pay features"):
        values = []
        for bcol in ["B_11", "B_14", "B_17"] + ["D_39", "D_131"] + ["S_16", "S_23"]:
            for pcol in ["P_2", "P_3"]:
                if bcol in df.columns:
                    name = f"{bcol}_{pcol}_diff"
                    values.append((df[bcol] - df[pcol]).rename(name).astype(pd.Float32Dtype()))
                    num_features.append(name)
        df = pd.concat([df] + values, axis=1)
        del values
        gc.collect()

    # dummies
    with trace.timer("process cat features"):
        df = pd.get_dummies(df, columns=cat_features, drop_first=False, dummy_na=True)
        cat_features = [col for col in df.columns if col not in num_features + ["S_2", "customer_ID"]]

    # isna
    with trace.timer("process num features"):
        values = []
        for col in num_features.copy():
            name = f"{col}_isna"
            values.append(df[col].isna().astype(np.int16).rename(name))
            cat_features.append(name)
        df = pd.concat([df] + values, axis=1)
        del values
        gc.collect()

    # dropcols
    with trace.timer("dropcols"):
        df.drop(columns="S_2", inplace=True)

    return df, num_features, cat_features


def process_na(df, num_features, cat_features):
    trace = Trace()

    with trace.timer("process na"):
        df.fillna(0, inplace=True)

    return df, num_features, cat_features


def scale_features(df, num_features, cat_features, type="train"):
    trace = Trace()

    with trace.timer("scale features"):
        if type == "train":

            scaler = StandardScaler(copy=True)
            # scaler = RobustScaler(copy=True)
            # scaler = QuantileTransformer(
            #     n_quantiles=1000,
            #     output_distribution="normal",
            #     copy=True,
            # )

            scaler.fit(df[num_features].astype(np.float32))
            joblib.dump(scaler, "scaler.pkl")
        elif type in ("public", "private"):
            scaler = joblib.load("scaler.pkl")
        else:
            raise ValueError()

        num_values = pd.DataFrame(
            scaler.transform(df[num_features].astype(np.float32)),
            columns=num_features,
            index=df.index,
        )

        df = pd.concat([num_values, df[cat_features]], axis=1)

    return df, num_features, cat_features


def prepare_data(TYPE):
    def reshape_pad(x: pd.DataFrame):
        l, c = x.shape
        return np.pad(x.to_numpy(), ((13 - l, 0), (0, 0)), constant_values=0)

    # train
    if TYPE == "train":
        print("#" * 30, "train", "#" * 30)
        train = pd.read_pickle(Path(INPUT_INTEGER_PICKLE_DIR, "train.pkl"))
        train_customer_ids = np.load(os.path.join(INPUT_CUSTOMER_IDS_DIR, "train.npy"), allow_pickle=True)
        train_labels = pd.read_csv(Path(INPUT_DIR, "train_labels.csv"))

        if DEBUG:
            train_sample_ids = pd.Series(train["customer_ID"].unique()).sample(1000)
            train = train.loc[train["customer_ID"].isin(train_sample_ids)].reset_index(drop=True)
            train_labels = train_labels.loc[train_labels["customer_ID"].isin(train_sample_ids)].reset_index(drop=True)

        # preprocessing
        train = preprocess(train)
        train_labels["customer_ID"] = pd.Categorical(train_labels["customer_ID"], ordered=True)
        train_labels = train_labels.set_index("customer_ID").sort_index().to_numpy().reshape(-1)

        # make features
        train, num_features, cat_features = make_features(train)

        # process na
        train, num_features, cat_features = process_na(train, num_features, cat_features)

        # scale features
        # train, num_features, cat_features = scale_features(train, num_features, cat_features, type=TYPE)

        # groupby, reshape, stack
        train_groups = train.astype(np.float32).groupby(level=0)
        train = np.stack([reshape_pad(g) for _, g in tqdm(train_groups, desc="train")], axis=0)

        if not DEBUG:
            np.save(os.path.join(INPUT_DATA_SEQUENTIAL_DIR, "train.npy"), train)
            np.save(os.path.join(INPUT_DATA_SEQUENTIAL_DIR, "train_ids.npy"), train_customer_ids)
            np.save(os.path.join(INPUT_DATA_SEQUENTIAL_DIR, "train_labels.npy"), train_labels)

        print(f"train.shape={train.shape}")

        assert train.dtype == "float32"
        assert np.isnan(train).sum() == 0

        del train_groups, train
        gc.collect()

    # public
    elif TYPE == "public":
        print("#" * 30, "public", "#" * 30)
        test = pd.read_pickle(Path(INPUT_INTEGER_PICKLE_DIR, "test.pkl"))
        public_customer_ids = np.load(os.path.join(INPUT_CUSTOMER_IDS_DIR, "public.npy"), allow_pickle=True)

        public = test.loc[test["customer_ID"].isin(public_customer_ids)].reset_index(drop=True)
        del test

        if DEBUG:
            public_sample_ids = pd.Series(public_customer_ids).sample(1000)
            public = public.loc[public["customer_ID"].isin(public_sample_ids)].reset_index(drop=True)

        # preprocessing
        public = preprocess(public)

        # make features
        public, num_features, cat_features = make_features(public)

        # process na
        public, num_features, cat_features = process_na(public, num_features, cat_features)

        # scale features
        # public, num_features, cat_features = scale_features(public, num_features, cat_features, type=TYPE)

        # groupby, reshape, stack
        public_groups = public.astype(np.float32).groupby(level=0)
        public = np.stack([reshape_pad(g) for _, g in tqdm(public_groups, desc="public")], axis=0)

        print(f"public.shape={public.shape}")

        assert public.dtype == "float32"
        assert np.isnan(public).sum() == 0

        if not DEBUG:
            np.save(os.path.join(INPUT_DATA_SEQUENTIAL_DIR, "public.npy"), public)
            np.save(os.path.join(INPUT_DATA_SEQUENTIAL_DIR, "public_ids.npy"), public_customer_ids)
        del public_groups, public
        gc.collect()

    # private
    elif TYPE == "private":
        print("#" * 30, "private", "#" * 30)
        test = pd.read_pickle(Path(INPUT_INTEGER_PICKLE_DIR, "test.pkl"))
        private_customer_ids = np.load(os.path.join(INPUT_CUSTOMER_IDS_DIR, "private.npy"), allow_pickle=True)

        private = test.loc[test["customer_ID"].isin(private_customer_ids)].reset_index(drop=True)
        del test

        if DEBUG:
            private_sample_ids = pd.Series(private_customer_ids).sample(1000)
            private = private.loc[private["customer_ID"].isin(private_sample_ids)].reset_index(drop=True)

        # preprocessing
        private = preprocess(private)

        # make features
        private, num_features, cat_features = make_features(private)

        # process na
        private, num_features, cat_features = process_na(private, num_features, cat_features)

        # scale features
        # private, num_features, cat_features = scale_features(private, num_features, cat_features, type=TYPE)

        # groupby, reshape, stack
        private_groups = private.astype(np.float32).groupby(level=0)
        private = np.stack([reshape_pad(g) for _, g in tqdm(private_groups, desc="private")], axis=0)

        print(f"private.shape={private.shape}")

        assert private.dtype == "float32"
        assert np.isnan(private).sum() == 0

        if not DEBUG:
            np.save(os.path.join(INPUT_DATA_SEQUENTIAL_DIR, "private.npy"), private)
            np.save(os.path.join(INPUT_DATA_SEQUENTIAL_DIR, "private_ids.npy"), private_customer_ids)
        del private_groups, private
        gc.collect()

    if not DEBUG:
        # feature names
        joblib.dump(num_features, os.path.join(INPUT_DATA_SEQUENTIAL_DIR, "num_features.pkl"))
        joblib.dump(cat_features, os.path.join(INPUT_DATA_SEQUENTIAL_DIR, "cat_features.pkl"))


def main():
    os.chdir("/workspaces/amex-default-prediction/work")
    seed_everything(SEED)

    prepare_data("train")
    prepare_data("public")
    prepare_data("private")


if __name__ == "__main__":
    main()
