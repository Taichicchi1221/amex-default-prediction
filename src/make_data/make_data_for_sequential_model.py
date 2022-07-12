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
import yaml

from tqdm.auto import tqdm

import joblib


import numpy as np
import pandas as pd

import cupy
import cudf
import cuml


# ====================================================
# config
# ====================================================
DEBUG = False

TYPE = "private"

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
    # sorting
    df.sort_values(["customer_ID", "S_2"], inplace=True)

    # customer_ID
    df["customer_ID"] = pd.Categorical(df["customer_ID"], ordered=True)
    df.set_index("customer_ID", inplace=True)

    # S_2
    df["S_2"] = pd.to_datetime(df["S_2"], format="%Y-%m-%d")


def make_features(train, test):
    START_TIME = time.perf_counter()

    train_customer_ids = list(train.index.unique())
    test_customer_ids = list(test.index.unique())

    df = pd.concat([train, test], axis=0)
    cat_features = CAT_FEATURES.copy()
    num_features = [col for col in df.columns if col not in cat_features]

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

    del date_df
    gc.collect()

    print(f"make date features: {time.perf_counter() - START_TIME:.2f} seconds")

    # fillna
    for col in num_features:
        df[col].fillna(0, inplace=True)
    print(f"process num features: {time.perf_counter() - START_TIME:.2f} seconds")

    # dummies
    df = pd.get_dummies(df, columns=cat_features, drop_first=True)
    cat_features = [col for col in df.columns if col not in num_features]

    print(f"process cat features: {time.perf_counter() - START_TIME:.2f} seconds")

    # dropcols
    df.drop(columns="S_2", inplace=True)
    num_features.remove("S_2")

    print(f"all process completed: {time.perf_counter() - START_TIME:.2f} seconds")

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
    train_labels = train_labels.set_index("customer_ID").sort_index().to_numpy().reshape(-1)

    # temporal
    train.to_pickle("train.pkl")
    test.to_pickle("test.pkl")

    train_ids = np.array(train.index.unique().sort_values())
    test_ids = np.array(test.index.unique().sort_values())

    def reshape_pad(x):
        l, c = x.shape
        return np.pad(x, ((13 - l, 0), (0, 0))).astype(np.float32)

    # train
    if TYPE == "train":
        train_groups = train.groupby(level=0)
        train = np.stack([reshape_pad(g.to_numpy()) for _, g in tqdm(train_groups)], axis=0)

        if not DEBUG:
            np.save(os.path.join(INPUT_DATA_SEQUENTIAL_DIR, "train_ids.npy"), train_ids)
            np.save(os.path.join(INPUT_DATA_SEQUENTIAL_DIR, "train_labels.npy"), train_labels)
            np.save(os.path.join(INPUT_DATA_SEQUENTIAL_DIR, "train.npy"), train)
        del train
        gc.collect()

    # public
    elif TYPE == "public":
        public_customer_ids = np.load(os.path.join(INPUT_CUSTOMER_IDS_DIR, "public.npy"), allow_pickle=True)
        if DEBUG:
            public_groups = test.loc[test.index.isin(public_customer_ids)].groupby(level=0)
        else:
            public_groups = test.loc[public_customer_ids].groupby(level=0)
        public = np.stack([reshape_pad(g.to_numpy()) for _, g in tqdm(public_groups)], axis=0)

        if not DEBUG:
            np.save(os.path.join(INPUT_DATA_SEQUENTIAL_DIR, "public.npy"), public)
            np.save(os.path.join(INPUT_DATA_SEQUENTIAL_DIR, "public_ids.npy"), public_customer_ids)
        del public
        gc.collect()

    # private
    elif TYPE == "private":
        private_customer_ids = np.load(os.path.join(INPUT_CUSTOMER_IDS_DIR, "private.npy"), allow_pickle=True)
        if DEBUG:
            private_groups = test.loc[test.index.isin(private_customer_ids)].groupby(level=0)
        else:
            private_groups = test.loc[private_customer_ids].groupby(level=0)
        private = np.stack([reshape_pad(g.to_numpy()) for _, g in tqdm(private_groups)], axis=0)

        if not DEBUG:
            np.save(os.path.join(INPUT_DATA_SEQUENTIAL_DIR, "private.npy"), private)
            np.save(os.path.join(INPUT_DATA_SEQUENTIAL_DIR, "private_ids.npy"), private_customer_ids)
        del private
        gc.collect()

    if not DEBUG:
        # feature names
        joblib.dump(num_features, os.path.join(INPUT_DATA_SEQUENTIAL_DIR, "num_features.pkl"))
        joblib.dump(cat_features, os.path.join(INPUT_DATA_SEQUENTIAL_DIR, "cat_features.pkl"))


def main():
    os.chdir("/workspaces/amex-default-prediction/work")
    prepare_data()


if __name__ == "__main__":
    main()
