import os
import sys

import numpy as np
import pandas as pd

INPUT_DIR = "../input/amex-default-prediction"
OUTPUT_DIR = "../input/amex-customer-ids"


def main():
    train = pd.read_csv(os.path.join(INPUT_DIR, "train_data.csv"), usecols=["customer_ID", "S_2"])
    test = pd.read_csv(os.path.join(INPUT_DIR, "test_data.csv"), usecols=["customer_ID", "S_2"])

    train["S_2"] = pd.to_datetime(train["S_2"], format="%Y-%m-%d")
    test["S_2"] = pd.to_datetime(test["S_2"], format="%Y-%m-%d")

    train.sort_values(["customer_ID", "S_2"], inplace=True)
    test.sort_values(["customer_ID", "S_2"], inplace=True)

    train_groups = train.groupby("customer_ID").agg("last")
    test_groups = test.groupby("customer_ID").agg("last")

    train_customer_ids = np.sort(train_groups.index.to_numpy())
    public_customer_ids = np.sort(test_groups.loc[test_groups["S_2"].dt.month == 4].index.to_numpy())
    private_customer_ids = np.sort(test_groups.loc[test_groups["S_2"].dt.month == 10].index.to_numpy())

    np.save(os.path.join(OUTPUT_DIR, "train.npy"), train_customer_ids)
    np.save(os.path.join(OUTPUT_DIR, "public.npy"), public_customer_ids)
    np.save(os.path.join(OUTPUT_DIR, "private.npy"), private_customer_ids)


if __name__ == "__main__":
    os.chdir("/workspaces/amex-default-prediction/work")
    main()
