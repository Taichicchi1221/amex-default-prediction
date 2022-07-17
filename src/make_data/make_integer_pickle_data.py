# read dataset from https://www.kaggle.com/datasets/raddar/amex-data-integer-dtypes-parquet-format and convert dtypes to use pd.NA in integer dtypes

import os
import gc
from tqdm import tqdm
import pandas as pd


def convert_dtypes(df):
    for col in tqdm(df.columns):
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype(pd.Float32Dtype())
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].replace(-1, pd.NA).astype(pd.Int16Dtype())
        if pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].astype(pd.StringDtype())


def main():
    os.chdir("../../work")
    for TYPE in ["train", "test"]:
        print("#" * 30, TYPE, "#" * 30)
        df = pd.read_parquet(f"../input/amex-data-integer-dtypes-parquet-format/{TYPE}.parquet")
        convert_dtypes(df)
        df.sort_values(["customer_ID", "S_2"], inplace=True)
        df.to_pickle(f"../input/amex-integer-pickle/{TYPE}.pkl")

        del df
        gc.collect()


if __name__ == "__main__":
    main()
