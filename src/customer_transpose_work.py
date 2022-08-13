import gc
from pathlib import Path
from tqdm.auto import tqdm

from box import Box


from utils import *
from customer_aggregation_work import training_main, inference_main, process_input

# ====================================================
# config
# ====================================================
DEBUG = False

SEED = 42
N_SPLITS = 5


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
    "num_boost_round": 10000,
    "early_stopping_rounds": 10000,
    "target_encoding": False,
    "seed": SEED,
    "n_splits": N_SPLITS,
    "params": {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "dart",  # {gbdt, dart}
        "learning_rate": 0.03,
        "num_leaves": 100,
        "min_data_in_leaf": 40,
        "reg_alpha": 0.0,
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
#     "metric_name": "amex",  # {amex, logloss}
#     "num_boost_round": 10000,
#     "early_stopping_rounds": 10000,
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
    ]
    df.drop(columns=dropcols, inplace=True)

    return df


def transpose_features(df):

    values = []

    # dates
    df = pd.concat(
        [
            df,
            df["S_2"].dt.month.rename("month"),
            df["S_2"].dt.weekday.rename("weekday"),
        ],
        axis=1,
    )
    df.drop(columns="S_2", inplace=True)

    # transpose
    for d in tqdm(range(1, 14), desc="transpose"):
        values.append(df.groupby("customer_ID").nth(-d).add_suffix(f"-{14 - d}"))

    transpose_result = pd.concat(values, axis=1)

    del values
    gc.collect()

    cat_features = []
    num_features = []
    for col in transpose_result.columns:
        for cat in CAT_FEATURES:
            if col.startswith(cat):
                cat_features.append(col)
                break
        else:
            num_features.append(col)

    return transpose_result, num_features, cat_features


def prepare_data(debug):
    ### train
    train_ids = np.load(Path(INPUT_CUSTOMER_IDS_DIR, "train.npy"), allow_pickle=True)
    train = pd.read_pickle(Path(INPUT_INTEGER_PICKLE_DIR, "train.pkl"))
    if debug:
        train_ids = np.random.choice(train_ids, size=1000, replace=False)
        train = train.loc[train["customer_ID"].isin(train_ids)].reset_index(drop=True)

    # preprocessing
    train = preprocess(train)

    # main process
    print("#" * 10, "train", "#" * 10)
    train, num_features, cat_features = transpose_features(train)
    num_features, cat_features = process_input(train, filename="train.pkl", num_features=num_features, cat_features=cat_features, TYPE="train")

    ### public
    public_ids = np.load(Path(INPUT_CUSTOMER_IDS_DIR, "public.npy"), allow_pickle=True)
    test = pd.read_pickle(Path(INPUT_INTEGER_PICKLE_DIR, "test.pkl"))
    if debug:
        public_ids = np.random.choice(public_ids, size=1000, replace=False)

    public = test.loc[test["customer_ID"].isin(public_ids)]
    del test
    gc.collect()

    # preprocessing
    public = preprocess(public)

    # main process
    print("#" * 10, "public", "#" * 10)
    public, _, _ = transpose_features(public)
    _, _ = process_input(public, filename="public.pkl", num_features=num_features, cat_features=cat_features, TYPE="public")

    ### ptivate
    private_ids = np.load(Path(INPUT_CUSTOMER_IDS_DIR, "private.npy"), allow_pickle=True)
    test = pd.read_pickle(Path(INPUT_INTEGER_PICKLE_DIR, "test.pkl"))
    if debug:
        private_ids = np.random.choice(private_ids, size=1000, replace=False)

    private = test.loc[test["customer_ID"].isin(private_ids)]
    del test
    gc.collect()

    # preprocessing
    private = preprocess(private)

    # main process
    print("#" * 10, "private", "#" * 10)
    private, _, _ = transpose_features(private)
    _, _ = process_input(private, filename="private.pkl", num_features=num_features, cat_features=cat_features, TYPE="private")

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


def main():
    seed_everything(SEED)
    prepare_data(DEBUG)
    oof_score, g, d = training_main(PARAMS)
    inference_main(PARAMS)

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
