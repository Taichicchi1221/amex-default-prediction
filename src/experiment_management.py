import os
import subprocess
import sys
import glob
import typing
import argparse
import shutil
import joblib


import mlflow


MLFLOW_DIR = "../mlruns"

########################## customer aggregation ##########################
WORKFILE_NAME = "customer_aggregation_work.py"
DEPENDENT_FILES = ["utils.py"]
MLFLOW_EXPERIMENT = "CUSTOMER_ID_AGGREGATION"
WORKFILE_TO_CLEAR = [
    "train.pkl",
    "train_labels.pkl",
    "public.pkl",
    "private.pkl",
    "additive_features.pkl",
    "num_features.pkl",
    "cat_features.pkl",
    "train_ids.npy",
    "public_ids.npy",
    "private_ids.npy",
]
EXPERIMENT_NAME = "exp039"
EXPERIMENT_DESC = "lgbmdart"
########################## customer aggregation ##########################


########################## customer transpose ##########################
# WORKFILE_NAME = "customer_transpose_work.py"
# DEPENDENT_FILES = ["utils.py", "customer_aggregation_work.py"]
# MLFLOW_EXPERIMENT = "CUSTOMER_ID_TRANSPOSE"
# WORKFILE_TO_CLEAR = [
#     "train.pkl",
#     "train_labels.pkl",
#     "public.pkl",
#     "private.pkl",
#     "additive_features.pkl",
#     "num_features.pkl",
#     "cat_features.pkl",
#     "train_ids.npy",
#     "public_ids.npy",
#     "private_ids.npy",
# ]
# EXPERIMENT_NAME = "trn_exp001"
# EXPERIMENT_DESC = "xgboost"
########################## customer aggregation ##########################

########################## sequential model ##############################
# WORKFILE_NAME = "sequential_model_work.py"
# DEPENDENT_FILES = ["utils.py", "process_data_for_sequential_model.py"]
# MLFLOW_EXPERIMENT = "SEQUENTIAL_MODEL"
# WORKFILE_TO_CLEAR = [
#     "train.npy",
#     "train_labels.npy",
#     "public.npy",
#     "private.npy",
#     "num_features.pkl",
#     "cat_features.pkl",
#     "train_ids.npy",
#     "public_ids.npy",
#     "private_ids.npy",
# ]
# EXPERIMENT_NAME = "seq_exp026"
# EXPERIMENT_DESC = "TransformerEncoder + MultiSampleDropoutHead"
########################## sequential model ##############################


########################## adversarial_validation ##########################
# WORKFILE_NAME = "customer_aggregation_adversarial_validation_work.py"
# DEPENDENT_FILES = ["utils.py", "customer_aggregation_work.py"]
# MLFLOW_EXPERIMENT = "ADVERSARIAL_VALIDATION"
# WORKFILE_TO_CLEAR = [
#     "train.pkl",
#     "train_labels.pkl",
#     "public.pkl",
#     "private.pkl",
#     "additive_features.pkl",
#     "num_features.pkl",
#     "cat_features.pkl",
#     "train_ids.npy",
#     "public_ids.npy",
#     "private_ids.npy",
# ]
# EXPERIMENT_NAME = "adv_expxxx"
# EXPERIMENT_DESC = ""
########################## adversarial_validation ##########################


# ====================================================
# util
# ====================================================
def flatten_dict(params: typing.Dict[typing.Any, typing.Any], delimiter: str = "/") -> typing.Dict[str, typing.Any]:
    """
    https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/loggers/base.py
    Flatten hierarchical dict, e.g. ``{'a': {'b': 'c'}} -> {'a/b': 'c'}``.
    Args:
        params: Dictionary containing the hyperparameters
        delimiter: Delimiter to express the hierarchy. Defaults to ``'/'``.
    Returns:
        Flattened dict.
    Examples:
        >>> LightningLoggerBase._flatten_dict({'a': {'b': 'c'}})
        {'a/b': 'c'}
        >>> LightningLoggerBase._flatten_dict({'a': {'b': 123}})
        {'a/b': 123}
        >>> LightningLoggerBase._flatten_dict({5: {'a': 123}})
        {'5/a': 123}
    """

    def _dict_generator(input_dict, prefixes=None):
        prefixes = prefixes[:] if prefixes else []
        if isinstance(input_dict, typing.MutableMapping):
            for key, value in input_dict.items():
                key = str(key)
                if isinstance(value, (typing.MutableMapping, argparse.Namespace)):
                    value = vars(value) if isinstance(value, argparse.Namespace) else value
                    yield from _dict_generator(value, prefixes + [key])
                else:
                    yield prefixes + [
                        key,
                        value if value is not None else str(None),
                    ]
        else:
            yield prefixes + [input_dict if input_dict is None else str(input_dict)]

    return {delimiter.join(keys): val for *keys, val in _dict_generator(params)}


# ====================================================
# work
# ====================================================
def premain(directory):
    shutil.rmtree(directory)
    os.makedirs(directory)
    os.chdir(directory)
    sys.path.append(directory)


# ====================================================
# experiment management
# ====================================================
def manage_experiment():
    # copy src -> exp
    dir_ = f"../src/{EXPERIMENT_NAME}"
    os.makedirs(dir_, exist_ok=False)
    shutil.copy(f"../src/{WORKFILE_NAME}", f"{dir_}/work.py")

    for dependent_file in DEPENDENT_FILES:
        shutil.copy(f"../src/{dependent_file}", f"{dir_}/{dependent_file}")

    # copy src -> work
    shutil.copy(f"../src/{WORKFILE_NAME}", "work.py")

    for dependent_file in DEPENDENT_FILES:
        shutil.copy(f"../src/{dependent_file}", dependent_file)

    # write DESC -> exp
    with open(os.path.join(dir_, "desc.txt"), "w") as f:
        f.write(EXPERIMENT_DESC)


# ====================================================
# save results
# ====================================================
def save_results_main():
    results = joblib.load("results.pkl")

    print(results.params)

    # identify experiment
    client = mlflow.tracking.MlflowClient(MLFLOW_DIR)
    try:
        experiment_id = client.create_experiment(MLFLOW_EXPERIMENT)
    except:
        experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
        experiment_id = experiment.experiment_id

    run = client.create_run(experiment_id)

    # params
    client.log_param(run.info.run_id, "__EXPERIMENT_NAME__", EXPERIMENT_NAME)
    client.log_param(run.info.run_id, "__EXPERIMENT_DESC__", EXPERIMENT_DESC)
    for key, value in flatten_dict(results.params).items():
        client.log_param(run.info.run_id, key, value)

    # metric
    for key, value in results.metrics.items():
        client.log_metric(run.info.run_id, key, value)

    # artifacts
    for filename in glob.glob("./*"):
        client.log_artifact(run.info.run_id, filename)

    return results


def remove_work_files(paths):
    for path in paths:
        if os.path.exists(path):
            os.remove(path)


if __name__ == "__main__":

    # prepare
    premain("/workspaces/amex-default-prediction/work")
    manage_experiment()

    # call work.main
    print("=" * 25, "PROCESS", "=" * 25)
    subprocess.run("python -u work.py", shell=True)
    print("=" * 25, "PROCESS", "=" * 25)

    remove_work_files(WORKFILE_TO_CLEAR)

    # save results
    save_results_main()
