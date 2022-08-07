import os
import shutil
from pathlib import Path

os.chdir("/workspaces/amex-default-prediction/work")

experiment_run_ids = [
    "0/1e0bfa22eacb4767ac3c5ec04cbbdd7b",
    "0/c4e380ce15c845f59948c116deaa9754",
    "2/895d05be47974f709f433457eddadbaf",
]

os.makedirs("../experiments", exist_ok=True)

for experiment_run_id in experiment_run_ids:
    param_path = Path("../mlruns", experiment_run_id, "params")
    metric_path = Path("../mlruns", experiment_run_id, "metrics")
    artifact_path = Path("../mlruns", experiment_run_id, "artifacts")

    with open(Path(param_path, "__EXPERIMENT_NAME__")) as f:
        experiment_name = f.read()

    archive_path = f"../experiments/{experiment_name}"
    os.makedirs(archive_path, exist_ok=True)

    shutil.copy(Path(artifact_path, "oof.csv"), archive_path)
    shutil.copy(Path(artifact_path, "submission.csv"), archive_path)

    with open(Path(metric_path, "valid_score")) as f:
        score = f.read().split()[1]
        with open(Path(archive_path, "valid_score"), "w") as ff:
            ff.write(score)
    with open(Path(metric_path, "public_score")) as f:
        score = f.read().split()[1]
        with open(Path(archive_path, "public_score"), "w") as ff:
            ff.write(score)
