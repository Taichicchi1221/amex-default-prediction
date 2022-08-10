import os
import shutil
import subprocess
from pathlib import Path

os.chdir("/workspaces/amex-default-prediction/work")

experiment_run_ids = [
    "0/1e0bfa22eacb4767ac3c5ec04cbbdd7b",
    "0/c4e380ce15c845f59948c116deaa9754",
    "0/fe661d3d02354259b8cd13b7e6e1357f",
    "0/14ba6fc86d4f472badb4bed386c1e718",
    "2/895d05be47974f709f433457eddadbaf",
    "2/ad26ad078dd54a0b81f35a1682432e00",
]

shutil.rmtree("../experiments")
os.makedirs("../experiments")

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


os.chdir("../experiments")
subprocess.run("kaggle datasets metadata hutch1221/amex-experiments", shell=True)
subprocess.run("kaggle datasets version --dir-mode zip -m 'update'", shell=True)
