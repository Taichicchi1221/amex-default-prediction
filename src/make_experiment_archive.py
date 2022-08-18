import os
import shutil
import subprocess
from pathlib import Path

os.chdir("/workspaces/amex-default-prediction/work")

experiment_run_ids = [
    "0/2c1b142665554faea89d5165a2876cf4",  # exp039
    "0/1e0bfa22eacb4767ac3c5ec04cbbdd7b",  # exp034
    "0/c778d2c6a263487a875e22b1009403d8",  # exp040
    "0/8d41275b922a4cb8ab5396394baaf2f9",  # exp041
    "0/14ba6fc86d4f472badb4bed386c1e718",  # exp037
    "0/f13040b6a1a641e88a9de22ccf65ebf4",  # exp042
    "0/d8e48b61feee49a79b6b0b54b596649a",  # exp043
    "2/895d05be47974f709f433457eddadbaf",  # seq_exp015
    "2/ad26ad078dd54a0b81f35a1682432e00",  # seq_exp026
    "2/0cdebde31d594d5f8d33b4876c9fb9c6",  # seq_exp031
    "2/e6318d18be3547edbd7c137cb0b86a42",  # seq_exp038
    "2/5943053fbe5947d6adb6e0468cb35660",  # seq_exp045
    "2/2ffa4863ffbe421ab93b6104bf001b86",  # seq_exp051
    "2/9f77823610d24b42934f79b37a09b1ad",  # seq_exp055
    "2/e1baea5f5e9349ccb2781c077af2137f",  # seq_exp056
    "3/89b345229e9d4fb283f50fda2850a24f",  # trn_exp001
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
