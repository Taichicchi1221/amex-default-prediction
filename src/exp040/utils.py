import os
import sys
import math
import time
import subprocess
import random
import psutil
import joblib
from contextlib import contextmanager

import numpy as np
import pandas as pd

import torch

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


@contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


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
