"""通用工具: 日志 / 随机种子 / DataLoader worker 初始化.

供训练脚本 (train.py) 使用, 不依赖具体实验。
"""
import os
import sys
import logging
import numpy as np
import torch


def setup_logger(name, logfile):
    """创建同时写文件和控制台的 logger. 返回 logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile, mode="a")
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%m/%d %H:%M:%S"))
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(sh)
    return logger


def set_seed(s=42):
    """固定随机种子 (python/numpy/torch/cuda), 保证可复现."""
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def seed_worker(w):
    """DataLoader worker 初始化函数 (传入 worker_init_fn).
    避免多 worker 从同一个随机状态取数据."""
    np.random.seed(torch.initial_seed() % 2 ** 32)
