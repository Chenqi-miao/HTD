import torch
import numpy as np
import random
import os # 可选，有时用于环境变量
import cv2

def set_seed(seed):
    """
    设置整个环境的随机种子
    """
    cv2.setRNGSeed(seed)  # OpenCV 的随机种子
    # 设置 Python 内建 random 模块的种子
    random.seed(seed)
    # 设置 NumPy 的种子
    np.random.seed(seed)
    # 设置 PyTorch CPU 的种子
    torch.manual_seed(seed)
    # 如果使用 GPU，设置 PyTorch GPU 的种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # 如果使用多个 GPU，需要为所有 GPU 设置种子
        torch.cuda.manual_seed_all(seed)
        # 解决 cuDNN 可能引入的随机性
        # 设置为 True，会让 cuDNN 使用确定性算法，可能会牺牲一些性能
        torch.backends.cudnn.deterministic = True
        # 禁用 cuDNN 的 benchmarking 功能，该功能会根据输入大小选择最优（但可能不确定）的算法
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def seed_worker(worker_id):
    """
    DataLoader worker 的初始化函数，用于为每个 worker 设置独立的、基于全局种子的种子。
    """
    # 获取在主进程中设置的全局种子（这里假设它已经设置好了）
    # 注意：直接从全局变量获取种子可能不是最健壮的方式，
    # 更好的方式是通过 functools.partial 将种子传递给 worker_init_fn
    # 或者在 DataLoader 创建时捕获它。
    # 这里我们用一个简单的方式，假设主进程的 torch 种子就是基础种子。
    worker_seed = torch.initial_seed() % 2**32 # 获取当前 PyTorch 种子
    np.random.seed(worker_seed + worker_id)   # 为 NumPy 设置种子
    random.seed(worker_seed + worker_id)      # 为 random 模块设置种子
    # 注意：worker 内部的 PyTorch 操作通常会自动继承主进程设置，
    # 但为了保险起见，有时也会在这里显式设置 torch.manual_seed(worker_seed + worker_id)
    # 不过，NumPy 和 random 的种子设置通常更关键，因为它们常用于数据增强。