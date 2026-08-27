"""lazy_npz_30 序列数据读写, 供可视化与分析脚本共用.

数据布局: data/SeqHand/lazy_npz_30/InterHand_test/{field}/<Capture>/.../<seq>.npy
数组形状: (seq_len, 21, 3), 单位 mm.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DATA_ROOT = Path(
    "/home/chenqi/workspace/hand_3d_reconstruction/HTD"
    "/data/SeqHand/lazy_npz_30/InterHand_test"
)
FIELDS = ("gt_joint", "in_joint", "gt_joint_valid", "in_joint_valid", "hand_type")


def list_sequences(dataset: Path = DATA_ROOT) -> list[str]:
    """枚举可用序列(以 gt_joint 为准), 返回相对路径列表."""
    root = Path(dataset) / "gt_joint"
    seqs = []
    for p in sorted(root.rglob("*.npy")):
        rel = p.relative_to(root)
        seqs.append(str(rel)[:-4])
    return seqs


def load_field(rel: str, field: str, dataset: Path = DATA_ROOT):
    """读一个字段, 不存在返回 None. rel 如 Capture0/.../cam400262."""
    p = Path(dataset) / field / f"{rel}.npy"
    if not p.is_file():
        return None
    return np.asarray(np.load(p, mmap_mode="r"), dtype=np.float64)


def load_sequence(rel: str, dataset: Path = DATA_ROOT) -> dict:
    """一次取回 gt/in/valid 等字段的 dict."""
    return {f: load_field(rel, f, dataset) for f in FIELDS}
