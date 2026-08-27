"""手部 21 关节拓扑(InterHand 约定): 关节名、骨架边、指色、父子表与骨架矩阵。

多个实验/可视化共用, 不要在各脚本里复制一份。
"""
from __future__ import annotations

import numpy as np

JOINT_NAMES = [
    "Wrist", "ThumbMCP", "ThumbPIP", "ThumbDIP", "ThumbTIP",
    "IndexMCP", "IndexPIP", "IndexDIP", "IndexTIP",
    "MidMCP", "MidPIP", "MidDIP", "MidTIP",
    "RingMCP", "RingPIP", "RingDIP", "RingTIP",
    "PinkyMCP", "PinkyPIP", "PinkyDIP", "PinkyTIP",
]

# 20 根骨: 腕→5 指根 + 每指 3 节(MCP→PIP→DIP→TIP)
EDGES = [
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
    (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
]

# 每个关节的父关节 (0=腕=root, 无父); EDGES[e] = (parent, child)
PARENT = [
    -1, 0, 1, 2, 3,          # 腕, 拇指链
    0, 5, 6, 7,              # 食指链
    0, 9, 10, 11,            # 中指链
    0, 13, 14, 15,           # 无名链
    0, 17, 18, 19,           # 小指链
]

# 掌骨(腕→指根)骨骼下标, 用于稳健手尺寸估计
METACARPAL_EDGES = [0, 1, 2, 3, 4]

# 每指根关节索引 (腕=0, 拇指=1, 食指=5, 中指=9, 无名=13, 小指=17)
FINGER_ROOTS = [1, 5, 9, 13, 17]

# RGB 0-1: 腕=灰, 拇指=红, 食指=蓝, 中指=绿, 无名=橙, 小指=紫
FINGER_COLORS = {
    0: (0.33, 0.33, 0.33),
    1: (0.84, 0.15, 0.16),
    5: (0.12, 0.47, 0.71),
    9: (0.17, 0.63, 0.17),
    13: (1.00, 0.50, 0.05),
    17: (0.58, 0.40, 0.74),
}


def bone_name(i: int, j: int) -> str:
    return f"{JOINT_NAMES[i]}→{JOINT_NAMES[j]}"


def joint_color(j: int):
    """返回关节 j 的颜色列表(RGB 0-1)。"""
    if j == 0:
        return list(FINGER_COLORS[0])
    for root, c in FINGER_COLORS.items():
        if root <= j <= root + 3:
            return list(c)
    return [0.7, 0.7, 0.7]


def build_M(edges: list | None = None) -> np.ndarray:
    """骨骼向量提取矩阵 M ∈ R^{20×21}: b_c = M p_c (每个坐标独立).

    M[e, child(e)] = +1, M[e, parent(e)] = -1.
    """
    edges = edges if edges is not None else EDGES
    M = np.zeros((len(edges), len(PARENT)), dtype=np.float32)
    for e, (p, c) in enumerate(edges):
        M[e, c] = 1.0
        M[e, p] = -1.0
    return M


def build_A(edges: list | None = None) -> np.ndarray:
    """位置重建矩阵 A ∈ R^{21×20}: p_c = r_c·1 + A b_c.

    A[j, e] = 1 当骨骼 e 在 root→关节 j 的路径上 (树上前向累加).
    """
    edges = edges if edges is not None else EDGES
    edge_idx = {(p, c): e for e, (p, c) in enumerate(edges)}
    A = np.zeros((len(PARENT), len(edges)), dtype=np.float32)
    for j in range(len(PARENT)):
        cur = j
        while PARENT[cur] != -1:
            p = PARENT[cur]
            A[j, edge_idx[(p, cur)]] = 1.0
            cur = p
    return A
