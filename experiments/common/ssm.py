"""可微 Kalman 前向滤波 + RTS 后向平滑(批处理).

对 S 个并行 3 维状态 [pos, vel, acc] 做平滑:
- 状态转移(物理积分器):  A = [[1,1,0],[0,1,1],[0,0,1]],  即 pos 积分 vel、vel 积分 acc;
- 观测模型:              C = [1, 0, 0],  只观测位置分量;
- 过程噪声 Q、观测噪声 R 由外部(模型)提供, 可学习。

纯函数, 无参数 —— 所有可学习量(A/Q/R)由调用方持有。
"""
from __future__ import annotations

import torch


def kalman_smooth(y: torch.Tensor, A: torch.Tensor, Q: torch.Tensor, R: torch.Tensor,
                  p0_scale=(1.0, 1e-3, 1e-3)) -> torch.Tensor:
    """可微 Kalman 前向 + RTS 后向平滑.

    Args:
        y: [B, S, T]   每状态一维位置观测.
        A: [3, 3]      状态转移矩阵(积分器).
        Q: [3, 3]      过程噪声协方差(正定).
        R: [B, S, T]   观测噪声方差.
        p0_scale:      初始位置/速度/加速度方差系数.

    Returns:
        z_sm: [B, S, T, 3]  平滑状态 (pos, vel, acc).
    """
    B, S, T = y.shape
    dev = y.device
    A = A.to(dev)
    Q = Q.to(dev)
    I3 = torch.eye(3, device=dev)

    # ── 初始化: 第一帧直接用观测, 速度/加速度方差给先验 ──
    z = torch.zeros(B, S, 3, device=dev)
    z[..., 0] = y[..., 0]
    P = torch.zeros(B, S, 3, 3, device=dev)
    P[..., 0, 0] = R[..., 0] * p0_scale[0]
    P[..., 1, 1] = p0_scale[1]
    P[..., 2, 2] = p0_scale[2]

    z_pred = torch.zeros(B, S, T, 3, device=dev)
    P_pred = torch.zeros(B, S, T, 3, 3, device=dev)
    z_upd = torch.zeros(B, S, T, 3, device=dev)
    P_upd = torch.zeros(B, S, T, 3, 3, device=dev)

    # ── 前向滤波 ──
    for t in range(T):
        if t > 0:
            z = z @ A.T                       # [B,S,3]
            P = A @ P @ A.T + Q[None, None]   # [B,S,3,3]
        z_pred[..., t, :] = z
        P_pred[..., t, :, :] = P
        # 更新 (C = [1,0,0])
        e = y[..., t] - z[..., 0]             # [B,S] 新息
        Svar = P[..., 0, 0] + R[..., t]       # [B,S] 新息方差
        K = P[..., :, 0] / Svar.unsqueeze(-1)  # [B,S,3] Kalman 增益
        z = z + K * e.unsqueeze(-1)
        P = P - K.unsqueeze(-1) * P[..., 0, :].unsqueeze(-2)  # (I - K C) P
        z_upd[..., t, :] = z
        P_upd[..., t, :, :] = P

    # ── RTS 后向平滑 ──
    z_sm = z_upd.clone()
    P_sm = P_upd.clone()
    for t in range(T - 2, -1, -1):
        # G = P_upd[t] A^T P_pred[t+1]^{-1}
        P_pred_inv = torch.linalg.solve(
            P_pred[..., t + 1, :, :],
            I3.expand(B, S, 3, 3).contiguous(),
        )
        G = P_upd[..., t, :, :] @ A.T @ P_pred_inv            # [B,S,3,3]
        diff = (z_sm[..., t + 1, :] - z_pred[..., t + 1, :]).unsqueeze(-1)  # [B,S,3,1]
        z_sm[..., t, :] = z_upd[..., t, :] + (G @ diff).squeeze(-1)
        Pdiff = P_sm[..., t + 1, :, :] - P_pred[..., t + 1, :, :]
        P_sm[..., t, :, :] = P_upd[..., t, :, :] + G @ Pdiff @ G.transpose(-1, -2)

    return z_sm
