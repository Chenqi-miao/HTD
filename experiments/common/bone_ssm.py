"""骨向量 + 状态空间平滑器:PVANet 重新设计(见 HTD/设计方案_骨向量SSM.md).

流程: 骨骼向量提取 → 手尺寸归一化 → Kalman+RTS 平滑(骨骼向量流 + root 流)
      → 位置重建 → 输出一致运动学 (pos/vel/acc 来自同一状态, 构造性一致).

- M1: 纯线性 —— 固定积分器 A, 可学习过程/观测噪声 (Q, R);
- M2: 遮挡自适应观测噪声 R_t = g_θ(x_t, valid_t)   (未实现, 预留);
- M3: 非线性动力学修正 f_θ                            (未实现, 预留).

输入: [B, 1, F, J, 3] 带噪 3D 关节(mm), meta 提供 joint_in_val/joint_gt_val/continuous_val.
输出(训练): (o, l, e);  输出(推理): o = {'pd_joint_xyz': [B,1,F,J,3], ...}.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_here = os.path.dirname(os.path.abspath(__file__))          # experiments/common
if _here not in sys.path:
    sys.path.insert(0, _here)
if os.path.dirname(_here) not in sys.path:                   # experiments/
    sys.path.insert(0, os.path.dirname(_here))

from common.hand_topology import EDGES, PARENT, METACARPAL_EDGES, build_M, build_A
from common.ssm import kalman_smooth

# 积分器状态转移: pos 积分 vel, vel 积分 acc
_AC = [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]


class BoneSSM(nn.Module):
    def __init__(self, num_frame=31, num_joints=21,
                 q_init=(1e-4, 1e-3, 1e-2), r_init=1e-3, r_root_init=1e-2,
                 w_vel=0.1, w_accel=0.05, learn_q=True, learn_r=True,
                 adaptive_r=False, nonlinear=False):
        super().__init__()
        self.num_frame, self.num_joints = num_frame, num_joints
        self.n_bones = len(EDGES)
        self.w_vel, self.w_accel = w_vel, w_accel

        # 骨架矩阵 (固定)
        M = build_M(EDGES)            # (20, 21) 提取
        A = build_A(EDGES)            # (21, 20) 重建
        self.register_buffer("M", torch.tensor(M))
        self.register_buffer("Ab", torch.tensor(A))
        self.register_buffer("Ac", torch.tensor(_AC))

        # 噪声 (M1: 可学习标量; q: 过程噪声 diag, r: 观测噪声, r_root: root 观测噪声)
        self.log_q = nn.Parameter(torch.log(torch.tensor(q_init, dtype=torch.float32)),
                                  requires_grad=learn_q)
        self.log_r = nn.Parameter(torch.log(torch.tensor(r_init, dtype=torch.float32)),
                                  requires_grad=learn_r)
        self.log_r_root = nn.Parameter(torch.log(torch.tensor(r_root_init, dtype=torch.float32)),
                                       requires_grad=learn_r)

        self.adaptive_r = adaptive_r
        self.nonlinear = nonlinear
        if self.adaptive_r or self.nonlinear:
            raise NotImplementedError("M2(自适应R)/M3(非线性f) 尚未实现, 先跑 M1 纯线性。")

        n = sum(p.numel() for p in self.parameters())
        print(f"[BoneSSM] M1 纯线性 | bones={self.n_bones} params={n:,}")

    # ─────────────────────────── 工具 ───────────────────────────

    def _bone_valid(self, joint_valid: torch.Tensor) -> torch.Tensor:
        """关节有效掩码 → 每根骨有效(两端都有效). joint_valid: [B,F,J] → [B,F,20]."""
        va = joint_valid[:, :, [p for p, _ in EDGES]]
        vb = joint_valid[:, :, [c for _, c in EDGES]]
        return (va * vb).bool()

    def _masked_smoothl1(self, pred, target, mask) -> torch.Tensor:
        d = torch.abs(pred - target)
        loss = torch.where(d < 1.0, 0.5 * d * d, d - 0.5)
        return (loss * mask).sum() / (mask.sum() + 1e-8)

    # ─────────────────────────── forward ───────────────────────────

    def forward(self, inputs, targets, meta_info):
        dev = self.M.device
        ji = inputs["joint_xyz"].to(dev)[:, 0]       # [B,F,J,3] 带噪输入
        jg = targets["joint_xyz"].to(dev)[:, 0]      # [B,F,J,3] GT
        B, F, J, _ = ji.shape

        iv = meta_info["joint_in_val"].to(dev)[:, 0]     # [B,F,J] 输入有效
        gv = meta_info["joint_gt_val"].to(dev)[:, 0]     # [B,F,J] GT 有效
        cv = meta_info["continuous_val"].to(dev)         # [B,F] 连续帧

        # ── ① 骨骼向量 + root ──
        r = ji[:, :, 0, :]                              # [B,F,3] 腕
        b = torch.einsum("ej,btjc->btec", self.M, ji)   # [B,F,20,3]

        # ── ② 手尺寸归一化 ──
        mc = b[:, :, METACARPAL_EDGES, :].norm(dim=-1).mean(dim=2)   # [B,F]
        h = torch.median(mc, dim=1).values.clamp(min=1e-3)          # [B]
        h4 = h.reshape(B, 1, 1, 1)
        b_n = b / h4
        r_n = r / h.reshape(B, 1, 1)

        # ── ③ 观测 (每状态一维位置) ──
        y = b_n.permute(0, 2, 3, 1).reshape(B, self.n_bones * 3, F)   # [B,60,F]
        yr = r_n.permute(0, 2, 1)                                     # [B,3,F]
        S = self.n_bones * 3

        # ── ④ 观测噪声 R: M1 = 基础标量 + 无效骨放大(M2 换 g_θ) ──
        r_base = torch.exp(self.log_r)
        R = r_base.expand(B, S, F).clone()
        bval = self._bone_valid(iv)                     # [B,F,20] 每骨输入有效
        bval3 = bval.unsqueeze(-1).repeat(1, 1, 1, 3)   # [B,F,20,3]
        invalid = (~bval3).permute(0, 2, 3, 1).reshape(B, S, F)
        R = R * (1.0 + 99.0 * invalid.float())          # 无效骨观测噪声放大 100×
        R_root = torch.exp(self.log_r_root) * torch.ones(B, 3, F, device=dev)

        # ── ⑤ 平滑 ──
        q = torch.exp(self.log_q)                       # [3]
        Q = torch.diag(q)
        z_sm = kalman_smooth(y, self.Ac, Q, R)          # [B,60,F,3]
        zr_sm = kalman_smooth(yr, self.Ac, Q, R_root)   # [B,3,F,3]

        # ── ⑥ 输出运动学 (归一化单位) ──
        nB = self.n_bones
        z3 = lambda z: z[..., 0].reshape(B, nB, 3, F).permute(0, 3, 1, 2)  # [B,F,20,3]
        b_hat_n = z3(z_sm)      # 位置 (骨向量)
        v_bone_n = z_sm[..., 1].reshape(B, nB, 3, F).permute(0, 3, 1, 2)
        a_bone_n = z_sm[..., 2].reshape(B, nB, 3, F).permute(0, 3, 1, 2)
        r_hat_n = zr_sm[..., 0].permute(0, 2, 1)        # [B,F,3]
        rv_n = zr_sm[..., 1].permute(0, 2, 1)
        ra_n = zr_sm[..., 2].permute(0, 2, 1)

        # ── ⑦ 位置重建 ──
        p_hat_n = r_hat_n.unsqueeze(2) + torch.einsum("je,btec->btjc", self.Ab, b_hat_n)  # [B,F,21,3]
        p_hat = p_hat_n * h4
        # 关节速度/加速度 = root 的 + 路径骨速度/加速度累加 (一致)
        v_joint = (rv_n.unsqueeze(2) + torch.einsum("je,btec->btjc", self.Ab, v_bone_n)) * h4
        a_joint = (ra_n.unsqueeze(2) + torch.einsum("je,btec->btjc", self.Ab, a_bone_n)) * h4

        jp = p_hat.unsqueeze(1)                         # [B,1,F,J,3]
        vp = v_joint.unsqueeze(1)
        ap = a_joint.unsqueeze(1)

        if not self.training:
            return {"pd_joint_xyz": jp, "vel_pred": vp, "acc_pred": ap}

        # ── ⑧ 损失 (位置 + 状态推导运动学, 掩码) ──
        cvf = (cv > 0.5).float().reshape(B, 1, F, 1, 1)
        gvf = (gv > 0.5).float().reshape(B, 1, F, J, 1)
        msk = cvf * gvf                                   # [B,1,F,J,1]

        jg_u = jg.unsqueeze(1)
        pl = self._masked_smoothl1(jp, jg_u, msk)

        # 前向差分 GT 速度/加速度 (对齐状态语义: b_{t+1}=b_t+v_t)
        gv_f = torch.zeros_like(jg)
        gv_f[:, :-1] = jg[:, 1:] - jg[:, :-1]
        ga_f = torch.zeros_like(jg)
        ga_f[:, :-2] = jg[:, 2:] - 2 * jg[:, 1:-1] + jg[:, :-2]
        msk_v = msk[:, :, :-1] * msk[:, :, 1:]            # t 和 t+1 都有效
        msk_a = msk[:, :, :-2] * msk[:, :, 1:-1] * msk[:, :, 2:]
        vl = self._masked_smoothl1(vp[:, :, :-1], gv_f[:, :-1].unsqueeze(1), msk_v)
        al = self._masked_smoothl1(ap[:, :, :-2], ga_f[:, :-2].unsqueeze(1), msk_a)

        mpos = msk[:, 0]                            # [B,F,J,1] 对齐 ji [B,F,J,3]
        d = ji * mpos - jg * mpos
        ie = torch.sqrt((d * d).sum(-1)).sum() / (mpos.sum() + 1e-8)
        d = jp[:, 0] * mpos - jg * mpos
        re = torch.sqrt((d * d).sum(-1)).sum() / (mpos.sum() + 1e-8)

        total = pl + self.w_vel * vl + self.w_accel * al
        return ({"pd_joint_xyz": jp},
                {"pos": pl, "vel": vl, "acc": al, "total": total},
                {"init": ie, "refine": re})
