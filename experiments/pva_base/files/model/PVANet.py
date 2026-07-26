"""
PVANet — Position-Velocity-Acceleration Network for 3D Hand Joint Denoising
============================================================================
在 DSTFormer 骨干上增加速度/加速度预测头，通过多任务学习约束时序一致性。

架构:
  Input: [B, V, F, J, 3] noisy 3D joints
    ↓ Normalize
    ↓ DSTFormer backbone
    ↓ Shared features [B, V, F, d_model]
  ┌──────┬──────────┬──────────────┐
  │ Pos  │   Vel    │     Acc      │
  │ Head │  Head    │    Head      │
  ├──────┼──────────┼──────────────┤
  │[B,V, │ [B,V,    │ [B,V,        │
  │ F,J,3]│ F,J,3]  │  F,J,3]      │
  └──────┴──────────┴──────────────┘

Loss:
  L = L_pos + w_v * L_vel + w_a * L_accel

用法 (seq_config.py):
  cfg.backbone = 'PVANet'

依赖:
  - DSTFormer from DSF.py (不修改)
  - SmoothNetLoss from loss.py (不修改)
"""

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

from model.DSF import DSTFormer
from model.loss import SmoothNetLoss, SmoothL1Loss


class PVANetModel(nn.Module):
    """
    Position-Velocity-Acceleration Network for 3D joint sequence refinement.

    Args:
        num_frame:   sequence length F (default 15)
        num_joints:  number of hand joints J (default 21)
        num_view:    number of input views V (default 1)
        dim_feat:    backbone feature dimension (default 256)
        depth:       number of DSTFormer blocks (default 2)
        w_vel:       velocity loss weight (default 0.1)
        w_accel:     acceleration loss weight (default 0.05)
        sm_w_accel:  SmoothNet acceleration weight (default 0.1)
        sm_w_pos:    SmoothNet position weight (default 1.0)
    """

    def __init__(self,
                 num_frame=15,
                 num_joints=21,
                 num_view=1,
                 dim_feat=256,
                 depth=2,
                 w_vel=0.1,
                 w_accel=0.05,
                 sm_w_accel=0.1,
                 sm_w_pos=1.0):
        super().__init__()
        self.num_frame = num_frame
        self.num_joints = num_joints
        self.num_view = num_view
        self.dim_feat = dim_feat
        self.w_vel = w_vel
        self.w_accel = w_accel

        # ─── 共享骨干 (DSTFormer) ─────────────────────────
        self.backbone = DSTFormer(
            dim_in=3, dim_feat=dim_feat, dim_rep=dim_feat,
            depth=depth, num_heads=8, mlp_ratio=4,
            num_joints=num_joints, maxlen=num_frame,
        )

        # ─── 三个并行预测头 ────────────────────────────────
        self.pos_head = nn.Sequential(
            nn.Linear(dim_feat, dim_feat),
            nn.ReLU(),
            nn.Linear(dim_feat, num_joints * 3),
        )
        self.vel_head = nn.Sequential(
            nn.Linear(dim_feat, dim_feat),
            nn.ReLU(),
            nn.Linear(dim_feat, num_joints * 3),
        )
        self.acc_head = nn.Sequential(
            nn.Linear(dim_feat, dim_feat),
            nn.ReLU(),
            nn.Linear(dim_feat, num_joints * 3),
        )

        # ─── Loss 函数 ─────────────────────────────────────
        self.sm_loss = SmoothNetLoss(w_accel=sm_w_pos, w_pos=sm_w_accel)
        self.l1_loss = nn.L1Loss(reduction='none')

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # ─── 辅助：从 GT 计算速度/加速度目标 ──────────────
    @staticmethod
    def compute_vel_accel(joints, mask):
        """
        joints: [B, V, F, J, 3]
        mask:   [B, V, F, J]   float, 1=valid
        returns:
          vel_target:  [B, V, F, J, 3]  (最后一帧重复)
          acc_target:  [B, V, F, J, 3]  (最后两帧重复)
          vel_mask:    [B, V, F, J]
          acc_mask:    [B, V, F, J]
        """
        B, V, F, J, _ = joints.shape
        device = joints.device

        # ── 速度 = J[t+1] - J[t] ──
        vel = joints[:, :, 1:, :, :] - joints[:, :, :-1, :, :]          # [B, V, F-1, J, 3]
        vel = torch.cat([vel, vel[:, :, -1:, :, :]], dim=2)              # [B, V, F, J, 3]

        # ── 加速度 = J[t+2] - 2*J[t+1] + J[t] ──
        acc = joints[:, :, 2:, :, :] - 2 * joints[:, :, 1:-1, :, :] + joints[:, :, :-2, :, :]
        acc = torch.cat([acc, acc[:, :, -1:, :, :], acc[:, :, -1:, :, :]], dim=2)

        # ── mask ──
        vel_mask = mask[:, :, :-1, :] * mask[:, :, 1:, :]                # 需要连续两帧有效
        vel_mask = torch.cat([vel_mask, vel_mask[:, :, -1:, :]], dim=2)

        acc_mask = mask[:, :, :-2, :] * mask[:, :, 1:-1, :] * mask[:, :, 2:, :]
        acc_mask = torch.cat([acc_mask, acc_mask[:, :, -1:, :], acc_mask[:, :, -1:, :]], dim=2)

        return vel, acc, vel_mask.float(), acc_mask.float()

    # ─── 归一化（与 FusionModel 一致） ─────────────────
    @staticmethod
    def normalize(joints, center, in_val):
        norm = 300
        B, V, F, J, _ = joints.shape
        seq_center = (center * in_val).sum(dim=2).sum(dim=2) / (in_val.sum(dim=2).sum(dim=2) + 1e-8)
        seq_center = seq_center.reshape([B, V, 1, 1, 3])
        return (joints - seq_center) / norm, seq_center, norm

    def forward(self, inputs, targets, meta_info):
        """
        Training:   return (outs, loss_dict, error_dict)
        Inference:  return (outs)
        """
        joints_in = inputs["joint_xyz"].cuda()
        joints_gt = targets["joint_xyz"].cuda()
        center = meta_info['center_xyz'].cuda()
        in_val = meta_info['joint_in_val'].cuda().reshape(joints_in.shape[:4] + (1,))
        continuous_val = meta_info['continuous_val'].cuda()

        B, V, F, J, _ = joints_in.size()

        # ── 归一化 ──
        joints_norm_in, seq_center, norm_val = self.normalize(joints_in, center, in_val)
        joints_norm_gt = (joints_gt - seq_center) / norm_val

        # ── 共享骨干 (需要 [B, F, J, 3]，展开 V 维度) ──
        joints_norm_in_flat = rearrange(joints_norm_in, 'b v f j c -> (b v) f j c')
        x = self.backbone(joints_norm_in_flat)  # [(B*V), F, J, dim_feat]

        # ── 三个预测头 ──
        # 在 J 维度做 mean pooling 得到 [(B*V), F, dim_feat]
        feat = x.mean(dim=2)  # [(B*V), F, dim_feat]
        feat = rearrange(feat, '(b v) f c -> b v f c', b=B, v=V)  # [B, V, F, dim_feat]

        joints_norm_pred = self.pos_head(feat).reshape(B, V, F, J, 3)     # [B,V,F,J,3]
        vel_pred = self.vel_head(feat).reshape(B, V, F, J, 3)             # [B,V,F,J,3]
        acc_pred = self.acc_head(feat).reshape(B, V, F, J, 3)             # [B,V,F,J,3]

        # ── 反归一化 ──
        joints_pred = joints_norm_pred * norm_val + seq_center

        # ── Mask 构建 ──
        val = continuous_val.cuda().view(B, 1, F, 1, 1) * meta_info['joint_gt_val'].cuda().view(B, 1, F, J, 1)
        val_V = val.repeat([1, V, 1, 1, 1])                               # [B,V,F,J,1]
        val_VP = val.repeat([1, V, 1, 1, 3])                              # [B,V,F,J,3]

        if self.training:
            # ── 位置 loss (SmoothNet: 位置 + 隐式加速度平滑) ──
            pos_loss = self.sm_loss(
                joints_norm_pred.reshape(B * V, F, J * 3),
                joints_norm_gt.reshape(B * V, F, J * 3),
                1 - val_VP.view(B * V, F, J * 3),
            )

            # ── 速度/加速度 loss ──
            gt_vel, gt_acc, vel_mask, acc_mask = self.compute_vel_accel(joints_norm_gt, val_V[..., 0])

            vel_loss = (self.l1_loss(vel_pred, gt_vel) * vel_mask.unsqueeze(-1)).sum() / (vel_mask.sum() + 1e-8)
            accel_loss = (self.l1_loss(acc_pred, gt_acc) * acc_mask.unsqueeze(-1)).sum() / (acc_mask.sum() + 1e-8)

            # ── 总 loss ──
            total_loss = pos_loss + self.w_vel * vel_loss + self.w_accel * accel_loss

            # ── 误差指标 ──
            init_error = self._calc_error(joints_in, joints_gt, val_V)
            refine_error = self._calc_error(joints_pred, joints_gt, val_V)

            outs = {'pd_joint_xyz': joints_pred}
            loss_dict = {
                'pos_loss': pos_loss,
                'vel_loss': vel_loss,
                'accel_loss': accel_loss,
                'total_loss': total_loss,
            }
            error_dict = {'init': init_error, 'refine': refine_error}

            return outs, loss_dict, error_dict
        else:
            outs = {'pd_joint_xyz': joints_pred}
            return outs

    @staticmethod
    def _calc_error(joint, gt, val):
        diff = (joint * val - gt * val)
        error = torch.sqrt(torch.sum(diff * diff, dim=-1))
        return error.sum() / (val.sum() + 1e-8)
