"""
PVANet — Position-Velocity-Acceleration Network
================================================
遵循 HTD-Refine 论文架构（Sec 3.2, 图 3），适配手部 3D 关节输入。

架构:
  3D 关节 [B, F, J, 3] → JointEmbed → [B, F, D]
    → 8×TemporalTransformer(RoPE) → [B, F, D]
      ├→ Position Head (MLP)         → [B, F, J, 3]
      ├→ Velocity Decoder (conv→MLP)  → [B, F, J, 3]
      └→ Accel Decoder   (conv→MLP)  → [B, F, J, 3]
"""

import torch
import torch.nn as nn
from einops import rearrange
from temporal_transformer import TemporalTransformerDecoder as TemporalTransformer
from model.loss import SmoothNetLoss


# ═══════════════════════════════════════════════════════════
#  速度/加速度 Decoder（适配手部：去掉冗余 transformer）
# ═══════════════════════════════════════════════════════════

class MotionDecoder(nn.Module):
    """
    速度/加速度解码器（手部简化版）。
    论文用 conv → pool → transformer → MLP，但我们的特征已经经过 8 层 transformer，
    所以 MotionDecoder 内部不需要再套 transformer，conv + MLP 就够。

    Args:
        dim: 特征维度
        num_joints: 关节数
        out_dim: 输出维度（速度=3, 加速度=3）
    """
    def __init__(self, dim=256, num_joints=21, out_dim=3):
        super().__init__()
        # Conv1d: 聚合局部时序信息（kernel=3）
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(dim)
        # 输出 MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, num_joints * out_dim),
        )

    def forward(self, x):
        """
        x: [B, F, dim]
        returns: [B, F, J, out_dim]
        """
        B, F, C = x.shape
        x = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B, F, dim]
        x = self.norm(x)
        x = self.mlp(x)  # [B, F, J*3]
        return x.reshape(B, F, -1, 3)  # [B, F, J, 3]


# ═══════════════════════════════════════════════════════════
#  PVANet 主模型
# ═══════════════════════════════════════════════════════════

class PVANetModel(nn.Module):
    """
    PVA-Net: 位置-速度-加速度网络（手部关节版）
    使用 temporal_transformer.py 中的 TemporalTransformer（带 RoPE）。

    输入: [B, V, F, J, 3]
    输出:
      pos: [B, V, F, J, 3]  细化关节位置
      vel: [B, V, F, J, 3]  关节速度
      acc: [B, V, F, J, 3]  关节加速度
    """

    def __init__(self,
                 num_frame=15,
                 num_joints=21,
                 dim_feat=256,
                 depth=8,
                 num_heads=8,
                 dim_feedforward=1024,  # 4x 扩展比（适合 dim=256）
                 dropout=0.1,
                 w_vel=0.1,
                 w_accel=0.05):
        super().__init__()
        self.num_frame = num_frame
        self.num_joints = num_joints
        self.dim_feat = dim_feat
        self.w_vel = w_vel
        self.w_accel = w_accel

        # ─── Joint Embedding ─────────────────────────
        self.joint_embed = nn.Linear(3, dim_feat)
        self.pos_dropout = nn.Dropout(dropout)

        # ─── 主时序 Transformer（8层, RoPE）─────────
        self.temporal_transformer = TemporalTransformer(
            dim=dim_feat,
            depth=depth,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=num_frame,
        )

        # ─── 三个预测头 ────────────────────────────────
        self.pos_head = nn.Sequential(
            nn.Linear(dim_feat, dim_feat),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feat, num_joints * 3),
        )

        self.vel_head = MotionDecoder(
            dim=dim_feat, num_joints=num_joints, out_dim=3,
        )

        self.acc_head = MotionDecoder(
            dim=dim_feat, num_joints=num_joints, out_dim=3,
        )

        # ─── Loss ──────────────────────────────────────
        self.sm_loss = SmoothNetLoss(w_accel=0.1, w_pos=1.0)
        self.l1_loss = nn.L1Loss(reduction='none')

        self._init_weights()
        total = sum(p.numel() for p in self.parameters())
        print(f'[PVANet] {depth} layers, {num_heads} heads, '
              f'dim={dim_feat}, ffn={dim_feedforward}, params={total:,}')

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # ─── 辅助函数 ──────────────────────────────────────

    @staticmethod
    def compute_vel_accel(joints, mask):
        """从 GT 计算速度/加速度目标。joints: [B,V,F,J,3], mask: [B,V,F,J]"""
        B, V, F, J, _ = joints.shape
        vel = joints[:, :, 1:, :, :] - joints[:, :, :-1, :, :]
        vel = torch.cat([vel, vel[:, :, -1:, :, :]], dim=2)
        acc = joints[:, :, 2:, :, :] - 2 * joints[:, :, 1:-1, :, :] + joints[:, :, :-2, :, :]
        acc = torch.cat([acc, acc[:, :, -1:, :, :], acc[:, :, -1:, :, :]], dim=2)
        vel_mask = mask[:, :, :-1, :] * mask[:, :, 1:, :]
        vel_mask = torch.cat([vel_mask, vel_mask[:, :, -1:, :]], dim=2)
        acc_mask = mask[:, :, :-2, :] * mask[:, :, 1:-1, :] * mask[:, :, 2:, :]
        acc_mask = torch.cat([acc_mask, acc_mask[:, :, -1:, :], acc_mask[:, :, -1:, :]], dim=2)
        return vel, acc, vel_mask.float(), acc_mask.float()

    @staticmethod
    def normalize(joints, center, in_val):
        norm = 300
        B, V, F, J, _ = joints.shape
        sc = (center * in_val).sum(dim=2).sum(dim=2) / (in_val.sum(dim=2).sum(dim=2) + 1e-8)
        return (joints - sc.reshape(B, V, 1, 1, 3)) / norm, sc.reshape(B, V, 1, 1, 3), norm

    # ─── 前向 ──────────────────────────────────────────

    def forward(self, inputs, targets, meta_info):
        joints_in = inputs["joint_xyz"].cuda()
        joints_gt = targets["joint_xyz"].cuda()
        center = meta_info['center_xyz'].cuda()
        in_val = meta_info['joint_in_val'].cuda().reshape(joints_in.shape[:4] + (1,))
        continuous_val = meta_info['continuous_val'].cuda()
        B, V, F, J, _ = joints_in.size()

        # 归一化
        jn_in, sc, nv = self.normalize(joints_in, center, in_val)
        jn_gt = (joints_gt - sc) / nv

        # Joint Embed → Temporal Transformer
        j_flat = rearrange(jn_in, 'b v f j c -> (b v) f j c')      # [B*V, F, J, 3]
        feat = self.joint_embed(j_flat).mean(dim=2)                 # [B*V, F, dim]
        feat = self.pos_dropout(feat)
        feat = self.temporal_transformer(feat)                      # [B*V, F, dim]
        feat = rearrange(feat, '(b v) f c -> b v f c', b=B, v=V)   # [B, V, F, dim]

        # 三个预测头
        jn_pred = self.pos_head(feat).reshape(B, V, F, J, 3)
        f_main = feat[:, 0] if V == 1 else feat.mean(dim=1)        # [B, F, dim]
        vel_pred = self.vel_head(f_main).unsqueeze(1)               # [B, 1, F, J, 3]
        acc_pred = self.acc_head(f_main).unsqueeze(1)
        if V > 1:
            vel_pred = vel_pred.repeat(1, V, 1, 1, 1)
            acc_pred = acc_pred.repeat(1, V, 1, 1, 1)

        j_pred = jn_pred * nv + sc

        # Mask
        val = (continuous_val.view(B, 1, F, 1, 1)
               * meta_info['joint_gt_val'].cuda().view(B, 1, F, J, 1))
        val_V = val.repeat(1, V, 1, 1, 1)
        val_VP = val.repeat(1, V, 1, 1, 3)

        if self.training:
            pos_loss = self.sm_loss(
                jn_pred.reshape(B * V, F, J * 3),
                jn_gt.reshape(B * V, F, J * 3),
                1 - val_VP.view(B * V, F, J * 3),
            )
            gt_vel, gt_acc, vm, am = self.compute_vel_accel(jn_gt, val_V[..., 0])
            vel_loss = (self.l1_loss(vel_pred, gt_vel) * vm.unsqueeze(-1)).sum() / (vm.sum() + 1e-8)
            accel_loss = (self.l1_loss(acc_pred, gt_acc) * am.unsqueeze(-1)).sum() / (am.sum() + 1e-8)
            total = pos_loss + self.w_vel * vel_loss + self.w_accel * accel_loss

            init_err = self._calc_error(joints_in, joints_gt, val_V)
            ref_err = self._calc_error(j_pred, joints_gt, val_V)

            return {'pd_joint_xyz': j_pred}, \
                   {'pos_loss': pos_loss, 'vel_loss': vel_loss,
                    'accel_loss': accel_loss, 'total_loss': total}, \
                   {'init': init_err, 'refine': ref_err}
        else:
            return {'pd_joint_xyz': j_pred, 'vel_pred': vel_pred, 'acc_pred': acc_pred}

    @staticmethod
    def _calc_error(joint, gt, val):
        d = (joint * val - gt * val)
        return torch.sqrt(torch.sum(d * d, dim=-1)).sum() / (val.sum() + 1e-8)
