"""
PVANet — Position-Velocity-Acceleration Network (Pure Temporal)
===============================================================
纯时序 Transformer + RoPE，每关节独立处理。

问题定位: 之前 mean pool over J 丢失了关节信息，导致不收敛。
修复: reshape [B, F, J, D] → [B*J, F, D] 保留 J 维度。

架构:
  [B, F, J, 3]
    ↓ Joint Embed
  [B, F, J, D]
    ↓ reshape: J → batch
  [B*J, F, D]
    ↓ TemporalTransformer ×N (RoPE)
  [B*J, F, D]
    ↓ reshape back
  [B, F, J, D]
    ├→ Pos Head (per-joint Linear) → [B, F, J, 3]
    ├→ Vel Head (per-joint Conv1d+MLP) → [B, F, J, 3]
    └→ Acc Head (per-joint Conv1d+MLP) → [B, F, J, 3]
"""

import torch
import torch.nn as nn
from einops import rearrange
from model.temporal_transformer import TemporalTransformerDecoder
from model.loss import SmoothNetLoss


class MotionDecoder(nn.Module):
    """速度/加速度解码器（每关节独立 Conv1d + MLP）"""
    def __init__(self, dim=256, out_dim=3):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, out_dim),
        )

    def forward(self, x):
        B, F, J, C = x.shape
        x = x.reshape(B * J, F, C).permute(0, 2, 1)   # [B*J, C, F]
        x = self.conv(x).permute(0, 2, 1)               # [B*J, F, C]
        x = self.norm(x)
        x = self.head(x)                                 # [B*J, F, 3]
        return x.reshape(B, J, F, 3).permute(0, 2, 1, 3) # [B, F, J, 3]


class PVANetModel(nn.Module):
    def __init__(self,
                 num_frame=15,
                 num_joints=21,
                 dim_feat=256,
                 depth=8,
                 num_heads=8,
                 ff_rate=4,
                 dropout=0.1,
                 w_vel=0.05,
                 w_accel=0.02):
        super().__init__()
        self.num_frame = num_frame
        self.num_joints = num_joints
        self.w_vel = w_vel
        self.w_accel = w_accel

        # Joint Embed (保持 J 维)
        self.joint_embed = nn.Linear(3, dim_feat)
        self.pos_drop = nn.Dropout(dropout)

        # 纯时序 Transformer（J 维展进 batch）
        self.temporal = TemporalTransformerDecoder(
            dim=dim_feat, depth=depth, num_heads=num_heads,
            mlp_ratio=ff_rate, drop_rate=dropout,
            attn_drop_rate=dropout, use_rope=True,
        )

        # 三个预测头（每关节独立输出）
        self.pos_head = nn.Linear(dim_feat, 3)
        self.vel_head = MotionDecoder(dim=dim_feat)
        self.acc_head = MotionDecoder(dim=dim_feat)

        # Loss
        self.sm_loss = SmoothNetLoss(w_accel=0.1, w_pos=1.0)
        self.l1 = nn.L1Loss(reduction='none')

        p = sum(p.numel() for p in self.parameters())
        print(f'[PVANet] depth={depth}, dim={dim_feat}, heads={num_heads}, ff={ff_rate}x, params={p:,}')

    def forward(self, inputs, targets, meta_info):
        ji = inputs["joint_xyz"].cuda()
        jg = targets["joint_xyz"].cuda()
        center = meta_info['center_xyz'].cuda()
        iv = meta_info['joint_in_val'].cuda().reshape(ji.shape[:4] + (1,))
        cv = meta_info['continuous_val'].cuda()
        B, V, F, J = ji.shape[:4]

        # Normalize
        norm = 300
        sc = (center * iv).sum(dim=(2, 3)) / (iv.sum(dim=(2, 3)) + 1e-8)  # [B, V, 1]
        sc = sc.reshape(B, V, 1, 1, 3)
        jn_in = (ji - sc) / norm
        jn_gt = (jg - sc) / norm

        # Joint Embed → [B, V, F, J, D]
        x = self.joint_embed(jn_in)
        x = self.pos_drop(x)

        # 展平 V, J → batch，过时序 Transformer
        x = rearrange(x, 'b v f j d -> (b v j) f d')      # [B*V*J, F, D]
        x = self.temporal(x)                                # [B*V*J, F, D]
        x = rearrange(x, '(b v j) f d -> b v f j d',
                      b=B, v=V, j=J)                      # [B, V, F, J, D]

        # 三个头
        jn_pred = self.pos_head(x)                         # [B, V, F, J, 3]
        fm = x[:, 0] if V == 1 else x.mean(dim=1)          # [B, F, J, D]
        vp = self.vel_head(fm).unsqueeze(1)                 # [B, 1, F, J, 3]
        ap = self.acc_head(fm).unsqueeze(1)
        if V > 1:
            vp = vp.repeat(1, V, 1, 1, 1)
            ap = ap.repeat(1, V, 1, 1, 1)

        j_pred = jn_pred * norm + sc

        # Mask
        val = (cv.view(B, 1, F, 1, 1)
               * meta_info['joint_gt_val'].cuda().view(B, 1, F, J, 1))
        vV = val.repeat(1, V, 1, 1, 1)
        vP = val.repeat(1, V, 1, 1, 3)

        if self.training:
            # 位置 loss
            pl = self.sm_loss(jn_pred.reshape(B * V, F, J * 3),
                              jn_gt.reshape(B * V, F, J * 3),
                              1 - vP.view(B * V, F, J * 3))

            # 速度/加速度 loss
            gv = jn_gt[:, :, 1:] - jn_gt[:, :, :-1]
            gv = torch.cat([gv, gv[:, :, -1:]], dim=2)
            ga = gv[:, :, 1:] - gv[:, :, :-1]
            ga = torch.cat([ga, ga[:, :, -1:]], dim=2)
            vm = (vV[..., :-1, :] * vV[..., 1:, :])
            vm = torch.cat([vm, vm[:, :, -1:]], dim=2)
            am = (vV[..., :-2, :] * vV[..., 1:-1, :] * vV[..., 2:, :])
            am = torch.cat([am, am[:, :, -1:], am[:, :, -1:]], dim=2)

            vl = (self.l1(vp, gv) * vm).sum() / (vm.sum() + 1e-8)
            al = (self.l1(ap, ga) * am).sum() / (am.sum() + 1e-8)

            total = pl + self.w_vel * vl + self.w_accel * al
            return {'pd_joint_xyz': j_pred}, \
                   {'pos': pl, 'vel': vl, 'acc': al, 'total': total}, \
                   {'init': self._ce(ji, jg, vV), 'refine': self._ce(j_pred, jg, vV)}
        else:
            return {'pd_joint_xyz': j_pred, 'vel_pred': vp, 'acc_pred': ap}

    @staticmethod
    def _ce(j, g, v):
        d = (j * v - g * v)
        return torch.sqrt(torch.sum(d * d, dim=-1)).sum() / (v.sum() + 1e-8)
