"""
PVANet — Position-Velocity-Acceleration Network
================================================
遵循 HTD-Refine 论文架构（Sec 3.2, 图 3），适配 3D 手部关节输入。

论文架构:
  ViTPose encoder (frozen) ─→ 8-layer Temporal Transformer w/ RoPE ─→ 3 decoders
  │  (图片特征)                    │                                       │
  │                                │  ├─ Keypoint Decoder (deconv→MLP→heatmap)
  │                                │  ├─ Velocity Decoder (conv→pool→trans→MLP)
  │                                │  └─ Accel Decoder   (conv→pool→trans→MLP)
  3D 关节输入 (我们替换 ViTPose):
  JointEmbed + PosEmbed ─→ 8-layer Temp Transformer w/ RoPE ─→ 3 decoders

依赖:
  - RoPE: 自定义 Rotary Position Embedding
  - 纯 temporal Transformer（非 DSTFormer 的 spatio-temporal）
"""

import torch
import torch.nn as nn
import math
from einops import rearrange


# ═══════════════════════════════════════════════════════════
#  Rotary Position Embedding (RoPE)
#  遵循论文引用的 RoPE [Su et al., 2024]
# ═══════════════════════════════════════════════════════════

class RotaryPositionEmbedding(nn.Module):
    """
    一维旋转位置编码 (RoPE) — 用于 temporal dimension。
    公式: RoPE(x_t) = x_t · cos(mθ) + rotate_half(x_t) · sin(mθ)

    Args:
        dim: 特征维度（需为偶数）
        max_len: 最大序列长度
        base: theta 的 base（论文默认 10000）
    """

    def __init__(self, dim, max_len=243, base=10000.0):
        super().__init__()
        assert dim % 2 == 0, f'RoPE dim ({dim}) must be even'
        self.dim = dim
        self.base = base

        # theta_i = base^(-2i/dim), i ∈ [0, dim/2)
        inv_freq = base ** (-torch.arange(0, dim, 2).float() / dim)
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # 预计算 cos/sin 表
        pos = torch.arange(max_len).float()
        freqs = torch.einsum('i,j->ij', pos, inv_freq)           # [max_len, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)                  # [max_len, dim]
        self.register_buffer('cos', emb.cos(), persistent=False)  # [max_len, dim]
        self.register_buffer('sin', emb.sin(), persistent=False)

    def forward(self, x):
        """
        x: [B, F, dim] 或 [F, B, dim]
        returns: x with RoPE applied to query/key
        """
        seq_len = x.shape[-3]
        cos = self.cos[:seq_len].unsqueeze(-2)   # [F, 1, dim] 或 [F, dim]
        sin = self.sin[:seq_len].unsqueeze(-2)
        return cos, sin


def rotate_half(x):
    """将 x 的最后一维分为两半并交换旋转"""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(q, k, cos, sin):
    """
    q, k: [F, B, num_heads, dim] (seq_first format)
    cos, sin: [F, 1, dim]
    """
    cos = cos.unsqueeze(-2)  # [F, 1, 1, dim]
    sin = sin.unsqueeze(-2)
    q_out = q * cos + rotate_half(q) * sin
    k_out = k * cos + rotate_half(k) * sin
    return q_out, k_out


# ═══════════════════════════════════════════════════════════
#  Temporal Transformer Layer (with RoPE)
# ═══════════════════════════════════════════════════════════

class TemporalSelfAttention(nn.Module):
    """带 RoPE 的时序自注意力 — 在全特征维度上应用 RoPE，再拆头"""

    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dim = dim

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x, rope_cos, rope_sin):
        """
        x: [F, B, dim] (seq_first)
        rope_cos, rope_sin: [F, 1, dim] (全维，广播到 B)
        """
        F, B, C = x.shape

        # QKV 投影: [F, B, 3*dim]
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)  # 每个 [F, B, dim]

        # 在全特征维度上应用 RoPE（论文标准做法：Q、K 旋转）
        q = q * rope_cos + rotate_half(q) * rope_sin
        k = k * rope_cos + rotate_half(k) * rope_sin

        # 拆多头: [F, B, dim] → [F, B, H, D]
        q = q.reshape(F, B, self.num_heads, self.head_dim)
        k = k.reshape(F, B, self.num_heads, self.head_dim)
        v = v.reshape(F, B, self.num_heads, self.head_dim)

        # 注意力
        attn = torch.einsum('f b h d, g b h d -> b h f g', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = torch.einsum('b h f g, g b h d -> f b h d', attn, v)
        out = out.reshape(F, B, C)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class TemporalTransformerBlock(nn.Module):
    """单层时序 Transformer block (Pre-Norm + RoPE)"""

    def __init__(self, dim, num_heads=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = TemporalSelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, rope_cos, rope_sin):
        x = x + self.attn(self.norm1(x), rope_cos, rope_sin)
        x = x + self.ffn(self.norm2(x))
        return x


class TemporalTransformer(nn.Module):
    """纯时序 Transformer（N 层，Pre-Norm，RoPE）— 对应论文 8-layer decoder"""

    def __init__(self, dim=256, depth=8, num_heads=8, dim_feedforward=2048, dropout=0.1, max_len=243):
        super().__init__()
        self.rope = RotaryPositionEmbedding(dim, max_len)
        self.blocks = nn.ModuleList([
            TemporalTransformerBlock(dim, num_heads, dim_feedforward, dropout)
            for _ in range(depth)
        ])

    def forward(self, x):
        """
        x: [B, F, dim] (batch_first)
        returns: [B, F, dim]
        """
        # 转 seq_first (F, B, dim) 便于 RoPE
        x = x.permute(1, 0, 2)  # [F, B, dim]
        cos, sin = self.rope(x)

        for blk in self.blocks:
            x = blk(x, cos, sin)

        x = x.permute(1, 0, 2)  # [B, F, dim]
        return x


# ═══════════════════════════════════════════════════════════
#  速度/加速度 Decoder（论文设计: conv → pool → transformer → MLP）
# ═══════════════════════════════════════════════════════════

class MotionDecoder(nn.Module):
    """
    速度/加速度解码器（论文 Sec 3.2, 图 3）

    架构:
      conv1d (时序) → spatial avg pool → transformer → MLP → 输出

    Args:
        input_dim: 输入特征维度
        hidden_dim: 隐层维度
        num_joints: 关节数
        out_dim: 输出维度（速度=3, 加速度=3）
    """

    def __init__(self, input_dim=256, hidden_dim=256, num_joints=21, out_dim=3):
        super().__init__()
        # Conv1d: 沿时间维度卷积，聚合局部时序信息
        self.conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)

        # 空间池化后的小 transformer
        self.temp_transformer = TemporalTransformer(
            dim=hidden_dim, depth=2, num_heads=4,
            dim_feedforward=1024, max_len=243,
        )

        # 输出 MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_joints * out_dim),
        )

    def forward(self, x):
        """
        x: [B, F, dim] 时序特征
        returns: [B, F, J, out_dim]
        """
        B, F, C = x.shape

        # Conv1d: [B, C, F] → [B, hidden, F]
        x = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B, F, hidden]

        # 小 transformer 进一步建模时序
        x = self.temp_transformer(x)  # [B, F, hidden]

        # MLP 预测
        x = self.mlp(x)  # [B, F, J*3]
        x = x.reshape(B, F, -1, 3)  # [B, F, J, 3]
        return x


# ═══════════════════════════════════════════════════════════
#  PVANet 主模型
# ═══════════════════════════════════════════════════════════

class PVANetModel(nn.Module):
    """
    PVA-Net: 位置-速度-加速度网络（手部关节版）
    遵循 HTD-Refine 论文架构。

    输入: [B, F, J, 3] 归一化 3D 关节序列
    输出:
      pos:      [B, F, J, 3]  细化关节位置
      vel:      [B, F, J, 3]  关节速度
      acc:      [B, F, J, 3]  关节加速度

    架构:
      Joint Embed → Pos Embed → 8×TemporalTransformer(RoPE)
        → [位置头 (MLP)]
        → [速度头 (conv→pool→trans→MLP)]
        → [加速度头 (conv→pool→trans→MLP)]
    """

    def __init__(self,
                 num_frame=15,
                 num_joints=21,
                 dim_feat=256,
                 depth=8,              # alias for transformer_depth
                 transformer_depth=None,
                 num_heads=8,
                 dim_feedforward=2048,
                 dropout=0.1,
                 w_vel=0.1,
                 w_accel=0.05):
        super().__init__()
        td = transformer_depth if transformer_depth is not None else depth
        self.num_frame = num_frame
        self.num_joints = num_joints
        self.dim_feat = dim_feat
        self.w_vel = w_vel
        self.w_accel = w_accel

        # ─── Joint Embedding（替换论文的 ViTPose） ───────
        self.joint_embed = nn.Linear(3, dim_feat)
        self.pos_dropout = nn.Dropout(dropout)

        # ─── 主时序 Transformer（8 层，RoPE）─────────────
        self.temporal_transformer = TemporalTransformer(
            dim=dim_feat,
            depth=td,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=num_frame,
        )

        # ─── 三个解码器 ────────────────────────────────────
        # 位置头: 直接 MLP（回归 3D 关节坐标）
        self.pos_head = nn.Sequential(
            nn.Linear(dim_feat, dim_feat),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feat, num_joints * 3),
        )

        # 速度头: conv → transformer → MLP
        self.vel_head = MotionDecoder(
            input_dim=dim_feat, hidden_dim=dim_feat,
            num_joints=num_joints, out_dim=3,
        )

        # 加速度头: conv → transformer → MLP（与速度头相同架构）
        self.acc_head = MotionDecoder(
            input_dim=dim_feat, hidden_dim=dim_feat,
            num_joints=num_joints, out_dim=3,
        )

        # ─── Loss ──────────────────────────────────────────
        from model.loss import SmoothNetLoss
        self.sm_loss = SmoothNetLoss(w_accel=0.1, w_pos=1.0)
        self.l1_loss = nn.L1Loss(reduction='none')

        self._init_weights()
        print(f'[PVANet] {td} layers, {num_heads} heads, '
              f'dim={dim_feat}, ffn={dim_feedforward}')

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

    @staticmethod
    def compute_vel_accel(joints, mask):
        """从 GT 关节计算速度/加速度目标"""
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
        norm_val = 300
        B, V, F, J, _ = joints.shape
        seq_center = (center * in_val).sum(dim=2).sum(dim=2) / (in_val.sum(dim=2).sum(dim=2) + 1e-8)
        seq_center = seq_center.reshape([B, V, 1, 1, 3])
        return (joints - seq_center) / norm_val, seq_center, norm_val

    def forward(self, inputs, targets, meta_info):
        joints_in = inputs["joint_xyz"].cuda()
        joints_gt = targets["joint_xyz"].cuda()
        center = meta_info['center_xyz'].cuda()
        in_val = meta_info['joint_in_val'].cuda().reshape(joints_in.shape[:4] + (1,))
        continuous_val = meta_info['continuous_val'].cuda()
        B, V, F, J, _ = joints_in.size()

        # 归一化
        joints_norm_in, seq_center, norm_val = self.normalize(joints_in, center, in_val)
        joints_norm_gt = (joints_gt - seq_center) / norm_val

        # 展平 V 维度 (多视角处理)
        joints_flat = rearrange(joints_norm_in, 'b v f j c -> (b v) f j c')  # [B*V, F, J, 3]

        # Joint Embedding: [B*V, F, J, 3] → [B*V, F, J, dim] → [B*V, F, dim] (J 维度 pooling)
        feat = self.joint_embed(joints_flat)  # [B*V, F, J, dim]
        feat = feat.mean(dim=2)               # [B*V, F, dim] 空间池化
        feat = self.pos_dropout(feat)

        # 主时序 Transformer (8层, RoPE)
        feat = self.temporal_transformer(feat)  # [B*V, F, dim]

        # 恢复 batch/view
        feat = rearrange(feat, '(b v) f c -> b v f c', b=B, v=V)

        # ─── 三个预测头 ───
        joints_norm_pred = self.pos_head(feat).reshape(B, V, F, J, 3)

        # 速度/加速度头需要 [B, F, dim] 形式（共用 V 维度的特征）
        feat_main = feat[:, 0] if V == 1 else feat.mean(dim=1)  # [B, F, dim]

        vel_pred = self.vel_head(feat_main).unsqueeze(1)     # [B, 1, F, J, 3]
        acc_pred = self.acc_head(feat_main).unsqueeze(1)     # [B, 1, F, J, 3]

        # 多视角重复
        if V > 1:
            vel_pred = vel_pred.repeat(1, V, 1, 1, 1)
            acc_pred = acc_pred.repeat(1, V, 1, 1, 1)

        # 反归一化
        joints_pred = joints_norm_pred * norm_val + seq_center

        # Mask
        val = (continuous_val.view(B, 1, F, 1, 1)
               * meta_info['joint_gt_val'].cuda().view(B, 1, F, J, 1))
        val_V = val.repeat(1, V, 1, 1, 1)
        val_VP = val.repeat(1, V, 1, 1, 3)

        if self.training:
            # 位置 loss
            pos_loss = self.sm_loss(
                joints_norm_pred.reshape(B * V, F, J * 3),
                joints_norm_gt.reshape(B * V, F, J * 3),
                1 - val_VP.view(B * V, F, J * 3),
            )

            # 速度/加速度 loss
            gt_vel, gt_acc, vel_mask, acc_mask = self.compute_vel_accel(
                joints_norm_gt, val_V[..., 0])

            vel_loss = (self.l1_loss(vel_pred, gt_vel)
                        * vel_mask.unsqueeze(-1)).sum() / (vel_mask.sum() + 1e-8)
            accel_loss = (self.l1_loss(acc_pred, gt_acc)
                          * acc_mask.unsqueeze(-1)).sum() / (acc_mask.sum() + 1e-8)

            total_loss = pos_loss + self.w_vel * vel_loss + self.w_accel * accel_loss

            init_error = self._calc_error(joints_in, joints_gt, val_V)
            refine_error = self._calc_error(joints_pred, joints_gt, val_V)

            outs = {'pd_joint_xyz': joints_pred}
            loss_dict = {
                'pos_loss': pos_loss, 'vel_loss': vel_loss,
                'accel_loss': accel_loss, 'total_loss': total_loss,
            }
            error_dict = {'init': init_error, 'refine': refine_error}
            return outs, loss_dict, error_dict
        else:
            return {'pd_joint_xyz': joints_pred}

    @staticmethod
    def _calc_error(joint, gt, val):
        diff = (joint * val - gt * val)
        error = torch.sqrt(torch.sum(diff * diff, dim=-1))
        return error.sum() / (val.sum() + 1e-8)
