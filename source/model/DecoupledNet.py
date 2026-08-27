"""
DecoupledNet — 时空解耦的 3D 手部关节点精化网络
===============================================

设计思路:
  把 MotionBERT 的 DSTFormer（时空耦合）拆成独立的两阶段：
    Stage 1 — 空间编码: HaMeR 风格的 cross-attention（query ↔ joints），每帧独立
    Stage 2 — 时序编码: 纯 RoPE Transformer 建模帧间关系

  这样做的优势:
    - 可单独分析/可视化空间注意力和时序注意力
    - 可分别调节空间和时序的建模深度
    - 空间建模更高效（cross-attn 替代 O(J²) 的 full self-attn）

架构:
  Input: [B, V, F, J, 3]  noisy 3D joints
    │
    ├── Normalize: (joints - center) / 300
    │
    ├── [Stage 1] SpatialEncoder (HaMeR-style, per-frame)
    │    Joint Embed (3→D) + Joint Position Embed
    │    Cross-Attn Encoder × N:
    │      Query tokens ↔ Joint tokens
    │    → joint_feat [B*V, F, J, D] + frame_feat [B*V, F, D]
    │
    ├── [Stage 2] TemporalEncoder (RoPE)
    │    frame_feat → TemporalTransformerDecoder × M (RoPE)
    │    → temp_feat [B*V, F, D]
    │
    ├── [Fusion] frame_feat + temp_feat → expand + combine with joint_feat
    │    → feat [B, V, F, J, D]
    │
    ├── [Stage 3] Prediction Heads (exp2_spatialmix style)
    │    Pos Head: Linear(D → 3) → [B, V, F, J, 3]
    │    Vel Head: Conv1d + LN + Linear → [B, V, F, J, 3]
    │    Acc Head: Conv1d + LN + Linear → [B, V, F, J, 3]
    │
    └── Loss (PVAnet style)
         L = SmoothNetLoss(pos) + w_v * L1(vel) + w_a * L1(acc)

References:
  HaMeR: https://github.com/geopavlakos/hamer (cross-attn spatial encoder)
  MotionBERT: DSTFormer (original coupled baseline)
  HTD-Refine (PVAnet): position + velocity + acceleration multi-task loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from HTD.source.model.drop import DropPath
from HTD.pvanet.rope import RotaryEmbedding, apply_rotary


# ========================================================================
# 工具函数
# ========================================================================

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    import warnings, math
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_.")
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)


def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


# ========================================================================
# Temporal Transformer Decoder (RoPE) — 内嵌实现，避免导入依赖
# ========================================================================

class _TemporalMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TemporalSelfAttention(nn.Module):
    """纯时序自注意力 (带 RoPE) — 参考 HTD/pvanet/temporal_transformer.py"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., use_rope=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.use_rope = use_rope
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        if use_rope:
            self.rope = RotaryEmbedding(dim=head_dim, max_len=243)

    def forward(self, x):
        B, F, C = x.shape
        qkv = self.qkv(x).reshape(B, F, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.use_rope:
            cos, sin = self.rope(q)
            q = apply_rotary(q, cos, sin)
            k = apply_rotary(k, cos, sin)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, F, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class _TemporalBlock(nn.Module):
    """时序 Transformer Block: Self-Attn + MLP + LN + DropPath"""
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, use_rope=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = TemporalSelfAttention(
            dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, use_rope)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = _TemporalMLP(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TemporalTransformerDecoder(nn.Module):
    """
    N 层时序 Transformer Decoder (RoPE)

    Input:  [B, F, D]
    Output: [B, F, D]
    """
    def __init__(self, dim=256, depth=8, num_heads=8, mlp_ratio=4.,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 use_rope=True, norm_layer=nn.LayerNorm):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            _TemporalBlock(dim, num_heads, mlp_ratio, True, None,
                           drop_rate, attn_drop_rate, dpr[i],
                           norm_layer=norm_layer, use_rope=use_rope)
            for i in range(depth)])
        self.norm = norm_layer(dim)
        self.apply(_init_weights)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x


# ========================================================================
# Stage 1: 空间编码器 — HaMeR 风格 Cross-Attention
# ========================================================================

class CrossAttention(nn.Module):
    """Query ↔ Context 交叉注意力（HaMeR 风格）"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, context):
        """query: [B, Nq, D], context: [B, Nc, D] → [B, Nq, D]"""
        B = query.shape[0]
        q = self.q(query).reshape(B, -1, self.num_heads, query.shape[-1] // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(context).reshape(B, -1, self.num_heads, context.shape[-1] // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(context).reshape(B, -1, self.num_heads, context.shape[-1] // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, query.shape[-1])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionBlock(nn.Module):
    """
    Cross-Attention Block（HaMeR 风格）
    Self-Attn(query) → Cross-Attn(query→joints) → FFN
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = CrossAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, query, joints):
        """query: [B, Nq, D], joints: [B, J, D] → query_out: [B, Nq, D]"""
        # Self-attention on query
        q = query + self.drop_path(self.self_attn(self.norm1(query), self.norm1(query)))
        # Cross-attention: query → joints
        q = q + self.drop_path(self.cross_attn(self.norm2(q), joints))
        # FFN
        q = q + self.drop_path(self.ffn(self.norm3(q)))
        return q


class SpatialEncoder(nn.Module):
    """
    HaMeR 风格的空间编码器

    每帧独立处理：用 query tokens cross-attend 到 joint tokens，
    提取空间结构特征。

    Args:
        dim: 特征维度 (默认 256)
        num_joints: 关节数 (默认 21)
        num_queries: query token 数 (默认 4)
        depth: Cross-attention 层数 (默认 2)
        num_heads: 注意力头数 (默认 8)
        mlp_ratio: MLP 扩展比 (默认 4)

    Input:  [B, F, J, 3]   3D 关节序列
    Output: joint_feat  [B, F, J, D]  关节特征
            frame_feat  [B, F, D]     帧级特征（query 聚合）
    """
    def __init__(self, dim=256, num_joints=21, num_queries=4,
                 depth=2, num_heads=8, mlp_ratio=4,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1):
        super().__init__()
        self.dim = dim
        self.num_joints = num_joints
        self.num_queries = num_queries

        # Joint embedding: 3D coordinates → features
        self.joint_embed = nn.Linear(3, dim)

        # Joint position encoding (learnable, 关节点索引)
        self.joint_pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim))
        trunc_normal_(self.joint_pos_embed, std=.02)

        # Query tokens (learnable, 每帧独立)
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, dim))
        trunc_normal_(self.query_tokens, std=.02)

        # Cross-attention blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(dim, num_heads, mlp_ratio,
                                drop=drop_rate, attn_drop=attn_drop_rate,
                                drop_path=dpr[i])
            for i in range(depth)
        ])

        # Query → frame feature aggregator
        self.query_pool = nn.Sequential(
            nn.Linear(dim * num_queries, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

        self.norm = nn.LayerNorm(dim)
        self.apply(_init_weights)

    def forward(self, x):
        """
        x: [B, F, J, 3] — 归一化后的 3D 关节序列
        """
        B, F, J, _ = x.shape

        # 拍平 batch 和 frame 维度，每帧独立编码
        x_flat = x.reshape(B * F, J, 3)                  # [B*F, J, 3]

        # Joint embedding + position
        j_feat = self.joint_embed(x_flat)                 # [B*F, J, D]
        j_feat = j_feat + self.joint_pos_embed            # + 位置编码

        # Query tokens（扩展 batch）
        q = self.query_tokens.expand(B * F, -1, -1)       # [B*F, Nq, D]

        # Cross-attention 编码（每帧独立）
        for blk in self.blocks:
            q = blk(q, j_feat)                            # [B*F, Nq, D]

        q = self.norm(q)

        # 从 query 聚合帧级特征
        q_flat = q.reshape(B * F, -1)                     # [B*F, Nq*D]
        frame_feat = self.query_pool(q_flat)              # [B*F, D]

        # Joint 特征（用于后续与时序特征融合）
        joint_feat = j_feat                               # [B*F, J, D]

        # 恢复时序维度
        joint_feat = joint_feat.reshape(B, F, J, -1)      # [B, F, J, D]
        frame_feat = frame_feat.reshape(B, F, -1)         # [B, F, D]

        return joint_feat, frame_feat


# ========================================================================
# Stage 2: 时序编码器 — RoPE Transformer
# ========================================================================

# 直接复用 HTD/pvanet/temporal_transformer.py 中的 TemporalTransformerDecoder
# 输入: [B, F, D] → 输出: [B, F, D]


# ========================================================================
# Stage 3: 预测头 — exp2_spatialmix 风格
# ========================================================================

class VelHead(nn.Module):
    """速度/加速度预测头 (Conv1d + LayerNorm + Linear)"""
    def __init__(self, dim=256):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, padding=1)
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, 3)

    def forward(self, x):
        """x: [B, F, J, D] → [B, F, J, 3]"""
        B, F, J, D = x.shape
        x = x.reshape(B * J, F, D).permute(0, 2, 1)      # [B*J, D, F]
        x = self.conv(x).permute(0, 2, 1)                 # [B*J, F, D]
        out = self.fc(self.norm(x))                       # [B*J, F, 3]
        return out.reshape(B, J, F, 3).permute(0, 2, 1, 3)  # [B, F, J, 3]


class AccHead(nn.Module):
    """加速度预测头（与 VelHead 相同结构，独立参数）"""
    def __init__(self, dim=256):
        super().__init__()
        self.core = VelHead(dim)
    def forward(self, x):
        return self.core(x)


# ========================================================================
# Loss — PVAnet 风格
# ========================================================================

class SmoothNetLoss(nn.Module):
    """SmoothNetLoss: L1(pos) + w_a * L1(accel)"""
    def __init__(self, w_accel=0.1, w_pos=1.0):
        super().__init__()
        self.w_accel = w_accel
        self.w_pos = w_pos

    def _masked_l1(self, pred, target, mask):
        mask = 1 - mask  # SmoothNet 用 1-mask（0=有效）
        N = mask.sum(dtype=torch.float32)
        return F.l1_loss(pred * mask, target * mask, reduction='sum') / (N + 1e-8)

    def forward(self, pred, gt, mask):
        """pred, gt: [B, F, J*3], mask: [B, F, J*3] (0=valid)"""
        pred = pred.permute(0, 2, 1)   # [B, J*3, F]
        gt = gt.permute(0, 2, 1)
        mask = mask.permute(0, 2, 1)

        # 位置 loss
        loss_pos = self._masked_l1(pred, gt, mask)

        # 加速度 loss（时序平滑性）
        acc_pred = pred[:, :, :-2] - 2 * pred[:, :, 1:-1] + pred[:, :, 2:]
        acc_gt = gt[:, :, :-2] - 2 * gt[:, :, 1:-1] + gt[:, :, 2:]
        acc_valid = (mask[:, :, :-2] + mask[:, :, 1:-1] + mask[:, :, 2:]) > 0
        loss_acc = self._masked_l1(acc_pred, acc_gt, (1 - acc_valid.float()))

        return self.w_pos * loss_pos + self.w_accel * loss_acc


class PVALoss(nn.Module):
    """
    PVAnet 风格的多任务 Loss

    L = w_p * pos_loss + w_v * vel_loss + w_a * accel_loss

    pos_loss:  SmoothNetLoss (位置 L1 + 隐式加速度平滑)
    vel_loss:  Masked L1 (预测速度 vs GT 差分速度)
    accel_loss: Masked L1 (预测加速度 vs GT 差分加速度)

    推荐权重 (基于实测误差 pos~10mm, vel/acc~3mm):
      均衡:       w_p=1.0, w_v=3.0, w_a=3.0
      重 vel/acc: w_p=0.1, w_v=3.0, w_a=3.0
    """
    def __init__(self, w_pos=0.1, w_vel=3.0, w_accel=3.0, sm_w_accel=0.1, sm_w_pos=1.0):
        super().__init__()
        self.w_pos = w_pos
        self.w_vel = w_vel
        self.w_accel = w_accel
        self.sm_loss = SmoothNetLoss(w_accel=sm_w_accel, w_pos=sm_w_pos)
        self.l1 = nn.L1Loss(reduction='none')

    @staticmethod
    def compute_vel_accel_targets(joints, mask):
        """
        从 GT 计算速度/加速度目标

        joints: [B, V, F, J, 3]
        mask:   [B, V, F, J]  1=valid
        returns:
          vel_target: [B, V, F, J, 3] (最后一帧 pad)
          acc_target: [B, V, F, J, 3] (最后两帧 pad)
          vel_mask:   [B, V, F, J]
          acc_mask:   [B, V, F, J]
        """
        B, V, F, J, _ = joints.shape

        # 速度: ΔJ[t] = J[t+1] - J[t]
        vel = joints[:, :, 1:, :, :] - joints[:, :, :-1, :, :]           # [B,V,F-1,J,3]
        vel = torch.cat([vel, vel[:, :, -1:, :, :]], dim=2)               # [B,V,F,J,3]

        # 加速度: Δ²J[t] = J[t+2] - 2*J[t+1] + J[t]
        acc = joints[:, :, 2:, :, :] - 2 * joints[:, :, 1:-1, :, :] + joints[:, :, :-2, :, :]
        acc = torch.cat([acc, acc[:, :, -1:, :, :], acc[:, :, -1:, :, :]], dim=2)

        # Mask: 需要连续帧都有效
        vel_mask = mask[:, :, :-1, :] * mask[:, :, 1:, :]
        vel_mask = torch.cat([vel_mask, vel_mask[:, :, -1:, :]], dim=2)

        acc_mask = mask[:, :, :-2, :] * mask[:, :, 1:-1, :] * mask[:, :, 2:, :]
        acc_mask = torch.cat([acc_mask, acc_mask[:, :, -1:, :], acc_mask[:, :, -1:, :]], dim=2)

        return vel, acc, vel_mask.float(), acc_mask.float()

    def forward(self, jn_pred, jn_gt, vel_pred, acc_pred, val_V, vP=None):
        """
        jn_pred/norm_gt: 归一化后的坐标  [B, V, F, J, 3]
        vel_pred/acc_pred:               [B, V, F, J, 3]
        val_V: joint + continuous mask   [B, V, F, J, 1]
        vP: position mask for SmoothNet  [B, V, F, J, 3] (1-vP = valid mask)
        """
        B, V, F, J = jn_pred.shape[:4]

        if vP is None:
            vP = val_V.repeat(1, 1, 1, 1, 3)

        # 1. 位置 loss (SmoothNetLoss)
        pos_loss = self.sm_loss(
            jn_pred.reshape(B * V, F, J * 3),
            jn_gt.reshape(B * V, F, J * 3),
            (1 - vP).view(B * V, F, J * 3),
        )

        # 2. 速度/加速度目标 & mask
        gt_vel, gt_acc, vel_mask, acc_mask = self.compute_vel_accel_targets(jn_gt, val_V[..., 0])

        # 3. 速度 loss (masked L1)
        vel_loss = (self.l1(vel_pred, gt_vel) * vel_mask.unsqueeze(-1)).sum() / (vel_mask.sum() + 1e-8)

        # 4. 加速度 loss (masked L1)
        accel_loss = (self.l1(acc_pred, gt_acc) * acc_mask.unsqueeze(-1)).sum() / (acc_mask.sum() + 1e-8)

        # 5. 总 loss
        total_loss = self.w_pos * pos_loss + self.w_vel * vel_loss + self.w_accel * accel_loss

        return {
            'pos_loss': pos_loss,
            'vel_loss': vel_loss,
            'accel_loss': accel_loss,
            'total_loss': total_loss,
        }


# ========================================================================
# 主模型: DecoupledNet
# ========================================================================

class DecoupledNet(nn.Module):
    """
    时空解耦的 3D 手部关节点精化网络

    Architecture:
      SpatialEncoder (HaMeR cross-attn, per-frame)
        → TemporalEncoder (RoPE Transformer)
          → Prediction Heads (exp2_spatialmix style)
            → Loss (PVAnet style)

    Args:
        num_frame:  序列帧数 F (默认 15)
        num_joints: 关节数 J (默认 21)
        num_view:   视角数 V (默认 1)
        dim_feat:   特征维度 (默认 256)
        spatial_depth: 空间 cross-attn 层数 (默认 2)
        temporal_depth: 时序 Transformer 层数 (默认 8)
        num_queries: 空间 query token 数 (默认 4)
        w_vel:      速度 loss 权重 (默认 3.0, 基于 pos 10mm/vel 3mm 误差比)
        w_accel:    加速度 loss 权重 (默认 3.0)

    Input:  [B, V, F, J, 3]  noisy 3D joint sequence
    Output: {
        'pd_joint_xyz': [B, V, F, J, 3]  refined 3D joints
        'vel_pred':     [B, V, F, J, 3]  predicted velocity
        'acc_pred':     [B, V, F, J, 3]  predicted acceleration
    }
    """
    def __init__(self,
                 num_frame=15,
                 num_joints=21,
                 num_view=1,
                 dim_feat=256,
                 spatial_depth=2,
                 temporal_depth=8,
                 num_queries=4,
                 num_heads=8,
                 w_pos=0.1,
                 w_vel=3.0,
                 w_accel=3.0):
        super().__init__()
        self.num_frame = num_frame
        self.num_joints = num_joints
        self.num_view = num_view
        self.dim_feat = dim_feat
        self.w_vel = w_vel
        self.w_accel = w_accel

        # ─── Stage 1: 空间编码 (HaMeR 风格) ────────────
        self.spatial_encoder = SpatialEncoder(
            dim=dim_feat,
            num_joints=num_joints,
            num_queries=num_queries,
            depth=spatial_depth,
            num_heads=num_heads,
            mlp_ratio=4,
            drop_rate=0.1,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
        )

        # ─── Stage 2: 时序编码 (RoPE Transformer) ──────
        self.temporal_encoder = TemporalTransformerDecoder(
            dim=dim_feat,
            depth=temporal_depth,
            num_heads=num_heads,
            mlp_ratio=4,
            drop_rate=0.1,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            use_rope=True,
        )

        # ─── Fusion: 时序特征 → 关节特征融合 ──────────
        self.temp_to_joint = nn.Linear(dim_feat, dim_feat)
        self.fusion_norm = nn.LayerNorm(dim_feat)

        # ─── Stage 3: 预测头 (exp2_spatialmix 风格) ────
        self.pos_head = nn.Linear(dim_feat, 3)
        self.vel_head = VelHead(dim_feat)
        self.acc_head = AccHead(dim_feat)

        # ─── Loss ─────────────────────────────────────
        self.pva_loss = PVALoss(w_pos=w_pos, w_vel=w_vel, w_accel=w_accel)

        self.apply(_init_weights)

        p = sum(p.numel() for p in self.parameters())
        print(f'[DecoupledNet] spatial={spatial_depth}, temporal={temporal_depth}, queries={num_queries}, params={p:,}')

    # ─── 归一化 ─────────────────────────────────────
    @staticmethod
    def normalize(joints, center, in_val):
        """中心化 + 缩放"""
        norm = 300
        B, V, F, J, _ = joints.shape
        seq_center = (center * in_val).sum(dim=2).sum(dim=2) / (in_val.sum(dim=2).sum(dim=2) + 1e-8)
        seq_center = seq_center.reshape([B, V, 1, 1, 3])
        return (joints - seq_center) / norm, seq_center, norm

    def forward(self, inputs, targets, meta_info):
        """
        Training:   → (outs, loss_dict, error_dict)
        Inference:  → outs
        """
        ji = inputs["joint_xyz"].cuda()
        jg = targets["joint_xyz"].cuda()
        center = meta_info['center_xyz'].cuda()
        in_val = meta_info['joint_in_val'].cuda().reshape(ji.shape[:4] + (1,))
        cv = meta_info['continuous_val'].cuda()
        B, V, F, J = ji.shape[:4]

        # ── 归一化 ──
        jn_in, seq_center, norm_val = self.normalize(ji, center, in_val)
        jn_gt = (jg - seq_center) / norm_val

        # ══════════════════════════════════════════════
        # 前向传播
        # ══════════════════════════════════════════════

        # ── Stage 1: 空间编码 (每帧独立) ────────────
        # 展开 V 维度，每帧独立处理
        jn_in_flat = rearrange(jn_in, 'b v f j c -> (b v) f j c')  # [B*V, F, J, 3]
        joint_feat, frame_feat = self.spatial_encoder(jn_in_flat)
        # joint_feat: [B*V, F, J, D], frame_feat: [B*V, F, D]

        # ── Stage 2: 时序编码 ──────────────────────
        temp_feat = self.temporal_encoder(frame_feat)  # [B*V, F, D]

        # ── Fusion: 时序特征注入关节特征 ────────────
        temp_expand = self.temp_to_joint(temp_feat).unsqueeze(2)  # [B*V, F, 1, D]
        feat = joint_feat + temp_expand                            # [B*V, F, J, D]
        feat = self.fusion_norm(feat)
        feat = rearrange(feat, '(b v) f j d -> b v f j d', b=B, v=V)  # [B, V, F, J, D]

        # ── Stage 3: 预测头 ─────────────────────────
        # Position head: 每个关节独立预测
        jn_pred = self.pos_head(feat)                                # [B, V, F, J, 3]

        # 从帧级特征预测速度/加速度（取 V 的平均或用 V=0）
        feat_frame = feat[:, 0] if V == 1 else feat.mean(dim=1)     # [B, F, J, D]
        vel_pred = self.vel_head(feat_frame).unsqueeze(1)           # [B, 1, F, J, 3]
        acc_pred = self.acc_head(feat_frame).unsqueeze(1)           # [B, 1, F, J, 3]
        if V > 1:
            vel_pred = vel_pred.repeat(1, V, 1, 1, 1)
            acc_pred = acc_pred.repeat(1, V, 1, 1, 1)

        # ── 反归一化 ──
        jp = jn_pred * norm_val + seq_center

        # ── Mask 构建 ──
        val = cv.view(B, 1, F, 1, 1) * meta_info['joint_gt_val'].cuda().view(B, 1, F, J, 1)
        val_V = val.repeat(1, V, 1, 1, 1)                            # [B, V, F, J, 1]
        val_VP = val.repeat(1, V, 1, 1, 3)                           # [B, V, F, J, 3]

        # ── 输出 ──
        outs = {
            'pd_joint_xyz': jp,
            'vel_pred': vel_pred,
            'acc_pred': acc_pred,
        }

        if self.training:
            # ── Loss ──
            loss_dict = self.pva_loss(
                jn_pred, jn_gt, vel_pred, acc_pred, val_V, val_VP)

            # ── 误差指标 ──
            init_error = self._calc_error(ji, jg, val_V)
            refine_error = self._calc_error(jp, jg, val_V)
            error_dict = {'init': init_error, 'refine': refine_error}

            return outs, loss_dict, error_dict

        return outs

    @staticmethod
    def _calc_error(joint, gt, val):
        diff = (joint * val - gt * val)
        error = torch.sqrt(torch.sum(diff * diff, dim=-1))
        return error.sum() / (val.sum() + 1e-8)


# ========================================================================
# 快速测试
# ========================================================================

if __name__ == '__main__':
    B, V, F, J = 2, 1, 15, 21
    model = DecoupledNet(num_frame=F, num_joints=J, num_view=V,
                         spatial_depth=2, temporal_depth=4, num_queries=4).cuda()

    # 构造 dummy 数据
    ji = torch.randn(B, V, F, J, 3).cuda()
    jg = torch.randn(B, V, F, J, 3).cuda()
    center = torch.ones(B, V, F, 1, 3).cuda()
    iv = torch.ones(B, V, F, J).cuda()
    gv = torch.ones(B, 1, F, J).float().cuda()
    cv = torch.ones(B, F, 1).float().cuda()

    inputs = {'joint_xyz': ji}
    targets = {'joint_xyz': jg}
    meta = {'center_xyz': center, 'joint_in_val': iv,
            'joint_gt_val': gv, 'continuous_val': cv}

    model.train()
    outs, loss, error = model(inputs, targets, meta)
    print(f"Input:       {list(ji.shape)}")
    print(f"Output:      {list(outs['pd_joint_xyz'].shape)}")
    print(f"Vel pred:    {list(outs['vel_pred'].shape)}")
    print(f"Acc pred:    {list(outs['acc_pred'].shape)}")
    for k, v in loss.items():
        print(f"Loss {k}: {v.item():.6f}")
    for k, v in error.items():
        print(f"Error {k}: {v.item():.4f}")
