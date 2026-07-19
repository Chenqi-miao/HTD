"""
MotionEncoder — ViTPose+ 风格的运动编码器（手部版本）
====================================================
从 3D 关节序列中同时提取 2D 关键点、3D 速度、3D 加速度。

架构:
  JointEmbedding (21×3 → d_model)
    → TemporalTransformerDecoder × 8 (RoPE)
      → KeypointDecoder   → 2D 关节位置 [B, F, J, 2]
      → VelocityDecoder   → 3D 速度     [B, F, J, 3]
      → AccelDecoder      → 3D 加速度    [B, F, J, 3]

复用现有组件:
  - DropPath from model/drop.py
  - TemporalTransformerDecoder from model/temporal_transformer.py
  - trunc_normal_ from model/temporal_transformer.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from model.temporal_transformer import TemporalTransformerDecoder, trunc_normal_
from model.vit_encoder import ViTEncoder


class JointEmbedding(nn.Module):
    """
    帧级 Joint Embedding: 将每帧的关节坐标转成 token

    输入: [B, F, J, C=3]
    输出: [B, F, D]   (每帧一个 token)

    支持两种模式:
      - 'concat': 将 J*C 展平后线性映射到 D (默认)
      - 'mean':    对关节维度求均值后映射到 D
    """
    def __init__(self, dim_in=3, num_joints=21, dim_out=256, mode='concat'):
        super().__init__()
        self.mode = mode
        if mode == 'concat':
            self.proj = nn.Linear(num_joints * dim_in, dim_out)
        elif mode == 'mean':
            self.proj = nn.Linear(dim_in, dim_out)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward(self, x):
        # x: [B, F, J, C]
        B, F, J, C = x.shape
        if self.mode == 'concat':
            x = x.reshape(B, F, J * C)         # [B, F, J*C]
        else:
            x = x.mean(dim=-2)                  # [B, F, C]
        x = self.proj(x)                         # [B, F, D]
        return x


class TemporalPositionEncoding(nn.Module):
    """
    可学习时序位置编码
    与 DSF.py 中的 temp_embed 保持一致

    输入: [B, F, D]
    输出: [B, F, D] (加位置编码)
    """
    def __init__(self, dim=256, max_len=243):
        super().__init__()
        self.temp_embed = nn.Parameter(torch.zeros(1, max_len, 1, dim))
        trunc_normal_(self.temp_embed, std=.02)

    def forward(self, x):
        # x: [B, F, D]
        F = x.shape[1]
        return x + self.temp_embed[:, :F, 0, :]


class KeypointDecoder2D(nn.Module):
    """
    2D 关键点头

    当前原型: 简单的 MLP 映射（无图像时使用）
    未来替换: Deconv → Heatmap → Argmax（有图像时使用）

    输入: [B, F, D]
    输出: [B, F, J, 2]
    """
    def __init__(self, dim=256, num_joints=21, use_heatmap=False):
        super().__init__()
        self.use_heatmap = use_heatmap
        self.num_joints = num_joints

        if use_heatmap:
            # 热图解码器 (预留，接入图像时使用)
            self.fc = nn.Linear(dim, 64 * 64)  # H=W=64 的热图
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(1, 32, kernel_size=4, stride=2, padding=1),  # 64→128
                nn.ReLU(),
                nn.ConvTranspose2d(32, num_joints, kernel_size=4, stride=2, padding=1),  # 128→256
            )
        else:
            # 简单 MLP 映射 (原型阶段使用)
            self.proj = nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, num_joints * 2),
            )

    def forward(self, x):
        # x: [B, F, D]
        if self.use_heatmap:
            # 热图路径: [B, F, D] → [B, F, 1, 64, 64] → deconv → [B, F, J, 256, 256]
            B, F, D = x.shape
            x = self.fc(x)                                       # [B, F, 4096]
            x = x.reshape(B * F, 1, 64, 64)                      # [B*F, 1, 64, 64]
            x = self.deconv(x)                                   # [B*F, J, 256, 256]
            # softmax over spatial dims
            heatmap = x.reshape(B * F, self.num_joints, -1)      # [B*F, J, 65536]
            heatmap = F.softmax(heatmap, dim=-1)
            heatmap = heatmap.reshape(B * F, self.num_joints, 256, 256)
            # argmax（可微用 soft-argmax）
            ys, xs = torch.meshgrid(
                torch.arange(256, device=x.device, dtype=torch.float32),
                torch.arange(256, device=x.device, dtype=torch.float32),
                indexing='ij')
            coords_x = (heatmap * xs).view(B * F, self.num_joints, -1).sum(-1)  # [B*F, J]
            coords_y = (heatmap * ys).view(B * F, self.num_joints, -1).sum(-1)
            coords = torch.stack([coords_x, coords_y], dim=-1)   # [B*F, J, 2]
            return coords.reshape(B, F, self.num_joints, 2)
        else:
            x = self.proj(x)                                       # [B, F, J*2]
            return x.reshape(x.shape[0], x.shape[1], self.num_joints, 2)


class _TemporalConvBlock(nn.Module):
    """
    Conv1D + Pool + Tiny Transformer → 时序特征提取

    用于 VelocityDecoder 和 AccelDecoder 的共享子模块

    输入: [B, F, D]
    输出: [B, F, D]
    """
    def __init__(self, dim=256):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.ReLU()

    def forward(self, x):
        # x: [B, F, D]
        residual = x
        x = x.permute(0, 2, 1)       # [B, D, F]
        x = self.conv(x)              # [B, D, F]
        x = x.permute(0, 2, 1)       # [B, F, D]
        x = self.norm(x)
        x = self.act(x)
        return x + residual


class VelocityDecoder(nn.Module):
    """
    3D 速度头

    结构:
      Conv1D → Pool → TemporalTransformer → MLP → [B, F, J, 3]

    输入: [B, F, D]
    输出: [B, F, J, 3]
    """
    def __init__(self, dim=256, num_joints=21, num_heads=4, mlp_ratio=2):
        super().__init__()
        self.conv_block = _TemporalConvBlock(dim)
        # 小型时序 Transformer
        self.transformer = TemporalTransformerDecoder(
            dim=dim, depth=2, num_heads=num_heads, mlp_ratio=mlp_ratio,
            drop_rate=0.1, attn_drop_rate=0., drop_path_rate=0.1)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, num_joints * 3),
        )

    def forward(self, x):
        # x: [B, F, D]
        x = self.conv_block(x)         # [B, F, D]
        x = self.transformer(x)        # [B, F, D]
        x = self.proj(x)               # [B, F, J*3]
        return x.reshape(x.shape[0], x.shape[1], -1, 3)


class AccelDecoder(nn.Module):
    """
    3D 加速度头

    结构与 VelocityDecoder 相同

    输入: [B, F, D]
    输出: [B, F, J, 3]
    """
    def __init__(self, dim=256, num_joints=21, num_heads=4, mlp_ratio=2):
        super().__init__()
        self.conv_block = _TemporalConvBlock(dim)
        self.transformer = TemporalTransformerDecoder(
            dim=dim, depth=2, num_heads=num_heads, mlp_ratio=mlp_ratio,
            drop_rate=0.1, attn_drop_rate=0., drop_path_rate=0.1)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, num_joints * 3),
        )

    def forward(self, x):
        # x: [B, F, D]
        x = self.conv_block(x)
        x = self.transformer(x)
        x = self.proj(x)
        return x.reshape(x.shape[0], x.shape[1], -1, 3)


class MotionEncoder(nn.Module):
    """
    完整 ViTPose+ 风格运动编码器

    Args:
        dim_in: 输入关节坐标维度 (默认 3)
        dim_feat: 特征维度 (默认 256)
        num_joints: 手部关节数 (默认 21)
        max_len: 最大序列长度 (默认 243)
        depth: 主 Transformer 层数 (默认 8)
        num_heads: 注意力头数 (默认 8)
        mlp_ratio: MLP 扩展比 (默认 4)
        drop_rate: Dropout 率
        attn_drop_rate: 注意力 Dropout 率
        drop_path_rate: Stochastic Depth 率
        use_rope: 是否使用 RoPE (默认 True)
        use_temp_embed: 是否使用可学习位置编码 (当 use_rope=False 时用)
        use_heatmap: 是否使用热图解码器 (默认 False，用简单 MLP)
        embed_mode: JointEmbedding 模式 ('concat' | 'mean')

    Input:  [B, F, J, C]   3D 关节坐标序列
    Output: dict{
        'joint_2d':     [B, F, J, 2]   2D 关节位置
        'joint_vel':    [B, F, J, 3]   3D 关节速度
        'joint_accel':  [B, F, J, 3]   3D 关节加速度
        'feat':         [B, F, D]      中间特征 (供 HTD 复用)
    }
    """
    def __init__(self,
                 dim_in=3,
                 dim_feat=256,
                 num_joints=21,
                 max_len=243,
                 depth=8,
                 num_heads=8,
                 mlp_ratio=4,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 use_rope=True,
                 use_temp_embed=False,
                 use_heatmap=False,
                 embed_mode='concat'):
        super().__init__()
        self.dim_feat = dim_feat
        self.num_joints = num_joints

        # 1. 帧级 Token 提取
        self.joint_embed = JointEmbedding(
            dim_in=dim_in, num_joints=num_joints,
            dim_out=dim_feat, mode=embed_mode)

        # 2. 时序位置编码
        self.use_rope = use_rope
        self.use_temp_embed = use_temp_embed
        if use_temp_embed:
            self.temp_embed = TemporalPositionEncoding(dim=dim_feat, max_len=max_len)

        # 3. 主时序 Transformer
        self.transformer = TemporalTransformerDecoder(
            dim=dim_feat, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            use_rope=use_rope)

        # 4. 三个解码头
        self.kp_decoder = KeypointDecoder2D(
            dim=dim_feat, num_joints=num_joints, use_heatmap=use_heatmap)
        self.vel_decoder = VelocityDecoder(
            dim=dim_feat, num_joints=num_joints)
        self.accel_decoder = AccelDecoder(
            dim=dim_feat, num_joints=num_joints)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        """
        x: [B, F, J, C=3]   3D 关节坐标
        """
        B, F, J, C = x.shape

        # 1. 帧级 Token
        feat = self.joint_embed(x)  # [B, F, D]

        # 2. 位置编码 (RoPE 在 Transformer 内部完成)
        if self.use_temp_embed:
            feat = self.temp_embed(feat)

        # 3. 主时序 Transformer
        feat = self.transformer(feat)  # [B, F, D]

        # 4. 三个解码头
        out = {
            'joint_2d': self.kp_decoder(feat),        # [B, F, J, 2]
            'joint_vel': self.vel_decoder(feat),       # [B, F, J, 3]
            'joint_accel': self.accel_decoder(feat),   # [B, F, J, 3]
            'feat': feat,                               # [B, F, D]
        }
        return out


# ============================================================
# VideoMotionEncoder — 视频输入版（完整 ViTPose 风格）
# ============================================================

class PatchHeatmapDecoder(nn.Module):
    """
    ViTPose 风格热图解码器：从 ViT patch tokens 重建热图

    输入 patch tokens [B, N, D]  reshape 到 [B, D, H/P, W/P]
    然后逐级反卷积上采样到 [B, J, H_out, W_out]

    结构:
      Norm → Reshape to grid → Deconv×3 (×2 up each) → Heatmap → Soft-argmax

    Input:  [B, F, N, D]   patch tokens
    Output: [B, F, J, 2]   2D 关键点坐标
    """
    def __init__(self, embed_dim=768, num_joints=21, patch_size=16, img_size=256):
        super().__init__()
        self.num_joints = num_joints
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size  # e.g. 256/16=16

        self.norm = nn.LayerNorm(embed_dim)

        # Deconv 上采样: 16×16 → 32×32 → 64×64 → 256×256 (×2 each)
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=4, stride=2, padding=1),  # 16→32
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),        # 32→64
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),          # 64→128
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, num_joints, kernel_size=4, stride=2, padding=1),   # 128→256
        )

    def forward(self, patch_tokens):
        """
        patch_tokens: [B, F, N, D]
        returns: [B, F, J, 2]  2D coordinates via soft-argmax
        """
        B, T, N, D = patch_tokens.shape
        G = self.grid_size  # e.g. 16

        # Norm + reshape to spatial grid
        x = self.norm(patch_tokens)                          # [B, T, N, D]
        x = x.reshape(B * T, G, G, D).permute(0, 3, 1, 2)    # [B*T, D, G, G]

        # Deconv to heatmap
        heatmap = self.deconv_layers(x)                       # [B*T, J, 256, 256]

        # Soft-argmax
        H, W = heatmap.shape[-2:]
        heatmap_flat = heatmap.reshape(B * T, self.num_joints, -1)  # [B*T, J, 65536]
        heatmap_flat = torch.softmax(heatmap_flat, dim=-1)
        heatmap_flat = heatmap_flat.reshape(B * T, self.num_joints, H, W)

        # 坐标网格
        ys, xs = torch.meshgrid(
            torch.arange(H, device=x.device, dtype=torch.float32),
            torch.arange(W, device=x.device, dtype=torch.float32),
            indexing='ij')
        coords_x = (heatmap_flat * xs).view(B * T, self.num_joints, -1).sum(-1)  # [B*T, J]
        coords_y = (heatmap_flat * ys).view(B * T, self.num_joints, -1).sum(-1)
        coords = torch.stack([coords_x, coords_y], dim=-1)   # [B*T, J, 2]

        return coords.reshape(B, T, self.num_joints, 2)


class VideoMotionEncoder(nn.Module):
    """
    ViTPose+ 视频运动编码器（完整版本）

    从视频帧序列中同时提取 2D 关键点、3D 速度、3D 加速度。

    架构:
      ViTEncoder (per-frame) → [B, F, D] CLS + [B, F, N, D] patches
        ├── CLS tokens ──→ TemporalTransformerDecoder × 8 (RoPE)
        │                     ├── KeypointDecoder2D (MLP)   → [B, F, J, 2]
        │                     ├── VelocityDecoder           → [B, F, J, 3]
        │                     └── AccelDecoder              → [B, F, J, 3]
        └── Patch tokens → PatchHeatmapDecoder              → [B, F, J, 2]

    两种 2D 关键点路径（可同时使用，也可选其一）:
      - heatmap_kp=True:  从 patch tokens 经 deconv → heatmap → soft-argmax
      - mlp_kp=True:      从 CLS tokens 经 MLP 直接回归

    Input:  [B, F, 3, H, W]  视频帧（H, W 通常为 256）
    Output: dict{
        'joint_2d':      [B, F, J, 2]   2D 关节位置
        'joint_2d_heatmap': [B, F, J, 2]  热图版本的 2D（可选）
        'joint_vel':    [B, F, J, 3]   3D 关节速度
        'joint_accel':  [B, F, J, 3]   3D 关节加速度
        'feat':         [B, F, D]      中间特征
    }
    """
    def __init__(self,
                 img_size=256,
                 patch_size=16,
                 in_chans=3,
                 vit_backbone='base',       # 'base' or 'small'
                 vit_embed_dim=768,
                 vit_depth=12,
                 vit_heads=12,
                 dim_feat=256,
                 num_joints=21,
                 max_len=243,
                 temporal_depth=8,
                 temporal_heads=8,
                 mlp_ratio=4,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 heatmap_kp=True,            # 使用热图解码器输出 2D
                 mlp_kp=True,                # 同时使用 MLP 解码器输出 2D
                 ):
        super().__init__()
        self.dim_feat = dim_feat
        self.num_joints = num_joints
        self.heatmap_kp = heatmap_kp
        self.mlp_kp = mlp_kp

        # 1. 帧级 ViT 图像编码器
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            backbone=vit_backbone,
        )
        if vit_backbone == 'base':
            vit_kwargs.update(embed_dim=vit_embed_dim, depth=vit_depth, num_heads=vit_heads)
        self.vit_encoder = ViTEncoder(**vit_kwargs)

        # 2. 特征压缩: ViT D → dim_feat
        actual_vit_dim = self.vit_encoder.embed_dim
        self.vit_proj = nn.Linear(actual_vit_dim, dim_feat)

        # 3. 主时序 Transformer
        self.transformer = TemporalTransformerDecoder(
            dim=dim_feat, depth=temporal_depth, num_heads=temporal_heads,
            mlp_ratio=mlp_ratio, drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            use_rope=True)

        # 4. 三个解码头

        # MLP 版 2D 关键点头（从 CLS 特征）
        if mlp_kp:
            self.kp_decoder_mlp = KeypointDecoder2D(
                dim=dim_feat, num_joints=num_joints, use_heatmap=False)

        # 热图版 2D 关键点头（从 patch tokens）
        if heatmap_kp:
            self.kp_decoder_heatmap = PatchHeatmapDecoder(
                embed_dim=actual_vit_dim, num_joints=num_joints,
                patch_size=patch_size, img_size=img_size)

        self.vel_decoder = VelocityDecoder(dim=dim_feat, num_joints=num_joints)
        self.accel_decoder = AccelDecoder(dim=dim_feat, num_joints=num_joints)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, video_frames):
        """
        video_frames: [B, F, 3, H, W]  视频帧序列
        """
        B, F, C, H, W = video_frames.shape

        # 1. ViT 编码: 每帧独立编码
        # return_patches=True → 同时返回 CLS (用于 MLP/Motion) 和 patch tokens (用于热图)
        if self.heatmap_kp:
            cls_feat, patch_feat = self.vit_encoder(video_frames, return_patches=True)
            # cls_feat: [B, F, vit_dim], patch_feat: [B, F, N, vit_dim]
        else:
            cls_feat = self.vit_encoder(video_frames, return_patches=False)

        # 2. 特征压缩: vit_dim (768) → dim_feat (256)
        feat = self.vit_proj(cls_feat)                          # [B, F, dim_feat]

        # 3. 时序建模
        feat = self.transformer(feat)                            # [B, F, dim_feat]

        # 4. 三个解码头
        out = {'feat': feat}

        # 2D 关键点
        if self.mlp_kp:
            out['joint_2d'] = self.kp_decoder_mlp(feat)          # [B, F, J, 2]
        if self.heatmap_kp:
            out['joint_2d_heatmap'] = self.kp_decoder_heatmap(patch_feat)  # [B, F, J, 2]

        # 速度 & 加速度
        out['joint_vel'] = self.vel_decoder(feat)                # [B, F, J, 3]
        out['joint_accel'] = self.accel_decoder(feat)            # [B, F, J, 3]

        return out

class MotionEncoderLoss(nn.Module):
    """
    Motion Encoder 的多任务 Loss

    L = λ_kp * L1(2D) + λ_v * L1(vel) + λ_a * L1(accel)

    所有项都带 mask 支持
    """
    def __init__(self, w_kp=1.0, w_vel=1.0, w_accel=0.5):
        super().__init__()
        self.w_kp = w_kp
        self.w_vel = w_vel
        self.w_accel = w_accel

    def _masked_l1(self, pred, target, mask=None):
        diff = (pred - target).abs()
        if mask is not None:
            return (diff * mask).sum() / (mask.sum() + 1e-8)
        return diff.mean()

    def forward(self, pred, target, mask=None):
        """
        pred:   MotionEncoder 的输出 dict
        target: {
            'joint_2d':    [B, F, J, 2],
            'joint_vel':   [B, F, J, 3],
            'joint_accel': [B, F, J, 3],
        }
        mask:   {
            'vel':   [B, F, 1] 或 [B, F, J, 1]  (可选)
            'accel': 同上
        }
        """
        vel_mask = mask.get('vel', None) if mask else None
        accel_mask = mask.get('accel', None) if mask else None
        kp_mask = mask.get('kp', None) if mask else None

        loss_dict = {}
        if 'joint_2d' in pred and 'joint_2d' in target:
            loss_dict['kp_loss'] = self._masked_l1(
                pred['joint_2d'], target['joint_2d'], kp_mask)
        loss_dict['vel_loss'] = self._masked_l1(
            pred['joint_vel'], target['joint_vel'], vel_mask)
        loss_dict['accel_loss'] = self._masked_l1(
            pred['joint_accel'], target['joint_accel'], accel_mask)

        total = (self.w_kp * loss_dict.get('kp_loss', 0.) +
                 self.w_vel * loss_dict['vel_loss'] +
                 self.w_accel * loss_dict['accel_loss'])
        loss_dict['total'] = total
        return loss_dict
