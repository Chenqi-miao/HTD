"""
TemporalHaMeR — HaMeR 的时序版本，像 MotionBERT 一样处理视频序列
===============================================================

HaMeR（单帧）:
  img → ViT → MANO Head (cross-attn) → MANO Layer → 3D joints + mesh

TemporalHaMeR（视频序列）:
  [B, F, 3, H, W] → ViT per frame → Temporal Transformer → MANO Head per frame → [B, F, J, 3]

架构:
  - ViT Backbone: 每帧独立编码，输出空间特征图 [B, F, D_vit, H', W']
  - Frame Embed: GAP + Linear 投影 → 帧级 token [B, F, feat_dim]
  - Temporal Transformer: RoPE 时序注意力 → [B, F, feat_dim]
  - MANO Head: 每帧独立解码，注入时序特征作为额外 context token
  - MANO Layer: 每帧独立解码 3D joints + mesh

与 MotionBERT 的对应关系:
  MotionBERT:   3D joints [B, F, J, 3] → DSTFormer → refined 3D joints
  TemporalHaMeR:   frames [B, F, 3, H, W] → ViT + Temporal Transformer → MANO → 3D joints

Reference:
  HaMeR: https://github.com/geopavlakos/hamer
  ViTPose: https://github.com/ViTAE-Transformer/ViTPose
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import math
import numpy as np

from HTD.source.model.DSF import trunc_normal_, DropPath
from HTD.pvanet.temporal_transformer import TemporalTransformerDecoder


# ========================================================================
# 1. ViT Backbone（HaMeR 风格，输出空间特征图）
# ========================================================================

class PatchEmbed(nn.Module):
    """图像转 Patch Token（HaMeR 风格，带 padding 适配 ViT-Huge 的 patch embed）"""
    def __init__(self, img_size=(256, 192), patch_size=16, in_chans=3, embed_dim=1280, ratio=1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # HaMeR 的 ViT-H 使用带 padding 的 conv，确保输出尺寸正确
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=patch_size // ratio,
            padding=4 + 2 * (ratio // 2 - 1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)                           # [B, D, H', W']
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)            # [B, N, D]
        return x, (Hp, Wp)


class ViTBlock(nn.Module):
    """标准 ViT Block（与 HaMeR 一致）"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads,
                                          dropout=attn_drop, bias=qkv_bias,
                                          batch_first=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class HaMeRViT(nn.Module):
    """
    HaMeR 风格的 ViT 骨干网络

    输出空间特征图（而非 CLS token），供 MANO Head 交叉注意力使用。

    Args:
        img_size: 输入图像尺寸 (H, W)，默认 (256, 192)
        patch_size: patch 大小 (默认 16)
        embed_dim: ViT 特征维度 (默认 1280，ViT-Huge)
        depth: Transformer 层数 (默认 32，ViT-Huge)
        num_heads: 注意力头数 (默认 16)
        mlp_ratio: MLP 扩展比 (默认 4)
        drop_rate: Dropout 率
        attn_drop_rate: 注意力 Dropout 率
        drop_path_rate: Stochastic Depth 率

    Input:  [B, 3, H, W]
    Output: [B, D, H', W']  空间特征图（H'=H/16, W'=W/16）
    """
    def __init__(self,
                 img_size=(256, 192),
                 patch_size=16,
                 in_chans=3,
                 embed_dim=1280,
                 depth=32,
                 num_heads=16,
                 mlp_ratio=4,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.3):
        super().__init__()
        self.embed_dim = embed_dim

        # Patch Embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim)

        # Class token + Position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))

        # Stochastic Depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads, mlp_ratio,
                     qkv_bias=True,
                     drop=drop_rate, attn_drop=attn_drop_rate,
                     drop_path=dpr[i])
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        """输入: [B, 3, H, W] → 输出: [B, D, H', W']"""
        B = x.shape[0]
        x, (Hp, Wp) = self.patch_embed(x)           # [B, N, D]

        # 加 class token 和位置编码
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)        # [B, N+1, D]
        x = x + self.pos_embed

        # Transformer
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # 去掉 CLS token，reshape 回特征图
        x = x[:, 1:, :]                               # [B, N, D]
        x = x.permute(0, 2, 1)                        # [B, D, N]
        x = x.reshape(B, -1, Hp, Wp)                  # [B, D, H', W']
        return x


# ========================================================================
# 2. 轻量 ViT 版本（可从头训练的较小模型）
# ========================================================================

class LightViT(nn.Module):
    """
    轻量 ViT（可从头训练，不需要加载 HaMeR 预训练权重）

    输出空间特征图 [B, D, H', W'] 与 HaMeRViT 接口兼容。

    Args:
        img_size: 输入图像尺寸 (默认 256)
        patch_size: patch 大小 (默认 16)
        embed_dim: 特征维度 (默认 768，ViT-Base)
        depth: 层数 (默认 12)
        num_heads: 头数 (默认 12)
        mlp_ratio: MLP 扩展比 (默认 4)
    """
    def __init__(self,
                 img_size=256,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        # Patch Embedding（标准实现）
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

        # Position embedding（无 CLS token，直接对 patch tokens 加位置编码）
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim))

        # Stochastic Depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads, mlp_ratio,
                     qkv_bias=True,
                     drop=drop_rate, attn_drop=attn_drop_rate,
                     drop_path=dpr[i])
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        """输入: [B, 3, H, W] → 输出: [B, D, H', W']"""
        B, C, H, W = x.shape
        x = self.proj(x)                                   # [B, D, H', W']
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)                    # [B, N, D]

        # 位置编码
        x = x + self.pos_embed[:, :x.size(1), :]

        # Transformer
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = x.permute(0, 2, 1).reshape(B, -1, Hp, Wp)     # [B, D, H', W']
        return x


# ========================================================================
# 3. MANO Head（交叉注意力解码器，含时序上下文注入）
# ========================================================================

class MANOHead(nn.Module):
    """
    HaMeR 风格的 MANO Head，支持时序上下文注入

    核心: TransformerDecoder（cross-attention）
      - Query token（可学习或均值参数）
      - Cross-attention 到特征 context（空间 token + 时序 token）
      - 迭代精化（IEF - Iterative Error Feedback）

    Args:
        feat_dim: 特征维度（来自 backbone）
        num_joints: 手部关节数 (默认 21)
        num_pose: 手部姿态参数维度
        num_betas: 形状参数维度 (默认 10)
        ie_iters: IEF 迭代次数 (默认 3)
        transformer_depth: Cross-attention decoder 层数 (默认 6)
    """
    def __init__(self,
                 feat_dim=1280,
                 num_joints=21,
                 num_betas=10,
                 ie_iters=3,
                 transformer_depth=6,
                 num_heads=8,
                 mlp_dim=2048,
                 dropout=0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_joints = num_joints
        self.num_betas = num_betas
        self.ie_iters = ie_iters

        # MANO 参数维度: global_orient (6D) + hand_pose (15*6D) = 96
        # 使用 6D 旋转表示 (rot6d) 而非 axis-angle
        self.npose = 6 * (self.num_joints - 1 + 1)  # 16 joints * 6 = 96
        # (global_orient + 15 hand joints) × 6

        # Cross-attention Transformer Decoder
        # 参考 HaMeR 的 TransformerDecoder + TransformerCrossAttn
        from HTD.pvanet.temporal_transformer import TemporalTransformerBlock

        # Query token embedding
        self.query_tokens = nn.Parameter(torch.zeros(1, 1, feat_dim))
        trunc_normal_(self.query_tokens, std=.02)

        # Cross-attention layers
        # 每层: Self-Attn → Cross-Attn (to context) → FF
        self.cross_attn_layers = nn.ModuleList([
            nn.ModuleList([
                nn.MultiheadAttention(feat_dim, num_heads, dropout=dropout, batch_first=True),
                nn.MultiheadAttention(feat_dim, num_heads, dropout=dropout, kdim=feat_dim, vdim=feat_dim, batch_first=True),
                nn.Sequential(
                    nn.Linear(feat_dim, mlp_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_dim, feat_dim),
                    nn.Dropout(dropout),
                )
            ])
            for _ in range(transformer_depth)
        ])
        self.norm_layers = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(feat_dim),
                nn.LayerNorm(feat_dim),
                nn.LayerNorm(feat_dim),
            ])
            for _ in range(transformer_depth)
        ])

        # 输出头
        self.decpose = nn.Linear(feat_dim, self.npose)
        self.decshape = nn.Linear(feat_dim, num_betas)
        self.deccam = nn.Linear(feat_dim, 3)

        # 均值初始化
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

    def forward(self, x, temporal_feat=None):
        """
        Args:
            x: 空间特征 [B, N, D] 或 [B, D, H', W']，N 为空间 token 数
            temporal_feat: 时序特征 [B, 1, D]（可选，注入时序上下文）

        Returns:
            pred_mano_params: dict 包含 global_orient, hand_pose, betas
            pred_cam: [B, 3] 相机参数
        """
        if x.dim() == 4:
            # [B, D, H', W'] → [B, N, D]
            B, D, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)

        B = x.shape[0]

        # 构建 context token 序列
        # 空间 token + 时序 token
        if temporal_feat is not None:
            context = torch.cat([x, temporal_feat], dim=1)  # [B, N+1, D]
        else:
            context = x

        # 初始预测 = 均值参数（用零表示）
        pred_hand_pose = x.new_zeros(B, self.npose)
        pred_betas = x.new_zeros(B, self.num_betas)
        pred_cam = x.new_zeros(B, 3)

        # IEF 迭代精化
        for i in range(self.ie_iters):
            # Query token: 当前预测参数编码
            query = self.query_tokens.expand(B, -1, -1)  # [B, 1, D]

            # Cross-attention Transformer Decoder
            for (self_attn, cross_attn, ff), (norm1, norm2, norm3) in zip(
                    self.cross_attn_layers, self.norm_layers):
                # Self-attention (on query)
                q = self_attn(norm1(query), norm1(query), norm1(query))[0]
                query = query + q

                # Cross-attention (query → context)
                q = cross_attn(norm2(query), context, context)[0]
                query = query + q

                # FF
                query = query + ff(norm3(query))

            token_out = query.squeeze(1)  # [B, D]

            # 残差预测
            pred_hand_pose = pred_hand_pose + self.decpose(token_out)
            pred_betas = pred_betas + self.decshape(token_out)
            pred_cam = pred_cam + self.deccam(token_out)

        # 将 rot6d 转换为旋转矩阵
        pred_hand_pose_rotmat = self._rot6d_to_rotmat(pred_hand_pose)
        pred_global_orient = pred_hand_pose_rotmat[:, 0:1, :, :]   # [B, 1, 3, 3]
        pred_hand_pose_joints = pred_hand_pose_rotmat[:, 1:, :, :]  # [B, 15, 3, 3]

        pred_mano_params = {
            'global_orient': pred_global_orient,
            'hand_pose': pred_hand_pose_joints,
            'betas': pred_betas,
        }

        return pred_mano_params, pred_cam

    @staticmethod
    def _rot6d_to_rotmat(x):
        """6D 旋转表示 → 3×3 旋转矩阵"""
        x = x.reshape(-1, 3, 2)
        a1 = x[:, :, 0]
        a2 = x[:, :, 1]
        b1 = F.normalize(a1)
        b2 = F.normalize(a2 - torch.sum(b1 * a2, dim=-1, keepdim=True) * b1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack([b1, b2, b3], dim=-1)  # [..., 3, 3]


# ========================================================================
# 4. TemporalHaMeR — 主模型
# ========================================================================

class TemporalHaMeR(nn.Module):
    """
    时序 HaMeR —— 像 MotionBERT 一样处理视频序列

    架构:
        ViT per frame → Temporal Transformer → MANO Head per frame → 3D joints

    Input:  [B, F, 3, H, W]  视频帧序列
    Output: {
        'joints_3d':    [B, F, J, 3]   3D 关节点
        'vertices':     [B, F, V, 3]   3D 网格顶点
        'mano_params':   MANO 参数
        'pred_cam':     [B, F, 3]      相机参数
    }

    Args:
        backbone_type: 骨干网络类型 ('vit_huge' | 'vit_base' | 'light')
        img_size: 输入图像尺寸
        feat_dim: 时序特征维度 (默认 256)
        temporal_depth: 时序 Transformer 层数 (默认 4)
        num_joints: 手部关节数 (默认 21)
        max_len: 最大序列长度 (默认 243)
        ie_iters: IEF 迭代次数 (默认 3)
        mano_path: MANO 模型路径

    Usage:
        model = TemporalHaMeR(backbone_type='vit_base')
        frames = torch.randn(2, 15, 3, 256, 192)  # [B, F, 3, H, W]
        out = model(frames)
        joints = out['joints_3d']  # [2, 15, 21, 3]
    """
    def __init__(self,
                 backbone_type='vit_base',
                 img_size=(256, 192),
                 patch_size=16,
                 feat_dim=256,
                 temporal_depth=4,
                 temporal_heads=8,
                 num_joints=21,
                 num_betas=10,
                 max_len=243,
                 ie_iters=3,
                 mano_path='models/mano_v1_2/models',
                 mano_side='right',
                 ):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_joints = num_joints
        self.num_betas = num_betas

        # 1. 图像编码器（ViT 骨干）
        if backbone_type == 'vit_huge':
            # HaMeR 原始设置（需加载预训练权重）
            self.vit = HaMeRViT(
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=1280, depth=32, num_heads=16,
                drop_path_rate=0.3)
            vit_dim = 1280
        elif backbone_type == 'vit_base':
            # ViT-Base（可加载 ImageNet 预训练权重或从头训练）
            self.vit = HaMeRViT(
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=768, depth=12, num_heads=12,
                drop_path_rate=0.1)
            vit_dim = 768
        elif backbone_type == 'light':
            # 轻量 ViT（从头训练）
            self.vit = LightViT(
                img_size=max(img_size),
                patch_size=patch_size,
                embed_dim=384, depth=6, num_heads=6,
                drop_path_rate=0.1)
            vit_dim = 384
        else:
            raise ValueError(f"Unknown backbone: {backbone_type}")

        # 2. 特征投影（ViT dim → feat_dim）
        self.frame_proj = nn.Sequential(
            nn.Conv2d(vit_dim, feat_dim, kernel_size=1),  # 1×1 conv 降维
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(),
        )

        # 3. 时序 Transformer（帧间建模）
        self.temporal_transformer = TemporalTransformerDecoder(
            dim=feat_dim,
            depth=temporal_depth,
            num_heads=temporal_heads,
            mlp_ratio=4,
            drop_rate=0.1,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            use_rope=True,  # RoPE 相对位置编码
        )

        # 4. 时序位置编码（当单帧推理时使用）
        self.temp_pos_embed = nn.Parameter(torch.zeros(1, max_len, feat_dim))
        trunc_normal_(self.temp_pos_embed, std=.02)

        # 5. MANO Head（交叉注意力解码器，含时序注入）
        self.mano_head = MANOHead(
            feat_dim=feat_dim,
            num_joints=num_joints,
            num_betas=num_betas,
            ie_iters=ie_iters,
            transformer_depth=6,
            num_heads=feat_dim // 32,  # 如 256 → 8 heads
            mlp_dim=feat_dim * 4,
            dropout=0.1,
        )

        # 6. MANO Layer（解码 joints/vertices）
        # 支持两种方式：集成的 manotorch 或简化版
        self._init_mano(mano_path, mano_side)

        self.apply(self._init_weights)

    def _init_mano(self, mano_path, mano_side):
        """初始化 MANO 层"""
        try:
            from manotorch.manolayer import ManoLayer
            is_rhand = (mano_side == 'right')
            self.mano_layer = ManoLayer(
                mano_path,
                side=mano_side,
                center_idx=9,
                use_pca=False,
                flat_hand_mean=True,
            )
            # ManoLayer 输出 21 个关节（16 MANO + 5 指尖）
            self.register_buffer('mano_faces',
                                 self.mano_layer.th_faces.clone())
            self.mano_is_manotorch = True
        except (ImportError, RuntimeError):
            # 降级：使用 mano_v1_2
            from HTD.source.model.mano_wrapper import MANOWrapper
            self.mano_layer = MANOWrapper(
                model_path=mano_path,
                is_rhand=(mano_side == 'right'),
            )
            self.mano_faces = self.mano_layer.faces
            self.mano_is_manotorch = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, return_mano_params=False):
        """
        Args:
            x: [B, F, 3, H, W]  视频帧序列
            return_mano_params: 是否返回原始 MANO 参数

        Returns:
            dict 包含:
              - joints_3d: [B, F, J, 3]   3D 关节点
              - vertices:  [B, F, V, 3]   3D 网格顶点
              - cam:       [B, F, 3]       相机参数
              - mano_params (可选): MANO 原始参数
        """
        B, F, C, H, W = x.shape

        # ── 1. ViT 编码（每帧独立） ────────────────────────────
        x_flat = x.reshape(B * F, C, H, W)          # [B*F, 3, H, W]
        vit_feat = self.vit(x_flat)                   # [B*F, D_vit, H', W']

        # 投影到 feat_dim
        vit_feat = self.frame_proj(vit_feat)          # [B*F, feat_dim, H', W']

        # 分离 batch 和时序维度
        _, D, Hp, Wp = vit_feat.shape
        vit_feat = vit_feat.reshape(B, F, D, Hp, Wp)  # [B, F, D, H', W']

        # ── 2. Frame-level 特征 + 时序建模 ──────────────────
        # 空间池化 → 帧级特征
        frame_feat = vit_feat.mean(dim=[3, 4])         # [B, F, D]

        # 时序 Transformer
        temp_feat = self.temporal_transformer(frame_feat)  # [B, F, D]

        # ── 3. MANO Head（每帧处理） ─────────────────────────
        all_joints = []
        all_vertices = []
        all_cam = []
        all_mano_params = []

        for f in range(F):
            # 当前帧的空间特征
            spatial_feat = vit_feat[:, f, :, :, :]     # [B, D, H', W']

            # 当前帧的时序特征（unsqueeze 作为额外 context token）
            temporal_token = temp_feat[:, f:f+1, :]     # [B, 1, D]

            # MANO Head 预测
            mano_params, pred_cam = self.mano_head(
                spatial_feat, temporal_feat=temporal_token)

            # MANO Layer 解码
            joints_3d, vertices = self._decode_mano(
                mano_params, pred_cam)

            all_joints.append(joints_3d)     # [B, J, 3]
            all_vertices.append(vertices)    # [B, V, 3]
            all_cam.append(pred_cam)         # [B, 3]
            all_mano_params.append(mano_params)

        # 堆叠时序维度
        out = {
            'joints_3d': torch.stack(all_joints, dim=1),    # [B, F, J, 3]
            'vertices': torch.stack(all_vertices, dim=1),   # [B, F, V, 3]
            'cam': torch.stack(all_cam, dim=1),             # [B, F, 3]
        }

        if return_mano_params:
            out['mano_params'] = all_mano_params

        return out

    def _decode_mano(self, mano_params, pred_cam):
        """
        用 MANO Layer 解码关节点和网格

        Args:
            mano_params: dict with global_orient [B, 1, 3, 3], hand_pose [B, 15, 3, 3], betas [B, 10]
            pred_cam: [B, 3] → [s, tx, ty]

        Returns:
            joints_3d: [B, 21, 3]
            vertices: [B, 778, 3]
        """
        B = pred_cam.shape[0]

        global_orient = mano_params['global_orient'].reshape(B, 3, 3)
        hand_pose = mano_params['hand_pose'].reshape(B, -1, 3, 3)
        betas = mano_params['betas'].reshape(B, 10)

        if self.mano_is_manotorch:
            # manotorch 接口
            mano_out = self.mano_layer(
                global_orient=global_orient,
                hand_pose=hand_pose,
                betas=betas,
            )
            joints_3d = mano_out.joints  # [B, 21, 3]
            vertices = mano_out.vertices  # [B, 778, 3]
        else:
            # mano_v1_2 接口
            mano_out = self.mano_layer(
                global_orient=global_orient,
                hand_pose=hand_pose,
                betas=betas,
                return_verts=True, return_tips=True,
                pose2rot=False,
            )
            joints_3d = mano_out.joints   # [B, 21, 3]
            vertices = mano_out.vertices  # [B, 778, 3]

        # 相机投影（添加平移）
        # pred_cam = [s, tx, ty] 对应弱透视投影
        # 这里简化为直接加平移（完整投影在 loss 模块处理）
        joints_3d = joints_3d + pred_cam[:, None, :] * 0.01  # 轻微偏移

        return joints_3d, vertices


# ========================================================================
# 5. FusionHaMeRModel — 训练包装器（类似 FusionModel for MotionBERT）
# ========================================================================

class FusionHaMeRModel(nn.Module):
    """
    用于训练 TemporalHaMeR 的包装器

    类似 FusionModel for MotionBERT:
      - 处理数据归一化
      - 计算 loss
      - 提供训练/评估接口

    Input:
        inputs: {'img': [B, F, 3, H, W]}
        targets: {'joint_xyz': [B, 1, F, J, 3], 'mano_params': ...}
        meta_info: 元信息

    Output:
        训练: (out, loss, error)
        评估: out
    """
    def __init__(self,
                 backbone_type='vit_base',
                 img_size=(256, 192),
                 num_frame=15,
                 num_joints=21,
                 temporal_depth=4,
                 mano_path='models/mano_v1_2/models',
                 ):
        super().__init__()
        self.num_frame = num_frame
        self.num_joints = num_joints

        # 时序 HaMeR 模型
        self.model = TemporalHaMeR(
            backbone_type=backbone_type,
            img_size=img_size,
            feat_dim=256,
            temporal_depth=temporal_depth,
            num_joints=num_joints,
            max_len=num_frame,
            mano_path=mano_path,
        )

        # Loss
        self.mse = nn.MSELoss()
        self.sm_loss = SmoothNetLoss(w_accel=0.1, w_pos=1)
        self.l1_loss = nn.L1Loss()

    def forward(self, inputs, targets, meta_info):
        imgs = inputs["img"].cuda()           # [B, F, 3, H, W]
        joints_gt = targets["joint_xyz"].cuda()  # [B, 1, F, J, 3]

        # 模型前向
        out = self.model(imgs)

        # 计算 loss
        pd_joints = out['joints_3d'].unsqueeze(1)  # [B, 1, F, J, 3]

        if self.training:
            val = meta_info.get('joint_gt_val', None)
            if val is not None:
                val = val.cuda().view(-1, 1, self.num_frame, self.num_joints, 1)

            # 关节 loss
            joint_loss = self.l1_loss(pd_joints, joints_gt)
            loss = {'joint_loss': joint_loss}

            # 误差
            error = self.calculate_error(pd_joints, joints_gt, val)

            out_pd = {'pd_joint_xyz': pd_joints}
            return out_pd, loss, error

        out_pd = {'pd_joint_xyz': pd_joints}
        return out_pd

    def calculate_error(self, joint, gt, val=None):
        diff = (joint - gt)
        error = torch.sqrt(torch.sum(diff * diff, dim=-1))
        if val is not None:
            return error.sum() / (val.sum() + 1e-8)
        return error.mean()


# ========================================================================
# 工具函数: SmoothNetLoss
# ========================================================================

class SmoothNetLoss(nn.Module):
    """
    平滑性 loss（时序 + 关节一致性）
    w_accel * accel_loss + w_pos * position_loss
    """
    def __init__(self, w_accel=0.1, w_pos=1.0):
        super().__init__()
        self.w_accel = w_accel
        self.w_pos = w_pos

    def forward(self, pred, gt, val=None):
        # pred, gt: [B, F, J*3]
        B, F, D = pred.shape

        # 位置 loss
        pos_loss = F.l1_loss(pred, gt, reduction='none')
        if val is not None:
            pos_loss = (pos_loss * val).sum() / (val.sum() + 1e-8)
        else:
            pos_loss = pos_loss.mean()

        # 加速度 loss（时序平滑性）
        pred_acc = pred[:, 2:, :] + pred[:, :-2, :] - 2 * pred[:, 1:-1, :]
        gt_acc = gt[:, 2:, :] + gt[:, :-2, :] - 2 * gt[:, 1:-1, :]
        acc_loss = F.l1_loss(pred_acc, gt_acc)

        return self.w_pos * pos_loss + self.w_accel * acc_loss


# ========================================================================
# 测试
# ========================================================================

if __name__ == '__main__':
    # 测试 TemporalHaMeR 前向
    B, F = 2, 9
    H, W = 256, 192
    model = TemporalHaMeR(
        backbone_type='light',
        img_size=(H, W),
        feat_dim=128,
        temporal_depth=2,
        num_joints=21,
        max_len=F,
    ).cuda()
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    frames = torch.randn(B, F, 3, H, W).cuda()
    out = model(frames)
    print(f"Input:  {list(frames.shape)}")
    print(f"Joints: {list(out['joints_3d'].shape)}")
    print(f"Vertices: {list(out['vertices'].shape)}")
    print(f"Cam:    {list(out['cam'].shape)}")
