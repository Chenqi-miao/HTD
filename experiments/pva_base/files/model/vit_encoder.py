"""
ViTEncoder — 帧级 Vision Transformer 图像编码器
================================================
对视频的每帧独立编码，输出帧级特征 token。

架构（ViT-B 风格）：
  PatchEmbed (16×16) → CLS token + PosEmbed
    → N × Transformer Block
      → CLS head 输出帧级特征 [B*F, D]

Reference:
  An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
  https://arxiv.org/abs/2010.11929
"""

import torch
import torch.nn as nn
import math


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """截断正态初始化"""
    import warnings
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


class Mlp(nn.Module):
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


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """
    图像转 Patch Token

    Input:  [B, 3, H, W]
    Output: [B, N, D]  其中 N = (H/P) * (W/P)
    """
    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                           # [B, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)           # [B, N, D]
        return x


class ViTEncoder(nn.Module):
    """
    帧级 ViT 图像编码器

    对每帧独立编码，输出 CLS token 作为帧级特征。

    Input:  [B, F, 3, H, W]   视频帧序列（F 帧）
    Output: [B, F, D]          帧级特征

    当单独处理单帧时：
    Input:  [B, 3, H, W]
    Output: [B, D]

    支持两种规模：
      - 'base':  D=768, depth=12, heads=12 (ViT-B)
      - 'small': D=384, depth=6,  heads=6  (ViT-S)
    """
    def __init__(self,
                 img_size=256,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 backbone='base'):
        super().__init__()
        if backbone == 'small':
            embed_dim = 384
            depth = 6
            num_heads = 6

        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        # Patch Embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,
                  drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)])

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

    def forward_features(self, x, return_patches=False):
        """
        单帧特征提取

        Args:
            x: [B, 3, H, W]
            return_patches: 是否同时返回 patch tokens

        Returns:
            default: [B, D]  CLS token
            return_patches=True: ([B, D] CLS, [B, N, D] patches)
        """
        B = x.shape[0]
        x = self.patch_embed(x)                # [B, N, D]

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, N+1, D]
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        cls_feat = x[:, 0]                      # [B, D]
        if return_patches:
            patch_feat = x[:, 1:, :]             # [B, N, D]
            return cls_feat, patch_feat
        return cls_feat

    def forward(self, x, return_patches=False):
        """
        x: [B, F, 3, H, W]  或  [B, 3, H, W]

        Args:
            return_patches: 是否同时返回 patch tokens（用于热图解码器）

        returns:
            default: [B, F, D]  或  [B, D]
            return_patches=True: ([B, F, D] CLS, [B, F, N, D] patches) 或 ([B, D], [B, N, D])
        """
        if x.dim() == 4:
            # 单帧模式
            return self.forward_features(x, return_patches=return_patches)

        # 视频模式: [B, F, 3, H, W]
        B, F, C, H, W = x.shape
        x_flat = x.reshape(B * F, C, H, W)                    # [B*F, 3, H, W]

        if return_patches:
            cls_feat, patch_feat = self.forward_features(x_flat, return_patches=True)
            cls_feat = cls_feat.reshape(B, F, -1)             # [B, F, D]
            patch_feat = patch_feat.reshape(B, F, -1, self.embed_dim)  # [B, F, N, D]
            return cls_feat, patch_feat
        else:
            feat = self.forward_features(x_flat)               # [B*F, D]
            feat = feat.reshape(B, F, -1)                      # [B, F, D]
            return feat
