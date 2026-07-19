"""
TemporalTransformer — 纯时序 Transformer Decoder（带 RoPE）
========================================================
参考 DSF.py 的 Block/Attention 模式，剥离空间分支，保留时序注意力 + RoPE。

结构:
  TemporalTransformerDecoder
    └── N × TemporalTransformerBlock
          ├── TemporalSelfAttention (带 RoPE)
          └── MLP (GELU, Dropout)

与 DSF.py 中 DSTFormer 的关键区别：
  - 无空间注意力分支（只关注帧间关系，不关注关节间关系）
  - 使用 RoPE 替代 learned temporal embedding
  - 输入 [B, F, D] 而非 [B, F, J, D]
"""

import torch
import torch.nn as nn
import math

from model.drop import DropPath
from model.rope import RotaryEmbedding, apply_rotary


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """截断正态初始化 (从 DSF.py 移植)"""
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


class MLP(nn.Module):
    """双层 MLP (与 DSF.py 一致)"""
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
    """
    纯时序自注意力 (带 RoPE)

    输入: x [B, F, D]
    输出: [B, F, D]
    """
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
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, n_head, F, D_head]

        if self.use_rope:
            cos, sin = self.rope(q)  # [1, 1, F, D/2]
            q = apply_rotary(q, cos, sin)
            k = apply_rotary(k, cos, sin)

        # 缩放点积注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, F, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TemporalTransformerBlock(nn.Module):
    """
    时序 Transformer Block
    TemporalSelfAttention + MLP + LayerNorm + DropPath
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, use_rope=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = TemporalSelfAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, use_rope=use_rope)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TemporalTransformerDecoder(nn.Module):
    """
    N 层时序 Transformer Decoder

    输入: [B, F, D]
    输出: [B, F, D]

    Args:
        dim: 特征维度 (默认 256)
        depth: 层数 (默认 8)
        num_heads: 注意力头数 (默认 8)
        mlp_ratio: MLP 隐藏层扩展比 (默认 4)
        drop_rate: Dropout 率
        attn_drop_rate: 注意力 Dropout 率
        drop_path_rate: Stochastic Depth 率
        use_rope: 是否使用 RoPE
    """
    def __init__(self, dim=256, depth=8, num_heads=8, mlp_ratio=4.,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 use_rope=True, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_rope = use_rope

        # Stochastic Depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            TemporalTransformerBlock(
                dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, qk_scale=None,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer, use_rope=use_rope)
            for i in range(depth)])

        self.norm = norm_layer(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x
