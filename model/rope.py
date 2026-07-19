"""
RoPE — Rotary Position Embedding
==================================
从 3D 空间旋转推广到高维特征的逐维度旋转。
核心思想：对 token 序列的每对维度 (2i, 2i+1) 按位置 t 施加旋转。

Reference:
  RoFormer: Enhanced Transformer with Rotary Position Embedding
  https://arxiv.org/abs/2104.09864
"""

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """
    RoPE cos/sin 查找表

    Cache: [1, max_len, D/2]  (3D)
    配合 apply_rotary() 使用。

    输入 x: [B, F, D] 或 [B, n_head, F, D_head]
    返回 cos, sin: [1, F, D/2] 或 [1, 1, F, D/2] (与 x 维度对齐)
    """

    def __init__(self, dim: int, max_len: int = 243):
        super().__init__()
        assert dim % 2 == 0, f"RoPE requires even dim, got {dim}"

        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        position = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.outer(position, inv_freq)            # [max_len, D/2]
        cos_cached = freqs.cos().unsqueeze(0)              # [1, max_len, D/2]
        sin_cached = freqs.sin().unsqueeze(0)

        self.register_buffer('cos_cached', cos_cached)
        self.register_buffer('sin_cached', sin_cached)

    def forward(self, x: torch.Tensor, start_pos: int = 0):
        """
        Args:
            x: 占位张量，用于获取 seq_len 和维度
               [B, F, D]        → 返回 [1, F, D/2]
               [B, n_head, F, D_head] → 返回 [1, 1, F, D/2]
        Returns:
            cos, sin: 与 x 的前两维广播兼容
        """
        seq_len = x.shape[-2]
        cos = self.cos_cached[:, start_pos:start_pos + seq_len, :]  # [1, F, D/2]
        sin = self.sin_cached[:, start_pos:start_pos + seq_len, :]

        if x.dim() == 4:
            cos = cos.unsqueeze(1)  # [1, 1, F, D/2]
            sin = sin.unsqueeze(1)

        return cos, sin


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    在 Q 或 K 上施加 RoPE (in-place, shape 不变)

    x:   [B, n_head, F, D_head]  或  [B, F, D]
    cos: [1,    1,     F, D/2]   或  [1, F, D/2]
    sin: 同上
    """
    D = x.shape[-1]
    # cos/sin 当前是 [..., D/2]，需要扩展到 [..., D]
    cos = cos.repeat_interleave(2, dim=-1)
    sin = sin.repeat_interleave(2, dim=-1)

    # rotate_half: [x0, x1, x2, x3, ...] → [-x2, -x3, x0, x1, ...]
    x_half = x.chunk(2, dim=-1)
    x_rotated = torch.cat([-x_half[1], x_half[0]], dim=-1)

    return x * cos + x_rotated * sin
