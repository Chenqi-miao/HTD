"""帧长分析共享模块 — 加载纯 Transformer 模型 + 测试数据 + 批量预测.

供 experiments/tools/analysis/ 下四个帧长验证脚本复用
(analyze_neff / analyze_noise_sweep / analyze_error_spectrum / analyze_bias_aF2).

口径:
  - 模型 = exp2 纯 TemporalTransformer (net.py 无 LRU 替换)。
    s31 的 checkpoint 必须用 checkpoint/best_orig_full.pth (原版 12.77),
    不要用 best.pth (那是 LRU 混合版 12.18), 否则跨 F 对比混入架构差异。
  - 测试集 = SeqHandTest 原生 pkl 管线, data_num 限制窗数 (前 N 窗)。
  - 输出 numpy 数组 [W, F, J, 3]; msk 为 [W, F, J] (continuous_val × joint_gt_val).
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))   # experiments/common
EXPERIMENTS = os.path.dirname(HERE)                 # experiments
HTD = os.path.dirname(EXPERIMENTS)                  # HTD
for p in (HTD, EXPERIMENTS, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)


def load_config(exp_dir):
    spec = importlib.util.spec_from_file_location("exp_config", os.path.join(exp_dir, "config.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = mod.Config()
    cfg.exp_dir = exp_dir
    return cfg


def load_model(cfg, ckpt_rel):
    """加载 exp2 纯 Transformer 模型 + checkpoint, 返回 cuda eval 模型."""
    sys.path.insert(0, cfg.exp_dir)
    sys.path.insert(0, os.path.join(cfg.exp_dir, "model"))
    from net import Net
    model = Net(num_frame=cfg.seq_len, num_joints=cfg.joint_num,
                dim_feat=cfg.dim_feat, depth=cfg.transformer_depth,
                w_vel=cfg.w_vel, w_accel=cfg.w_accel).cuda()
    ck = torch.load(os.path.join(cfg.exp_dir, ckpt_rel), map_location="cuda")
    sd = ck["net"] if "net" in ck else ck
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  [frame_len] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"  missing: {list(missing)[:6]}")
    model.eval()
    return model


def load_test_loader(cfg, n_test=None, batch=32, num_workers=2):
    from dataset.seqhand import SeqHandTest
    ds = SeqHandTest(cfg.data_dir, cfg.test_list, min_seq_len=cfg.min_seq_len,
                     seq_len=cfg.seq_len, view_num=cfg.view_num, data_num=n_test)
    return DataLoader(ds, batch, shuffle=False, num_workers=num_workers)


def predict(model, test_ld, n_test=None, noise_fn=None, seed=0):
    """批量预测, 返回 dict:
      ji/gj/pj: [W, F, J, 3] (mm, view0);  ji 含注入噪声
      msk:      [W, F, J]  有效掩码
      F:        窗口帧数
    noise_fn: ji_np[W,F,J,3] -> 同形噪声数组 (mm), 会在模型 forward 前加到输入.
    """
    ji_l, gj_l, pj_l, msk_l = [], [], [], []
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        for inp, tgt, meta in test_ld:
            B, V, F, J = inp["joint_xyz"].shape[:4]
            ji = inp["joint_xyz"].numpy()[:, 0].copy()
            gj = tgt["joint_xyz"].numpy()[:, 0]
            if noise_fn is not None:
                ji = ji + noise_fn(ji, rng).astype(np.float32)
                inp["joint_xyz"] = torch.from_numpy(ji[:, None, :, :, :])
            o = model(inp, tgt, meta)
            pj = o["pd_joint_xyz"].cpu().numpy()[:, 0]
            cv = meta["continuous_val"].numpy().reshape(B, -1)
            gv = meta["joint_gt_val"].numpy()[:, 0]
            msk = cv[:, :, None] * gv
            ji_l.append(ji); gj_l.append(gj); pj_l.append(pj); msk_l.append(msk)
            if n_test is not None and len(ji_l) * B >= n_test:
                break
    ji = np.concatenate(ji_l)[:n_test] if n_test else np.concatenate(ji_l)
    return {"ji": ji, "gj": np.concatenate(gj_l)[:n_test] if n_test else np.concatenate(gj_l),
            "pj": np.concatenate(pj_l)[:n_test] if n_test else np.concatenate(pj_l),
            "msk": np.concatenate(msk_l)[:n_test] if n_test else np.concatenate(msk_l),
            "F": ji.shape[1]}


def _apply_rotary(x, cos, sin):
    """与 experiments/model/rope.py 的 apply_rotary 完全一致 (内联避免 import 歧义)."""
    D = x.shape[-1]
    cos = cos.repeat_interleave(2, dim=-1)
    sin = sin.repeat_interleave(2, dim=-1)
    x_half = x.chunk(2, dim=-1)
    x_rotated = torch.cat([-x_half[1], x_half[0]], dim=-1)
    return x * cos + x_rotated * sin


class AttnAccumulator:
    """包住 TemporalSelfAttention.forward, 逐层逐头累计注意力统计.

    不存原始 attn (内存 O(1)), 只累计:
      sum_sq[l][h] = Σ_rows Σ_s w²      → n_eff_h = rows / sum_sq
      dist[l][h]   = Σ_rows Σ_s w·|s−t|  → 平均注意力距离
    用后需 model.eval() 状态一致 (drop 为 Identity, 无影响).
    """

    def __init__(self, model):
        self.rows = {}                      # 每层行数 (每个 (batch, frame) 行)
        self.sum_sq = {}
        self.dist = {}
        for idx, blk in enumerate(model.temporal.blocks):
            a = blk.attn
            self.rows[idx] = 0
            self.sum_sq[idx] = np.zeros(a.num_heads)
            self.dist[idx] = np.zeros(a.num_heads)
            orig = a.forward

            def make(idx, a, orig):
                def forward(x):
                    B, F, C = x.shape
                    qkv = a.qkv(x).reshape(B, F, 3, a.num_heads, C // a.num_heads).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv[0], qkv[1], qkv[2]
                    if a.use_rope:
                        cos, sin = a.rope(q)
                        q = _apply_rotary(q, cos, sin)
                        k = _apply_rotary(k, cos, sin)
                    attn = (q @ k.transpose(-2, -1)) * a.scale
                    attn = attn.softmax(dim=-1)
                    attn = a.attn_drop(attn)
                    self._accumulate(idx, attn)
                    out = (attn @ v).transpose(1, 2).reshape(B, F, C)
                    out = a.proj(out)
                    return a.proj_drop(out)
                return forward

            a.forward = make(idx, a, orig)

    def _accumulate(self, idx, attn):            # attn [B, H, F, F]
        B, H, F, _ = attn.shape
        sq = attn.square().sum(dim=-1)                          # [B, H, F]
        t = torch.arange(F, device=attn.device).float()
        d = (t[None, None, :] - t[None, :, None]).abs()         # [1, 1, F, F]
        dist = (attn * d).sum(dim=-1)                           # [B, H, F]
        self.sum_sq[idx] += sq.sum(dim=(0, 2)).cpu().numpy()
        self.dist[idx] += dist.sum(dim=(0, 2)).cpu().numpy()
        self.rows[idx] += B * F

    def stats(self, F):
        out = {}
        for idx in sorted(self.sum_sq):
            n_eff = self.rows[idx] / (self.sum_sq[idx] + 1e-9)       # per head
            dist_mean = self.dist[idx] / (self.rows[idx] + 1e-9)     # per head
            out[idx] = {
                "n_eff_mean": float(n_eff.mean()),
                "n_eff_std": float(n_eff.std()),
                "n_eff_min": float(n_eff.min()),
                "n_eff_max": float(n_eff.max()),
                "n_eff_head": [float(v) for v in n_eff],
                "dist_mean": float(dist_mean.mean()),
                "util_frac": float(n_eff.mean() / F),           # 有效帧占窗口比例
            }
        return out
