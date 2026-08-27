"""实验1: 逐层注意力 n_eff 统计 — 验证"长窗核更平 / 有效帧饱和".

n_eff = rows / Σw²  (每头), 有效参与帧数 ∈ [1, F].
util_frac = n_eff / F  (有效帧占窗口比例).
dist_mean = 平均注意力距离 (帧).

用法:
  CUDA_VISIBLE_DEVICES=0 python tools/analysis/analyze_neff.py --n-test 3000
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))          # tools/analysis
EXPERIMENTS = os.path.dirname(os.path.dirname(HERE))       # experiments
sys.path.insert(0, EXPERIMENTS)

from common.frame_len import (AttnAccumulator, load_config, load_model, load_test_loader)

# 纯 Transformer 模型 (s31 必须用原版 checkpoint, 不是 LRU 混合版)
MODELS = [
    ("exp2_s8_att", "checkpoint/best.pth"),
    ("exp2_s15_att", "checkpoint/best.pth"),
    ("exp2_s30_att", "checkpoint/best_orig_full.pth"),
]


def run_neff(exp, ckpt, n_test):
    exp_dir = os.path.join(EXPERIMENTS, exp)
    cfg = load_config(exp_dir)
    model = load_model(cfg, ckpt)
    test_ld = load_test_loader(cfg, n_test)
    acc = AttnAccumulator(model)
    n = 0
    with torch.no_grad():
        for inp, tgt, meta in test_ld:
            model(inp, tgt, meta)
            n += inp["joint_xyz"].shape[0]
            if n_test is not None and n >= n_test:
                break
    return cfg.seq_len, acc.stats(cfg.seq_len)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+", default=[m[0] for m in MODELS])
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--n-test", type=int, default=3000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    results = {}
    print(f"{'exp':14s} F   layer  n_eff_mean  n_eff_std  util_frac  dist_mean")
    print("-" * 70)
    for exp in args.exps:
        ckpt = args.ckpt or dict(MODELS).get(exp, "checkpoint/best.pth")
        F, layer_stats = run_neff(exp, ckpt, args.n_test)
        n_eff_all = [s["n_eff_mean"] for s in layer_stats.values()]
        results[exp] = {"F": F, "layers": layer_stats,
                        "overall": {"n_eff_mean": float(np.mean(n_eff_all)),
                                    "n_eff_avg_F": float(np.mean(n_eff_all) / F),
                                    "dist_mean": float(np.mean([s["dist_mean"] for s in layer_stats.values()]))}}
        for idx, s in layer_stats.items():
            print(f"{exp:14s} {F:<3d} {idx:5d}  {s['n_eff_mean']:9.2f}  {s['n_eff_std']:8.2f}  "
                  f"{s['util_frac']:8.2f}  {s['dist_mean']:9.2f}")
        ov = results[exp]["overall"]
        print(f"{'':14s} {'':3s} avg   {ov['n_eff_mean']:9.2f}                {ov['n_eff_avg_F']:8.2f}  {ov['dist_mean']:9.2f}")
    print()

    out_path = args.out or os.path.join(EXPERIMENTS, "eval_data", "frame_len_neff.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"报告: {out_path}")


if __name__ == "__main__":
    main()
