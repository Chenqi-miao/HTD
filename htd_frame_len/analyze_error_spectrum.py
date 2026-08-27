"""实验3: 误差频谱分解 — 验证长窗低通效应 (长窗低频修得多、高频修正为负).

对预测误差沿帧轴做 rFFT, 分低/中/高频带 (cycles/frame):
  low   [0, 0.10)
  mid   [0.10, 0.25)
  high  [0.25, 0.50]
每带报告 输入误差功率 vs 模型误差功率 及 修正率 (1 − model/input).
若 s31 高频带修正率为负 → 长窗把高频真信号(尖峰)抹平, 直接支持低通解释.

用法:
  CUDA_VISIBLE_DEVICES=0 python tools/analysis/analyze_error_spectrum.py --n-test 3000
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, EXPERIMENTS)

from common.frame_len import load_config, load_model, load_test_loader, predict

MODELS = [
    ("exp2_s8_att", "checkpoint/best.pth"),
    ("exp2_s15_att", "checkpoint/best.pth"),
    ("exp2_s30_att", "checkpoint/best_orig_full.pth"),
]

BANDS = [("low", 0.0, 0.10), ("mid", 0.10, 0.25), ("high", 0.25, 0.50)]


def interior_errors(d, F):
    """只取内圈帧 t∈[1,F−2] (加速度可定义), 且整窗该关节全 valid 才纳入."""
    ji, gj, pj, msk = d["ji"], d["gj"], d["pj"], d["msk"]
    W, F_, J, _ = ji.shape
    lo, hi = 1, F_ - 2
    seqs_in, seqs_md = [], []
    for w in range(W):
        for j in range(J):
            if msk[w, lo:hi + 1, j].all():
                seqs_in.append(ji[w, lo:hi + 1, j] - gj[w, lo:hi + 1, j])
                seqs_md.append(pj[w, lo:hi + 1, j] - gj[w, lo:hi + 1, j])
    if not seqs_in:
        return None, None
    return np.stack(seqs_in), np.stack(seqs_md)     # [S, F_in, 3]


def band_stats(seqs_in, seqs_md):
    """每带: 输入/模型误差功率(均值) + 修正率. 也返回整体 MPJPE 近似."""
    F = seqs_in.shape[1]
    spec_in = np.abs(np.fft.rfft(seqs_in, axis=1, norm="ortho")) ** 2   # [S, nf, 3]
    spec_md = np.abs(np.fft.rfft(seqs_md, axis=1, norm="ortho")) ** 2
    freqs = np.fft.rfftfreq(F)                                          # cycles/frame
    out = {}
    for name, f0, f1 in BANDS:
        sel = (freqs >= f0) & (freqs < f1)
        if sel.sum() == 0:
            out[name] = None
            continue
        pin = spec_in[:, sel, :].mean()
        pmd = spec_md[:, sel, :].mean()
        out[name] = {"freqs": f"{f0:.2f}-{f1:.2f}", "input_power": float(pin),
                     "model_power": float(pmd),
                     "reduction_pct": float((1 - pmd / (pin + 1e-9)) * 100)}
    return out


def run(exp, ckpt, n_test):
    cfg = load_config(os.path.join(EXPERIMENTS, exp))
    model = load_model(cfg, ckpt)
    test_ld = load_test_loader(cfg, n_test)
    d = predict(model, test_ld, n_test)
    seqs_in, seqs_md = interior_errors(d, d["F"])
    if seqs_in is None:
        return None
    return band_stats(seqs_in, seqs_md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+", default=[m[0] for m in MODELS])
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--n-test", type=int, default=3000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    results = {}
    print(f"{'exp':14s} F   band   input_power  model_power  reduction%")
    print("-" * 66)
    for exp in args.exps:
        ckpt = args.ckpt or dict(MODELS).get(exp, "checkpoint/best.pth")
        st = run(exp, ckpt, args.n_test)
        results[exp] = st
        F = {"exp2_s8_att": 8, "exp2_s15_att": 15, "exp2_s30_att": 31}.get(exp, "?")
        for name, b in (st or {}).items():
            if b is None:
                continue
            print(f"{exp:14s} {F:<3d} {name:5s}   {b['input_power']:10.3f}  {b['model_power']:10.3f}  {b['reduction_pct']:9.1f}")
        print()

    out_path = args.out or os.path.join(EXPERIMENTS, "eval_data", "frame_len_error_spectrum.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"报告: {out_path}")


if __name__ == "__main__":
    main()
