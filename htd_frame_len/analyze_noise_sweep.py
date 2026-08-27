"""实验2: 输入噪声扫描 — 验证 F*∝σ^{2/5} (高斯) 与短窗重尾鲁棒 (尖峰).

对固定测试子集人为加噪声, 看"最优 F"随噪声强度右移:
  高斯 σ=0,5,10,20mm  → 若 σ↑ 时最佳窗口从短窗移到长窗, 则支持 bias-variance 权衡;
  稀疏尖峰 (1-2 帧大偏移) → 若短窗尤其抗衰, 则支持"短窗对离群帧结构性免疫".

每个模型只加载一次 (数据加载 ~30s/次), 循环所有噪声配置, 各自独立跑前向.

用法:
  CUDA_VISIBLE_DEVICES=0 python tools/analysis/analyze_noise_sweep.py --n-test 3000
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

GAUSS = [0.0, 5.0, 10.0, 20.0]
SPIKES = [(1, 30.0), (2, 30.0), (1, 60.0)]     # (n_spike_frames, delta_mm)


def gauss_noise(sigma_mm):
    def f(ji, rng):
        return rng.normal(0.0, sigma_mm, size=ji.shape).astype(np.float32)
    return f


def spike_noise(n_spike, delta_mm):
    """返回增量 δ (与 gauss_noise 一致: predict 里 ji = ji + noise_fn(ji))."""
    def f(ji, rng):
        W, F, J, C = ji.shape
        noise = np.zeros_like(ji)
        for w in range(W):
            idx = rng.choice(F, size=n_spike, replace=False)
            noise[w, idx] += rng.normal(delta_mm, delta_mm * 0.3, size=(n_spike, J, C))
        return noise
    return f


def masked_mpjpe(x, g, m):
    e = np.sqrt(((x - g) ** 2).sum(-1))
    return float((e * m).sum() / (m.sum() + 1e-8))


def run_all_noises(exp, ckpt, n_test, configs):
    """加载一次模型+数据, 对每个噪声配置跑 predict (内部按 n_test 截断)."""
    cfg = load_config(os.path.join(EXPERIMENTS, exp))
    model = load_model(cfg, ckpt)
    test_ld = load_test_loader(cfg, None)   # data_num 不限制窗数, 截断交给 predict
    out = {}
    for name, noise_fn in configs:
        d = predict(model, test_ld, n_test, noise_fn=noise_fn)
        out[name] = {"input_mpjpe": masked_mpjpe(d["ji"], d["gj"], d["msk"]),
                     "model_mpjpe": masked_mpjpe(d["pj"], d["gj"], d["msk"])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+", default=[m[0] for m in MODELS])
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--n-test", type=int, default=3000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    configs = [(f"gauss-{s:.0f}", gauss_noise(s)) for s in GAUSS]
    configs += [(f"spike-{n}x{d:.0f}", spike_noise(n, d)) for n, d in SPIKES]

    res = {}
    for exp in args.exps:
        print(f"[{exp}] 加载模型+数据...", flush=True)
        ckpt = args.ckpt or dict(MODELS).get(exp, "checkpoint/best.pth")
        res[exp] = run_all_noises(exp, ckpt, args.n_test, configs)

    # 打印表格: 每行噪声, 每模型 model_mpjpe, 标记最优
    print(f"\n{'noise':12s} {'':>8s} " + "".join(f"{e[-6:]:>14s}" for e in args.exps))
    print(" " * 22 + "".join(f"{'model':>7s}{'best':>7s}" for _ in args.exps))
    for name, _ in configs:
        vals = [res[e][name] for e in args.exps]
        best_i = int(np.argmin([v["model_mpjpe"] for v in vals]))
        line = f"{name:12s}"
        for i, v in enumerate(vals):
            mark = " *" if i == best_i else "  "
            line += f"{v['model_mpjpe']:9.2f}{mark:6s}"
        print(line)
        line = f"{'':12s} input "
        for v in vals:
            line += f"{v['input_mpjpe']:9.2f}{'':6s}"
        print(line)
    print()

    results = {"n_test": args.n_test, "models": list(res.keys()),
               "gauss_sigmas": GAUSS, "spikes": SPIKES, "per_noise": res}
    out_path = args.out or os.path.join(EXPERIMENTS, "eval_data", "frame_len_noise_sweep.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"报告: {out_path}")


if __name__ == "__main__":
    main()
