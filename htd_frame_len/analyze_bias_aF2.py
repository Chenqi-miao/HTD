"""实验4: 量化 bias∝a·F² — 按 GT 加速度分桶, 看误差随加速度增长是否按 F² 放大.

理论: bias ≈ ½·a·μ₂ ∝ a·F² → 误差 vs 加速度 的斜率应 ∝ F².
判读: 对每个模型, 分桶算 MPJPE, 对 (平均a, MPJPE) 做线性拟合得斜率;
      slope_s31/slope_s8 应接近 (31/8)² ≈ 15, 否则偏差项不成立.

只统计内圈帧 t∈[1,F−2] (加速度可定义, 且三帧全 valid).

用法:
  CUDA_VISIBLE_DEVICES=0 python tools/analysis/analyze_bias_aF2.py --n-test 3000
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
N_BUCKETS = 6


def collect(exp, ckpt, n_test):
    """返回 (a[], err_in[], err_md[]) 按内圈帧·关节打平."""
    cfg = load_config(os.path.join(EXPERIMENTS, exp))
    model = load_model(cfg, ckpt)
    test_ld = load_test_loader(cfg, n_test)
    d = predict(model, test_ld, n_test)
    ji, gj, pj, msk = d["ji"], d["gj"], d["pj"], d["msk"]
    W, F, J, _ = ji.shape
    lo, hi = 1, F - 2
    a_l, in_l, md_l = [], [], []
    # 加速度: 内圈帧 + 相邻三帧 GT, 逐关节
    for w in range(W):
        for t in range(lo, hi + 1):
            for j in range(J):
                if msk[w, t - 1, j] * msk[w, t, j] * msk[w, t + 1, j] < 0.5:
                    continue
                ag = np.linalg.norm(gj[w, t + 1, j] - 2 * gj[w, t, j] + gj[w, t - 1, j])
                a_l.append(ag)
                in_l.append(np.linalg.norm(ji[w, t, j] - gj[w, t, j]))
                md_l.append(np.linalg.norm(pj[w, t, j] - gj[w, t, j]))
    return np.array(a_l), np.array(in_l), np.array(md_l)


def linear_fit(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    if n < 2:
        return None
    # y = c0 + c1*x  (最小二乘)
    c1 = (np.cov(x, y, ddof=0)[0, 1]) / (np.var(x) + 1e-9)
    c0 = y.mean() - c1 * x.mean()
    yhat = c0 + c1 * x
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum() + 1e-9
    return {"c0": float(c0), "c1": float(c1), "r2": float(1 - ss_res / ss_tot)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+", default=[m[0] for m in MODELS])
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--n-test", type=int, default=3000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    data = {}
    all_a = []
    for exp in args.exps:
        ckpt = args.ckpt or dict(MODELS).get(exp, "checkpoint/best.pth")
        a, in_e, md_e = collect(exp, ckpt, args.n_test)
        data[exp] = (a, in_e, md_e)
        all_a.append(a)
    all_a = np.concatenate(all_a)
    edges = np.percentile(all_a, np.linspace(0, 100, N_BUCKETS + 1))

    Fmap = {e: {"exp2_s8_att": 8, "exp2_s15_att": 15, "exp2_s30_att": 31}[e] for e in args.exps}
    print("按 GT 加速度分桶 (帧·关节级), 内圈帧:")
    print(f"{'bucket':>8s} {'n':>7s} {'mean_a':>7s} " + "".join(f"{e[-4:]:>9s}" for e in args.exps))
    table = {}
    for b in range(N_BUCKETS):
        lo, hi = edges[b], edges[b + 1]
        row = {"mean_a": float((all_a[(all_a >= lo) & (all_a < hi)]).mean())}
        row["n"] = 0
        for exp in args.exps:
            a, in_e, md_e = data[exp]
            sel = (a >= lo) & (a < hi)
            row["n"] += int(sel.sum())
            row[exp] = float(md_e[sel].mean()) if sel.sum() else float("nan")
        table[b] = row
        print(f"{b:>8d} {row['n']:>7d} {row['mean_a']:>7.2f} " +
              "".join(f"{row.get(e, float('nan')):>9.3f}" for e in args.exps))

    # 斜率拟合: 每模型 MPJPE vs a (分桶点), 期望 slope ∝ F²
    print()
    print("线性拟合 MPJPE ≈ c0 + c1·a (分桶点):")
    fit_by_exp = {}
    for exp in args.exps:
        a = np.array([row["mean_a"] for row in table.values()])
        y = np.array([row[exp] for row in table.values()])
        f = linear_fit(a, y)
        fit_by_exp[exp] = f
        F = Fmap[exp]
        print(f"  {exp:14s} F={F:2d}  c1={f['c1']:8.4f}  c0={f['c0']:8.3f}  R²={f['r2']:.3f}")

    # slope 比值 vs (F 比值)²
    exps = args.exps
    if len(exps) >= 2:
        print()
        print("斜率比值 vs (F1/F2)² 理论值:")
        base = exps[0]
        for e in exps[1:]:
            r_slope = fit_by_exp[e]["c1"] / (fit_by_exp[base]["c1"] + 1e-9)
            r_F2 = (Fmap[e] / Fmap[base]) ** 2
            print(f"  {e}/{base}: 实测 slope 比={r_slope:.2f}   理论 F² 比={r_F2:.2f}")

    results = {"F": Fmap, "buckets": table, "fits": fit_by_exp}
    out_path = args.out or os.path.join(EXPERIMENTS, "eval_data", "frame_len_bias_aF2.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n报告: {out_path}")


if __name__ == "__main__":
    main()
