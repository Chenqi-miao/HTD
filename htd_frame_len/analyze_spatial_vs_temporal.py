"""实验: 空间 vs 时序功能消融 — 证明"短窗位置靠空间注意力".

对 exp2 纯 Transformer 模型 (s8/s15/s31) 在**同一测试子集**上做推理期消融:
  把 joint_attn_pre / temporal / joint_attn_post 逐个换成 Identity,
测 MPJPE(位置) / MPJVE(速度) / MPJAE(加速度) / corr_v / corr_a。

变体:
  full          = 全部模块
  pure_temporal = joint_attn_pre + joint_attn_post 置 Identity   (只留时序)
  pure_spatial  = temporal 置 Identity                           (只留空间)
  no_pre        = 仅 joint_attn_pre 置 Identity
  no_post       = 仅 joint_attn_post 置 Identity

假设 (用户): 短窗位置由空间注意力(帧内关节结构先验)承载。
  预期 s8: pure_spatial ≈ full (空间能独立修位置), pure_temporal ≫ full (时序修不动位置)。

⚠️ 推理期零掉模块是 OOD (下游按完整管线训练), 结论是方向性的。

用法:
  CUDA_VISIBLE_DEVICES=0 python tools/analysis/analyze_spatial_vs_temporal.py --n-test 3000
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, EXPERIMENTS)

from common.frame_len import load_config, load_model, load_test_loader, predict
from common.metrics import corr_kinematics

MODELS = [
    ("exp2_s8_att", "checkpoint/best.pth"),
    ("exp2_s15_att", "checkpoint/best.pth"),
    ("exp2_s30_att", "checkpoint/best_orig_full.pth"),
]

VARIANTS = ["full", "pure_temporal", "pure_spatial", "no_pre", "no_post"]


def apply_variant(model, orig, variant):
    def _id_or(key, want):
        return nn.Identity() if key in want else orig[key]
    if variant == "full":
        model.joint_attn_pre = orig["joint_attn_pre"]
        model.temporal = orig["temporal"]
        model.joint_attn_post = orig["joint_attn_post"]
    elif variant == "pure_temporal":          # 去掉全部空间注意力
        model.joint_attn_pre = nn.Identity()
        model.joint_attn_post = nn.Identity()
        model.temporal = orig["temporal"]
    elif variant == "pure_spatial":           # 去掉时序
        model.joint_attn_pre = orig["joint_attn_pre"]
        model.temporal = nn.Identity()
        model.joint_attn_post = orig["joint_attn_post"]
    elif variant == "no_pre":
        model.joint_attn_pre = nn.Identity()
        model.temporal = orig["temporal"]
        model.joint_attn_post = orig["joint_attn_post"]
    elif variant == "no_post":
        model.joint_attn_pre = orig["joint_attn_pre"]
        model.temporal = orig["temporal"]
        model.joint_attn_post = nn.Identity()


def weighted_metrics(ji, gj, pj, msk):
    """累计加权指标, 返回 {src: {metric: value}}."""
    acc = {src: {k: [0.0, 0.0] for k in ["mpjpe", "mpjve", "mpjae"]} for src in ("input", "model")}
    corr = {src: {"v": [], "a": []} for src in ("input", "model")}
    W, F, J = ji.shape[:3]
    for b in range(W):
        m = msk[b]
        for src, x in (("input", ji[b]), ("model", pj[b])):
            e = np.sqrt(((x - gj[b]) ** 2).sum(-1))
            acc[src]["mpjpe"][0] += (e * m).sum(); acc[src]["mpjpe"][1] += m.sum()
            vp = x[1:] - x[:-1]; vg = gj[b][1:] - gj[b][:-1]
            vm = m[:-1] * m[1:]
            ev = np.sqrt(((vp - vg) ** 2).sum(-1))
            acc[src]["mpjve"][0] += (ev * vm).sum(); acc[src]["mpjve"][1] += vm.sum()
            ap = vp[1:] - vp[:-1]; ag = vg[1:] - vg[:-1]
            am = m[:-2] * m[1:-1] * m[2:]
            ea = np.sqrt(((ap - ag) ** 2).sum(-1))
            acc[src]["mpjae"][0] += (ea * am).sum(); acc[src]["mpjae"][1] += am.sum()
            cv_, ca_ = corr_kinematics(x, gj[b], m)[:2]
            corr[src]["v"].append(cv_); corr[src]["a"].append(ca_)
    out = {}
    for src in ("input", "model"):
        out[src] = {k: acc[src][k][0] / (acc[src][k][1] + 1e-8) for k in ["mpjpe", "mpjve", "mpjae"]}
        out[src]["corr_v"] = float(np.nanmean(corr[src]["v"]))
        out[src]["corr_a"] = float(np.nanmean(corr[src]["a"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+", default=[m[0] for m in MODELS])
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--n-test", type=int, default=3000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    results = {}
    for exp in args.exps:
        ckpt = args.ckpt or dict(MODELS).get(exp, "checkpoint/best.pth")
        cfg = load_config(os.path.join(EXPERIMENTS, exp))
        model = load_model(cfg, ckpt)
        test_ld = load_test_loader(cfg, None)
        orig = {"joint_attn_pre": model.joint_attn_pre, "temporal": model.temporal,
                "joint_attn_post": model.joint_attn_post}
        print(f"\n===== {exp} (seq_len={cfg.seq_len}) =====", flush=True)
        print(f"{'variant':14s} {'MPJPE':>7s} {'MPJVE':>7s} {'MPJAE':>7s} {'corr_v':>7s} {'corr_a':>7s}  (input MPJPE)")
        results[exp] = {}
        for variant in VARIANTS:
            apply_variant(model, orig, variant)
            d = predict(model, test_ld, args.n_test)
            wm = weighted_metrics(d["ji"], d["gj"], d["pj"], d["msk"])
            m, im = wm["model"], wm["input"]
            results[exp][variant] = m
            print(f"{variant:14s} {m['mpjpe']:7.2f} {m['mpjve']:7.2f} {m['mpjae']:7.2f} "
                  f"{m['corr_v']:7.3f} {m['corr_a']:7.3f}  (input {im['mpjpe']:.2f})", flush=True)
        apply_variant(model, orig, "full")   # 恢复

    out_path = args.out or os.path.join(EXPERIMENTS, "eval_data", "frame_len_spatial_vs_temporal.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n报告: {out_path}")


if __name__ == "__main__":
    main()
