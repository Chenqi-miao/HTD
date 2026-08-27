"""骨长不变性完整统计图.

读取 verify_bone_length.py 保存的每序列统计 npz, 出:
  1. cv_per_bone.png  —— 每根骨 GT vs 输入骨长时间稳定性(中位 CV%, 按指分组着色)
  2. cv_dist.png      —— CV 跨序列分布(箱线图, GT vs 输入)
  3. dev_per_bone.png —— 输入相对 GT 骨长的偏离(中位 + IQR)
  4. hand_size.png    —— 跨受试手尺寸分布(掌骨长直方图 + 每骨 GT 骨长分布)
  5. panel.png        —— 以上 2x2 汇总

用法:
  .venv/bin/python experiments/plot_bone_length.py --npz bone_stats.npz --out vis_bone_length
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent          # experiments/viz/bone_length
sys.path.insert(0, str(HERE.parent.parent))    # experiments → 使 common 可导入
from common import hand_topology as H

GT_C = "#5b9bd5"      # 蓝
IN_C = "#ed7d31"      # 橙
DEV_C = "#70ad47"     # 绿
GRAY = "#8c8c8c"

# 每根骨所属手指组: 0=掌骨(腕→指根), 1..5 = 拇/食/中/无名/小
FINGER_OF_BONE = [0] * 5 + [1] * 3 + [2] * 3 + [3] * 3 + [4] * 3 + [5] * 3
FINGER_LABELS = ["Palm", "Thumb", "Index", "Middle", "Ring", "Pinky"]
# 分组分隔位置(前 5 根掌骨之间不隔)
SEPARATORS = [4.5, 7.5, 10.5, 13.5, 16.5]

BONE_LABELS = [H.bone_name(i, j) for i, j in H.EDGES]


def _style(ax) -> None:
    ax.set_xticks(np.arange(20))
    ax.set_xticklabels(BONE_LABELS, rotation=90, fontsize=6.5)
    for s in SEPARATORS:
        ax.axvline(s, color=GRAY, lw=0.6, alpha=0.6)
    for k, lbl in enumerate(FINGER_LABELS):
        # 每组的中心位置
        start = sum(5 if k == 0 else 3 for kk in range(k))
        width = 5 if k == 0 else 3
        ax.text(start + width / 2 - 0.5, ax.get_ylim()[1], lbl,
                ha="center", va="bottom", fontsize=8, color=GRAY)


def draw_cv_per_bone(ax, gt_cv, in_cv) -> None:
    x = np.arange(20)
    w = 0.38
    ax.bar(x - w / 2, np.median(gt_cv, axis=0) * 100, w, color=GT_C, label="GT")
    ax.bar(x + w / 2, np.median(in_cv, axis=0) * 100, w, color=IN_C, label="Input")
    ax.set_ylabel("CV% (median)")
    ax.set_title("Bone-length temporal stability CV% (median)", fontsize=10)
    ax.legend(loc="upper left")
    ax.set_ylim(bottom=0)
    _style(ax)


def draw_cv_dist(ax, gt_cv, in_cv) -> None:
    x = np.arange(20)
    bp_gt = ax.boxplot([gt_cv[:, b] * 100 for b in range(20)],
                       positions=x - 0.22, widths=0.4, showfliers=False,
                       patch_artist=True, boxprops=dict(facecolor=GT_C, alpha=0.7),
                       medianprops=dict(color="black", lw=1))
    bp_in = ax.boxplot([in_cv[:, b] * 100 for b in range(20)],
                       positions=x + 0.22, widths=0.4, showfliers=False,
                       patch_artist=True, boxprops=dict(facecolor=IN_C, alpha=0.7),
                       medianprops=dict(color="black", lw=1))
    ax.set_ylabel("CV% (all sequences)")
    ax.set_title("CV% distribution across sequences (box: quartiles)", fontsize=10)
    ax.bar([0], [0], color=GT_C, label="GT")
    ax.bar([0], [0], color=IN_C, label="Input")
    ax.legend(loc="upper left")
    _style(ax)


def draw_dev(ax, dev) -> None:
    x = np.arange(20)
    med = np.median(dev, axis=0) * 100
    lo = np.percentile(dev, 25, axis=0) * 100
    hi = np.percentile(dev, 75, axis=0) * 100
    ax.bar(x, med, color=DEV_C, yerr=[med - lo, hi - med], capsize=2,
           error_kw=dict(lw=0.8, color="black"))
    ax.axhline(np.median(dev) * 100, color=GRAY, ls="--", lw=1)
    ax.set_ylabel("Input deviation from GT bone length %")
    ax.set_title("Input rigidity violation (median + IQR)", fontsize=10)
    _style(ax)


def draw_hand_size(ax, gt_mu) -> None:
    # (a) 掌骨长分布 = 手尺寸代理
    palm = gt_mu[:, 0:5].mean(axis=1)
    ax.hist(palm, bins=60, color=GT_C, alpha=0.8)
    ax.axvline(np.median(palm), color="black", ls="--", lw=1,
               label=f"median {np.median(palm):.0f} mm")
    ax.set_xlabel("Mean metacarpal length mm (hand-size proxy)")
    ax.set_ylabel("# sequences")
    ax.set_title("Hand size across subjects", fontsize=10)
    ax.legend()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="verify_bone_length.py 的 --npz 输出")
    ap.add_argument("--out", default=str(HERE / "vis_bone_length"), help="输出目录")
    args = ap.parse_args()

    d = np.load(args.npz)
    gt_cv, in_cv, dev, gt_mu = d["gt_cv"], d["in_cv"], d["dev"], d["gt_mu"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"统计: {gt_cv.shape[0]} 序列 × {gt_cv.shape[1]} 骨")

    # 四张单图
    fig, ax = plt.subplots(figsize=(14, 4.5))
    draw_cv_per_bone(ax, gt_cv, in_cv)
    fig.tight_layout(); fig.savefig(out / "cv_per_bone.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 4.5))
    draw_cv_dist(ax, gt_cv, in_cv)
    fig.tight_layout(); fig.savefig(out / "cv_dist.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 4.5))
    draw_dev(ax, dev)
    fig.tight_layout(); fig.savefig(out / "dev_per_bone.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    draw_hand_size(ax, gt_mu)
    fig.tight_layout(); fig.savefig(out / "hand_size.png", dpi=150); plt.close(fig)

    # 2x2 汇总面板
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    draw_cv_per_bone(axes[0, 0], gt_cv, in_cv)
    draw_cv_dist(axes[0, 1], gt_cv, in_cv)
    draw_dev(axes[1, 0], dev)
    draw_hand_size(axes[1, 1], gt_mu)
    fig.tight_layout()
    fig.savefig(out / "panel.png", dpi=150)
    plt.close(fig)

    print(f"已输出到 {out}/: cv_per_bone.png cv_dist.png dev_per_bone.png hand_size.png panel.png")


if __name__ == "__main__":
    main()
