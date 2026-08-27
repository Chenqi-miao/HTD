"""可视化: 滤波后数据 vs 原始输入 vs GT, 同格式(上误差/下位置).

上: 原始输入误差(红) 与 滤波后误差(绿);
下: 原始输入位置(橙)、滤波后位置(绿)、GT(蓝).
"""
import argparse
import sys
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(os.path.dirname(HERE))     # tools/viz → experiments
sys.path.insert(0, os.path.dirname(EXPERIMENTS))          # HTD 根
sys.path.insert(0, EXPERIMENTS)
from common import seq_io
from common.hand_topology import JOINT_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="Capture0/ROM03_LT_No_Occlusion/InterWild/cam400262")
    ap.add_argument("--joint", type=int, default=8, help="关节 (8=食指TIP)")
    ap.add_argument("--coord", type=int, default=1)
    ap.add_argument("--sg-window", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out", default="experiments/vis_filtered_error_motion.png")
    args = ap.parse_args()

    rel = args.seq
    ji = seq_io.load_field(rel, "in_joint")
    gj = seq_io.load_field(rel, "gt_joint")
    valid = seq_io.load_field(rel, "gt_joint_valid")
    T = len(ji)
    if args.max_frames:
        T = min(T, args.max_frames)
    ji, gj = ji[:T], gj[:T]
    if valid is not None:
        valid = valid[:T, args.joint] > 0.5

    # SG 滤波 (沿时间轴, 每关节每坐标)
    ji_sg = savgol_filter(ji, args.sg_window, 2, axis=0)

    j, c = args.joint, args.coord
    err_in = np.linalg.norm(ji[:, j] - gj[:, j], axis=-1)
    err_sg = np.linalg.norm(ji_sg[:, j] - gj[:, j], axis=-1)
    in_pos, sg_pos, gt_pos = ji[:, j, c], ji_sg[:, j, c], gj[:, j, c]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.2]})
    ax1.plot(err_in, color="#d62728", lw=1, label=f"input err (mean {err_in.mean():.2f}mm)")
    ax1.plot(err_sg, color="#2ca02c", lw=1, label=f"SG-filtered err (mean {err_sg.mean():.2f}mm)")
    ax1.set_ylabel("error (mm)")
    ax1.set_title(f"{rel}\n{JOINT_NAMES[j]}: error (top) vs position (bottom)", fontsize=11)
    ax1.legend(loc="upper right"); ax1.grid(alpha=0.3)

    ax2.plot(in_pos, color="#ff7f0e", lw=1, alpha=0.8, label="input")
    ax2.plot(sg_pos, color="#2ca02c", lw=1.3, label="SG-filtered")
    ax2.plot(gt_pos, color="#1f77b4", lw=1.3, label="GT")
    ax2.set_ylabel(f"position {['x','y','z'][c]} (mm)")
    ax2.set_xlabel("frame")
    ax2.legend(loc="upper right"); ax2.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"已存: {args.out}")
    print(f"输入误差均值 {err_in.mean():.2f}mm → SG滤波后 {err_sg.mean():.2f}mm "
          f"({(1-err_sg.mean()/err_in.mean())*100:+.1f}%)")


if __name__ == "__main__":
    main()
