"""可视化: 输入误差(上) vs 运动本身(下), 共享时间轴.

上: 某关节每帧的输入误差 |input - GT| (3D 距离 mm);
下: 同一关节同一坐标的 input 位置 与 GT 位置 (mm), 看误差与运动速度的关系.

用法: uv run python experiments/viz_error_motion.py --seq 'Capture1/.../cam400262'
"""
import argparse
import sys
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(os.path.dirname(HERE))     # tools/viz → experiments
sys.path.insert(0, os.path.dirname(EXPERIMENTS))          # HTD 根
sys.path.insert(0, EXPERIMENTS)
from common import seq_io
from common.hand_topology import JOINT_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="", help="序列相对路径; 空则自动挑一条运动大的")
    ap.add_argument("--joint", type=int, default=8, help="关节索引 (8=食指TIP)")
    ap.add_argument("--coord", type=int, default=1, help="坐标 (0=x 1=y 2=z)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out", default="experiments/vis_input_error_motion.png")
    args = ap.parse_args()

    seqs = seq_io.list_sequences()
    if args.seq:
        rel = args.seq
    else:
        # 挑"误差适中(5-20mm, 接近测试集均值13.6) + 运动丰富"的序列
        best, best_score = None, -1
        for s in seqs[:300]:
            ji = seq_io.load_field(s, "in_joint")
            gj = seq_io.load_field(s, "gt_joint")
            if ji is None or len(ji) < 120:
                continue
            e = np.linalg.norm(ji - gj, axis=-1).mean()
            if not (5 < e < 20):
                continue
            v = np.abs(np.diff(gj[:, 8], axis=0)).mean()   # 指尖运动幅度
            if v > best_score:
                best_score, best = v, s
        rel = best
        print(f"自动挑选: {rel} (运动量 {best_score:.1f})")

    ji = seq_io.load_field(rel, "in_joint")
    gj = seq_io.load_field(rel, "gt_joint")
    valid = seq_io.load_field(rel, "gt_joint_valid")
    if ji is None or gj is None:
        print("序列数据缺失"); return
    T = len(ji)
    if args.max_frames:
        T = min(T, args.max_frames)
    ji, gj = ji[:T], gj[:T]
    if valid is not None:
        valid = valid[:T, args.joint] > 0.5

    j = args.joint
    c = args.coord
    err = np.linalg.norm(ji[:, j] - gj[:, j], axis=-1)          # 每帧 3D 误差
    in_pos = ji[:, j, c]
    gt_pos = gj[:, j, c]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.2]})
    ax1.plot(err, color="#d62728", lw=1)
    ax1.axhline(err.mean(), color="gray", ls="--", lw=0.8, label=f"mean {err.mean():.2f}mm")
    ax1.set_ylabel("input error (mm)")
    ax1.set_title(f"{rel}\n{JOINT_NAMES[j]}: per-frame input error (top) vs motion (bottom)",
                  fontsize=11)
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)

    ax2.plot(in_pos, color="#ff7f0e", lw=1.2, label="input")
    ax2.plot(gt_pos, color="#1f77b4", lw=1.2, label="GT")
    if valid is not None:
        inv = ~valid
        ax2.scatter(np.where(inv)[0], gt_pos[inv], color="gray", s=4, label="invalid")
    ax2.set_ylabel(f"position {['x','y','z'][c]} (mm)")
    ax2.set_xlabel("frame")
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"已存: {args.out}")
    # 误差与运动的关系: 该关节 3D 速度 vs 输入误差
    gt_speed = np.linalg.norm(np.diff(gj[:, j], axis=0), axis=-1)   # 关节3D速度
    r1 = np.corrcoef(gt_speed, err[1:])[0, 1]
    # 误差分桶: 慢/快 半帧
    med = np.median(gt_speed)
    err_slow = err[1:][gt_speed < med].mean()
    err_fast = err[1:][gt_speed >= med].mean()
    print(f"关节3D速度 vs 输入误差 相关: {r1:.3f}")
    print(f"慢半输入误差均值 {err_slow:.2f}mm, 快半 {err_fast:.2f}mm (差 {err_fast-err_slow:+.2f}mm)")


if __name__ == "__main__":
    main()
