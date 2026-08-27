"""Rerun 可视化: exp1_s31 vs exp2_s31 vs input vs GT (~1分钟连续手部运动).

4 个实体并排 (x 偏移): input(橙) / GT(蓝) / exp1(绿) / exp2(红).
数据: exp_compare.npz (服务器 pkl 处理 + 两个模型输出).

用法:
  .venv_vis/bin/python experiments/viz_rerun_compare.py            # spawn 原生 viewer
  .venv_vis/bin/python experiments/viz_rerun_compare.py --save out.rrd
"""
import argparse
import sys
import os
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

HERE = Path(__file__).resolve().parent          # experiments/viz/win_compare
sys.path.insert(0, str(HERE.parent.parent))    # experiments → 使 common 可导入
from common.hand_topology import JOINT_NAMES, EDGES

DATA = HERE / "exp_compare.npz"

# 每实体颜色
ENTITIES = [
    ("input",  "#ff7f0e", -0.30),
    ("GT",     "#1f77b4", -0.10),
    ("exp1",   "#2ca02c",  0.10),
    ("exp2",   "#d62728",  0.30),
]

def rgb(hexc, tint=None):
    r, g, b = int(hexc[1:3], 16) / 255, int(hexc[3:5], 16) / 255, int(hexc[5:7], 16) / 255
    if tint:
        r, g, b = (r + tint[0]) / 2, (g + tint[1]) / 2, (b + tint[2]) / 2
    return r, g, b

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default="", help="存 .rrd 文件 (否则 spawn viewer)")
    ap.add_argument("--trail", type=int, default=10, help="轨迹窗口帧数")
    ap.add_argument("--max-frames", type=int, default=0, help="0=全部")
    args = ap.parse_args()

    d = np.load(DATA)
    ji, gj, o1, o2 = d["in_joint"], d["gt_joint"], d["exp1_out"], d["exp2_out"]
    T = len(ji)
    if args.max_frames:
        T = min(T, args.max_frames)
    print(f"帧数: {T} (~{T/30:.0f}s), 4 实体并排")

    rr.init("exp1_vs_exp2_s31")
    if args.save:
        rr.save(args.save)
    else:
        rr.spawn()

    # 轨迹时间窗
    if args.trail:
        rr.send_blueprint(rrb.Spatial3DView(
            origin="/",
            overrides={f"{name}/trail": rrb.VisibleTimeRanges(
                timeline="frame",
                start=rrb.TimeRangeBoundary.cursor_relative(seq=-args.trail),
                end=rrb.TimeRangeBoundary.cursor_relative(),
            ) for name, _, _ in ENTITIES},
        ))

    datas = {"input": ji, "GT": gj, "exp1": o1, "exp2": o2}
    for t in range(T):
        rr.set_time("frame", sequence=t)
        for name, hexc, off in ENTITIES:
            pts = datas[name][t] / 1000.0          # mm → m
            pts = pts + np.array([off, 0, 0])
            c = [rgb(hexc)] * 21
            rr.log(f"{name}/points", rr.Points3D(pts, colors=c, radii=0.003, labels=JOINT_NAMES))
            rr.log(f"{name}/skeleton",
                   rr.LineStrips3D(strips=[pts[[i, j]] for i, j in EDGES],
                                   colors=[rgb("#ffffff")], radii=0.0015))
            if args.trail and t > 0:
                prev = datas[name][t - 1] / 1000.0 + np.array([off, 0, 0])
                rr.log(f"{name}/trail",
                       rr.LineStrips3D(strips=[[prev[j], pts[j]] for j in range(21)],
                                       colors=[rgb(hexc)] * 21, radii=0.001))
        if t % 300 == 0:
            print(f"frame {t}", flush=True)

    print("完成。左侧实体树开关 input/GT/exp1/exp2, 时间轴播放。")


if __name__ == "__main__":
    main()
