"""Rerun 可视化: exp1 的 8/15/31 帧窗口对比 (~1分钟).

5 实体并排: input(橙) / GT(蓝) / exp1_s8(绿) / exp1_s15(青) / exp1_s31(红).
数据: exp1_windows.npz (滑窗中心帧推理).

用法:
  .venv_vis/bin/python experiments/viz_rerun_exp1_windows.py            # spawn
  .venv_vis/bin/python experiments/viz_rerun_exp1_windows.py --save out.rrd
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

HERE = Path(__file__).resolve().parent          # experiments/viz/win_compare
sys.path.insert(0, str(HERE.parent.parent))    # experiments → 使 common 可导入
from common.hand_topology import JOINT_NAMES, EDGES

DATA = HERE / "exp1_windows.npz"
ENTITIES = [
    ("input",     "#ff7f0e", -0.40),
    ("GT",        "#1f77b4", -0.20),
    ("exp1_s8",   "#2ca02c",  0.00),
    ("exp1_s15",  "#17becf",  0.20),
    ("exp1_s31",  "#d62728",  0.40),
]

def rgb(hexc):
    return (int(hexc[1:3], 16) / 255, int(hexc[3:5], 16) / 255, int(hexc[5:7], 16) / 255)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default="")
    ap.add_argument("--trail", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    d = np.load(DATA)
    ji, gj = d["in_joint"], d["gt_joint"]
    o8, o15, o31 = d["exp1_s8"], d["exp1_s15"], d["exp1_s31"]
    T = len(ji)
    if args.max_frames:
        T = min(T, args.max_frames)
    print(f"帧数: {T} (~{T/30:.0f}s), 5 实体并排 (exp1 窗口对比)")

    rr.init("exp1_windows_s8_s15_s31")
    if args.save:
        rr.save(args.save)
    else:
        rr.spawn()

    if args.trail:
        rr.send_blueprint(rrb.Spatial3DView(
            origin="/",
            overrides={f"{name}/trail": rrb.VisibleTimeRanges(
                timeline="frame",
                start=rrb.TimeRangeBoundary.cursor_relative(seq=-args.trail),
                end=rrb.TimeRangeBoundary.cursor_relative(),
            ) for name, _, _ in ENTITIES},
        ))

    datas = {"input": ji, "GT": gj, "exp1_s8": o8, "exp1_s15": o15, "exp1_s31": o31}
    for t in range(T):
        rr.set_time("frame", sequence=t)
        for name, hexc, off in ENTITIES:
            pts = datas[name][t] / 1000.0 + np.array([off, 0, 0])
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
    print("完成。")


if __name__ == "__main__":
    main()
