"""Rerun 可视化: exp1_s8(拼31) vs exp2_s31 vs GT vs input, 长连续片段 (~56s, 1674帧).

用法: .venv_vis/bin/python experiments/viz_rerun_long.py [--save out.rrd] [--start 0] [--n 1674]
"""
import argparse, sys
from pathlib import Path
import numpy as np
import rerun as rr

HERE = Path(__file__).resolve().parent          # experiments/viz/longclip
sys.path.insert(0, str(HERE.parent.parent))    # experiments → 使 common 可导入
from common.hand_topology import JOINT_NAMES, EDGES

DATA = HERE / "longclip.npz"
ENTITIES = [
    ("input",    "#ff7f0e", -0.30),
    ("GT",       "#1f77b4", -0.10),
    ("exp1_s8",  "#2ca02c",  0.10),
    ("exp2_s31", "#d62728",  0.30),
]

def rgb(hexc):
    return tuple(int(hexc[i:i+2], 16) / 255 for i in (1, 3, 5))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default="", help="存 .rrd (缺省 spawn)")
    ap.add_argument("--trail", type=int, default=5)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()

    d = np.load(DATA)
    ji, gj, o1, o2, mk = d["in_joint"], d["gt_joint"], d["exp1_s8_out"], d["exp2_out"], d["msk"] > 0
    T = len(ji)
    if args.n: T = min(T, args.n)
    print(f"长片段: {T} 帧 (~{T/30:.0f}s), 4 实体并排, mask 置灰")

    rr.init("exp1_s8_vs_s31_long")
    if args.save:
        rr.save(args.save)
    else:
        rr.spawn()

    datas = {"input": ji, "GT": gj, "exp1_s8": o1, "exp2_s31": o2}
    for t in range(args.start, args.start + T):
        rr.set_time("frame", sequence=t)
        for name, hexc, off in ENTITIES:
            pts = datas[name][t] / 1000.0 + np.array([off, 0, 0])
            valid = mk[t]
            colors = [rgb(hexc) if v else (0.5, 0.5, 0.5) for v in valid]
            rr.log(f"{name}/points", rr.Points3D(pts, colors=colors, radii=0.004, labels=JOINT_NAMES))
            edge_cols = [rgb("#ffffff") if valid[i] and valid[j] else (0.4, 0.4, 0.4)
                         for i, j in EDGES]
            rr.log(f"{name}/skeleton",
                   rr.LineStrips3D(strips=[pts[[i, j]] for i, j in EDGES],
                                   colors=edge_cols, radii=0.002))
            if args.trail and t > args.start:
                prev = datas[name][t - 1] / 1000.0 + np.array([off, 0, 0])
                tr = [([prev[j], pts[j]] if valid[j] else [pts[j], pts[j]]) for j in range(21)]
                rr.log(f"{name}/trail",
                       rr.LineStrips3D(strips=tr, colors=[rgb(hexc)] * 21, radii=0.001))
        if t % 300 == 0:
            print(f"frame {t}", flush=True)
    print("完成。实体树: input/GT/exp1_s8/exp2_s31, 时间轴播放。")


if __name__ == "__main__":
    main()
