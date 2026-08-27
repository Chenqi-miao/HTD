"""Rerun 可视化: exp1_s8(拼31) vs exp2_s31 vs GT vs input (5 个 31 帧窗口并排).

用法: .venv_vis/bin/python experiments/viz_rerun_s8_vs_s31.py [--save out.rrd]
      --save 缺省则 spawn viewer.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import rerun as rr

HERE = Path(__file__).resolve().parent          # experiments/viz/win_compare
sys.path.insert(0, str(HERE.parent.parent))    # experiments → 使 common 可导入
from common.hand_topology import JOINT_NAMES, EDGES

DATA = HERE / "rerun_s8_s31_sample.npz"

ENTITIES = [
    ("GT",       "#1f77b4", -0.30),
    ("exp1_s8",  "#2ca02c", -0.10),
    ("exp2_s31", "#d62728",  0.10),
    ("input",    "#ff7f0e",  0.30),
]

def rgb(hexc):
    return tuple(int(hexc[i:i+2], 16) / 255 for i in (1, 3, 5))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default="")
    ap.add_argument("--trail", type=int, default=5)
    args = ap.parse_args()

    d = np.load(DATA)
    sel = [int(x) for x in d["sel"]]
    print("窗口:", sel)

    rr.init("exp1_s8_vs_s31")
    if args.save:
        rr.save(args.save)
    else:
        rr.spawn()

    for wi, idx in enumerate(sel):
        wname = "w%d" % idx
        datas = {"GT": d["gt_%d" % idx], "exp1_s8": d["p8_%d" % idx],
                 "exp2_s31": d["p31_%d" % idx], "input": d["inp_%d" % idx]}
        for t in range(31):
            rr.set_time("frame", sequence=wi * 31 + t)
            for name, hexc, off in ENTITIES:
                pts = datas[name][t] / 1000.0 + np.array([off, 0, 0])
                c = [rgb(hexc)] * 21
                rr.log(f"{wname}/{name}/points", rr.Points3D(pts, colors=c, radii=0.004, labels=JOINT_NAMES))
                rr.log(f"{wname}/{name}/skeleton",
                       rr.LineStrips3D(strips=[pts[[i, j]] for i, j in EDGES],
                                       colors=[rgb("#ffffff")], radii=0.002))
                if args.trail and t > 0:
                    prev = datas[name][t - 1] / 1000.0 + np.array([off, 0, 0])
                    rr.log(f"{wname}/{name}/trail",
                           rr.LineStrips3D(strips=[[prev[j], pts[j]] for j in range(21)],
                                           colors=[rgb(hexc)] * 21, radii=0.001))
    print("完成。实体树: w{idx}/GT|exp1_s8|exp2_s31|input, 时间轴播放。")


if __name__ == "__main__":
    main()
