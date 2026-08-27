"""Rerun 误差热力图: input / exp1_s8 / exp2_s31 各关节 |pred-GT| 着色 (蓝0 → 红≥ERR_MAX mm).

GT 用中性灰作参考。用法: .venv_vis/bin/python experiments/viz_rerun_error.py [--save out.rrd]
"""
import argparse, sys
from pathlib import Path
import numpy as np
import rerun as rr

HERE = Path(__file__).resolve().parent          # experiments/viz/longclip
sys.path.insert(0, str(HERE.parent.parent))    # experiments → 使 common 可导入
from common.hand_topology import JOINT_NAMES, EDGES

DATA = HERE / "longclip.npz"
ERR_MAX = 15.0

def jet(v):
    v = max(0.0, min(1.0, v)) * 4.0
    if v < 1: return (0.0, v, 1.0)
    if v < 2: return (0.0, 1.0, 2.0 - v)
    if v < 3: return (v - 2.0, 1.0, 0.0)
    return (1.0, 4.0 - v, 0.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default="")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()

    d = np.load(DATA)
    ji, gj, o1, o2, mk = d["in_joint"], d["gt_joint"], d["exp1_s8_out"], d["exp2_out"], d["msk"] > 0
    T = len(ji)
    if args.n: T = min(T, args.n)
    print(f"误差热力图: {T} 帧, 色标 0-{ERR_MAX}mm (蓝=0 → 红=≥{ERR_MAX}mm)", flush=True)

    rr.init("hand_error_heatmap")
    if args.save:
        rr.save(args.save)
    else:
        rr.spawn()

    for t in range(args.start, args.start + T):
        rr.set_time("frame", sequence=t)
        for name, off in [("input", -0.30), ("GT", -0.10), ("exp1_s8", 0.10), ("exp2_s31", 0.30)]:
            g = gj[t]
            if name == "GT":
                pts = g / 1000.0 + np.array([off, 0, 0])
                jc = [(0.8, 0.8, 0.8) if v else (0.4, 0.4, 0.4) for v in mk[t]]
                ec = [(0.6, 0.6, 0.6)] * len(EDGES)
                rad = 0.003
            else:
                pred = {"input": ji, "exp1_s8": o1, "exp2_s31": o2}[name][t]
                err = np.linalg.norm(pred - g, axis=-1)
                pts = pred / 1000.0 + np.array([off, 0, 0])
                jc = [jet(e / ERR_MAX) if v else (0.4, 0.4, 0.4) for e, v in zip(err, mk[t])]
                ec = [jet(((err[i] + err[j]) / 2) / ERR_MAX) if mk[t][i] and mk[t][j] else (0.4, 0.4, 0.4)
                      for i, j in EDGES]
                rad = 0.004
            rr.log(f"{name}/points", rr.Points3D(pts, colors=jc, radii=rad, labels=JOINT_NAMES))
            rr.log(f"{name}/skeleton", rr.LineStrips3D(strips=[pts[[i, j]] for i, j in EDGES],
                                                       colors=ec, radii=0.002))
        if t % 300 == 0:
            print("frame", t, flush=True)
    print("完成。蓝=0mm → 红=≥%dmm。实体树 input/GT/exp1_s8/exp2_s31。" % ERR_MAX, flush=True)


if __name__ == "__main__":
    main()
