"""rerun 渲染 3 份 rrd (数据 viz_out.npz: 测试集大动作段 465帧):
  1) input vs GT
  2) input vs GT vs exp2原版15帧
  3) input vs GT vs stage2_s31
用法: .venv_vis/bin/python experiments/viz_rrd_stage2.py
"""
import sys
from pathlib import Path
import numpy as np
import rerun as rr

HERE = Path(__file__).resolve().parent          # experiments/viz/stage2
sys.path.insert(0, str(HERE.parent.parent))    # experiments → 使 common 可导入
from common.hand_topology import JOINT_NAMES, EDGES

ROOT = HERE
DATA = ROOT / "viz_out.npz"


def rgb(hexc):
    return tuple(int(hexc[i:i + 2], 16) / 255 for i in (1, 3, 5))


def make_rrd(datas, msk, entities, out_path, T):
    """entities: [(key, hexcolor, x_offset)]; key 是 datas 的键."""
    rr.init(out_path.stem)
    rr.save(str(out_path))
    for t in range(T):
        rr.set_time("frame", sequence=t)
        for key, hexc, off in entities:
            pts = datas[key][t] / 1000.0 + np.array([off, 0, 0])
            valid = msk[t]
            colors = [rgb(hexc) if v else (0.5, 0.5, 0.5) for v in valid]
            rr.log(f"{key}/points", rr.Points3D(pts, colors=colors, radii=0.004, labels=JOINT_NAMES))
            edge_cols = [rgb("#ffffff") if valid[i] and valid[j] else (0.4, 0.4, 0.4)
                         for i, j in EDGES]
            rr.log(f"{key}/skeleton",
                   rr.LineStrips3D(strips=[pts[[i, j]] for i, j in EDGES],
                                   colors=edge_cols, radii=0.002))
            if t > 0:
                prev = datas[key][t - 1] / 1000.0 + np.array([off, 0, 0])
                tr = [([prev[j], pts[j]] if valid[j] else [pts[j], pts[j]]) for j in range(21)]
                rr.log(f"{key}/trail",
                       rr.LineStrips3D(strips=tr, colors=[rgb(hexc)] * 21, radii=0.001))
        if t % 100 == 0:
            print(f"{out_path.name} frame {t}", flush=True)
    print(f"done: {out_path}")


def main():
    d = np.load(DATA)
    datas = {
        "input": d["in_joint"],
        "GT": d["gt_joint"],
        "exp2_s15": d["out15"],
        "stage2_s31": d["out31"],
    }
    msk = (d["msk"] > 0).astype(bool)
    T = len(datas["input"])
    outdir = ROOT
    print(f"T={T} 帧 (~{T/30:.0f}s)", flush=True)

    make_rrd(datas, msk, [("input", "#ff7f0e", -0.15), ("GT", "#1f77b4", 0.15)],
             outdir / "viz_input_gt.rrd", T)
    make_rrd(datas, msk, [("input", "#ff7f0e", -0.30), ("GT", "#1f77b4", -0.10), ("exp2_s15", "#2ca02c", 0.10)],
             outdir / "viz_exp2_s15.rrd", T)
    make_rrd(datas, msk, [("input", "#ff7f0e", -0.30), ("GT", "#1f77b4", -0.10), ("stage2_s31", "#d62728", 0.10)],
             outdir / "viz_stage2_s31.rrd", T)


if __name__ == "__main__":
    main()
