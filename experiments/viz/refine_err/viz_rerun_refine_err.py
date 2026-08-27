"""Rerun 可视化: refine 输出误差热力图 + 逐帧误差曲线 (定位误差大的连续帧段)."""
import argparse, sys
from pathlib import Path
import numpy as np
import rerun as rr

HERE = Path(__file__).resolve().parent          # experiments/viz/refine_err
sys.path.insert(0, str(HERE.parent.parent))    # experiments → 使 common 可导入
from common.hand_topology import JOINT_NAMES, EDGES

DATA = HERE / "hotspot_seq.npz"
ERR_MAX = 20.0

def jet(v):
    v = max(0.0, min(1.0, v)) * 4.0
    if v < 1: return (0.0, v, 1.0)
    if v < 2: return (0.0, 1.0, 2.0 - v)
    if v < 3: return (v - 2.0, 1.0, 0.0)
    return (1.0, 4.0 - v, 0.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default="")
    args = ap.parse_args()
    d = np.load(DATA)
    pr, g, m, fe = d["pr"], d["g"], d["m"] > 0, d["frame_err"]
    T = len(pr)
    print(f"{T} 帧, 逐帧误差 均值={fe.mean():.1f} 最大={fe.max():.1f}mm", flush=True)
    rr.init("refine_error_hotspots")
    if args.save: rr.save(args.save)
    else: rr.spawn()
    for t in range(T):
        rr.set_time("frame", sequence=t)
        gp = g[t]/1000.0 + np.array([-0.15, 0, 0])
        gcol = [(0.8,0.8,0.8) if v else (0.4,0.4,0.4) for v in m[t]]
        rr.log("GT/points", rr.Points3D(gp, colors=gcol, radii=0.004, labels=JOINT_NAMES))
        rr.log("GT/skeleton", rr.LineStrips3D(strips=[gp[[i,j]] for i,j in EDGES], colors=[(0.6,0.6,0.6)]*len(EDGES), radii=0.002))
        rp = pr[t]/1000.0 + np.array([0.15, 0, 0])
        err = np.linalg.norm(pr[t]-g[t], axis=-1)
        rcol = [jet(e/ERR_MAX) if v else (0.4,0.4,0.4) for e, v in zip(err, m[t])]
        recol = [jet(((err[i]+err[j])/2)/ERR_MAX) if m[t][i] and m[t][j] else (0.4,0.4,0.4) for i,j in EDGES]
        rr.log("refine/points", rr.Points3D(rp, colors=rcol, radii=0.004, labels=JOINT_NAMES))
        rr.log("refine/skeleton", rr.LineStrips3D(strips=[rp[[i,j]] for i,j in EDGES], colors=recol, radii=0.002))
        rr.log("plot/frame_error_mm", rr.Scalars([float(fe[t])]))
        sp = float(np.linalg.norm(g[t+1]-g[t],axis=-1).mean()) if t<T-1 else 0.0
        rr.log("plot/GT_speed_mm", rr.Scalars([sp]))
        if t % 300 == 0: print("frame", t, flush=True)
    print("完成。plot/frame_error_mm 尖峰=误差大的帧; 实体 GT/refine; 色标 蓝0→红≥%dmm。" % ERR_MAX, flush=True)

if __name__ == "__main__":
    main()
