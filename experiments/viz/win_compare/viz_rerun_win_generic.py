"""Rerun: N 个对比窗, 每窗 GT(灰)/sliding(绿)/refine(误差着色). --data 指定 npz."""
import argparse, sys
from pathlib import Path
import numpy as np
import rerun as rr

HERE = Path(__file__).resolve().parent          # experiments/viz/win_compare
sys.path.insert(0, str(HERE.parent.parent))    # experiments → 使 common 可导入
from common.hand_topology import JOINT_NAMES, EDGES
DATA = HERE / "refine_win_windows.npz"
ERR_MAX = 25.0
def jet(v):
    v = max(0.0, min(1.0, v)) * 4.0
    if v < 1: return (0.0, v, 1.0)
    if v < 2: return (0.0, 1.0, 2.0 - v)
    if v < 3: return (v - 2.0, 1.0, 0.0)
    return (1.0, 4.0 - v, 0.0)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--save", default=""); ap.add_argument("--data", default=""); ap.add_argument("--title", default="windows")
    args = ap.parse_args()
    src = HERE / args.data if args.data else DATA
    d = np.load(src); n = int(d["n"])
    rr.init(args.title)
    if args.save: rr.save(args.save)
    else: rr.spawn()
    for k in range(n):
        sl, rf, g, mk = d[f"slide_{k}"], d[f"ref_{k}"], d[f"gt_{k}"], d[f"msk_{k}"] > 0
        w = int(d[f"w_{k}"][0])
        es = (np.sqrt(((sl-g)**2).sum(-1)*mk).sum()/(mk.sum()+1e-8))
        er = (np.sqrt(((rf-g)**2).sum(-1)*mk).sum()/(mk.sum()+1e-8))
        print(f"窗{w}: 滑动={es:.1f} refine={er:.1f}", flush=True)
        for t in range(31):
            rr.set_time("frame", sequence=k*40 + t)
            gp = g[t]/1000.0 + np.array([-0.15,0,0])
            sp = sl[t]/1000.0 + np.array([0.10,0,0])
            rp = rf[t]/1000.0 + np.array([0.30,0,0])
            rr.log(f"w{w}/GT/points", rr.Points3D(gp, colors=[(0.8,0.8,0.8) if v else (0.4,0.4,0.4) for v in mk[t]], radii=0.004, labels=JOINT_NAMES))
            rr.log(f"w{w}/GT/skeleton", rr.LineStrips3D(strips=[gp[[i,j]] for i,j in EDGES], colors=[(0.6,0.6,0.6)]*len(EDGES), radii=0.002))
            rr.log(f"w{w}/sliding/points", rr.Points3D(sp, colors=[(0.17,0.63,0.17) if v else (0.4,0.4,0.4) for v in mk[t]], radii=0.003, labels=JOINT_NAMES))
            rr.log(f"w{w}/sliding/skeleton", rr.LineStrips3D(strips=[sp[[i,j]] for i,j in EDGES], colors=[(0.3,0.7,0.3)]*len(EDGES), radii=0.0015))
            err = np.linalg.norm(rf[t]-g[t], axis=-1)
            rcol = [jet(e/ERR_MAX) if v else (0.4,0.4,0.4) for e,v in zip(err, mk[t])]
            rr.log(f"w{w}/refine/points", rr.Points3D(rp, colors=rcol, radii=0.004, labels=JOINT_NAMES))
            rr.log(f"w{w}/refine/skeleton", rr.LineStrips3D(strips=[rp[[i,j]] for i,j in EDGES], colors=[jet(((err[i]+err[j])/2)/ERR_MAX) for i,j in EDGES], radii=0.002))
    print("完成。每窗 GT(灰)/sliding(绿)/refine(误差着色)。", flush=True)
if __name__ == "__main__":
    main()
