"""Rerun: 5 段高误差序列, 每段 input(橙) + GT(灰) + refine(误差着色) + 逐帧误差曲线."""
import argparse, sys
from pathlib import Path
import numpy as np
import rerun as rr

HERE = Path(__file__).resolve().parent          # experiments/viz/high_err
sys.path.insert(0, str(HERE.parent.parent))    # experiments → 使 common 可导入
from common.hand_topology import JOINT_NAMES, EDGES
DATA = HERE / "high_err_seqs.npz"
ERR_MAX = 25.0
INPUT_COL = (1.0, 0.5, 0.1)

def jet(v):
    v = max(0.0, min(1.0, v)) * 4.0
    if v < 1: return (0.0, v, 1.0)
    if v < 2: return (0.0, 1.0, 2.0 - v)
    if v < 3: return (v - 2.0, 1.0, 0.0)
    return (1.0, 4.0 - v, 0.0)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--save", default=""); args = ap.parse_args()
    d = np.load(DATA)
    nseq = len([k for k in d.files if k.startswith("range")])
    rr.init("high_err_sequences")
    if args.save: rr.save(args.save)
    else: rr.spawn()
    toff = 0
    for k in range(nseq):
        pr, g, inn, mk = d[f"pr_{k}"], d[f"gt_{k}"], d[f"in_{k}"], d[f"msk_{k}"] > 0
        T = len(pr)
        fe = np.sqrt(((pr-g)**2).sum(-1)*mk).sum(1)/(mk.sum(1)+1e-8)
        w0, w1 = d[f"range_{k}"]
        print(f"seq{k} 窗{w0}-{w1} {T}帧 平均误差={fe.mean():.1f}mm", flush=True)
        for t in range(T):
            rr.set_time("frame", sequence=toff + t)
            # input 橙色 (中性)
            ip = inn[t]/1000.0 + np.array([-0.30,0,0])
            icol = [INPUT_COL if v else (0.4,0.4,0.4) for v in mk[t]]
            rr.log(f"seq{k}/input/points", rr.Points3D(ip, colors=icol, radii=0.003, labels=JOINT_NAMES))
            rr.log(f"seq{k}/input/skeleton", rr.LineStrips3D(strips=[ip[[i,j]] for i,j in EDGES], colors=[(0.9,0.6,0.3)]*len(EDGES), radii=0.0015))
            # GT 灰
            gp = g[t]/1000.0 + np.array([-0.10,0,0])
            rr.log(f"seq{k}/GT/points", rr.Points3D(gp, colors=[(0.8,0.8,0.8) if v else (0.4,0.4,0.4) for v in mk[t]], radii=0.003, labels=JOINT_NAMES))
            rr.log(f"seq{k}/GT/skeleton", rr.LineStrips3D(strips=[gp[[i,j]] for i,j in EDGES], colors=[(0.6,0.6,0.6)]*len(EDGES), radii=0.0015))
            # refine 误差着色
            rp = pr[t]/1000.0 + np.array([0.15,0,0])
            err = np.linalg.norm(pr[t]-g[t], axis=-1)
            rcol = [jet(e/ERR_MAX) if v else (0.4,0.4,0.4) for e,v in zip(err, mk[t])]
            rr.log(f"seq{k}/refine/points", rr.Points3D(rp, colors=rcol, radii=0.004, labels=JOINT_NAMES))
            rr.log(f"seq{k}/refine/skeleton", rr.LineStrips3D(strips=[rp[[i,j]] for i,j in EDGES], colors=[jet(((err[i]+err[j])/2)/ERR_MAX) for i,j in EDGES], radii=0.002))
            rr.log(f"plot/seq{k}_error_mm", rr.Scalars([float(fe[t])]))
        toff += T
    print("完成。每段 3 实体: input(橙)/GT(灰)/refine(误差着色 蓝0→红≥%dmm)。" % ERR_MAX, flush=True)

if __name__ == "__main__":
    main()
