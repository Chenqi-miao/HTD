"""rerun 画 input/GT/stage2_s31 各关节 3D 运动轨迹, 同一坐标系(不 offset), 随时间生长动画.

数据: viz_out.npz (in_joint/gt_joint/out31/msk, mm).
- 动画(默认): 每帧 log "到当前帧为止的轨迹线(细)" + "当前帧的点", rerun 播放时轨迹随时间逐渐画出
- 静态(--static): 一次性完整轨迹线
用法: .venv_vis/bin/python experiments/viz_traj_stage2.py [--start N] [--len M] [--static]
      .venv_vis/bin/rerun viz_traj_stage2.rrd    (点播放 ▶)
"""
import sys, argparse
from pathlib import Path
import numpy as np
import rerun as rr

HERE = Path(__file__).resolve().parent          # experiments/viz/stage2
sys.path.insert(0, str(HERE.parent.parent))    # experiments → 使 common 可导入
from common.hand_topology import JOINT_NAMES       # 21 关节名

ROOT = HERE
DATA = ROOT / "viz_out.npz"
SRC = {"input": "#ff7f0e", "GT": "#1f77b4", "exp2_s15": "#2ca02c", "stage2_s31": "#d62728"}
TRAJ_R = 0.0003       # 轨迹线半径 (细)
POINT_R = 0.004       # 当前帧点半径


def rgb(h):
    return tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))


def valid_prefix(arr_j, v_j, t):
    """0..t 帧内的有效点序列 [K,3] (跳过 mask 无效帧)."""
    pts = arr_j[:t + 1][v_j[:t + 1]]
    return pts.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--len", type=int, default=0)   # 0 = 全部
    ap.add_argument("--static", action="store_true")
    args = ap.parse_args()

    d = np.load(DATA)
    datas = {"input": d["in_joint"], "GT": d["gt_joint"], "exp2_s15": d["out15"],
             "stage2_s31": d["out31"]}
    msk = (d["msk"] > 0).astype(bool)
    T = len(datas["input"])
    s = args.start
    e = T if args.len == 0 else min(args.start + args.len, T)
    tag = "" if (s == 0 and e == T) else f"_{s}_{e}"
    if args.static:
        tag += "_static"

    out = ROOT / f"viz_traj_stage2{tag}.rrd"
    rr.init(out.stem)
    rr.save(str(out))

    if args.static:
        for j, name in enumerate(JOINT_NAMES):
            for src, arr in datas.items():
                seg = arr[s:e] / 1000.0
                v = msk[s:e, j]
                strips = [sg.astype(np.float32) for sg in segs_of(seg[:, j], v)]
                rr.log(f"{name}/{src}", rr.LineStrips3D(strips=strips,
                        colors=rgb(SRC[src]), radii=TRAJ_R))
    else:
        # 动画: 每帧 log 前缀轨迹 + 当前点
        for t in range(s, e):
            rr.set_time("frame", sequence=t - s)
            for j, name in enumerate(JOINT_NAMES):
                for src, arr in datas.items():
                    pts = valid_prefix(arr[s:e][:, j] / 1000.0, msk[s:e, j], t - s)
                    if len(pts) >= 2:
                        rr.log(f"{name}/{src}/traj",
                               rr.LineStrips3D(strips=[pts], colors=rgb(SRC[src]), radii=TRAJ_R))
                    cur = (arr[t] / 1000.0)[j]
                    rr.log(f"{name}/{src}/point",
                           rr.Points3D([cur], colors=rgb(SRC[src]), radii=POINT_R))
            if (t - s) % 50 == 0:
                print(f"frame {t - s}/{e - s}", flush=True)
    print(f"done: {out}  ({T} 帧, 画 {s}:{e}, {'static' if args.static else 'animate'})")


def segs_of(arr_j, v_j):
    """arr_j [T,3], v_j [T] bool → 有效连续段列表 (供静态模式)."""
    segs, cur = [], []
    for t in range(len(arr_j)):
        if v_j[t]:
            cur.append(arr_j[t])
        elif cur:
            segs.append(np.stack(cur)); cur = []
    if cur:
        segs.append(np.stack(cur))
    return segs or [arr_j]


if __name__ == "__main__":
    main()
