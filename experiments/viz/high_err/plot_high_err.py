"""5 段高误差序列: 逐帧误差图 (refine 红 / 输入 灰)."""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent          # experiments/viz/high_err
DATA = HERE / "high_err_seqs.npz"

def main():
    d = np.load(DATA)
    nseq = len([k for k in d.files if k.startswith("range")])
    fig, axes = plt.subplots(nseq, 1, figsize=(14, 3*nseq))
    for k, ax in enumerate(axes):
        pr, g, mk, inn = d[f"pr_{k}"], d[f"gt_{k}"], d[f"msk_{k}"] > 0, d[f"in_{k}"]
        mkf = mk.astype(float)
        fe = np.sqrt(((pr-g)**2).sum(-1)*mkf).sum(1)/(mkf.sum(1)+1e-8)
        ie = np.sqrt(((inn-g)**2).sum(-1)*mkf).sum(1)/(mkf.sum(1)+1e-8)
        w0, w1 = d[f"range_{k}"]
        x = np.arange(len(fe))
        ax.plot(x, ie, color="#888888", lw=0.8, alpha=0.7, label=f"input (均{ie.mean():.1f}mm)")
        ax.plot(x, fe, color="#d62728", lw=1.0, label=f"refine (均{fe.mean():.1f}mm)")
        ax.axhline(fe.mean(), color="#d62728", ls="--", lw=0.8)
        ax.fill_between(x, fe, color="#d62728", alpha=0.15)
        ax.set_title(f"seq{k}  窗{w0}-{w1}  ({len(fe)}帧)", fontsize=11)
        ax.set_ylabel("误差(mm)")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("帧号")
    fig.tight_layout()
    out = HERE / "high_err_loss.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}", flush=True)

if __name__ == "__main__":
    main()
