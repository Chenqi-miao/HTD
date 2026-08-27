from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

JOINT_NAMES = ['Wrist','ThumbMCP','ThumbPIP','ThumbDIP','ThumbTIP',
               'IndexMCP','IndexPIP','IndexDIP','IndexTIP',
               'MidMCP','MidPIP','MidDIP','MidTIP',
               'RingMCP','RingPIP','RingDIP','RingTIP',
               'PinkyMCP','PinkyPIP','PinkyDIP','PinkyTIP']

parser = argparse.ArgumentParser()
parser.add_argument('--joint', type=int, default=20)
parser.add_argument('--seg', type=int, default=850)
parser.add_argument('--len', type=int, default=150)
parser.add_argument('--dim', type=int, default=0, help='0=x 1=y 2=z')
parser.add_argument('--out', type=str, default='/mnt/c/Users/ChenQi/ppt_figs/fig.png')
args = parser.parse_args()

d = np.load(Path(__file__).resolve().parent / 'longclip.npz')
inp, gt = d['in_joint'], d['gt_joint']
j, s, W, dim = args.joint, args.seg, args.len, args.dim
a = inp[s:s+W, j, dim].astype(float)
b = gt[s:s+W, j, dim].astype(float)

vel_in, vel_gt = np.gradient(a), np.gradient(b)
acc_in, acc_gt = np.gradient(vel_in), np.gradient(vel_gt)
err = np.abs(a - b).mean()

fig, axes = plt.subplots(3, 1, figsize=(11, 9.5), sharex=True)
y_lab = ['Position (mm)', 'Velocity (mm/frame)', 'Acceleration (mm/frame$^2$)']
pairs = [(a, b), (vel_in, vel_gt), (acc_in, acc_gt)]
for ax, (pi, pg), yl in zip(axes, pairs, y_lab):
    ax.plot(pi, color='#d62728', lw=1.0, alpha=0.95, label='Input (noisy)')
    ax.plot(pg, color='#1f77b4', lw=2.2, label='GT')
    ax.set_ylabel(yl, fontsize=11)
    ax.legend(loc='upper right', fontsize=10, ncol=2)
    ax.grid(alpha=0.3)
axes[-1].set_xlabel('Frame (within segment)', fontsize=11)

dimname = 'xyz'[dim]
fig.suptitle(f'Joint {j} {JOINT_NAMES[j]} | dim {dimname} | segment [{s}, {s+W})  '
             f'(input mean |err| = {err:.1f} mm)', fontsize=13, y=0.995)
fig.tight_layout()
fig.savefig(args.out, dpi=170)
print('saved', args.out, '| joint', j, JOINT_NAMES[j], '| mean_err %.1fmm' % err)
