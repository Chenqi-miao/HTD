"""
run_eval.py — 原版模型评估脚本（流式版）
========================================
单次遍历，流式写 TXT，不攒内存。

用法：
    uv run --no-sync python3 run_eval.py

输出产物：
  [1] Terminal metrics       — MPJPE / Jitter / MPJVE
  [2] eval_results/results.json
  [3] eval_results/init.txt / refine.txt / gt.txt  (流式追加)
  [4] eval_results/vis/batch{0,1,2}/  (骨架 PNG)
  [5] eval_results/metric_comparison.png
"""

import os, sys, json, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ─── 运行时配置 ──────────────────────────────────
import seq_config as sc
sc.cfg.phase = 'test'
sc.cfg.data_list = ['InterHand_test']
sc.cfg.checkpoint = './InterHand_train-train-MotionBERT-Seq15-Window27/checkpoint/best.pth'
sc.cfg.view_num = 1
sc.cfg.batch_size = 256                      # 加大 batch 减少迭代
sc.cfg.num_worker = 0
sc.cfg.test_data_num = None

OUT_DIR = os.path.join(PROJECT_ROOT, 'eval_results')
VIS_DIR = os.path.join(OUT_DIR, 'vis')
os.makedirs(VIS_DIR, exist_ok=True)

# ─── 指标函数 ──────────────────────────────────

def compute_jitter(joint, mask=None):
    vel = joint[:, :, 1:, :, :] - joint[:, :, :-1, :, :]
    accel = vel[:, :, 1:, :, :] - vel[:, :, :-1, :, :]
    accel_mag = torch.sqrt(torch.sum(accel * accel, dim=-1) + 1e-8)
    if mask is not None:
        m = mask[:, :, :-2, :] * mask[:, :, 1:-1, :] * mask[:, :, 2:, :]
        return (accel_mag * m).sum().item() / (m.sum().item() + 1e-8)
    return accel_mag.mean().item()

def compute_vel_error(joint, gt, mask=None):
    vp = joint[:, :, 1:, :, :] - joint[:, :, :-1, :, :]
    vg = gt[:, :, 1:, :, :] - gt[:, :, :-1, :, :]
    err = torch.sqrt(torch.sum((vp - vg) ** 2, dim=-1) + 1e-8)
    if mask is not None:
        m = mask[:, :, :-1, :] * mask[:, :, 1:, :]
        return (err * m).sum().item() / (m.sum().item() + 1e-8)
    return err.mean().item()

# ─── 模型 & 数据 ────────────────────────────────
from model.FusionFormer import FusionModel
from dataset.seqhand import SeqHandTest

t0 = time.time()
print('█▓▒░ Loading model ...')
model = FusionModel(
    model_name=sc.cfg.backbone, num_frame=sc.cfg.seq_len,
    num_joints=sc.cfg.joint_num, num_view=sc.cfg.view_num,
    global_temporal=sc.cfg.global_temporal, window_size=sc.cfg.window_size,
).cuda()
for p in model.parameters(): p.requires_grad = False
ckpt = torch.load(sc.cfg.checkpoint)
model.load_state_dict(ckpt['net'])
model.eval()
print(f'  ✓ epoch={ckpt.get("last_epoch","?")}  ({time.time()-t0:.1f}s)')

t0 = time.time()
print('█▓▒░ Loading InterHand_test ...')
dataset = SeqHandTest(
    sc.cfg.data_dir, sc.cfg.data_list,
    min_seq_len=sc.cfg.min_seq_len, seq_len=sc.cfg.seq_len,
    view_num=sc.cfg.view_num, data_num=sc.cfg.test_data_num,
)
loader = DataLoader(dataset, batch_size=sc.cfg.batch_size, shuffle=False, num_workers=sc.cfg.num_worker)
print(f'  ✓ {len(dataset)} samples / {len(loader)} batches  ({time.time()-t0:.1f}s)\n')

# ─── 单次遍历 ──────────────────────────────────
init_f = open(os.path.join(OUT_DIR, 'init.txt'), 'w')
refine_f = open(os.path.join(OUT_DIR, 'refine.txt'), 'w')
gt_f = open(os.path.join(OUT_DIR, 'gt.txt'), 'w')

meta_acc = {k: [] for k in ['init_mpjpe', 'refine_mpjpe',
                             'init_jitter', 'refine_jitter', 'gt_jitter',
                             'init_mpjve', 'refine_mpjve']}
vis_count = 0

from utils.visualize import draw_2d_skeleton
import cv2

print('█▓▒░ Running inference ...')
t0 = time.time()
for iteration, (inputs, targets, meta_infos) in tqdm(enumerate(loader), total=len(loader)):
    outs = model(inputs, targets, meta_infos)

    # ── MPJPE ──
    init_err, refine_err = dataset.evaluate(outs, inputs, targets, meta_infos)
    meta_acc['init_mpjpe'].append(init_err.item() if hasattr(init_err, 'item') else init_err)
    meta_acc['refine_mpjpe'].append(refine_err.item() if hasattr(refine_err, 'item') else refine_err)

    # ── 抖动指标 ──
    device = outs['pd_joint_xyz'].device
    in_joint = torch.as_tensor(inputs['joint_xyz']).float().to(device)
    gt_joint = torch.as_tensor(targets['joint_xyz']).float().to(device)
    pd_joint = outs['pd_joint_xyz']
    mask_bool = (meta_infos['joint_gt_val'] * meta_infos['joint_in_val']).to(device).bool()
    meta_acc['init_jitter'].append(compute_jitter(in_joint, mask_bool))
    meta_acc['refine_jitter'].append(compute_jitter(pd_joint, mask_bool))
    meta_acc['gt_jitter'].append(compute_jitter(gt_joint, mask_bool))
    meta_acc['init_mpjve'].append(compute_vel_error(in_joint, gt_joint, mask_bool))
    meta_acc['refine_mpjve'].append(compute_vel_error(pd_joint, gt_joint, mask_bool))

    # ── 流式写 TXT（避免内存爆炸） ──
    B = inputs['joint_xyz'].shape[0]
    in_np = torch.as_tensor(inputs['joint_xyz']).numpy().reshape(B, -1)
    gt_np = torch.as_tensor(targets['joint_xyz']).numpy().reshape(B, -1)
    pd_np = outs['pd_joint_xyz'].detach().cpu().numpy().reshape(B, -1)
    for b in range(B):
        init_f.write(' '.join(f'{v:.4f}' for v in in_np[b]) + '\n')
        gt_f.write(' '.join(f'{v:.4f}' for v in gt_np[b]) + '\n')
        refine_f.write(' '.join(f'{v:.4f}' for v in pd_np[b]) + '\n')

    # ── 可视化（前 3 个 batch 的 sample 0） ──
    if vis_count < 3:
        center = meta_infos['center_xyz'].cuda()
        b0 = 0
        seq_dir = os.path.join(VIS_DIR, f'batch{iteration}')
        os.makedirs(seq_dir, exist_ok=True)
        init_uv = ((torch.as_tensor(inputs['joint_xyz']).cuda()[b0:b0+1] - center[b0:b0+1])[..., :2] / 150 * 128 + 128).detach().cpu().numpy()[0]
        pd_uv   = ((outs['pd_joint_xyz'][b0:b0+1] - center[b0:b0+1])[..., :2] / 150 * 128 + 128).detach().cpu().numpy()[0]
        gt_uv   = ((torch.as_tensor(targets['joint_xyz']).cuda()[b0:b0+1] - center[b0:b0+1])[..., :2] / 150 * 128 + 128).detach().cpu().numpy()[0]
        blank = np.zeros([256, 256, 3], dtype=np.uint8) + 255
        for t in range(sc.cfg.seq_len):
            for label, uv in [('init', init_uv), ('pd', pd_uv), ('gt', gt_uv)]:
                img = draw_2d_skeleton(blank.copy(), uv[0, t])
                cv2.imwrite(os.path.join(seq_dir, f'joint_{label}_t{t:02d}.png'), img)
        vis_count += 1

init_f.close(); refine_f.close(); gt_f.close()
elapsed = time.time() - t0
print(f'\n  ✓ Done in {elapsed:.0f}s ({len(dataset)/elapsed:.0f} samples/s)')

# ─── 汇总 ─────────────────────────────────────
init_mpjpe = float(np.mean(meta_acc['init_mpjpe']))
refine_mpjpe = float(np.mean(meta_acc['refine_mpjpe']))
results = {k: float(np.mean(v)) for k, v in meta_acc.items()}
jt = results['init_jitter']; rjt = results['refine_jitter']
ive = results['init_mpjve']; rve = results['refine_mpjve']

# ══════════════════════════════════════════════════
#  [PRODUCED 1] 终端指标输出
# ══════════════════════════════════════════════════
print('\n' + '═' * 55)
print('           EVALUATION RESULTS')
print('═' * 55)
print(f'  Dataset : InterHand_test (seq_len={sc.cfg.seq_len})')
print(f'  Model   : {sc.cfg.backbone} | Samples: {len(dataset)}')
print('─' * 55)
print(f'  ┌──────────────┬──────────┬──────────┬──────┐')
print(f'  │   Metric     │  Init    │ Refine   │  ↓%  │')
print(f'  ├──────────────┼──────────┼──────────┼──────┤')
print(f'  │ MPJPE (mm)   │ {init_mpjpe:>7.3f} │ {refine_mpjpe:>7.3f} │ {(1-refine_mpjpe/init_mpjpe)*100:>4.1f}% │')
print(f'  │ Jitter (mm)  │ {jt:>7.3f} │ {rjt:>7.3f} │ {(1-rjt/jt)*100:>4.1f}% │')
print(f'  │ GT Jitter    │ {results["gt_jitter"]:>7.3f} │          │      │')
print(f'  │ MPJVE (mm)   │ {ive:>7.3f} │ {rve:>7.3f} │ {(1-rve/ive)*100:>4.1f}% │')
print(f'  └──────────────┴──────────┴──────────┴──────┘')

# ══════════════════════════════════════════════════
#  [PRODUCED 2] JSON
# ══════════════════════════════════════════════════
result_path = os.path.join(OUT_DIR, 'results.json')
json.dump({
    'dataset': 'InterHand_test', 'seq_len': sc.cfg.seq_len,
    'backbone': sc.cfg.backbone, 'samples': len(dataset),
    'init_mpjpe_mm': round(init_mpjpe, 4), 'refine_mpjpe_mm': round(refine_mpjpe, 4),
    'init_jitter_mm': round(jt, 4), 'refine_jitter_mm': round(rjt, 4),
    'gt_jitter_mm': round(results['gt_jitter'], 4),
    'init_mpjve_mm': round(ive, 4), 'refine_mpjve_mm': round(rve, 4),
}, result_path, indent=2)
print(f'\n  ✓ {result_path}')

# ══════════════════════════════════════════════════
#  [PRODUCED 3] TXT
# ══════════════════════════════════════════════════
for f in ['init.txt', 'refine.txt', 'gt.txt']:
    fp = os.path.join(OUT_DIR, f)
    sz = os.path.getsize(fp) / 1024 / 1024
    print(f'  ✓ {f:<15} ({sz:.1f} MB)')

# ══════════════════════════════════════════════════
#  [PRODUCED 4] 指标对比图
# ══════════════════════════════════════════════════
print('  ✓ Generating metric_comparison.png ...')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle(f'HTD — {sc.cfg.backbone} (seq_len={sc.cfg.seq_len})  InterHand_test', fontsize=14)
data_bars = {
    'MPJPE (mm)': (init_mpjpe, refine_mpjpe, None),
    'Jitter (mm)': (jt, rjt, results['gt_jitter']),
    'MPJVE (mm)': (ive, rve, None),
}
colors = {'Init': '#e74c3c', 'Refine': '#2ecc71', 'GT': '#3498db'}
for ax, (title, (vi, vr, vg)) in zip(axes, data_bars.items()):
    labels = ['Init', 'Refine'] + (['GT'] if vg is not None else [])
    vals = [vi, vr] + ([vg] if vg is not None else [])
    bars = ax.bar(labels, vals, color=[colors[l] for l in labels], width=0.5, edgecolor='white')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(vals)*0.02, f'{v:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    pct = (1-vr/vi)*100 if vi > 0 else 0
    ax.set_title(f'{title}\n↓ {pct:.1f}%', fontsize=12, fontweight='bold')
    ax.set_ylabel('mm'); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
chart_path = os.path.join(OUT_DIR, 'metric_comparison.png')
plt.savefig(chart_path, dpi=150, bbox_inches='tight'); plt.close()

# ══════════════════════════════════════════════════
#  FINAL
# ══════════════════════════════════════════════════
print('\n' + '█' * 55)
print('  ALL PRODUCED OUTPUTS')
print('█' * 55)
print(f'  [1] Terminal metrics         → 如上表')
print(f'  [2] JSON                    → {result_path}')
print(f'  [3] Coordinate TXTs          → {OUT_DIR}/ (init/refine/gt.txt)')
print(f'  [4] Skeleton vis             → {VIS_DIR}/batch{{0,1,2}}/')
print(f'  [5] Metric chart             → {chart_path}')
print('█' * 55 + '\n')
