"""
eval.py — 全量指标评估（MPJPE + Jitter + MPJVE + MPJAE）
=========================================================
可加载任意 checkpoint 进行评估。

用法:
  .venv/bin/python3 eval.py --checkpoint ../pva_base/checkpoint/best.pth
  .venv/bin/python3 eval.py --checkpoint checkpoint/best.pth

依赖: config.py (实验配置)
"""

import os, sys, argparse, json
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))

from config import Config as ExpConfig
from dataset.seqhand import SeqHandTest


# ─── 指标函数 ──────────────────────────────────────

def mpjpe(pred, gt, mask):
    d = torch.sqrt(torch.sum((pred * mask - gt * mask) ** 2, dim=-1))
    return d.sum().item() / (mask.sum().item() + 1e-8)

def jitter(joints, mask):
    acc = joints[:, :, 2:] - 2 * joints[:, :, 1:-1] + joints[:, :, :-2]
    m = mask[:, :, :-2] * mask[:, :, 1:-1] * mask[:, :, 2:]
    mag = torch.sqrt(torch.sum(acc ** 2, dim=-1) + 1e-8)
    return (mag * m).sum().item() / (m.sum().item() + 1e-8)

def vel_error(pred, gt, mask):
    vp = pred[:, :, 1:] - pred[:, :, :-1]
    vg = gt[:, :, 1:] - gt[:, :, :-1]
    m = mask[:, :, :-1] * mask[:, :, 1:]
    d = torch.sqrt(torch.sum((vp - vg) ** 2, dim=-1) + 1e-8)
    return (d * m).sum().item() / (m.sum().item() + 1e-8)

def acc_error(pred, gt, mask):
    ap = pred[:, :, 2:] - 2 * pred[:, :, 1:-1] + pred[:, :, :-2]
    ag = gt[:, :, 2:] - 2 * gt[:, :, 1:-1] + gt[:, :, :-2]
    m = mask[:, :, :-2] * mask[:, :, 1:-1] * mask[:, :, 2:]
    d = torch.sqrt(torch.sum((ap - ag) ** 2, dim=-1) + 1e-8)
    return (d * m).sum().item() / (m.sum().item() + 1e-8)


# ─── 主评估 ────────────────────────────────────────

def load_model(checkpoint_path, cfg):
    """根据 checkpoint 的 key 自动判断架构并加载"""
    device = 'cuda'
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt['net']
    keys = list(state.keys())

    # 判断架构: 含有 blocks_st 的是 PVANet (DSTFormer), 否则是 PVANetm (纯时序)
    has_dstf = any('blocks_st' in str(k) for k in keys)
    arch = 'DSTFormer' if has_dstf else 'TemporalTransformer'
    print(f'  Detected backbone: {arch}')
    if has_dstf:
        print('  Architecture: PVANet (DSTFormer + 3 heads)')
        from model.PVANet import PVANetModel as Model
        model = Model(
            num_frame=cfg.seq_len, num_joints=cfg.joint_num,
            num_view=cfg.view_num, dim_feat=cfg.dim_feat, depth=2,
        ).to(device)
    else:
        print('  Architecture: PVANetm (TemporalTransformer + RoPE)')
        from model.PVANetm import PVANetModel as Model
        model = Model(
            num_frame=cfg.seq_len, num_joints=cfg.joint_num,
            dim_feat=cfg.dim_feat, depth=cfg.transformer_depth,
            w_vel=0, w_accel=0,
        ).to(device)

    model.load_state_dict(state)
    model.eval()
    print(f'  Checkpoint: {checkpoint_path} (epoch {ckpt.get("last_epoch","?")})')
    return model


def evaluate(checkpoint_path, data_num=None):
    cfg = ExpConfig()
    model = load_model(checkpoint_path, cfg)

    # 数据
    dataset = SeqHandTest(
        cfg.data_dir, cfg.test_list,
        min_seq_len=cfg.min_seq_len, seq_len=cfg.seq_len,
        view_num=cfg.view_num, data_num=data_num or cfg.test_data_num,
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    print(f'Test samples: {len(dataset)}')

    # 采集
    metrics = {k: [] for k in [
        'init_mpjpe', 'refine_mpjpe',
        'init_jitter', 'refine_jitter', 'gt_jitter',
        'init_mpjve', 'refine_mpjve',
        'init_mpjae', 'refine_mpjae',
    ]}

    with torch.no_grad():
        for inputs, targets, meta_infos in tqdm(loader, desc='Evaluating'):
            outs = model(inputs, targets, meta_infos)
            device = outs['pd_joint_xyz'].device

            pd = outs['pd_joint_xyz']
            gt = torch.as_tensor(targets['joint_xyz']).float().to(device)
            inp = torch.as_tensor(inputs['joint_xyz']).float().to(device)

            mask = (meta_infos['joint_gt_val'] * meta_infos['joint_in_val']).to(device)
            mask_3d = mask.unsqueeze(-1)

            metrics['init_mpjpe'].append(mpjpe(inp, gt, mask_3d))
            metrics['refine_mpjpe'].append(mpjpe(pd, gt, mask_3d))

            metrics['init_jitter'].append(jitter(inp, mask))
            metrics['refine_jitter'].append(jitter(pd, mask))
            metrics['gt_jitter'].append(jitter(gt, mask))

            metrics['init_mpjve'].append(vel_error(inp, gt, mask))
            metrics['refine_mpjve'].append(vel_error(pd, gt, mask))

            metrics['init_mpjae'].append(acc_error(inp, gt, mask))
            metrics['refine_mpjae'].append(acc_error(pd, gt, mask))

    results = {k: float(np.mean(v)) for k, v in metrics.items()}
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_num', type=int, default=None, help='limit test samples')
    args = parser.parse_args()

    r = evaluate(args.checkpoint, args.data_num)

    print('\n' + '═' * 55)
    print('           EVALUATION RESULTS')
    print('═' * 55)
    print(f'  ┌──────────────┬──────────┬──────────┬──────┐')
    print(f'  │   Metric     │  Init    │ Refine   │  ↓%  │')
    print(f'  ├──────────────┼──────────┼──────────┼──────┤')
    print(f'  │ MPJPE (mm)   │ {r["init_mpjpe"]:>7.3f} │ {r["refine_mpjpe"]:>7.3f} │ '
          f'{(1-r["refine_mpjpe"]/r["init_mpjpe"])*100:>4.1f}% │')
    print(f'  │ Jitter (mm)  │ {r["init_jitter"]:>7.3f} │ {r["refine_jitter"]:>7.3f} │ '
          f'{(1-r["refine_jitter"]/r["init_jitter"])*100:>4.1f}% │')
    print(f'  │ GT Jitter    │ {r["gt_jitter"]:>7.3f} │          │      │')
    print(f'  ├──────────────┼──────────┼──────────┼──────┤')
    print(f'  │ MPJVE (mm)   │ {r["init_mpjve"]:>7.3f} │ {r["refine_mpjve"]:>7.3f} │ '
          f'{(1-r["refine_mpjve"]/r["init_mpjve"])*100:>4.1f}% │')
    print(f'  │ MPJAE (mm)   │ {r["init_mpjae"]:>7.3f} │ {r["refine_mpjae"]:>7.3f} │ '
          f'{(1-r["refine_mpjae"]/r["init_mpjae"])*100:>4.1f}% │')
    print(f'  └──────────────┴──────────┴──────────┴──────┘')

    result_path = os.path.join(os.path.dirname(__file__), 'eval_result.json')
    with open(result_path, 'w') as f:
        json.dump(r, f, indent=2)
    print(f'\n  → {result_path}')
