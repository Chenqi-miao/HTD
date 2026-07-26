"""
train.py — pva_paper 实验训练入口
===================================
用法:
  cd experiments/pva_paper
  uv run --no-sync python3 train.py

所有代码和输出都在本文件夹内，完全自包含。
"""

import os, sys, argparse, importlib.util, shutil
import cv2
import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 项目根路径（本文件在 experiments/pva_paper/train.py）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # ← pva_paper/
sys.path.insert(0, os.path.join(PROJECT_ROOT, '..', '..'))  # ← HTD/ (for dataset, utils)


def load_config(exp_name):
    """从 experiments/{exp_name}/config.py 加载配置"""
    config_path = os.path.join(PROJECT_ROOT, 'experiments', exp_name, 'config.py')
    if not os.path.exists(config_path):
        print(f'[ERROR] Config not found: {config_path}')
        sys.exit(1)
    spec = importlib.util.spec_from_file_location('exp_config', config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = mod.Config()

    # 确保输出目录存在
    for d in ['checkpoint', 'log', 'vis']:
        os.makedirs(os.path.join(cfg.exp_dir, d), exist_ok=True)

    return cfg, mod.EXP_NAME


def setup_logger(name, logfile):
    import logging
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile, mode='a')
    fh.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%m/%d %H:%M:%S'))
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(sh)
    return logger


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def draw_vis(inputs, targets, meta_infos, outs, step, cfg, seq_len):
    img = np.zeros([seq_len, 256, 256, 3])
    center = meta_infos['center_xyz'].cuda()
    init_uv = ((inputs['joint_xyz'].cuda() - center)[..., :2].detach().cpu().numpy() / 150 * 128 + 128)[0]
    pd_uv = (outs['pd_joint_xyz'] - center)[..., :2].detach().cpu().numpy()[0]
    gt_uv = ((targets['joint_xyz'].cuda() - center)[..., :2].detach().cpu().numpy() / 150 * 128 + 128)[0]
    vis_path = os.path.join(cfg.exp_dir, 'vis', f'step{step}')
    os.makedirs(vis_path, exist_ok=True)
    from utils.visualize import draw_2d_skeleton
    for t in range(seq_len):
        for label, uv in [('init', init_uv), ('pd', pd_uv), ('gt', gt_uv)]:
            cv2.imwrite(f'{vis_path}/{label}_t{t:02d}.png',
                        draw_2d_skeleton(img[t].copy(), uv[0, t]))


def evaluate(model, test_loader):
    """跑验证集，返回 (init_mpjpe, refine_mpjpe, mpjve, mpjae)"""
    model.eval()
    init_errs, refine_errs = [], []
    vel_errs, acc_errs = [], []
    with torch.no_grad():
        for inputs, targets, meta_infos in tqdm(test_loader, desc='Eval', leave=False):
            outs = model(inputs, targets, meta_infos)
            init_e, refine_e = test_loader.dataset.evaluate(outs, inputs, targets, meta_infos)
            init_errs.append(init_e)
            refine_errs.append(refine_e)

            # ── 速度/加速度误差（从 pred/gt 关节差分计算） ──
            device = outs['pd_joint_xyz'].device
            gt = torch.as_tensor(targets['joint_xyz']).float().to(device)
            pd = outs['pd_joint_xyz']
            mask = (meta_infos['joint_gt_val'] * meta_infos['joint_in_val']).to(device)

            gt_vel = gt[:, :, 1:] - gt[:, :, :-1]
            pd_vel = pd[:, :, 1:] - pd[:, :, :-1]
            gt_acc = gt_vel[:, :, 1:] - gt_vel[:, :, :-1]
            pd_acc = pd_vel[:, :, 1:] - pd_vel[:, :, :-1]

            v_mask = mask[:, :, :-1, :] * mask[:, :, 1:, :]
            a_mask = mask[:, :, :-2, :] * mask[:, :, 1:-1, :] * mask[:, :, 2:, :]

            ve = torch.sqrt(torch.sum((pd_vel - gt_vel) ** 2, dim=-1) + 1e-8)
            ae = torch.sqrt(torch.sum((pd_acc - gt_acc) ** 2, dim=-1) + 1e-8)
            vel_errs.append((ve * v_mask).sum().item() / (v_mask.sum().item() + 1e-8))
            acc_errs.append((ae * a_mask).sum().item() / (a_mask.sum().item() + 1e-8))

    model.train()
    return (float(np.mean(init_errs)), float(np.mean(refine_errs)),
            float(np.mean(vel_errs)), float(np.mean(acc_errs)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', default='', help='checkpoint path to resume from')
    args = parser.parse_args()

    # 实验目录 = 本文件所在目录
    exp_dir = PROJECT_ROOT
    exp_name = os.path.basename(exp_dir)  # 'pva_paper'

    # 加载本地 config.py
    config_path = os.path.join(exp_dir, 'config.py')
    spec = importlib.util.spec_from_file_location('exp_config', config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = mod.Config()
    cfg.exp_dir = exp_dir  # 覆盖为实际路径

    for d in ['checkpoint', 'log', 'vis']:
        os.makedirs(os.path.join(exp_dir, d), exist_ok=True)

    set_seed(42)

    # ── 日志 ──
    logfile = os.path.join(exp_dir, 'log', f'train_{exp_name}.log')
    logger = setup_logger(exp_name, logfile)
    writer = SummaryWriter(os.path.join(exp_dir, 'log'))

    # ── 模型（加载实验目录下的 model/，自包含） ──
    sys.path.insert(0, PROJECT_ROOT)  # 确保能找到 model/
    from model.PVANetm import PVANetModel
    model = PVANetModel(
        num_frame=cfg.seq_len,
        num_joints=cfg.joint_num,
        dim_feat=cfg.dim_feat,
        depth=cfg.transformer_depth,
        w_vel=cfg.w_vel,
        w_accel=cfg.w_accel,
    ).cuda()

    # ── 优化器 ──
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)
    if cfg.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.total_epoch, eta_min=0)
    elif cfg.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [30, 40], gamma=0.1)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume)
        model.load_state_dict(ckpt['net'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['schedule'])
        start_epoch = ckpt['last_epoch'] + 1
        logger.info(f'Resumed from {args.resume} (epoch {ckpt["last_epoch"]})')

    # ── 数据 ──
    from dataset.seqhand import SeqHand, SeqHandTest
    g = torch.Generator().manual_seed(42)

    logger.info(f'Loading training data...')
    train_dataset = SeqHand(cfg.data_dir, cfg.data_list,
                            min_seq_len=cfg.min_seq_len, seq_len=cfg.seq_len,
                            view_num=cfg.view_num, data_num=cfg.data_num)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
                              num_workers=cfg.num_worker, worker_init_fn=seed_worker,
                              generator=g, shuffle=True, drop_last=True)

    logger.info(f'Loading test data...')
    test_dataset = SeqHandTest(cfg.data_dir, cfg.test_list,
                               min_seq_len=cfg.min_seq_len, seq_len=cfg.seq_len,
                               view_num=cfg.view_num, data_num=cfg.test_data_num)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size,
                             shuffle=False, num_workers=cfg.num_worker)

    # ── 打印配置 ──
    total_params = sum(p.numel() for p in model.parameters())
    logger.info('=' * 55)
    logger.info(f'Experiment: {exp_name}')
    logger.info(f'Train: {len(train_dataset)} samples | Test: {len(test_dataset)} samples')
    logger.info(f'Model params: {total_params:,}')
    logger.info(f'Loss weights: w_vel={cfg.w_vel}, w_accel={cfg.w_accel}')
    logger.info(f'Batch={cfg.batch_size}, LR={cfg.lr}, Epochs={cfg.total_epoch}')
    logger.info('=' * 55)

    # ── 备份外部依赖（dataset, utils — model/ 已本地存在） ──
    htd_root = os.path.join(exp_dir, '..', '..')
    code_dir = os.path.join(exp_dir, 'files')
    shutil.rmtree(code_dir, ignore_errors=True)
    shutil.copytree(os.path.join(htd_root, 'dataset'), f'{code_dir}/dataset/', dirs_exist_ok=True)
    shutil.copytree(os.path.join(htd_root, 'utils'), f'{code_dir}/utils/', dirs_exist_ok=True)
    shutil.copy2(__file__, f'{code_dir}/train.py')

    # ── 训练循环 ──
    min_error = 100.0
    global_step = 0
    epochs_since_improvement = 0
    early_stop_patience = 15

    for epoch in range(start_epoch, cfg.total_epoch):
        model.train()
        epoch_losses = []

        pbar = tqdm(train_loader, desc=f'Epoch {epoch:2d}/{cfg.total_epoch}')
        for iteration, (inputs, targets, meta_infos) in enumerate(pbar):
            optimizer.zero_grad()
            outs, loss, error = model(inputs, targets, meta_infos)
            loss['total_loss'].backward()
            optimizer.step()

            # Log
            for k, v in loss.items():
                writer.add_scalar(f'train/{k}', v.detach(), global_step)
            for k, v in error.items():
                writer.add_scalar(f'train/{k}', v.detach(), global_step)
            global_step += 1

            epoch_losses.append(loss['total_loss'].item())

            if iteration % cfg.print_iter == 0:
                pbar.set_postfix({
                    'loss': f"{loss['total_loss'].item():.4f}",
                    'pos': f"{loss['pos_loss'].item():.4f}",
                    'vel': f"{loss['vel_loss'].item():.4f}",
                    'acc': f"{loss['accel_loss'].item():.4f}",
                    'init_err': f"{error['init'].item():.2f}",
                    'refine_err': f"{error['refine'].item():.2f}",
                })

            if cfg.vis_interval > 0 and iteration % cfg.vis_interval == 0:
                draw_vis(inputs, targets, meta_infos, outs, global_step, cfg, cfg.seq_len)

        scheduler.step()
        avg_loss = np.mean(epoch_losses)
        logger.info(f'Epoch {epoch:2d} | avg_loss={avg_loss:.4f} | lr={scheduler.get_last_lr()[0]:.2e}')

        # ── 保存 latest ──
        torch.save({
            'net': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'schedule': scheduler.state_dict(),
            'last_epoch': epoch,
        }, os.path.join(exp_dir, 'checkpoint', 'latest.pth'))

        # 重采样
        if cfg.loader_resample and hasattr(train_dataset, 'generator_seq'):
            train_dataset.generator_seq()

        # ── 验证 ──
        if epoch % cfg.eval_interval == 0:
            init_mpjpe, refine_mpjpe, mpjve, mpjae = evaluate(model, test_loader)
            logger.info(f'         Init MPJPE={init_mpjpe:.4f} | Refine MPJPE={refine_mpjpe:.4f} | '
                        f'MPJVE={mpjve:.4f} | MPJAE={mpjae:.4f}')
            writer.add_scalar('eval/init_mpjpe', init_mpjpe, epoch)
            writer.add_scalar('eval/refine_mpjpe', refine_mpjpe, epoch)
            writer.add_scalar('eval/mpjve', mpjve, epoch)
            writer.add_scalar('eval/mpjae', mpjae, epoch)

            if refine_mpjpe < min_error:
                min_error = refine_mpjpe
                torch.save({
                    'net': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'schedule': scheduler.state_dict(),
                    'last_epoch': epoch,
                }, os.path.join(exp_dir, 'checkpoint', 'best.pth'))
                logger.info(f'  → Best model saved (Refine MPJPE={refine_mpjpe:.4f} mm)')
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            if epochs_since_improvement >= early_stop_patience:
                logger.info(f'Early stopping after {epoch+1} epochs (no improvement for {early_stop_patience} evals)')
                break

    logger.info(f'Training complete! Best Refine MPJPE: {min_error:.4f} mm')

    # ── 训练完成后生成报告骨架 ──
    report_path = os.path.join(exp_dir, 'report.md')
    with open(report_path, 'w') as f:
        f.write(f"""# {exp_name} 实验报告

## 配置

| 参数 | 值 |
|------|-----|
| w_vel | {cfg.w_vel} |
| w_accel | {cfg.w_accel} |
| seq_len | {cfg.seq_len} |
| batch_size | {cfg.batch_size} |
| lr | {cfg.lr} |
| total_epoch | {cfg.total_epoch} |

## 结果

| 指标 | 值 |
|------|-----|
| Best Refine MPJPE | {min_error:.4f} mm |
| 训练样本数 | {len(train_dataset)} |
| 测试样本数 | {len(test_dataset)} |

## Loss 曲线

TensorBoard: `{os.path.join(exp_dir, 'log')}`

## 可视化

骨架图: `{os.path.join(exp_dir, 'vis/')}`
""")
    logger.info(f'Report skeleton → {report_path}')
    logger.info('Done!')


if __name__ == '__main__':
    main()
