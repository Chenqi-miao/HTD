"""exp2_spatialmix — 训练入口"""
import os, sys, argparse, importlib.util, shutil, cv2, torch, numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "model"))

def setup_logger(name, logfile):
    import logging
    logger = logging.getLogger(name); logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile, mode='a')
    fh.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%m/%d %H:%M:%S'))
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(sh)
    return logger

def set_seed(s=42):
    import random; random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def seed_worker(w):
    np.random.seed(torch.initial_seed() % 2**32)

def evaluate(model, test_loader):
    model.eval(); ie, re = [], []
    with torch.no_grad():
        for inp, tgt, meta in tqdm(test_loader, desc='Eval', leave=False):
            o = model(inp, tgt, meta)
            i, r = test_loader.dataset.evaluate(o, inp, tgt, meta)
            ie.append(i); re.append(r)
    model.train()
    return float(np.mean(ie)), float(np.mean(re))

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--resume', default='')
    parser.add_argument('--exp', default='exp2_spatialmix'); args = parser.parse_args()
    exp_dir = PROJECT_ROOT; exp_name = os.path.basename(exp_dir)
    config_path = os.path.join(exp_dir, 'config.py')
    spec = importlib.util.spec_from_file_location('exp_config', config_path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    cfg = mod.Config(); cfg.exp_dir = exp_dir
    for d2 in ['checkpoint', 'log', 'vis']: os.makedirs(os.path.join(exp_dir, d2), exist_ok=True)

    set_seed(42)
    logfile = os.path.join(exp_dir, 'log', f'train_{exp_name}.log')
    logger = setup_logger(exp_name, logfile)
    writer = SummaryWriter(os.path.join(exp_dir, 'log'))

    from model.net import Net
    model = Net(num_frame=cfg.seq_len, num_joints=cfg.joint_num,
                dim_feat=cfg.dim_feat, depth=cfg.transformer_depth,
                w_vel=cfg.w_vel, w_accel=cfg.w_accel).cuda()

    opt = optim.AdamW(model.parameters(), lr=cfg.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.total_epoch, eta_min=0)
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume); model.load_state_dict(ckpt['net'])
        opt.load_state_dict(ckpt['optimizer']); sched.load_state_dict(ckpt['schedule'])
        start_epoch = ckpt['last_epoch'] + 1; logger.info(f'Resumed epoch {ckpt["last_epoch"]}')

    from dataset.seqhand import SeqHand, SeqHandTest
    g = torch.Generator().manual_seed(42)
    train_ds = SeqHand(cfg.data_dir, cfg.data_list, min_seq_len=cfg.min_seq_len,
                        seq_len=cfg.seq_len, view_num=cfg.view_num, data_num=cfg.data_num)
    train_ld = DataLoader(train_ds, cfg.batch_size, num_workers=cfg.num_worker,
                          worker_init_fn=seed_worker, generator=g, shuffle=True, drop_last=True)
    test_ds = SeqHandTest(cfg.data_dir, cfg.test_list, min_seq_len=cfg.min_seq_len,
                          seq_len=cfg.seq_len, view_num=cfg.view_num, data_num=cfg.test_data_num)
    test_ld = DataLoader(test_ds, cfg.batch_size, shuffle=False, num_workers=cfg.num_worker)

    total_p = sum(p.numel() for p in model.parameters())
    logger.info(f'{exp_name} | train={len(train_ds)} test={len(test_ds)} params={total_p:,}')
    logger.info(f'w_vel={cfg.w_vel} w_accel={cfg.w_accel} batch={cfg.batch_size} lr={cfg.lr}')

    min_err = 100.0; global_step = 0; stagnant = 0

    for epoch in range(start_epoch, cfg.total_epoch):
        model.train(); losses = []
        pbar = tqdm(train_ld, desc=f'Epoch {epoch:2d}/{cfg.total_epoch}')
        for iteration, (inp, tgt, meta) in enumerate(pbar):
            opt.zero_grad(); o, l, e = model(inp, tgt, meta)
            l['total'].backward(); opt.step()
            for k, v in l.items(): writer.add_scalar(f'train/{k}', v.detach(), global_step)
            for k, v in e.items(): writer.add_scalar(f'train/{k}', v.detach(), global_step)
            global_step += 1; losses.append(l['total'].item())
            if iteration % cfg.print_iter == 0:
                pbar.set_postfix({'loss':f"{l['total'].item():.4f}", 'pos':f"{l['pos'].item():.4f}",
                                  'vel':f"{l['vel'].item():.4f}", 'acc':f"{l['acc'].item():.4f}"})

        sched.step()
        avg_l = np.mean(losses)
        logger.info(f'Epoch {epoch:2d} | loss={avg_l:.4f} | lr={sched.get_last_lr()[0]:.2e}')

        torch.save({'net': model.state_dict(), 'optimizer': opt.state_dict(),
                     'schedule': sched.state_dict(), 'last_epoch': epoch},
                    os.path.join(exp_dir, 'checkpoint', 'latest.pth'))

        if cfg.loader_resample and hasattr(train_ds, 'generator_seq'): train_ds.generator_seq()

        if epoch % cfg.eval_interval == 0:
            imp, rmp = evaluate(model, test_ld)
            logger.info(f'  Init MPJPE={imp:.4f} | Refine MPJPE={rmp:.4f}')
            writer.add_scalar('eval/init_mpjpe', imp, epoch)
            writer.add_scalar('eval/refine_mpjpe', rmp, epoch)
            if rmp < min_err:
                min_err = rmp; stagnant = 0
                torch.save({'net': model.state_dict(), 'optimizer': opt.state_dict(),
                             'schedule': sched.state_dict(), 'last_epoch': epoch},
                            os.path.join(exp_dir, 'checkpoint', 'best.pth'))
                logger.info(f'  → Best: {rmp:.4f}')
            else: stagnant += 1
            if stagnant >= 15: logger.info(f'Early stop at {epoch}'); break

    logger.info(f'Done! Best={min_err:.4f}')

if __name__ == '__main__':
    main()
