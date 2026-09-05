"""两阶段: s1(冻结, 纯空间单帧) → 原始 exp2(31帧, JointAttention+时序Transformer) 精修位置.

Stage 1: s1 逐帧去噪 → p_s1 (mm)
Stage 2: 原始 exp2 (exp2_s30_att net) 吃 p_s1 31帧窗 → 精修 (从原版 checkpoint 初始化, 微调)
exp2 自带 位置+速度+加速度 三头损失 (运动损失)。

评估: SeqHandTest 31帧 相机空间 masked MPJPE。
用法: python train_s1exp2.py [--resume checkpoint/latest.pth]
"""
import os, sys, argparse, importlib.util, torch, numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
EXP2_ROOT = "/home/pfren/pycharm/HTD/experiments/exp2_s30_att"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "model"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "..", ".."))

from model.exp2_net import Net as Exp2Net
from model.net_sf import Net as S1Net
from dataset.seqhand import SeqHand, SeqHandTest
from common.utils import setup_logger, seed_worker

CENTER_JOINT = 9


def run_s1(s1, ji, jg, B, V, F, J):
    """s1 冻结, 逐帧生成 p_s1. ji/jg [B,V,F,J,3] mm."""
    ji_flat = ji.reshape(B * F, 1, 1, J, 3)
    jg_flat = jg.reshape(B * F, 1, 1, J, 3)
    center = jg[:, :, :, CENTER_JOINT:CENTER_JOINT + 1, :].reshape(B * F, 1, 1, 1, 3)
    m = {"center_xyz": center,
         "joint_in_val": torch.ones(B * F, 1, 1, J, device=ji.device),
         "continuous_val": torch.ones(B * F, 1, 1, device=ji.device),
         "joint_gt_val": torch.ones(B * F, 1, 1, J, device=ji.device)}
    o, l, e = s1({"joint_xyz": ji_flat}, {"joint_xyz": jg_flat}, m)
    return o["pd_joint_xyz"].reshape(B, 1, F, J, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", default="")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--test_num", type=int, default=2000)
    args = ap.parse_args()

    exp_dir = PROJECT_ROOT
    exp_name = os.path.basename(exp_dir)
    logger = setup_logger(exp_name, os.path.join(exp_dir, "log", f"train_{exp_name}.log"))
    writer = SummaryWriter(os.path.join(exp_dir, "log"))

    spec = importlib.util.spec_from_file_location("cfg", os.path.join(exp_dir, "config.py"))
    C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
    C = C.Config
    F = C.seq_len   # 31

    # s1 冻结
    s1 = S1Net(num_frame=1, num_joints=21, dim_feat=256, depth=8).cuda().eval()
    ck1 = torch.load(os.path.join(PROJECT_ROOT, "..", "exp2_s1_att_grouped", "checkpoint", "best.pth"),
                     map_location="cuda")
    s1.load_state_dict(ck1["net"])
    for p in s1.parameters():
        p.requires_grad = False
    logger.info("s1 冻结加载 (exp2_s1_att_grouped best.pth)")

    # stage2 = 原始 exp2, 从原版 checkpoint 初始化
    model = Exp2Net(num_frame=F, num_joints=C.joint_num, dim_feat=C.dim_feat,
                    depth=C.transformer_depth, w_vel=C.w_vel, w_accel=C.w_accel).cuda()
    ck2 = torch.load(os.path.join(EXP2_ROOT, "checkpoint", "best.pth"), map_location="cuda")
    model.load_state_dict(ck2["net"], strict=False)
    logger.info(f"exp2 从原版 checkpoint 初始化 (best.pth, seq_len={F})")

    opt = optim.AdamW(model.parameters(), lr=args.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=0)
    start_epoch = 0
    if args.resume:
        ck = torch.load(args.resume)
        model.load_state_dict(ck["net"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["schedule"])
        start_epoch = ck["last_epoch"] + 1

    g = torch.Generator().manual_seed(42)
    train_ds = SeqHand(C.data_dir, C.data_list, min_seq_len=C.min_seq_len,
                       seq_len=F, view_num=C.view_num, data_num=C.data_num)
    train_ld = DataLoader(train_ds, args.batch, num_workers=4, worker_init_fn=seed_worker,
                          generator=g, shuffle=True, drop_last=True)
    test_ds = SeqHandTest(C.data_dir, C.test_list, min_seq_len=C.min_seq_len,
                          seq_len=F, view_num=C.view_num, data_num=args.test_num)
    test_ld = DataLoader(test_ds, args.batch, shuffle=False, num_workers=4)

    def evaluate():
        model.eval()
        se = ne = 0.0
        with torch.no_grad():
            for inp, tgt, meta in tqdm(test_ld, desc="Eval", leave=False):
                B, V, F_, J = inp["joint_xyz"].shape[:4]
                ji = inp["joint_xyz"].cuda(); jg = tgt["joint_xyz"].cuda()
                meta_c = {k: v.cuda() for k, v in meta.items()}
                p_s1 = run_s1(s1, ji, jg, B, V, F_, J)
                out = model({"joint_xyz": p_s1}, {"joint_xyz": jg}, meta_c)   # eval 模式返回 dict
                pj = out["pd_joint_xyz"]
                cv = meta_c["continuous_val"].reshape(B, 1, F_, 1, 1)
                gv = meta_c["joint_gt_val"].reshape(B, 1, F_, J, 1)
                m = cv * gv
                se += (torch.norm(pj - jg, dim=-1).unsqueeze(-1) * m).sum().item()
                ne += m.sum().item()
        model.train()
        return se / ne

    min_err = 1e9
    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses = []
        pbar = tqdm(train_ld, desc=f"Epoch {epoch:2d}/{args.epochs}")
        for it, (inp, tgt, meta) in enumerate(pbar):
            B, V, F_, J = inp["joint_xyz"].shape[:4]
            ji = inp["joint_xyz"].cuda(); jg = tgt["joint_xyz"].cuda()
            meta_c = {k: v.cuda() for k, v in meta.items()}
            with torch.no_grad():
                p_s1 = run_s1(s1, ji, jg, B, V, F_, J)
            opt.zero_grad()
            o, l, e = model({"joint_xyz": p_s1}, {"joint_xyz": jg}, meta_c)
            l["total"].backward()
            opt.step()
            losses.append(l["total"].item())
            if it % 100 == 0:
                pbar.set_postfix({"loss": f"{l['total'].item():.4f}"})
        sched.step()
        logger.info(f"Epoch {epoch:2d} | loss={np.mean(losses):.4f} | lr={sched.get_last_lr()[0]:.2e}")
        torch.save({"net": model.state_dict(), "optimizer": opt.state_dict(),
                    "schedule": sched.state_dict(), "last_epoch": epoch},
                   os.path.join(exp_dir, "checkpoint", "latest.pth"))
        if epoch % 5 == 0:
            m = evaluate()
            logger.info(f"  Test MPJPE={m:.4f}")
            if m < min_err:
                min_err = m
                torch.save({"net": model.state_dict(), "optimizer": opt.state_dict(),
                            "schedule": sched.state_dict(), "last_epoch": epoch},
                           os.path.join(exp_dir, "checkpoint", "best.pth"))
                logger.info(f"  -> Best: {min_err:.4f}")

    logger.info(f"Done! Best={min_err:.4f}")


if __name__ == "__main__":
    main()
