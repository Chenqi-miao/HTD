"""
PVA-Net 评估脚本 — 速度、加速度、3D 关键点全指标评估
=====================================================
支持 exp1_perjoint / exp2_spatialmix / exp3_reshape 三个实验。

参考 HTD-Refine 论文评估方法 (Supplementary Sec C):
  - 3D 关键点: MPJPE (Init vs Refine)
  - 速度: MPJVE + PCE@τ (τ = 0.10, 0.05, 0.01)
  - 加速度: MPJAE + PCE@τ (τ = 0.10, 0.05, 0.01)

用法:
  cd /path/to/HTD
  python3 experiments/eval_pva.py --exp exp1_perjoint
  python3 experiments/eval_pva.py --exp exp2_spatialmix
  python3 experiments/eval_pva.py --exp exp3_reshape
  python3 experiments/eval_pva.py --all          # 评估全部三个实验
"""

import os, sys, argparse, json
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader


# ═══════════════════════════════════════════════════════════════
#  实验配置映射
# ═══════════════════════════════════════════════════════════════

EXP_INFO = {
    "exp1_perjoint": {
        "name": "exp1_perjoint",
        "desc": "DSTFormer + 3 per-joint Linear heads",
        "model_cls": "Net",
        "model_module": "net",
        "model_dir": "model",
        "checkpoint": "checkpoints/best.pth",
    },
    "exp2_spatialmix": {
        "name": "exp2_spatialmix",
        "desc": "TemporalTransformer + SpatialMix + 3 per-joint heads",
        "model_cls": "Net",
        "model_module": "net",
        "model_dir": "model",
        "checkpoint": "checkpoint/latest.pth",
    },
    "exp3_reshape": {
        "name": "exp3_reshape",
        "desc": "PVANetm (pure temporal, per-joint reshape)",
        "model_cls": "PVANetModel",
        "model_module": "PVANetm",
        "model_dir": "model",
        "checkpoint": "checkpoint/best.pth",
    },
}


# ═══════════════════════════════════════════════════════════════
#  指标函数（参考论文 Eq.18, 19 + Supplementary Eq.21）
# ═══════════════════════════════════════════════════════════════

def mpjpe(pred, gt, mask):
    """Mean Per-Joint Position Error (mm)"""
    d = torch.sqrt(torch.sum((pred * mask - gt * mask) ** 2, dim=-1) + 1e-8)
    return d.sum().item() / (mask.sum().item() + 1e-8)


def mpjve(pred, gt, mask):
    """
    Mean Per-Joint Velocity Error (mm/frame)
    对时序维度 (dim=3) 差分，参考论文 Eq.(18)
    pred/gt: [B, V, J, F, 3],  mask: [B, V, J, F]
    """
    vp = pred[:, :, :, 1:] - pred[:, :, :, :-1]   # [B, V, J, F-1, 3]
    vg = gt[:, :, :, 1:] - gt[:, :, :, :-1]        # [B, V, J, F-1, 3]
    # mask 在 F 维上对齐
    m = mask[:, :, :, :-1] * mask[:, :, :, 1:]      # [B, V, J, F-1]
    d = torch.sqrt(torch.sum((vp - vg) ** 2, dim=-1) + 1e-8)  # [B, V, J, F-1]
    return (d * m).sum().item() / (m.sum().item() + 1e-8)


def mpjae(pred, gt, mask):
    """
    Mean Per-Joint Acceleration Error (mm/frame²)
    对时序维度 (dim=3) 二阶差分，参考论文 Eq.(19)
    pred/gt: [B, V, J, F, 3],  mask: [B, V, J, F]
    """
    ap = pred[:, :, :, 2:] - 2 * pred[:, :, :, 1:-1] + pred[:, :, :, :-2]
    ag = gt[:, :, :, 2:] - 2 * gt[:, :, :, 1:-1] + gt[:, :, :, :-2]
    # mask 在 F 维上对齐
    m = mask[:, :, :, :-2] * mask[:, :, :, 1:-1] * mask[:, :, :, 2:]
    d = torch.sqrt(torch.sum((ap - ag) ** 2, dim=-1) + 1e-8)
    return (d * m).sum().item() / (m.sum().item() + 1e-8)


def compute_pce(predictions, targets, mask, thresholds=(0.10, 0.05, 0.01)):
    """
    Percentage of Correct Estimates (PCE) — 参考论文 Supplementary Eq.(21)

    PCE@τ = 1/N · Σᵢ 𝟙(|D̂ᵢ − Dᵢ| < τ · (r_max − r_min))

    [r_min, r_max] 基于真值分布的经验 0.1 和 99.9 百分位。

    Args:
        predictions: [B, V, F, J, 3] 预测值（速度或加速度）
        targets:     [B, V, F, J, 3] 真值
        mask:        [B, V, F, J] 有效帧 mask
        thresholds:  阈值系数列表
    Returns:
        dict {f"PCE@{t}": value}
    """
    # 计算动态范围（基于真值数据的 0.1 和 99.9 百分位）
    gt_flat = targets[mask.bool()]  # [N, 3]
    if gt_flat.numel() == 0:
        return {f"pce_{str(t):s}": 0.0 for t in thresholds}

    # 计算每个分量（x, y, z）的误差
    diff = torch.abs(predictions - targets)  # [B, V, F, J, 3]

    # 使用所有真值数据确定动态范围
    r_min = torch.quantile(gt_flat.flatten(), 0.001).item()
    r_max = torch.quantile(gt_flat.flatten(), 0.999).item()
    dynamic_range = r_max - r_min

    if dynamic_range < 1e-8:
        return {f"pce_{str(t):s}": 0.0 for t in thresholds}

    # 扩展 mask 到最后一维
    m = mask.unsqueeze(-1)  # [B, V, F, J, 1]

    results = {}
    for tau in thresholds:
        correct = (diff < tau * dynamic_range).float()  # [B, V, F, J, 3]
        total = (m * correct).sum().item()
        count = m.sum().item()
        results[f"pce_{tau:.2f}"] = (total / (count + 1e-8)) * 100.0

    return results


# ═══════════════════════════════════════════════════════════════
#  GT 速度/加速度计算（从关节位置差分）
# ═══════════════════════════════════════════════════════════════

def compute_gt_vel_accel(joints_gt, mask):
    """从 GT 关节计算速度和加速度，用于 PCE 真值"""
    # 速度: 后向差分 + padding 保持时序对齐
    vel = joints_gt[:, :, 1:] - joints_gt[:, :, :-1]
    vel = torch.cat([vel, vel[:, :, -1:]], dim=2)

    # 加速度: 二阶差分 + padding
    acc = joints_gt[:, :, 2:] - 2 * joints_gt[:, :, 1:-1] + joints_gt[:, :, :-2]
    acc = torch.cat([acc, acc[:, :, -1:], acc[:, :, -1:]], dim=2)

    # mask 对齐
    vel_mask = mask[:, :, :-1] * mask[:, :, 1:]
    vel_mask = torch.cat([vel_mask, vel_mask[:, :, -1:]], dim=2)

    acc_mask = mask[:, :, :-2] * mask[:, :, 1:-1] * mask[:, :, 2:]
    acc_mask = torch.cat([acc_mask, acc_mask[:, :, -1:], acc_mask[:, :, -1:]], dim=2)

    return vel, acc, vel_mask.float(), acc_mask.float()


# ═══════════════════════════════════════════════════════════════
#  模型加载
# ═══════════════════════════════════════════════════════════════

def load_model(exp_name, exp_dir, cfg):
    exp_info = EXP_INFO[exp_name]
    model_path = os.path.join(exp_dir, exp_info["checkpoint"])

    if not os.path.exists(model_path):
        print(f"  ⚠  Checkpoint not found: {model_path}")
        return None

    # 设置导入路径
    sys.path.insert(0, os.path.join(exp_dir, exp_info["model_dir"]))
    sys.path.insert(0, exp_dir)
    sys.path.insert(0, os.path.join(exp_dir, '..', '..'))  # HTD root

    # 动态导入模型
    mod = __import__(exp_info["model_module"], fromlist=[exp_info["model_cls"]])
    ModelCls = getattr(mod, exp_info["model_cls"])

    # 构建模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ModelCls(
        num_frame=cfg.seq_len, num_joints=cfg.joint_num,
        dim_feat=cfg.dim_feat, depth=cfg.transformer_depth,
        w_vel=cfg.w_vel, w_accel=cfg.w_accel,
    ).to(device)

    # 加载权重
    ckpt = torch.load(model_path, map_location=device)
    state = ckpt['net'] if 'net' in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"  ✓ Loaded: {model_path} (epoch={ckpt.get('last_epoch', '?')})")
    return model


# ═══════════════════════════════════════════════════════════════
#  主评估流程
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_experiment(exp_name, data_num=None):
    print(f"\n{'=' * 55}")
    print(f"   Evaluating: {exp_name}")
    print(f"   {EXP_INFO[exp_name]['desc']}")
    print(f"{'=' * 55}")

    exp_dir = os.path.join(os.path.dirname(__file__), exp_name)
    if not os.path.exists(exp_dir):
        print(f"  ✗ Experiment directory not found: {exp_dir}")
        return None

    # 加载配置
    sys.path.insert(0, exp_dir)
    from config import Config as ExpConfig
    cfg = ExpConfig()

    # 加载模型
    model = load_model(exp_name, exp_dir, cfg)
    if model is None:
        return None

    device = next(model.parameters()).device

    # 数据
    from dataset.seqhand import SeqHandTest
    dataset = SeqHandTest(
        cfg.data_dir, cfg.test_list,
        joint_num=cfg.joint_num,
        min_seq_len=cfg.seq_len, seq_len=cfg.seq_len,
        view_num=cfg.view_num, data_num=data_num or cfg.test_data_num,
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    print(f"  Test samples: {len(dataset)}")

    # 收集指标
    metrics = {
        k: [] for k in [
            "init_mpjpe", "refine_mpjpe",
            "init_mpjve", "refine_mpjve",
            "init_mpjae", "refine_mpjae",
            "vel_pce_0.10", "vel_pce_0.05", "vel_pce_0.01",
            "acc_pce_0.10", "acc_pce_0.05", "acc_pce_0.01",
        ]
    }

    for inputs, targets, meta_infos in tqdm(loader, desc="  Evaluating"):
        # 前向（模型 eval 模式返回 pd_joint_xyz + vel_pred + acc_pred）
        outs = model(inputs, targets, meta_infos)

        pd = outs["pd_joint_xyz"].to(device)          # [B, V, F, J, 3]
        gt = torch.as_tensor(targets["joint_xyz"]).float().to(device)  # [B, V, F, J, 3]
        inp = torch.as_tensor(inputs["joint_xyz"]).float().to(device)  # [B, V, F, J, 3]

        vel_pred = outs.get("vel_pred", None)          # [B, V, F, J, 3]
        acc_pred = outs.get("acc_pred", None)          # [B, V, F, J, 3]

        # 构建 mask: [B, V, F, J]
        mask_gt = meta_infos["joint_gt_val"].to(device)   # [B, F, J] (view_num=1)
        mask_in = meta_infos["joint_in_val"].to(device)   # [B, V, F, J]
        cont_val = meta_infos["continuous_val"].to(device) # [B, F]
        # 如果 view_num > 1，扩展 gt_val
        if mask_gt.dim() == 3:
            mask_gt = mask_gt.unsqueeze(1).expand_as(mask_in)
        cont_val = cont_val.view(-1, 1, cfg.seq_len, 1)   # [B, 1, F, 1]
        mask = (mask_gt * mask_in * cont_val).float()     # [B, V, F, J]

        # data_repeat 过滤（测试集边界补齐的重复帧）
        if "data_repeat" in meta_infos:
            dr = meta_infos["data_repeat"].to(device)      # [B, V, F, 1]
            mask = mask * dr                                  # broadcast: [B,V,F,1] × [B,V,F,J]

        mask_3d = mask.unsqueeze(-1)  # [B, V, F, J, 1]

        # ── 位置误差 ──
        metrics["init_mpjpe"].append(mpjpe(inp, gt, mask_3d))
        metrics["refine_mpjpe"].append(mpjpe(pd, gt, mask_3d))

        # ── 速度误差（从关节差分计算） ──
        # 调整为 [B, V, J, F, 3] 格式以便时序差分
        pd_perm = pd.permute(0, 1, 3, 2, 4).contiguous()     # [B, V, J, F, 3]
        gt_perm = gt.permute(0, 1, 3, 2, 4).contiguous()     # [B, V, J, F, 3]
        inp_perm = inp.permute(0, 1, 3, 2, 4).contiguous()   # [B, V, J, F, 3]
        mask_perm = mask.permute(0, 1, 3, 2).contiguous()    # [B, V, J, F]

        metrics["init_mpjve"].append(mpjve(inp_perm, gt_perm, mask_perm))
        metrics["refine_mpjve"].append(mpjve(pd_perm, gt_perm, mask_perm))

        # ── 加速度误差 ──
        metrics["init_mpjae"].append(mpjae(inp_perm, gt_perm, mask_perm))
        metrics["refine_mpjae"].append(mpjae(pd_perm, gt_perm, mask_perm))

        # ── PCE：速度直接预测 ──
        if vel_pred is not None:
            gt_vel, _, vel_mask_gt, _ = compute_gt_vel_accel(gt, mask)
            pce_vel = compute_pce(vel_pred, gt_vel, vel_mask_gt)
            for k, v in pce_vel.items():
                metrics[f"vel_{k}"].append(v)

        # ── PCE：加速度直接预测 ──
        if acc_pred is not None:
            _, gt_acc, _, acc_mask_gt = compute_gt_vel_accel(gt, mask)
            pce_acc = compute_pce(acc_pred, gt_acc, acc_mask_gt)
            for k, v in pce_acc.items():
                metrics[f"acc_{k}"].append(v)

    # 汇总
    results = {}
    for k, v in metrics.items():
        if v:
            results[k] = float(np.mean(v))
        else:
            results[k] = 0.0

    # 附加信息
    results["exp_name"] = exp_name
    results["num_samples"] = len(dataset)
    results["model_params"] = sum(p.numel() for p in model.parameters())

    # 改进率
    if results["init_mpjpe"] > 0:
        results["mpjpe_improvement_pct"] = (
            1 - results["refine_mpjpe"] / results["init_mpjpe"]
        ) * 100.0

    return results


def print_results(results):
    """格式化打印结果"""
    if results is None:
        return

    print("\n" + "═" * 55)
    print(f"   RESULTS — {results['exp_name']}")
    print("═" * 55)

    # ── 位置精度 ──
    print(f"  ┌──────────────┬──────────┬──────────┬───────┐")
    print(f"  │ Metric       │ Init     │ Refine   │  ↓%   │")
    print(f"  ├──────────────┼──────────┼──────────┼───────┤")
    print(f"  │ MPJPE (mm)   │ {results['init_mpjpe']:>7.3f} │ {results['refine_mpjpe']:>7.3f} │ "
          f"{results.get('mpjpe_improvement_pct', 0):>5.1f}% │")
    print(f"  │ MPJVE (mm/f) │ {results['init_mpjve']:>7.3f} │ {results['refine_mpjve']:>7.3f} │ "
          f"{'—':>5s} │")
    print(f"  │ MPJAE (mm/f²)│ {results['init_mpjae']:>7.3f} │ {results['refine_mpjae']:>7.3f} │ "
          f"{'—':>5s} │")
    print(f"  └──────────────┴──────────┴──────────┴───────┘")

    # ── PCE: 速度 ──
    print(f"  ┌────────────────────┬──────────┐")
    print(f"  │ PCE — Velocity     │   %      │")
    print(f"  ├────────────────────┼──────────┤")
    print(f"  │ PCE@0.10           │ {results['vel_pce_0.10']:>7.2f}% │")
    print(f"  │ PCE@0.05           │ {results['vel_pce_0.05']:>7.2f}% │")
    print(f"  │ PCE@0.01           │ {results['vel_pce_0.01']:>7.2f}% │")
    print(f"  └────────────────────┴──────────┘")

    # ── PCE: 加速度 ──
    print(f"  ┌────────────────────┬──────────┐")
    print(f"  │ PCE — Acceleration │   %      │")
    print(f"  ├────────────────────┼──────────┤")
    print(f"  │ PCE@0.10           │ {results['acc_pce_0.10']:>7.2f}% │")
    print(f"  │ PCE@0.05           │ {results['acc_pce_0.05']:>7.2f}% │")
    print(f"  │ PCE@0.01           │ {results['acc_pce_0.01']:>7.2f}% │")
    print(f"  └────────────────────┴──────────┘")

    print(f"  Samples: {results['num_samples']}  |  "
          f"Params: {results['model_params']:,}")


def save_results(results, exp_name):
    """保存评估结果为 JSON"""
    exp_dir = os.path.join(os.path.dirname(__file__), exp_name)
    result_path = os.path.join(exp_dir, "eval_pva_result.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  → Saved: {result_path}")


# ═══════════════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="PVA-Net 评估: 速度/加速度/3D关键点精度"
    )
    parser.add_argument("--exp", type=str, default=None,
                        choices=list(EXP_INFO.keys()),
                        help="实验名称")
    parser.add_argument("--all", action="store_true",
                        help="评估全部实验")
    parser.add_argument("--data_num", type=int, default=None,
                        help="限制测试样本数（用于快速调试）")
    args = parser.parse_args()

    exps = []
    if args.all:
        exps = list(EXP_INFO.keys())
    elif args.exp:
        exps = [args.exp]
    else:
        parser.print_help()
        print("\n请指定 --exp 或 --all")
        sys.exit(1)

    all_results = {}
    for exp_name in exps:
        results = evaluate_experiment(exp_name, args.data_num)
        if results is not None:
            print_results(results)
            save_results(results, exp_name)
            all_results[exp_name] = results

    # 汇总对比
    if len(all_results) > 1:
        print("\n\n" + "█" * 55)
        print("   对比汇总")
        print("█" * 55)
        for exp_name, r in all_results.items():
            print(f"  {exp_name:20s}  "
                  f"MPJPE={r['refine_mpjpe']:.3f}mm  "
                  f"MPJVE={r['refine_mpjve']:.3f}  "
                  f"MPJAE={r['refine_mpjae']:.3f}  "
                  f"PCEvel@0.05={r['vel_pce_0.05']:.1f}%  "
                  f"PCEacc@0.05={r['acc_pce_0.05']:.1f}%")


if __name__ == "__main__":
    main()
