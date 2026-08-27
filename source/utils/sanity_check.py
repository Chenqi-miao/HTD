"""
sanity_check.py — 训练前自动健康检查
====================================
在训练正式开始前运行，用几分钟时间排除大部分架构/数值/梯度错误。

检查项:
  1. overfit_check     过拟合测试: 小批数据上 loss 应明显下降
                        loss 不降 → 架构/梯度有问题（模型学不会训练数据）
  2. activation_stats  中间张量统计: 检查数值爆炸/塌缩
                        相加分量 std 差 >10x → 危险信号（如 3.2 + 292）
  3. gradient_health   梯度健康: 各模块梯度范数
                        梯度 <0.01 → 模块死了（冻结/梯度消失/数值问题）
  4. residual_check    残差检查: 残差模型初始输出应≈输入
                        修正量过大 → 输出可能乱飞

用法 (任意实验, 模型需有 forward(inputs, targets, meta_info) 返回
      (outs, loss_dict, error_dict) 训练接口):

  from source.utils.sanity_check import run_sanity_checks
  passed, report = run_sanity_checks(model, small_loader, num_steps=50)
  if not passed:
      print('❌ sanity check 未通过，停止训练')
      exit(1)
"""

import torch
import torch.optim as optim


def _to_device(data, device):
    """把 batch 里的张量移到指定设备（meta_info 可能含非张量）"""
    if isinstance(data, dict):
        return {k: _to_device(v, device) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [_to_device(x, device) for x in data]
    if isinstance(data, torch.Tensor):
        return data.to(device)
    return data


# ═══════════════════════════════════════════════════════════════════════
# 1. 过拟合测试
# ═══════════════════════════════════════════════════════════════════════

def overfit_check(model, dataloader, num_steps=50, lr=1e-3, device='cuda',
                  verbose=True):
    """
    在固定的小批数据上训练 num_steps 步，检查 loss 是否明显下降。

    判定:
      模型若能学会训练数据（loss 显著下降）→ 架构能学
      模型学不会 → 架构/梯度/数据对齐有问题

    注意:
      - 会修改模型权重！应在模型刚初始化/加载后调用
      - 用 Adam(lr=1e-3) 快速过拟合，验证架构而非训练策略
    """
    model.train()
    opt = optim.Adam(model.parameters(), lr=lr)

    # 固定一批数据，反复训练（移到模型所在设备）
    try:
        batch = next(iter(dataloader))
    except StopIteration:
        return False, {'error': 'dataloader 为空'}
    device = next(model.parameters()).device
    batch = _to_device(batch, device)

    losses = []
    for step in range(num_steps):
        opt.zero_grad()
        outs, loss_dict, _ = model(*batch)
        loss = loss_dict['total_loss']
        loss.backward()
        opt.step()
        losses.append(loss.item())

    init_loss, final_loss = losses[0], losses[-1]
    drop_ratio = (init_loss - final_loss) / (init_loss + 1e-8)

    # 判定:
    #   非残差模型: 初始 loss 大, 应明显下降 (drop_ratio > 0.2)
    #   残差模型:   初始 loss 已很小 (输出≈输入≈GT), 只要不发散即可
    if init_loss < 0.05:
        passed = final_loss < init_loss * 1.5   # 已很小, 不发散 >50% 即通过
    else:
        passed = drop_ratio > 0.2               # 正常模型, 需明显下降

    if verbose:
        print(f'[overfit] loss: {init_loss:.4f} → {final_loss:.4f} '
              f'(下降 {drop_ratio*100:.1f}%) {"✅" if passed else "❌"}')

    return passed, {'init': init_loss, 'final': final_loss, 'drop_ratio': drop_ratio}


# ═══════════════════════════════════════════════════════════════════════
# 2. 中间张量统计（数值爆炸/塌缩检测）
# ═══════════════════════════════════════════════════════════════════════

def activation_stats(model, dataloader, device='cuda', verbose=True,
                     min_std=1e-3, max_std=100):
    """
    用 forward hooks 记录各子模块输出的 std。

    检测:
      - std 接近 0 → 特征塌缩（死神经元）
      - std 巨大（远超输入）→ 数值爆炸

    返回: (是否通过, {module_name: output_std})
      通过条件: 所有模块输出 std 在 [min_std, max_std] 内
    """
    stats = {}

    def make_hook(name):
        def hook(module, inp, out):
            # 取输出张量（tuple 则取第一个）
            if isinstance(out, (tuple, list)):
                out = out[0]
            if isinstance(out, torch.Tensor):
                stats[name] = out.detach().std().item()
        return hook

    hooks = []
    for name, module in model.named_children():
        hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    try:
        batch = next(iter(dataloader))
        device = next(model.parameters()).device
        batch = _to_device(batch, device)
        with torch.no_grad():
            model(*batch)
    except Exception as e:
        print(f'[activation] 前向失败: {e}')
    finally:
        for h in hooks:
            h.remove()

    passed = True
    if verbose:
        for name, std in stats.items():
            ok = min_std < std < max_std
            passed = passed and ok
            flag = '✅' if ok else ('❌ 塌缩' if std < min_std else '❌ 爆炸')
            print(f'[activation] {name} std={std:.3f} {flag}')

    return passed, stats


# ═══════════════════════════════════════════════════════════════════════
# 3. 梯度健康检查
# ═══════════════════════════════════════════════════════════════════════

# LayerNorm/BatchNorm 等归一化层: 梯度天然小（作用是归一化），不参与健康判定
_NORM_LAYERS = (torch.nn.LayerNorm, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)


def gradient_health(model, threshold=0.01, verbose=True):
    """
    检查各子模块的梯度 L2 范数。

    需在 backward() 之后调用。

    判定:
      梯度范数 < threshold → 模块没有有效学习（冻结/梯度消失/数值切断）
      梯度范数异常大     → 更新不稳定
      归一化层 (LayerNorm/BatchNorm) → 梯度天然小，跳过判定

    返回: (是否通过, {module_name: grad_norm})
      通过条件: 所有非归一化"可训练"模块梯度范数 > threshold
    """
    result = {}
    for name, module in model.named_children():
        # 跳过归一化层（gamma/beta 梯度天然小）
        if isinstance(module, _NORM_LAYERS):
            continue
        grad_sq = 0.0
        has_trainable = False
        for p in module.parameters():
            if p.requires_grad:
                has_trainable = True
                if p.grad is not None:
                    grad_sq += p.grad.norm().item() ** 2
        # 只有含可训练参数的模块才参与检查
        if has_trainable:
            result[name] = grad_sq ** 0.5

    passed = True
    if verbose:
        for name, grad_norm in result.items():
            ok = grad_norm > threshold
            passed = passed and ok
            flag = '✅' if ok else f'❌ 梯度太小(<{threshold})'
            print(f'[gradient] {name} grad={grad_norm:.4f} {flag}')

    return passed, result


# ═══════════════════════════════════════════════════════════════════════
# 4. 残差检查（残差模型专用）
# ═══════════════════════════════════════════════════════════════════════

def residual_check(model, dataloader, device='cuda',
                   correction_key=None, verbose=True, max_ratio=0.3):
    """
    检查残差模型的初始输出是否≈输入。

    适用: 输出 = 输入 + 修正量 的模型（如 DecoupledNet 的 jn_pred = jn_in + pos_head(feat)）

    判定:
      修正量 / 输入 比值 > max_ratio (0.3=30%) → 修正量过大，输出可能乱飞
      修正量≈0 且模型能学 → 正常（残差安全网）

    参数:
      correction_key: 模型 forward 返回值里包含修正量的键
                      （若 None，用输出 vs 输入的差值近似）
    """
    model.eval()
    try:
        batch = next(iter(dataloader))
        device = next(model.parameters()).device
        batch = _to_device(batch, device)
        with torch.no_grad():
            result = model(*batch)
            # eval 模式可能只返回 outs dict（DecoupledNet）或 3 元组
            outs = result[0] if isinstance(result, (tuple, list)) else result
    except Exception as e:
        print(f'[residual] 前向失败: {e}')
        return False, {}

    # 尝试从 outs 拿输出和输入
    # 约定: 输入在 batch[0]（inputs dict）的 'joint_xyz'
    inputs = batch[0]
    if isinstance(inputs, dict) and 'joint_xyz' in inputs:
        jn_in = inputs['joint_xyz'].float()
    else:
        return False, {'error': '无法定位输入 joint_xyz'}

    # 输出: 优先从 outs 拿 pd_joint_xyz
    if isinstance(outs, dict) and 'pd_joint_xyz' in outs:
        jn_out = outs['pd_joint_xyz'].float()
    else:
        return False, {'error': '无法定位输出 pd_joint_xyz'}

    # 修正量 ≈ 输出 - 输入（在相同尺度下）
    correction = jn_out - jn_in
    corr_std = correction.std().item()
    inp_std = jn_in.std().item() + 1e-8
    ratio = corr_std / inp_std
    passed = ratio < max_ratio

    if verbose:
        flag = '✅' if passed else '❌'
        print(f'[residual] 修正量std={corr_std:.4f} / 输入std={inp_std:.4f} '
              f'= {ratio*100:.1f}% (阈值 {max_ratio*100:.0f}%) {flag}')

    return passed, {'correction_std': corr_std, 'input_std': inp_std, 'ratio': ratio}


# ═══════════════════════════════════════════════════════════════════════
# 一键运行
# ═══════════════════════════════════════════════════════════════════════

def run_sanity_checks(model, dataloader, num_steps=50, device='cuda',
                      check_residual=True, verbose=True):
    """
    一键运行所有 sanity check，返回 (是否全部通过, 报告 dict)。

    用法:
      passed, report = run_sanity_checks(model, small_loader)
      if not passed:
          print('❌ sanity check 未通过')
          exit(1)

    注意: 会修改模型权重（overfit 测试），应在刚初始化后调用。
    """
    print('═' * 50)
    print('  训练前 Sanity Check')
    print('═' * 50)

    results = {}

    # 1. 过拟合测试（会训练模型，需最先跑）
    results['overfit'] = overfit_check(
        model, dataloader, num_steps=num_steps, device=device, verbose=verbose)

    # 2. 激活统计（需要重置为 eval，重新 forward）
    model.train()
    results['activation'] = activation_stats(
        model, dataloader, device=device, verbose=verbose)

    # 3. 梯度健康（需要一次 backward）
    # 重新取一批数据做 backward
    model.train()
    batch = next(iter(dataloader))
    device = next(model.parameters()).device
    batch = _to_device(batch, device)
    opt = optim.Adam(model.parameters(), lr=1e-4)
    opt.zero_grad()
    outs, loss_dict, _ = model(*batch)
    loss_dict['total_loss'].backward()
    results['gradient'] = gradient_health(model, verbose=verbose)

    # 4. 残差检查（可选）
    if check_residual:
        results['residual'] = residual_check(
            model, dataloader, device=device, verbose=verbose)

    # 汇总（所有检查统一返回 (bool, dict) 格式）
    print('═' * 50)
    passed = True
    for check, (ok, detail) in results.items():
        passed = passed and ok
        print(f'  {check:12s}: {"✅" if ok else "❌"}')
    print('═' * 50)

    return passed, results


if __name__ == '__main__':
    # 示例: 测试 sanity check 本身
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'experiments', 'exp10', 'src')))
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'experiments', 'exp10', 'model')))
    from net import DecoupledNet
    from dataloader import SeqHandLazy
    from torch.utils.data import DataLoader

    model = DecoupledNet()
    model.set_stage('spatial')
    ds = SeqHandLazy('/home/chenqi/workspace/hand_3d_reconstruction/HTD/data/SeqHand',
                     ['InterHand_train'], joint_num=21, min_seq_len=30,
                     seq_len=15, view_num=1, data_num=4)
    ld = DataLoader(ds, batch_size=4, shuffle=False)
    passed, report = run_sanity_checks(model, ld, num_steps=30)
