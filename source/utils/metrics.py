"""
metrics.py — 3D 关节点误差指标工具函数
=====================================
提供位置、速度、加速度的 masked 误差计算。
所有函数接受任意 batch 维度，只需最后一维是 3（xyz）即可。

用法:
  from source.utils.metrics import calc_mpjpe, calc_mpjve, calc_mpjae

  mask = continuous_val * joint_gt_val          # [B, V, F, J, 1]
  mpjpe = calc_mpjpe(pred_joints, gt_joints, mask)
  mpjve = calc_mpjve(pred_joints, gt_joints, mask)
  mpjae = calc_mpjae(pred_joints, gt_joints, mask)
"""

import torch


def _masked_mean(error, mask, eps=1e-8):
    """
    对 error 张量做 masked mean

    Args:
        error: [..., J]  per-joint 误差
        mask:  [..., J, 1]  或 [..., J]  0/1 mask
    Returns: scalar
    """
    if mask is None:
        return error.mean()
    if mask.dim() == error.dim():
        return (error * mask).sum() / (mask.sum() + eps)
    return (error * mask.squeeze(-1)).sum() / (mask.sum() + eps)


def calc_mpjpe(pred, target, mask=None):
    """
    Mean Per Joint Position Error (mm)

    MPJPE = mean(sqrt(||pred - target||²))

    Args:
        pred:   [..., J, 3]  预测关节 (mm)
        target: [..., J, 3]  GT 关节 (mm)
        mask:   [..., J, 1]  或 [..., J]  0/1 mask (1=有效)
    Returns: scalar (mm)
    """
    diff = pred - target
    per_joint = torch.sqrt(torch.sum(diff * diff, dim=-1))  # [..., J]
    return _masked_mean(per_joint, mask)


def calc_mpjve(pred, target, mask=None):
    """
    Mean Per Joint Velocity Error (mm/frame)

    速度 = Δpos = pos[t+1] - pos[t]
    MPJVE = mean(sqrt(||vel_pred - vel_target||²))

    注: 输出长度比输入少 1 帧（F→F-1），mask 会自动对齐。

    Args:
        pred:   [..., F, J, 3]  预测关节 (mm)
        target: [..., F, J, 3]  GT 关节 (mm)
        mask:   [..., F, J, 1]  0/1 mask (1=有效)
    Returns: scalar (mm/frame)
    """
    vel_pred = pred[..., 1:, :, :] - pred[..., :-1, :, :]     # [..., F-1, J, 3]
    vel_target = target[..., 1:, :, :] - target[..., :-1, :, :]
    diff = vel_pred - vel_target
    per_joint = torch.sqrt(torch.sum(diff * diff, dim=-1))     # [..., F-1, J]
    if mask is not None:
        mask_v = mask[..., :-1, :]                             # [..., F-1, J] (切 F 维度)
        return _masked_mean(per_joint, mask_v)
    return per_joint.mean()


def calc_mpjae(pred, target, mask=None):
    """
    Mean Per Joint Acceleration Error (mm/frame²)

    加速度 = Δ²pos = pos[t+2] - 2·pos[t+1] + pos[t]
    MPJAE = mean(sqrt(||acc_pred - acc_target||²))

    注: 输出长度比输入少 2 帧（F→F-2），mask 会自动对齐。

    Args:
        pred:   [..., F, J, 3]  预测关节 (mm)
        target: [..., F, J, 3]  GT 关节 (mm)
        mask:   [..., F, J, 1]  0/1 mask (1=有效)
    Returns: scalar (mm/frame²)
    """
    # 二阶差分: x[t+2] - 2*x[t+1] + x[t]
    acc_pred = pred[..., 2:, :, :] - 2 * pred[..., 1:-1, :, :] + pred[..., :-2, :, :]
    acc_target = target[..., 2:, :, :] - 2 * target[..., 1:-1, :, :] + target[..., :-2, :, :]
    diff = acc_pred - acc_target
    per_joint = torch.sqrt(torch.sum(diff * diff, dim=-1))     # [..., F-2, J]
    if mask is not None:
        mask_a = mask[..., :-2, :]                             # [..., F-2, J] (切 F 维度)
        return _masked_mean(per_joint, mask_a)
    return per_joint.mean()


def calc_all_errors(pred, target, mask=None):
    """
    同时计算 MPJPE + MPJVE + MPJAE

    返回值:
        dict {mpjpe, mpjve, mpjae}
    """
    return {
        'mpjpe': calc_mpjpe(pred, target, mask),
        'mpjve': calc_mpjve(pred, target, mask),
        'mpjae': calc_mpjae(pred, target, mask),
    }


def build_mask(continuous_val, joint_gt_val, data_repeat=None):
    """
    构建评估用的组合 mask

    对齐 HTD/source/dataset/seqhand.py 的 SeqHandTest.evaluate():
      mask = data_repeat × joint_gt_val × continuous_val
    其中 data_repeat=None 时等价于 1（训练时不需要，因为训练集无 padding）

    Args:
        continuous_val: [B, F]          帧连续性标记 (0/1)
        joint_gt_val:   [B, V, F, J]    GT 关节有效标记 (0/1)
        data_repeat:    [B, V, F, 1]    测试 padding 标记 (0/1, optional)
    Returns:
        mask: [B, V, F, J]   1=有效 (与数据集 evaluate 一致)
    """
    # continuous_val [B,F] → con_val [B,1,F,1]
    con_val = continuous_val.unsqueeze(-1).unsqueeze(1)   # [B,F] → [B,1,F,1]
    # joint_gt_val [B,V,F,J] 保持不动（mask 的关节维度）
    mask = con_val * joint_gt_val                          # [B,V,F,J]
    if data_repeat is not None:
        # data_repeat [B,V,F,1] → [B,V,F,1]
        dr = data_repeat if data_repeat.dim() == 4 else data_repeat.unsqueeze(-1)
        mask = mask * dr                                   # [B,V,F,J]
    return mask
