"""
PVAOptimizer — HTD-Refine 论文的 test-time 优化模块（手部适配版）
====================================================================
严格遵循 HTD-Refine 论文 Sec 3.3 的能量函数设计，适配 3D 手部关节。

论文核心公式 (10)-(15):
  E = λ_V·EV + λ_A·EA + λ_jerk·Ejerk + λ_reg·Ereg

其中:
  EV = ||pred_vel - computed_vel||²       速度一致性
  EA = ||pred_acc - computed_acc||²       加速度一致性
  Ejerk = ||J'''||²                       Jerk 平滑（三阶导）
  Ereg = ||joints - initial||²            正则化（防止跑偏）

原始论文还包含 2D 关键点投影项 EK，我们跳过（无相机参数）。

用法:
  optimizer = PVAOptimizer()
  refined = optimizer.optimize(
      init_joints,      # [B, V, F, J, 3] 初始预测
      pred_vel,         # [B, V, F, J, 3] PVA-Net 速度输出
      pred_acc,         # [B, V, F, J, 3] PVA-Net 加速度输出
      mask,             # [B, V, F, J] 有效帧 mask
  )
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm


class PVAOptimizer:
    """
    HTD-Refine 风格的运动优化器。

    Args:
        lr:         Adam 学习率 (default 0.01)
        steps:      优化迭代步数 (default 200)
        lambda_v:   速度一致性权重 (default 1.0, 论文值)
        lambda_a:   加速度一致性权重 (default 0.1, 论文值)
        lambda_jerk: Jerk 平滑权重 (default 1e4, 论文值)
        lambda_reg: 正则化权重 (default 1e4, 论文值)
        verbose:    是否打印优化进度
    """

    def __init__(self,
                 lr=0.01,
                 steps=200,
                 lambda_v=1.0,
                 lambda_a=0.1,
                 lambda_jerk=1e4,
                 lambda_reg=1e4,
                 verbose=True):
        self.lr = lr
        self.steps = steps
        self.lambda_v = lambda_v
        self.lambda_a = lambda_a
        self.lambda_jerk = lambda_jerk
        self.lambda_reg = lambda_reg
        self.verbose = verbose

    @staticmethod
    def _compute_vel(joints):
        """[B,V,F,J,3] → [B,V,F,J,3] 速度（最后一帧重复前一帧的值）"""
        vel = joints[:, :, 1:, :, :] - joints[:, :, :-1, :, :]
        return torch.cat([vel, vel[:, :, -1:, :, :]], dim=2)

    @staticmethod
    def _compute_acc(joints):
        """[B,V,F,J,3] → [B,V,F,J,3] 加速度（最后两帧重复）"""
        acc = joints[:, :, 2:, :, :] - 2 * joints[:, :, 1:-1, :, :] + joints[:, :, :-2, :, :]
        acc = torch.cat([acc, acc[:, :, -1:, :, :], acc[:, :, -1:, :, :]], dim=2)
        return acc

    @staticmethod
    def _compute_jerk(joints):
        """[B,V,F,J,3] → [B,V,F,J,3] Jerk = J'''（最后三帧重复）"""
        jerk = joints[:, :, 3:, :, :] - 3 * joints[:, :, 2:-1, :, :] \
               + 3 * joints[:, :, 1:-2, :, :] - joints[:, :, :-3, :, :]
        pad = jerk[:, :, -1:, :, :]
        return torch.cat([jerk, pad, pad, pad], dim=2)

    def optimize(self, init_joints, pred_vel, pred_acc, mask=None):
        """
        主入口：执行 test-time 优化。

        Args:
            init_joints: [B,V,F,J,3] 初始 3D 关节预测（模型原始输出）
            pred_vel:    [B,V,F,J,3] PVA-Net 预测的速度
            pred_acc:    [B,V,F,J,3] PVA-Net 预测的加速度
            mask:        [B,V,F,J]   Bool/float, 1=有效帧

        Returns:
            refined:     [B,V,F,J,3] 优化后的 3D 关节
            losses:      dict of final loss values
        """
        device = init_joints.device

        # 需要梯度的优化变量（深拷贝初始值，从初始值开始优化）
        joints = init_joints.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([joints], lr=self.lr)

        # 目标值（只用于计算 energy 中的 mask）
        if mask is None:
            mask = torch.ones(init_joints.shape[:4], device=device, dtype=torch.bool)

        mask_float = mask.float().unsqueeze(-1)  # [B,V,F,J,1] 用于加权

        pbar = range(self.steps)
        if self.verbose:
            pbar = tqdm(pbar, desc='PVA Optimize', leave=False)

        for step in pbar:
            optimizer.zero_grad()

            # 从当前优化的 joints 算速度/加速度/jerk
            vel = self._compute_vel(joints)    # [B,V,F,J,3]
            acc = self._compute_acc(joints)    # [B,V,F,J,3]
            jerk = self._compute_jerk(joints)  # [B,V,F,J,3]

            # ── 各项 loss ─────────────────────────────

            # EV: 速度一致性 (公式 11)
            loss_v = ((vel - pred_vel) ** 2 * mask_float).sum() / (mask_float.sum() + 1e-8)

            # EA: 加速度一致性 (公式 12)
            loss_a = ((acc - pred_acc) ** 2 * mask_float).sum() / (mask_float.sum() + 1e-8)

            # Ejerk: Jerk 平滑 (公式 14)
            loss_jerk = (jerk ** 2 * mask_float).sum() / (mask_float.sum() + 1e-8)

            # Ereg: 正则化 (公式 15) — 保持靠近初始值
            loss_reg = ((joints - init_joints.detach()) ** 2 * mask_float).sum() / (mask_float.sum() + 1e-8)

            # ── 总 loss ──
            total = (self.lambda_v * loss_v
                     + self.lambda_a * loss_a
                     + self.lambda_jerk * loss_jerk
                     + self.lambda_reg * loss_reg)

            total.backward()
            optimizer.step()

            if self.verbose and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({
                    'v': f'{loss_v.item():.4f}',
                    'a': f'{loss_a.item():.4f}',
                    'j': f'{loss_jerk.item():.1e}',
                    'r': f'{loss_reg.item():.4f}',
                })

        losses = {
            'loss_v': loss_v.item(),
            'loss_a': loss_a.item(),
            'loss_jerk': loss_jerk.item(),
            'loss_reg': loss_reg.item(),
        }

        return joints.detach(), losses
