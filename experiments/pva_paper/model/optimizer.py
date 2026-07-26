"""
PVAOptimizer — HTD-Refine 论文 Stage 3 运动优化（手部关节版）
=============================================================
严格遵循论文 Sec 3.3 能量函数 (公式 10-15)，适配 3D 关节坐标。

能量函数:
  E = λ_V·EV  + λ_A·EA  + λ_jerk·Ejerk  + λ_reg·Ereg

  EV     = ||V - V̂||²           速度一致性 (公式 11)
  EA     = ||A - Â||²           加速度一致性 (公式 12)
  Ejerk  = ||J'''||²            三阶导平滑 (公式 14)
  Ereg   = ||joints - init||²   正则化 (公式 15)

用法:
  optimizer = PVAOptimizer()
  refined = optimizer.optimize(
      init_joints, pred_vel, pred_acc, mask=None)
"""

import torch
from tqdm import tqdm


class PVAOptimizer:
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

    # ─── 运动学量计算 ──────────────────────────────────

    @staticmethod
    def _vel(joints):
        """[B,1,F,J,3] → [B,1,F,J,3], 末帧重复"""
        v = joints[:, :, 1:] - joints[:, :, :-1]
        return torch.cat([v, v[:, :, -1:]], dim=2)

    @staticmethod
    def _acc(joints):
        """2阶差分, 末两帧重复"""
        a = joints[:, :, 2:] - 2 * joints[:, :, 1:-1] + joints[:, :, :-2]
        return torch.cat([a, a[:, :, -1:], a[:, :, -1:]], dim=2)

    @staticmethod
    def _jerk(joints):
        """3阶差分, 末三帧重复"""
        j = joints[:, :, 3:] - 3 * joints[:, :, 2:-1] \
            + 3 * joints[:, :, 1:-2] - joints[:, :, :-3]
        return torch.cat([j, j[:, :, -1:], j[:, :, -1:], j[:, :, -1:]], dim=2)

    # ─── 优化入口 ──────────────────────────────────────

    def optimize(self, init_joints, pred_vel, pred_acc, mask=None):
        """
        Args:
            init_joints: [B,1,F,J,3] PVANet 输出的关节位置
            pred_vel:    [B,1,F,J,3] PVANet 预测的速度
            pred_acc:    [B,1,F,J,3] PVANet 预测的加速度
            mask:        [B,1,F,J]   有效帧标志 (1=有效)
        Returns:
            refined:     [B,1,F,J,3] 优化后的关节
            losses:      dict
        """
        device = init_joints.device
        joints = init_joints.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([joints], lr=self.lr)
        m = mask.float().unsqueeze(-1) if mask is not None else torch.ones_like(init_joints[..., :1])

        pbar = tqdm(range(self.steps), desc='Optimize', leave=False) if self.verbose else range(self.steps)

        for _ in pbar:
            opt.zero_grad()
            v = self._vel(joints)
            a = self._acc(joints)
            j = self._jerk(joints)

            loss_v = ((v - pred_vel) ** 2 * m).sum() / m.sum()
            loss_a = ((a - pred_acc) ** 2 * m).sum() / m.sum()
            loss_j = (j ** 2 * m).sum() / m.sum()
            loss_r = ((joints - init_joints.detach()) ** 2 * m).sum() / m.sum()

            total = self.lambda_v * loss_v + self.lambda_a * loss_a \
                    + self.lambda_jerk * loss_j + self.lambda_reg * loss_r
            total.backward()
            opt.step()

            if self.verbose:
                pbar.set_postfix(v=f'{loss_v.item():.3f}', a=f'{loss_a.item():.3f}',
                                 j=f'{loss_j.item():.1e}', r=f'{loss_r.item():.4f}')

        return joints.detach(), {
            'loss_v': loss_v.item(), 'loss_a': loss_a.item(),
            'loss_jerk': loss_j.item(), 'loss_reg': loss_r.item(),
        }
