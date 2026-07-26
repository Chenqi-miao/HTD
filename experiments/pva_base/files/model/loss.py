import torch
import torch.nn as nn
import torch.nn.functional as F

class SmoothL1Loss(torch.nn.Module):
    def __init__(self, size_average=False):
        super(SmoothL1Loss, self).__init__()
        self.size_average = size_average

    def forward(self, x, y, mask=None):
        total_loss = 0
        assert (x.shape == y.shape)
        z = (x - y).float()
        mse_mask = (torch.abs(z) < 0.01).float()
        l1_mask = (torch.abs(z) >= 0.01).float()
        mse = mse_mask * z
        l1 = l1_mask * z
        total_loss += self._calculate_MSE(mse)*mse_mask
        total_loss += self._calculate_L1(l1)*l1_mask

        if mask is not None:
            return (total_loss * mask).sum() / (mask.sum() + 1e-8)
        else:
            return total_loss.mean()

    def _calculate_MSE(self, z):
        return 0.5 * (torch.pow(z, 2))

    def _calculate_L1(self, z):
        return 0.01 * (torch.abs(z) - 0.005)


class SmoothNetLoss(nn.Module):

    def __init__(self, w_accel, w_pos):
        super().__init__()
        self.w_accel = w_accel
        self.w_pos = w_pos

    def mask_lr1_loss(self, inputs, targets, mask):
        not_mask = 1 - mask.int()
        not_mask = not_mask.float()

        N = not_mask.sum(dtype=torch.float32)
        loss = F.l1_loss(inputs * not_mask, targets * not_mask, reduction='sum') / (N+1e-8)
        return loss

    def forward(self, denoise, gt, mask):
        denoise = denoise.permute(0, 2, 1)
        gt = gt.permute(0, 2, 1)
        mask = mask.permute(0, 2, 1)

        loss_pos = self.mask_lr1_loss(denoise, gt, mask)

        accel_gt = gt[:, :, :-2] - 2 * gt[:, :, 1:-1] + gt[:, :, 2:]
        accel_denoise = denoise[:, :, :-2] - 2 * denoise[:, :, 1:-1] + denoise[:, :, 2:]

        mask = (mask[:, :, :-2] + mask[:, :, 1:-1] + mask[:, :, 2:]) >0
        loss_accel = self.mask_lr1_loss(accel_denoise, accel_gt, mask)

        weighted_loss = self.w_accel * loss_accel + self.w_pos * loss_pos

        return weighted_loss