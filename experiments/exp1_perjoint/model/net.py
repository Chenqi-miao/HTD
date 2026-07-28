"""
Exp1: DSTFormer backbone + 3 per-joint Linear(256→3) heads
原版 HTD 的骨干 + 速度/加速度头，无 mean pool。
"""
import torch
import torch.nn as nn
from einops import rearrange
from DSF import DSTFormer
from loss import SmoothNetLoss

class PosHead(nn.Module):
    """per-joint 位置头: Linear(256→3)"""
    def __init__(self, dim=256): super().__init__(); self.fc = nn.Linear(dim, 3)
    def forward(self, x): return self.fc(x)

class VelHead(nn.Module):
    """per-joint 速度头: Conv1d → Linear(256→3)"""
    def __init__(self, dim=256):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, padding=1)
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, 3)
    def forward(self, x):
        B, F, J, C = x.shape
        x = x.reshape(B*J, F, C).permute(0, 2, 1)
        x = self.conv(x).permute(0, 2, 1)
        return self.fc(self.norm(x)).reshape(B, J, F, 3).permute(0, 2, 1, 3)

class AccHead(nn.Module):
    """per-joint 加速度头（同速度头）"""
    def __init__(self, dim=256): super().__init__(); self.core = VelHead(dim)
    def forward(self, x): return self.core(x)

class Net(nn.Module):
    def __init__(self, num_frame=15, num_joints=21, dim_feat=256, depth=2,
                 w_vel=0.1, w_accel=0.05):
        super().__init__()
        self.num_frame, self.num_joints = num_frame, num_joints
        self.w_vel, self.w_accel = w_vel, w_accel
        self.backbone = DSTFormer(dim_in=3, dim_feat=dim_feat, dim_rep=dim_feat, depth=depth, num_heads=8, mlp_ratio=4, num_joints=num_joints, maxlen=num_frame)
        self.pos_head = PosHead(dim_feat)
        self.vel_head = VelHead(dim_feat)
        self.acc_head = AccHead(dim_feat)
        self.sm_loss = SmoothNetLoss(0.1, 1.0)
        self.l1 = nn.L1Loss(reduction='none')
        p = sum(p.numel() for p in self.parameters())
        print(f'[Exp1] params={p:,}')

    def forward(self, inputs, targets, meta_info):
        ji, jg = inputs["joint_xyz"].cuda(), targets["joint_xyz"].cuda()
        c = meta_info['center_xyz'].cuda()
        iv = meta_info['joint_in_val'].cuda().reshape(*ji.shape[:4], 1)
        cv = meta_info['continuous_val'].cuda()
        B, V, F, J = ji.shape[:4]
        norm = 300
        sc = (c * iv).sum(dim=(2,3)) / (iv.sum(dim=(2,3))+1e-8)
        sc = sc.reshape(B, V, 1, 1, 3)
        jn_in = (ji - sc) / norm
        jn_gt = (jg - sc) / norm

        feat = self.backbone(rearrange(jn_in, 'b v f j c -> (b v) f j c'))
        feat = rearrange(feat, '(b v) f j d -> b v f j d', b=B, v=V)

        jn_p = self.pos_head(feat)
        fm = feat[:, 0] if V == 1 else feat.mean(dim=1)
        vp = self.vel_head(fm).unsqueeze(1)
        ap = self.acc_head(fm).unsqueeze(1)
        if V > 1: vp = vp.repeat(1, V, 1, 1, 1); ap = ap.repeat(1, V, 1, 1, 1)

        jp = jn_p * norm + sc
        val = (cv.view(B,1,F,1,1) * meta_info['joint_gt_val'].cuda().view(B,1,F,J,1))
        vV = val.repeat(1, V, 1, 1, 1); vP = val.repeat(1, V, 1, 1, 3)

        if self.training:
            pl = self.sm_loss(jn_p.reshape(B*V, F, J*3), jn_gt.reshape(B*V, F, J*3), 1-vP.view(B*V, F, J*3))
            gv = torch.cat([jn_gt[:,:,1:]-jn_gt[:,:,:-1], (jn_gt[:,:,1:]-jn_gt[:,:,:-1])[:,:,-1:]], dim=2)
            ga = torch.cat([gv[:,:,1:]-gv[:,:,:-1], gv[:,:,-1:]], dim=2)
            vm = torch.cat([vV[:,:,:-1]*vV[:,:,1:], vV[:,:,-1:]], dim=2)
            am = torch.cat([vV[:,:,:-2]*vV[:,:,1:-1]*vV[:,:,2:], vV[:,:,-1:], vV[:,:,-1:]], dim=2)
            vl = (self.l1(vp, gv)*vm).sum()/(vm.sum()+1e-8)
            al = (self.l1(ap, ga)*am).sum()/(am.sum()+1e-8)
            d = (ji*vV - jg*vV); ie = torch.sqrt((d*d).sum(-1)).sum()/(vV.sum()+1e-8)
            d = (jp*vV - jg*vV); re = torch.sqrt((d*d).sum(-1)).sum()/(vV.sum()+1e-8)
            return {'pd_joint_xyz': jp}, {'pos': pl, 'vel': vl, 'acc': al, 'total': pl+self.w_vel*vl+self.w_accel*al}, {'init': ie, 'refine': re}
        return {'pd_joint_xyz': jp, 'vel_pred': vp, 'acc_pred': ap}
