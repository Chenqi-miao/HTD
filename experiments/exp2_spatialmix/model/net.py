"""
Exp2: Pure temporal transformer + per-frame spatial mixing
  每帧对 J 个关节做一次轻量线性混合（提供空间交互），
  然后 [B*J, F, D] 过时序 Transformer。

  架构:
  [B, F, J, 3] → Joint Embed → [B, F, J, D]
    → SpatialMix (per-frame Linear mixing over J)   ← 新增空间信息交换
    → Reshape [B*J, F, D] → TemporalTransformer ×8 (RoPE) → Reshape [B, F, J, D]
    → SpatialMix
    → 3× per-joint Linear(256→3)
"""
import torch
import torch.nn as nn
from einops import rearrange
from temporal_transformer import TemporalTransformerDecoder
from loss import SmoothNetLoss

class SpatialMix(nn.Module):
    """Per-frame 轻量线性空间混合: 每帧共享的 Linear(J·D → J·D)"""
    def __init__(self, num_joints=21, dim=256):
        super().__init__()
        self.mix = nn.Linear(num_joints * dim, num_joints * dim)
        self.norm = nn.LayerNorm(num_joints * dim)
    def forward(self, x):
        B, F, J, D = x.shape
        x = x.reshape(B, F, J * D)
        x = self.norm(x)
        x = self.mix(x)
        return x.reshape(B, F, J, D)

class VelHead(nn.Module):
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
    def __init__(self, dim=256): super().__init__(); self.core = VelHead(dim)
    def forward(self, x): return self.core(x)

class Net(nn.Module):
    def __init__(self, num_frame=15, num_joints=21, dim_feat=256, depth=8,
                 w_vel=0.1, w_accel=0.05):
        super().__init__()
        self.num_frame, self.num_joints = num_frame, num_joints
        self.w_vel, self.w_accel = w_vel, w_accel

        self.joint_embed = nn.Linear(3, dim_feat)
        self.pos_drop = nn.Dropout(0.1)

        # 空间混合（放在时序 transformer 前后）
        self.spatial_pre = SpatialMix(num_joints, dim_feat)
        self.spatial_post = SpatialMix(num_joints, dim_feat)

        self.temporal = TemporalTransformerDecoder(
            dim=dim_feat, depth=depth, num_heads=8, mlp_ratio=4,
            drop_rate=0.1, attn_drop_rate=0.1, use_rope=True)

        self.pos_head = nn.Linear(dim_feat, 3)
        self.vel_head = VelHead(dim_feat)
        self.acc_head = AccHead(dim_feat)

        self.sm_loss = SmoothNetLoss(0.1, 1.0)
        self.l1 = nn.L1Loss(reduction='none')
        p = sum(p.numel() for p in self.parameters())
        print(f'[Exp2] depth={depth}, params={p:,}')

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

        x = self.joint_embed(jn_in)           # [B, V, F, J, D]
        x = self.pos_drop(x)
        x = rearrange(x, 'b v f j d -> (b v) f j d')

        # 空间混合 → 时序 → 空间混合
        x = self.spatial_pre(x)               # [B*V, F, J, D]
        x = rearrange(x, 'b f j d -> (b j) f d')
        x = self.temporal(x)                  # [B*V*J, F, D]
        x = rearrange(x, '(b j) f d -> b f j d', j=J)
        x = self.spatial_post(x)

        x = rearrange(x, '(b v) f j d -> b v f j d', b=B, v=V)

        jn_p = self.pos_head(x)
        fm = x[:, 0] if V == 1 else x.mean(dim=1)
        vp = self.vel_head(fm).unsqueeze(1)
        ap = self.acc_head(fm).unsqueeze(1)
        if V > 1: vp = vp.repeat(1, V, 1, 1, 1); ap = ap.repeat(1, V, 1, 1, 1)

        jp = jn_p * norm + sc
        val = (cv.view(B,1,F,1,1) * meta_info['joint_gt_val'].cuda().view(B,1,F,J,1))
        vV = val.repeat(1, V, 1, 1, 1); vP = val.repeat(1, V, 1, 1, 3)

        if self.training:
            pl = self.sm_loss(jn_p.reshape(B*V,F,J*3), jn_gt.reshape(B*V,F,J*3), 1-vP.view(B*V,F,J*3))
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
