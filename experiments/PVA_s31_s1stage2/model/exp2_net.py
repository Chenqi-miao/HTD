"""
exp2_s30_att: 30帧窗口 + 关节注意力
===================================
修复两个诊断:
  1. 窗口太短(15帧)学不到运动规律 → seq_len=30
  2. SpatialMix 线性混合抹平关节动力学 → 换成 JointAttention
     (对 J 维做多头自注意力 + 关节身份嵌入, 内容相关且保留每关节身份)
"""
import torch
import torch.nn as nn
from einops import rearrange
from temporal_transformer import TemporalTransformerDecoder
from loss import SmoothNetLoss


class JointAttention(nn.Module):
    """关节维自注意力 + 关节身份嵌入 (保留每关节动力学差异), 支持多层"""
    def __init__(self, dim=256, num_heads=8, num_joints=21, n_layers=1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([nn.LayerNorm(dim), nn.MultiheadAttention(dim, num_heads, batch_first=True)])
            for _ in range(n_layers)])
        self.joint_emb = nn.Parameter(torch.randn(num_joints, dim) * 0.02)

    def forward(self, x):                             # [B, F, J, D]
        B, F, J, D = x.shape
        xe = x + self.joint_emb[None, None, :, :]      # 关节身份
        xr = xe.reshape(B * F, J, D)
        for norm, attn in self.layers:
            xn = norm(xr)
            out, _ = attn(xn, xn, xn)                  # 对 J 维注意力
            xr = xr + out                             # 残差
        return xr.reshape(B, F, J, D)


class PosHead(nn.Module):
    """时序 Conv1d 位置头 (同 exp1 的 vel/acc 头结构, 利用相邻帧时间上下文)"""
    def __init__(self, dim=256):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, padding=1)
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, 3)
    def forward(self, x):                                   # [B, V, F, J, D]
        B, V, F, J, C = x.shape
        x = x.permute(0, 1, 3, 2, 4)                        # [B, V, J, F, D] (J 提前)
        x = x.reshape(B * V * J, F, C).permute(0, 2, 1)      # [B*V*J, C, F]
        x = self.conv(x).permute(0, 2, 1)                    # [B*V*J, F, C]
        out = self.fc(self.norm(x))                          # [B*V*J, F, 3]
        return out.reshape(B, V, J, F, 3).permute(0, 1, 3, 2, 4)  # [B, V, F, J, 3]


class VelHead(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, padding=1)
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, 3)
    def forward(self, x):                                # [B, F, J, C]
        B, F, J, C = x.shape
        x = x.permute(0, 2, 1, 3)                        # [B, J, F, C] (J 提前)
        x = x.reshape(B * J, F, C).permute(0, 2, 1)       # [B*J, C, F]
        x = self.conv(x).permute(0, 2, 1)                 # [B*J, F, C]
        return self.fc(self.norm(x)).reshape(B, J, F, 3).permute(0, 2, 1, 3)  # [B, F, J, 3]


class AccHead(nn.Module):
    def __init__(self, dim=256): super().__init__(); self.core = VelHead(dim)
    def forward(self, x): return self.core(x)


class Net(nn.Module):
    def __init__(self, num_frame=30, num_joints=21, dim_feat=256, depth=8,
                 w_vel=0.1, w_accel=0.05):
        super().__init__()
        self.num_frame, self.num_joints = num_frame, num_joints
        self.w_vel, self.w_accel = w_vel, w_accel

        self.joint_embed = nn.Linear(9, dim_feat)   # 位置+速度+加速度
        self.pos_drop = nn.Dropout(0.1)
        self.joint_attn_pre = JointAttention(dim_feat, 8, num_joints, n_layers=1)
        self.joint_attn_post = JointAttention(dim_feat, 8, num_joints, n_layers=2)
        self.temporal = TemporalTransformerDecoder(
            dim=dim_feat, depth=depth, num_heads=8, mlp_ratio=4,
            drop_rate=0.1, attn_drop_rate=0.1, use_rope=True)
        self.pos_head = nn.Linear(dim_feat, 3)
        self.vel_head = VelHead(dim_feat)
        self.acc_head = AccHead(dim_feat)
        self.sm_loss = SmoothNetLoss(0.1, 1.0)
        self.l1 = nn.L1Loss(reduction='none')
        p = sum(p.numel() for p in self.parameters())
        print(f'[Exp2_s30_att] depth={depth}, frames={num_frame}, params={p:,}')

    def forward(self, inputs, targets, meta_info):
        ji, jg = inputs["joint_xyz"].cuda(), targets["joint_xyz"].cuda()
        c = meta_info['center_xyz'].cuda()
        iv = meta_info['joint_in_val'].cuda().reshape(*ji.shape[:4], 1)
        cv = meta_info['continuous_val'].cuda()
        B, V, F, J = ji.shape[:4]
        norm = 300
        sc = (c * iv).sum(dim=(2, 3)) / (iv.sum(dim=(2, 3)) + 1e-8)
        sc = sc.reshape(B, V, 1, 1, 3)
        jn_in = (ji - sc) / norm
        jn_gt = (jg - sc) / norm

        # ── 输入动力学: 位置 + 速度 + 加速度 (归一化系, 补全到 F 帧) ──
        in_v = jn_in[:, :, 1:] - jn_in[:, :, :-1]                    # [B,V,F-1,J,3]
        in_v = torch.cat([in_v, in_v[:, :, -1:]], dim=2)             # 补到 F
        in_a = in_v[:, :, 1:] - in_v[:, :, :-1]                      # [B,V,F-1,J,3]→差分
        in_a = torch.cat([in_a, in_a[:, :, -1:]], dim=2)             # 补到 F
        jin = torch.cat([jn_in, in_v, in_a], dim=-1)                 # [B,V,F,J,9]

        x = self.joint_embed(jin)
        x = self.pos_drop(x)
        x = rearrange(x, 'b v f j d -> (b v) f j d')
        x = self.joint_attn_pre(x)
        x = rearrange(x, 'b f j d -> (b j) f d')
        x = self.temporal(x)
        x = rearrange(x, '(b j) f d -> b f j d', j=J)
        x = self.joint_attn_post(x)
        x = rearrange(x, '(b v) f j d -> b v f j d', b=B, v=V)

        jn_p = self.pos_head(x)
        fm = x[:, 0] if V == 1 else x.mean(dim=1)
        vp = self.vel_head(fm).unsqueeze(1)
        ap = self.acc_head(fm).unsqueeze(1)
        if V > 1:
            vp = vp.repeat(1, V, 1, 1, 1); ap = ap.repeat(1, V, 1, 1, 1)

        jp = jn_p * norm + sc
        val = (cv.view(B, 1, F, 1, 1) * meta_info['joint_gt_val'].cuda().view(B, 1, F, J, 1))
        vV = val.repeat(1, V, 1, 1, 1); vP = val.repeat(1, V, 1, 1, 3)

        if self.training:
            pl = self.sm_loss(jn_p.reshape(B * V, F, J * 3), jn_gt.reshape(B * V, F, J * 3),
                              1 - vP.view(B * V, F, J * 3))
            gv = torch.cat([jn_gt[:, :, 1:] - jn_gt[:, :, :-1],
                            (jn_gt[:, :, 1:] - jn_gt[:, :, :-1])[:, :, -1:]], dim=2)
            ga = torch.cat([gv[:, :, 1:] - gv[:, :, :-1], gv[:, :, -1:]], dim=2)
            vm = torch.cat([vV[:, :, :-1] * vV[:, :, 1:], vV[:, :, -1:]], dim=2)
            am = torch.cat([vV[:, :, :-2] * vV[:, :, 1:-1] * vV[:, :, 2:],
                            vV[:, :, -1:], vV[:, :, -1:]], dim=2)
            vl = (self.l1(vp, gv) * vm).sum() / (vm.sum() + 1e-8)
            al = (self.l1(ap, ga) * am).sum() / (am.sum() + 1e-8)
            d = (ji * vV - jg * vV); ie = torch.sqrt((d * d).sum(-1)).sum() / (vV.sum() + 1e-8)
            d = (jp * vV - jg * vV); re = torch.sqrt((d * d).sum(-1)).sum() / (vV.sum() + 1e-8)
            return ({'pd_joint_xyz': jp},
                    {'pos': pl, 'vel': vl, 'acc': al, 'total': pl + self.w_vel * vl + self.w_accel * al},
                    {'init': ie, 'refine': re})
        return {'pd_joint_xyz': jp, 'vel_pred': vp, 'acc_pred': ap}
