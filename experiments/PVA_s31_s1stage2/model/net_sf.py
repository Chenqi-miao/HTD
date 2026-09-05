"""纯空间分组网络 (单帧): 组内 + 组间注意力, 无时序层.

空间为主方向的真正地板 (窗口=1, 无法用时序注意力)。
输入: 单帧位置 [B,V,1,J,3] (无速度/加速度, 单帧算不出帧间差分)。
复用 net.py 的空间模块: GroupJointAttn / CrossGroupAttn / 损失函数。
"""
import torch
import torch.nn as nn

from net import (GroupJointAttn, CrossGroupAttn,
                 joint_angles, bone_lengths, TIP, DIP)


class Outputs(tuple):
    """既是元组(可 o,l,e=model() 解包), 又能 model()[key] 按字符串索引."""
    def __new__(cls, o, l, e):
        return super().__new__(cls, (o, l, e))
    def __getitem__(self, key):
        if isinstance(key, str):
            return self[0][key]
        return super().__getitem__(key)


class Net(nn.Module):
    def __init__(self, num_frame=1, num_joints=21, dim_feat=256, depth=8,
                 w_vel=0.1, w_accel=0.05):
        super().__init__()
        self.num_frame, self.num_joints = num_frame, num_joints
        self.joint_embed = nn.Linear(3, dim_feat)          # 单帧纯位置
        self.pos_drop = nn.Dropout(0.1)
        # 空间块: 组内 + 组间 交错, 无时序
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(GroupJointAttn(dim_feat, 8))
            self.blocks.append(CrossGroupAttn(dim_feat, 8))
        self.pos_head = nn.Linear(dim_feat, 3)
        self.lam_bone = 0.2
        self.l1 = nn.L1Loss(reduction='none')
        p = sum(p.numel() for p in self.parameters())
        print(f'[SingleFrameGrouped] depth={depth} params={p:,}')

    def forward(self, inputs, targets, meta_info):
        ji = inputs["joint_xyz"].cuda()
        jg = targets["joint_xyz"].cuda()
        c = meta_info['center_xyz'].cuda()
        iv = meta_info['joint_in_val'].cuda().reshape(*ji.shape[:4], 1)
        cv = meta_info['continuous_val'].cuda()
        B, V, F, J = ji.shape[:4]
        norm = 300
        # 中心归一化 (MidMCP 加权均值, 同 exp2 口径)
        sc = (c * iv).sum(dim=(2, 3)) / (iv.sum(dim=(2, 3)) + 1e-8)
        sc = sc.reshape(B, V, 1, 1, 3)
        jn_in = (ji - sc) / norm
        jn_gt = (jg - sc) / norm

        x = jn_in.reshape(B, F, J, 3)                      # 单帧纯位置
        x = self.pos_drop(self.joint_embed(x))             # [B,F,J,C]
        for blk in self.blocks:
            x = blk(x)
        jn_p = self.pos_head(x)                            # [B,F,J,3]
        jp = jn_p.reshape(B, V, F, J, 3) * norm + sc       # 反归一化

        # 损失: 分指加权位置 + 骨长
        val = (cv.view(B, 1, F, 1, 1) * meta_info['joint_gt_val'].cuda().view(B, 1, F, J, 1))
        w = torch.ones(1, 1, 1, J, 1, device=ji.device)
        w[:, :, :, TIP, :] = 2.0
        w[:, :, :, DIP, :] = 1.5
        L_pos = (self.l1(jn_p.reshape(B, 1, F, J, 3), jn_gt.reshape(B, 1, F, J, 3)) * val * w).sum() \
            / ((val * w).sum() + 1e-8)
        b_p = bone_lengths(jn_p.reshape(B, F, J, 3))
        b_g = bone_lengths(jn_gt.reshape(B, F, J, 3))
        vf_b = cv.view(B, 1, F, 1).expand(B, 1, F, 20)
        L_bone = (self.l1(b_p, b_g) * vf_b).sum() / (vf_b.sum() + 1e-8)
        L_total = L_pos + self.lam_bone * L_bone

        o = {"pd_joint_xyz": jp}
        l = {"total": L_total, "pos": L_pos, "bone": L_bone}
        d = (ji * val - jg * val)
        ie = torch.sqrt((d * d).sum(-1)).sum() / (val.sum() + 1e-8)
        d = (jp * val - jg * val)
        re = torch.sqrt((d * d).sum(-1)).sum() / (val.sum() + 1e-8)
        e = {"init": ie, "refine": re}
        return Outputs(o, l, e)
