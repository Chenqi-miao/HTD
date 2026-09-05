"""
exp2_s8_att_grouped: 分组时空注意力网络 (空间为主).

设计 (用户确认):
  分组: 5指组(各4关节) + Wrist根token, 不做21关节全局自注意力
  U型: 组内×8 → 组间×4 → 组内×2
    组内层: 关节注意力×2(空间,组内) → 时序×1(RoPE)
    组间层: 组token注意力×2(空间,注意力+广播) → 时序×1(RoPE)
  头: 只位置头 (无 vel/acc)
  损失: L_pos(分指加权) + λ_ang·L_ang(15角,抓欠弯) + λ_bone·L_bone(20骨,抓刚性)

输入: [B,V,F,J,9] 相机空间 (center=MidMCP/300), 与 exp2 一致。
"""
import torch
import torch.nn as nn
import math
from einops import rearrange

FINGERS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
GROUP_JOINTS = [j for g in FINGERS for j in g]      # 20 指关节 (不含 wrist 0)
N_GROUPS, GROUP_SIZE = 5, 4
# 指尖/指中 加权 (TIP/DIP 重灾区)
TIP = [4, 8, 12, 16, 20]
DIP = [3, 7, 11, 15, 19]


def _rot(x, freqs):
    """RoPE: x[.., F, D], freqs[F, D//2]. 逐特征对旋转."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c = freqs.cos().unsqueeze(0)
    s = freqs.sin().unsqueeze(0)
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    return torch.stack([o1, o2], dim=-1).flatten(-2)


class TemporalAttn(nn.Module):
    """时序自注意力 over F (RoPE). x[B,N,F,C] -> [B,N,F,C]."""
    def __init__(self, dim, heads, max_len):
        super().__init__()
        self.dim, self.heads = dim, heads
        self.hdim = dim // heads
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        inv = 10000.0 ** (-torch.arange(0, self.hdim // 2).float() / (self.hdim // 2))
        t = torch.arange(max_len).float()
        self.register_buffer("freqs", torch.outer(t, inv))      # [max_len, hdim//2]
    def forward(self, x, seqlen):
        B, N, F, C = x.shape
        xr = rearrange(x, 'b n f c -> (b n) f c')
        xn = self.norm(xr)
        qkv = self.qkv(xn).reshape(-1, F, 3, self.heads, self.hdim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                        # [BN,H,F,hdim]
        q = _rot(q, self.freqs[:F])
        k = _rot(k, self.freqs[:F])
        attn = (q @ k.transpose(-2, -1)) * (self.hdim ** -0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(-1, F, C)
        xr = xr + self.proj(out)
        return rearrange(xr, '(b n) f c -> b n f c', b=B, n=N)


class GroupJointAttn(nn.Module):
    """组内关节注意力 (空间, 每组4关节自注意力, per frame)."""
    def __init__(self, dim, heads):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
    def forward(self, x):                                       # x[B,F,J,C]
        B, F, J, C = x.shape
        xg = x[:, :, GROUP_JOINTS, :]
        xg = rearrange(xg, 'b f (g k) c -> (b f g) k c', g=N_GROUPS, k=GROUP_SIZE)
        xn = self.norm(xg)
        out, _ = self.attn(xn, xn, xn)
        xg = xg + out
        xg = rearrange(xg, '(b f g) k c -> b f (g k) c', b=B, f=F, g=N_GROUPS)
        x = x.clone()
        x[:, :, GROUP_JOINTS, :] = xg
        return x


class GroupTemporalAttn(nn.Module):
    """组内时序: 每组关节在F帧上 RoPE 时序 (含 wrist)."""
    def __init__(self, dim, heads, max_len):
        super().__init__()
        self.attn = TemporalAttn(dim, heads, max_len)
    def forward(self, x, seqlen):
        B, F, J, C = x.shape
        xg = x[:, :, GROUP_JOINTS, :]
        xg = rearrange(xg, 'b f (g k) c -> b (g k) f c', g=N_GROUPS, k=GROUP_SIZE)   # N=20
        xg = self.attn(xg, seqlen)
        xg = rearrange(xg, 'b (g k) f c -> b f (g k) c', g=N_GROUPS, k=GROUP_SIZE)
        xw = rearrange(x[:, :, 0:1, :], 'b f 1 c -> b 1 f c')
        xw = self.attn(xw, seqlen)
        xw = rearrange(xw, 'b 1 f c -> b f 1 c')
        x = x.clone()
        x[:, :, GROUP_JOINTS, :] = xg
        x[:, :, 0:1, :] = xw
        return x


class CrossGroupAttn(nn.Module):
    """组间空间: pool组→[B,F,6,C] 组token注意力, 广播回 (注意力+广播)."""
    def __init__(self, dim, heads):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
    def forward(self, x):
        B, F, J, C = x.shape
        xg = rearrange(x[:, :, GROUP_JOINTS, :], 'b f (g k) c -> b f g k c', g=N_GROUPS, k=GROUP_SIZE)
        tok = xg.mean(dim=3)                                    # [B,F,5,C]
        tok = torch.cat([tok, x[:, :, 0:1, :]], dim=2)          # [B,F,6,C]
        tok_old = tok
        tokf = rearrange(tok, 'b f n c -> (b f) n c')
        tn = self.norm(tokf)
        out, _ = self.attn(tn, tn, tn)
        tok = rearrange(tokf + out, '(b f) n c -> b f n c', b=B, f=F)
        delta = tok - tok_old                                    # [B,F,6,C]
        x = x.clone()
        xg2 = rearrange(x[:, :, GROUP_JOINTS, :], 'b f (g k) c -> b f g k c', g=N_GROUPS, k=GROUP_SIZE)
        xg2 = xg2 + delta[:, :, :N_GROUPS, :].unsqueeze(3)       # 广播回该组每个关节
        x[:, :, GROUP_JOINTS, :] = rearrange(xg2, 'b f g k c -> b f (g k) c')
        x[:, :, 0:1, :] = x[:, :, 0:1, :] + delta[:, :, 5:6, :]
        return x


class CrossGroupTemporalAttn(nn.Module):
    """组间时序: 6组token在F帧上 RoPE."""
    def __init__(self, dim, heads, max_len):
        super().__init__()
        self.attn = TemporalAttn(dim, heads, max_len)
    def forward(self, x, seqlen):
        B, F, J, C = x.shape
        xg = rearrange(x[:, :, GROUP_JOINTS, :], 'b f (g k) c -> b f g k c', g=N_GROUPS, k=GROUP_SIZE)
        tok = xg.mean(dim=3)
        tok = torch.cat([tok, x[:, :, 0:1, :]], dim=2)
        tok_old = tok
        tokt = self.attn(rearrange(tok, 'b f n c -> b n f c'), seqlen)
        tok = rearrange(tokt, 'b n f c -> b f n c')
        delta = tok - tok_old
        x = x.clone()
        xg2 = rearrange(x[:, :, GROUP_JOINTS, :], 'b f (g k) c -> b f g k c', g=N_GROUPS, k=GROUP_SIZE)
        xg2 = xg2 + delta[:, :, :N_GROUPS, :].unsqueeze(3)
        x[:, :, GROUP_JOINTS, :] = rearrange(xg2, 'b f g k c -> b f (g k) c')
        x[:, :, 0:1, :] = x[:, :, 0:1, :] + delta[:, :, 5:6, :]
        return x


class GroupLayer(nn.Module):
    """组内: 关节注意力×2 → 时序×1."""
    def __init__(self, dim, heads, max_len):
        super().__init__()
        self.j1 = GroupJointAttn(dim, heads)
        self.j2 = GroupJointAttn(dim, heads)
        self.t = GroupTemporalAttn(dim, heads, max_len)
    def forward(self, x, seqlen):
        x = self.j1(x); x = self.j2(x); x = self.t(x, seqlen)
        return x


class CrossLayer(nn.Module):
    """组间: 组token注意力×2 → 时序×1."""
    def __init__(self, dim, heads, max_len):
        super().__init__()
        self.c1 = CrossGroupAttn(dim, heads)
        self.c2 = CrossGroupAttn(dim, heads)
        self.t = CrossGroupTemporalAttn(dim, heads, max_len)
    def forward(self, x, seqlen):
        x = self.c1(x); x = self.c2(x); x = self.t(x, seqlen)
        return x


def joint_angles(X):
    """X[B,F,J,3] -> [B,F,15] 关节角 (rad): 每指 MCP(vs Wrist)/PIP/DIP."""
    outs = []
    for f in FINGERS:
        m, p, d, t = f
        for j, par, chd in ((m, 0, p), (p, m, d), (d, p, t)):
            vi = X[..., j, :] - X[..., par, :]
            vo = X[..., chd, :] - X[..., j, :]
            vi = vi / (vi.norm(dim=-1, keepdim=True) + 1e-8)
            vo = vo / (vo.norm(dim=-1, keepdim=True) + 1e-8)
            cos = torch.clamp((vi * vo).sum(-1), -1, 1)
            outs.append(torch.acos(cos))
    return torch.stack(outs, dim=-1)


def bone_lengths(X):
    """X[B,F,J,3] -> [B,F,20] 骨长."""
    outs = []
    for f in FINGERS:
        m, p, d, t = f
        for a, b in ((0, m), (m, p), (p, d), (d, t)):
            outs.append((X[..., b, :] - X[..., a, :]).norm(dim=-1))
    return torch.stack(outs, dim=-1)


class Outputs(tuple):
    """既是元组(可 o,l,e=model() 解包), 又能 model()[key] 按字符串索引."""
    def __new__(cls, o, l, e):
        return super().__new__(cls, (o, l, e))
    def __getitem__(self, key):
        if isinstance(key, str):
            return self[0][key]          # 按输出 dict 索引
        return super().__getitem__(key)


class Net(nn.Module):
    def __init__(self, num_frame=8, num_joints=21, dim_feat=256, depth=8,
                 w_vel=0.1, w_accel=0.05):
        super().__init__()
        self.num_frame, self.num_joints = num_frame, num_joints
        self.joint_embed = nn.Linear(9, dim_feat)
        self.pos_drop = nn.Dropout(0.1)
        self.group_layers = nn.ModuleList([GroupLayer(dim_feat, 8, num_frame) for _ in range(depth)])
        self.cross_layers = nn.ModuleList([CrossLayer(dim_feat, 8, num_frame) for _ in range(depth // 2)])
        self.refine_layers = nn.ModuleList([GroupLayer(dim_feat, 8, num_frame) for _ in range(depth // 4)])
        self.pos_head = nn.Linear(dim_feat, 3)
        self.lam_bone = 0.2
        self.l1 = nn.L1Loss(reduction='none')
        p = sum(p.numel() for p in self.parameters())
        print(f'[GroupedNet] depth={depth} frames={num_frame} params={p:,}')

    def forward(self, inputs, targets, meta_info):
        ji = inputs["joint_xyz"].cuda()
        jg = targets["joint_xyz"].cuda()
        c = meta_info['center_xyz'].cuda()
        iv = meta_info['joint_in_val'].cuda().reshape(*ji.shape[:4], 1)
        cv = meta_info['continuous_val'].cuda()
        B, V, F, J = ji.shape[:4]
        norm = 300
        sc = (c * iv).sum(dim=(2, 3)) / (iv.sum(dim=(2, 3)) + 1e-8)
        sc = sc.reshape(B, V, 1, 1, 3)
        jn_in = (ji - sc) / norm
        jn_gt = (jg - sc) / norm
        # 9维输入: pos+vel+acc
        in_v = torch.cat([jn_in[:, :, 1:] - jn_in[:, :, :-1], jn_in[:, :, -1:] - jn_in[:, :, -2:-1]], dim=2)
        in_a = torch.cat([in_v[:, :, 1:] - in_v[:, :, :-1], in_v[:, :, -1:] - in_v[:, :, -2:-1]], dim=2)
        x = torch.cat([jn_in, in_v, in_a], dim=-1).reshape(B, F, J, 9)
        x = self.pos_drop(self.joint_embed(x))                  # [B,F,J,C]
        for l in self.group_layers: x = l(x, F)
        for l in self.cross_layers: x = l(x, F)
        for l in self.refine_layers: x = l(x, F)
        jn_p = self.pos_head(x)                                 # [B,F,J,3]
        if __import__('os').environ.get('DBG'):
            print('DBG jn_p', tuple(jn_p.shape), 'sc', tuple(sc.shape), 'x', tuple(x.shape), flush=True)
        jp = jn_p.reshape(B, V, F, J, 3) * norm + sc            # 反归一化 [B,V,F,J,3]

        # ── 损失 ──
        val = (cv.view(B, 1, F, 1, 1) * meta_info['joint_gt_val'].cuda().view(B, 1, F, J, 1))  # [B,1,F,J,1]
        # L_pos: 分指加权 (TIP/DIP 重)
        w = torch.ones(1, 1, 1, J, 1, device=ji.device)
        w[:, :, :, TIP, :] = 2.0
        w[:, :, :, DIP, :] = 1.5
        e_pos = (self.l1(jn_p.reshape(B, 1, F, J, 3), jn_gt.reshape(B, 1, F, J, 3)) * val * w).sum()
        L_pos = e_pos / ((val * w).sum() + 1e-8)
        # L_bone: 20 骨长 (刚性约束)
        b_p, b_g = bone_lengths(jn_p.reshape(B, F, J, 3)), bone_lengths(jn_gt.reshape(B, F, J, 3))
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
