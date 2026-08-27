"""骨向量 + RoPE Transformer (审计修正版) 模型.

相对 exp2 的修正 (审计结论, 见对话):
  1. 去掉 joint_attn_pre (消融实测有害);
  2. 骨向量输入 + root 流 (根相对 → 去 DC, 无 GT 中心泄露);
  3. 位置输出 + 中心差分得 v/a (构造性一致, 删独立 vel/acc 头);
  4. 去掉裸差分输入通道 (只喂坐标, 时序模型自己学运动学);
  5. depth/width 参数化, 供规模消融.

架构:
  骨提取 M → 手尺寸归一化 → 20 骨 + root 共 21 token
  → joint_embed(3→D) → RoPE TemporalTransformer (per-token 时序)
  → joint_attn_post (跨 token) → 局部窗口 Conv1d 读出位置
  → 重建位置 → 中心差分得 v/a.

输入: [B, 1, F, J, 3] 带噪 3D 关节 (mm).
输出(训练): (o, l, e); 推理: o = {'pd_joint_xyz': [B,1,F,J,3], ...}.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

_here = os.path.dirname(os.path.abspath(__file__))          # experiments/common
if _here not in sys.path:
    sys.path.insert(0, _here)
if os.path.dirname(_here) not in sys.path:
    sys.path.insert(0, os.path.dirname(_here))

from common.hand_topology import EDGES, METACARPAL_EDGES, build_M, build_A
from common.temporal_transformer import TemporalTransformerDecoder


class JointAttention(nn.Module):
    """关节/骨骼维自注意力 + 身份嵌入 (保留每 token 动力学差异), 多层. x: [B,F,J,D]."""
    def __init__(self, dim=256, num_heads=8, num_tokens=21, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([nn.LayerNorm(dim),
                           nn.MultiheadAttention(dim, num_heads, batch_first=True)])
            for _ in range(n_layers)])
        self.tok_emb = nn.Parameter(torch.randn(num_tokens, dim) * 0.02)

    def forward(self, x):                             # [B, F, J, D]
        B, F, J, D = x.shape
        xe = x + self.tok_emb[None, None, :, :]
        xr = xe.reshape(B * F, J, D)
        for norm, attn in self.layers:
            xn = norm(xr)
            out, _ = attn(xn, xn, xn)
            xr = xr + out
        return xr.reshape(B, F, J, D)


class BoneTransformer(nn.Module):
    def __init__(self, num_frame=31, num_joints=21, dim_feat=256, depth=8, num_heads=8,
                 k=2, w_vel=0.1, w_accel=0.05, joint_attn_post=2):
        """k: 位置解码器的局部窗口半径 (kernel = 2k+1)."""
        super().__init__()
        self.num_frame, self.num_joints = num_frame, num_joints
        self.n_bones = len(EDGES)
        self.n_tokens = self.n_bones + 1              # 20 骨 + root
        self.w_vel, self.w_accel = w_vel, w_accel

        self.register_buffer("M", torch.tensor(build_M(EDGES)))      # (20,21)
        self.register_buffer("Ab", torch.tensor(build_A(EDGES)))     # (21,20)

        self.joint_embed = nn.Linear(3, dim_feat)     # 只喂坐标 (3维, 无裸差分)
        self.pos_drop = nn.Dropout(0.1)
        # 无 joint_attn_pre (审计: 有害)
        self.temporal = TemporalTransformerDecoder(
            dim=dim_feat, depth=depth, num_heads=num_heads, mlp_ratio=4,
            drop_rate=0.1, attn_drop_rate=0.1, use_rope=True)
        self.joint_attn_post = JointAttention(dim_feat, 8, self.n_tokens, joint_attn_post)
        # 局部窗口位置读出 (Conv1d over F): kernel=2k+1, 输出位置 (3维/token)
        self.decoder = nn.Conv1d(dim_feat, 3, kernel_size=2 * k + 1, padding=k)

        n = sum(p.numel() for p in self.parameters())
        print(f"[BoneTransformer] bones={self.n_bones} tokens={self.n_tokens} "
              f"dim={dim_feat} depth={depth} k={k} params={n:,}")

    # ─────────────────────────── forward ───────────────────────────

    def forward(self, inputs, targets, meta_info):
        dev = self.M.device
        ji = inputs["joint_xyz"].to(dev)[:, 0]        # [B,F,J,3] 带噪输入
        jg = targets["joint_xyz"].to(dev)[:, 0]       # [B,F,J,3] GT
        B, F, J, _ = ji.shape

        iv = meta_info["joint_in_val"].to(dev)[:, 0]  # [B,F,J]
        gv = meta_info["joint_gt_val"].to(dev)[:, 0]  # [B,F,J]
        cv = meta_info["continuous_val"].to(dev)      # [B,F]

        # ── 骨向量 + root ──
        b = torch.einsum("ej,btjc->btec", self.M, ji)   # [B,F,20,3]
        r = ji[:, :, 0, :]                            # [B,F,3] 腕
        # 手尺寸归一化
        mc = b[:, :, METACARPAL_EDGES, :].norm(dim=-1).mean(dim=2)   # [B,F]
        h = torch.median(mc, dim=1).values.clamp(min=1e-3)          # [B]
        h4 = h.reshape(B, 1, 1, 1)
        b_n = b / h4
        r_n = r / h.reshape(B, 1, 1)

        # ── tokens: 20 骨 + root → [B,F,21,3] ──
        tok = torch.cat([b_n, r_n.unsqueeze(2)], dim=2)

        x = self.joint_embed(tok)                     # [B,F,21,D]
        x = self.pos_drop(x)
        x = x.reshape(B * self.n_tokens, F, -1)       # per-token 时序
        x = self.temporal(x)                          # RoPE 时序 (per-token)
        x = x.reshape(B, F, self.n_tokens, -1)
        x = self.joint_attn_post(x)                   # 跨 token 耦合

        # ── 局部窗口读出位置 (每 token 的坐标, 归一化单位) ──
        xd = x.reshape(B * self.n_tokens, F, -1).permute(0, 2, 1)   # [B*21, D, F]
        pred = self.decoder(xd).permute(0, 2, 1)                    # [B*21, F, 3]
        pred = pred.reshape(B, self.n_tokens, F, 3).permute(0, 2, 1, 3)  # [B,F,21,3]

        b_hat_n = pred[:, :, :self.n_bones, :]
        r_hat_n = pred[:, :, self.n_bones, :]         # [B,F,3]
        p_hat_n = r_hat_n.unsqueeze(2) + torch.einsum("je,btec->btjc", self.Ab, b_hat_n)
        p_hat = p_hat_n * h4                          # mm

        # ── 运动学: 中心差分 (构造性一致) ──
        v_hat = torch.zeros_like(p_hat)
        v_hat[:, 1:-1] = (p_hat[:, 2:] - p_hat[:, :-2]) / 2
        v_hat[:, 0] = p_hat[:, 1] - p_hat[:, 0]
        v_hat[:, -1] = p_hat[:, -1] - p_hat[:, -2]
        a_hat = torch.zeros_like(p_hat)
        a_hat[:, 1:-1] = p_hat[:, 2:] - 2 * p_hat[:, 1:-1] + p_hat[:, :-2]
        a_hat[:, 0] = a_hat[:, 1]
        a_hat[:, -1] = a_hat[:, -2]

        jp = p_hat.unsqueeze(1)
        vp = v_hat.unsqueeze(1)
        ap = a_hat.unsqueeze(1)

        if not self.training:
            return {"pd_joint_xyz": jp, "vel_pred": vp, "acc_pred": ap}

        # ── 损失 (位置 + 一致运动学, 掩码) ──
        cvf = (cv > 0.5).float().reshape(B, 1, F, 1, 1)
        gvf = (gv > 0.5).float().reshape(B, 1, F, J, 1)
        msk = cvf * gvf                                # [B,1,F,J,1]

        def _sl1(a, b, m):
            d = torch.abs(a - b)
            loss = torch.where(d < 1.0, 0.5 * d * d, d - 0.5)
            return (loss * m).sum() / (m.sum() + 1e-8)

        jg_u = jg.unsqueeze(1)
        pl = _sl1(jp, jg_u, msk)

        gv_gt = torch.zeros_like(jg)
        gv_gt[:, 1:-1] = (jg[:, 2:] - jg[:, :-2]) / 2
        gv_gt[:, 0] = jg[:, 1] - jg[:, 0]
        gv_gt[:, -1] = jg[:, -1] - jg[:, -2]
        ga_gt = torch.zeros_like(jg)
        ga_gt[:, 1:-1] = jg[:, 2:] - 2 * jg[:, 1:-1] + jg[:, :-2]
        ga_gt[:, 0] = ga_gt[:, 1]
        ga_gt[:, -1] = ga_gt[:, -2]

        msk_v = msk[:, :, 1:-1] * msk[:, :, :-2] * msk[:, :, 2:]
        vl = _sl1(vp[:, :, 1:-1], gv_gt[:, 1:-1].unsqueeze(1), msk_v)
        al = _sl1(ap[:, :, 1:-1], ga_gt[:, 1:-1].unsqueeze(1), msk_v)

        mpos = msk[:, 0]
        d = ji * mpos - jg * mpos
        ie = torch.sqrt((d * d).sum(-1)).sum() / (mpos.sum() + 1e-8)
        d = jp[:, 0] * mpos - jg * mpos
        re = torch.sqrt((d * d).sum(-1)).sum() / (mpos.sum() + 1e-8)

        total = pl + self.w_vel * vl + self.w_accel * al
        return ({"pd_joint_xyz": jp},
                {"pos": pl, "vel": vl, "acc": al, "total": total},
                {"init": ie, "refine": re})
