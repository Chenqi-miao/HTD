# MotionEncoder — ViTPose+ 风格运动编码器设计文档

> 本文档描述 MotionEncoder + VideoMotionEncoder 的架构设计、数据流、与 HTD 的集成方式以及当前状态。
> 
> **两个版本共存**：
> - `MotionEncoder`：输入 3D 关节坐标 [B, F, J, 3]（Phase 1 原型）
> - `VideoMotionEncoder`：输入视频帧 [B, F, 3, H, W]（完整 ViTPose 风格）

---

## 目录

1. [设计动机](#1-设计动机)
2. [整体架构](#2-整体架构)
3. [模块详解](#3-模块详解)
4. [数据流](#4-数据流)
5. [与 HTD 的集成（Phase 2）](#5-与-htd-的集成phase-2)
6. [当前状态与配置](#6-当前状态与配置)
7. [中间张量一览](#7-中间张量一览)
8. [参考文献](#8-参考文献)

---

## 1. 设计动机

### 问题

HTD 当前使用 `SmoothNetLoss` 做时序细化，但其加速度项只做**隐式平滑约束**（要求预测轨迹的二阶差分为 0），缺少对真实运动规律的显式建模：

| 当前 (HTD) | 问题 |
|-----------|------|
| 输入：抖动 3D 关键点 | 只用了 3D 坐标单模态 |
| Loss：SmoothNetLoss | 加速度惩罚 → 平滑但不区分"噪声"和"真实运动" |
| Refine Jitter (2.60) < GT Jitter (3.42) | **过度平滑**，抹掉了真实手部运动 |

### 解决方案

增加一个 ViTPose+ 风格的运动编码器，从 3D 关节序列中同时提取 **2D 关键点、3D 速度、3D 加速度** 三路输出：

- **2D 关键点 head** → 提供空间定位约束（未来可接入图像投影）
- **3D 速度 head** → 约束时序一阶运动（方向 + 速率）
- **3D 加速度 head** → 约束时序二阶运动（物理平滑，保留真实运动细节）

### 两阶段策略

| 阶段 | 内容 | 目标 |
|------|------|------|
| **Phase 1**（当前） | 单独训练 MotionEncoder | 验证架构可行，loss 下降 |
| **Phase 2**（后续） | 接入 HTD 作为额外 loss | 提升 HTD 细化质量，解决过度平滑 |

---

## 2. 整体架构

### 2.1 MotionEncoder（输入 3D 关节坐标）

```
输入: 3D 关节序列 [B, F, J=21, C=3]
  │
  ├── JointEmbedding → [B, F, D=256]
  ├── TemporalTransformerDecoder × 8 (RoPE)
  ├── Head 1: KeypointDecoder (MLP) → [B, F, J, 2]
  ├── Head 2: VelocityDecoder → [B, F, J, 3]
  └── Head 3: AccelDecoder → [B, F, J, 3]
```

### 2.2 VideoMotionEncoder（输入视频帧）

```
输入: 视频帧序列 [B, F, 3, H=256, W=256]
  │
  ├── ViTEncoder (per-frame, 共享权重)
  │    ├── PatchEmbed (16×16) → [B*F, N=256, D_vit]
  │    ├── CLS token + PosEmbed
  │    └── N × Transformer Block
  │         ├── CLS tokens   → [B, F, D_vit]    (MLP 路径 + 运动头)
  │         └── Patch tokens → [B, F, N, D_vit] (热图路径)
  │
  ├── 特征压缩 (Linear: D_vit → D=256)
  ├── TemporalTransformerDecoder × 8 (RoPE)
  ├── Head 1a: KeypointDecoder2D (MLP, from CLS)  → [B, F, J, 2]
  ├── Head 1b: PatchHeatmapDecoder (from Patch)
  │    ├── reshape → [B*F, D, 16, 16]
  │    ├── Deconv ×4 (×2: 16→256)
  │    └── Soft-argmax → [B, F, J, 2]
  ├── Head 2: VelocityDecoder → [B, F, J, 3]
  └── Head 3: AccelDecoder → [B, F, J, 3]
```

两者共享相同的 `TemporalTransformerDecoder`、`VelocityDecoder`、`AccelDecoder` 模块。

### 3.1 RoPE — Rotary Position Embedding

**文件**: `model/rope.py`

与 Learned Positional Encoding 的对比：

| | Learned PE | RoPE |
|---|---|---|
| 位置编码方式 | 可学习参数 | 旋转矩阵（无参数） |
| 序列长度外推 | ❌ 需预设 max_len | ✅ 可任意长度 |
| 相对位置感知 | ❌ 需要额外 bias | ✅ 注意力自然地感知相对位置 |
| 计算开销 | 低 | 与序列长度线性 |
| 多头兼容 | ✅ | ✅ |

RoPE 在注意力计算的 Q·Kᵀ 之前施加：

```python
q = apply_rotary(q, cos, sin)   # [B, n_head, F, D_head]
k = apply_rotary(k, cos, sin)
attn = softmax(q @ kᵀ / √d)     # RoPE 使 Q·Kᵀ 包含相对位置信息
```

Cache 表预计算：`[1, max_len=243, D/2=128]`，在 `RotaryEmbedding.forward()` 中按实际序列长度切片。

### 3.2 TemporalTransformerDecoder

**文件**: `model/temporal_transformer.py`

| 组件 | 实现 |
|------|------|
| Attention | `TemporalSelfAttention`：QKV projection → RoPE → Scaled Dot-Product |
| MLP | GELU 激活 + Dropout（与 DSF.py 一致） |
| Normalization | Pre-LayerNorm |
| Regularization | Dropout + Stochastic Depth (DropPath) |
| Depth | 8 层（可配置） |
| Heads | 8（可配置） |

与 DSF.py 中 DSTFormer 的区别：

| | DSTFormer | TemporalTransformerDecoder |
|---|---|---|
| 注意力维度 | 空间 + 时序（双分支） | **纯时序** |
| 输入形状 | [B, F, J, D] | [B, F, D] |
| 位置编码 | Learned temp_embed + pos_embed | **RoPE** |
| 融合方式 | 自适应 α 权重融合 | 单分支 |

### 3.3 KeypointDecoder

两种模式：

**原型模式**（`use_heatmap=False`，当前默认）：

```python
self.proj = nn.Sequential(
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 21 * 2),  # 直接回归 (x, y)
)
```

**热图模式**（`use_heatmap=True`，图像接入后使用）：

```python
FC(256 → 64×64) → ConvTranspose2d ×2 (64→128→256) → Heatmap → Soft-argmax
```

### 3.4 VelocityDecoder / AccelDecoder

结构对称，共享 `_TemporalConvBlock` 子模块：

```python
self.conv_block = _TemporalConvBlock(dim)        # Conv1D 3×1 + residual
self.transformer = TemporalTransformerDecoder×2   # 小 Transformer
self.proj = MLP(dim → J×3)                        # 投影到速度/加速度
```

两个 head 各自独立参数，不共享权重。

### 3.5 ViTEncoder — 帧级图像编码器

**文件**: `model/vit_encoder.py`

对视频的每帧独立编码，输出 CLS token + patch tokens 两路特征。

```
PatchEmbed (16×16 conv) → [B*F, N=256, D]
  → CLS token + 可学习 PosEmbed
    → N × Transformer Block (Pre-LN, GELU)
      → CLS token:   [B, F, D]      (全局帧特征)
      → Patch tokens: [B, F, N, D]   (空间特征，用于热图)
```

| 配置 | embed_dim | depth | heads | 参数量 |
|------|-----------|-------|-------|--------|
| `base` | 768 | 12 | 12 | ~85M |
| `small` | 384 | 6 | 6 | ~22M |

帧间共享权重：每次将 `[B, F]` 展平为 `[B*F]` 批量编码，再 reshape 回来。

### 3.6 PatchHeatmapDecoder — 热图解码器

**文件**: `model/motion_encoder.py` (class `PatchHeatmapDecoder`)

从 ViT 的 patch tokens 重建空间热图：

```
Patch tokens [B, T, N=256, D=384/768]
  → LayerNorm + reshape → [B*T, D, 16, 16]
  → Deconv ×4 (×2 up each: 16→32→64→128→256)
  → Heatmap [B*T, J, 256, 256]
  → Softmax over spatial → Soft-argmax
  → 2D coordinates [B, T, J, 2]
```

支持与 MLP 版 2D 头同时使用，也可单独启用。

---

## 4. 数据流

### 4.1 训练数据（Phase 1）

**文件**: `dataset/seqhand_motion.py`

`SeqHandMotion` 继承 `SeqHand` 的 NPZ 加载逻辑，额外在线计算伪 GT：

```
NPZ 坐标
  ├── inputs['joint_xyz']    抖动 3D [1, 15, 21, 3]      ← MotionEncoder 的输入
  ├── targets['joint_xyz']   GT 3D   [1, 15, 21, 3]
  │
  ├── pseudo_gt['joint_2d']   GT 取 (x, y)  [1, 15, 21, 2]    ← 2D 监督
  ├── pseudo_gt['joint_vel']  GT 差分      [1, 14, 21, 3]    ← 速度监督
  ├── pseudo_gt['joint_accel']GT 二阶差分   [1, 13, 21, 3]    ← 加速度监督
  │
  └── masks['vel']            有效帧标记    [1, 14, 21, 1]    ← mask 掉不连续帧
      masks['accel']          有效帧标记    [1, 13, 21, 1]
      masks['kp']             有效关节标记   [1, 15, 21, 1]
```

伪 GT 计算逻辑：

```python
vel_gt    = J_gt[:, 1:] - J_gt[:, :-1]           # 速度：位置差分
accel_gt  = vel_gt[:, 1:] - vel_gt[:, :-1]        # 加速度：速度差分
# mask 规则：帧间位移 > 30mm 标记为不连续，不参与 loss
```

### 4.2 与 HTD 共享什么

| | HTD 的 SeqHand | MotionEncoder 的 SeqHandMotion |
|---|---|---|
| NPZ 文件 | 相同 | 相同 |
| 序列切分 | 随机起始，滑窗 | 随机起始，滑窗 |
| 数据增强 | ✅ seq_aug + joint_aug | 当前无（计划加入） |
| 返回字段 | 3 个 (inputs, targets, meta) | 5 个 (inputs, targets, meta, pseudo, masks) |

---

## 5. 与 HTD 的集成（Phase 2）

### 集成方式

MotionEncoder 作为冻结的特征提取器，其输出作为额外的 loss 项：

```python
# 在 FusionModel.forward() 里新增

# 冻结的 motion encoder
with torch.no_grad():
    motion_out = motion_encoder(joints_in)  # [B, V, F, J, 3] → dict

# 从预测坐标算速度/加速度
pred_vel = joints_pred[:, :, 1:] - joints_pred[:, :, :-1]
pred_accel = pred_vel[:, :, 1:] - pred_vel[:, :, :-1]

# 额外 loss
vel_loss = F.l1_loss(pred_vel * vel_mask, motion_out['joint_vel'] * vel_mask)
accel_loss = F.l1_loss(pred_accel * accel_mask, motion_out['joint_accel'] * accel_mask)

loss = {
    'joint_loss': joint_loss,              # 原有 SmoothNetLoss
    'vel_loss': w_vel * vel_loss,          # 新增
    'accel_loss': w_accel * accel_loss,    # 新增
}
```

### Loss 组合效果

| Loss 项 | 约束目标 | 与 SmoothNetLoss 的关系 |
|---------|---------|----------------------|
| `SmoothNetLoss.w_pos * L1(pos)` | 位置精度 | 基础 |
| `SmoothNetLoss.w_accel * L1(accel)` | **隐式**平滑（加速度→0） | 可能会过度平滑 |
| `vel_loss` | 速度匹配 **真实**运动 | 互补：防止速度被过度抹平 |
| `accel_loss` | 加速度匹配 **真实**运动 | 互补：保留自然抖动 |

关键洞察：SmoothNetLoss 的加速项推着加速度 → 0，MotionEncoder 的加速项推着加速度 → **GT 的加速度**。两者平衡点就是最优解。

---

## 6. 当前状态与配置

### 已交付

| 文件 | 说明 | 验证 |
|------|------|------|
| `model/rope.py` | RoPE 旋转位置编码 | ✅ |
| `model/temporal_transformer.py` | 8 层时序 Transformer Decoder | ✅ |
| `model/motion_encoder.py` | MotionEncoder + VideoMotionEncoder + Loss | ✅ |
| `model/vit_encoder.py` | ViT 帧级图像编码器 (base/small) | ✅ |
| `dataset/seqhand_motion.py` | 运动数据加载器（3D 坐标版） | ✅ |

### 验证过的形状

**MotionEncoder（3D 坐标输入）**：
```python
x = torch.randn(2, 15, 21, 3)
out = MotionEncoder(...)(x)
out['joint_2d'].shape     # → [2, 15, 21, 2]  ✓
out['joint_vel'].shape    # → [2, 15, 21, 3]  ✓
out['joint_accel'].shape  # → [2, 15, 21, 3]  ✓
out['feat'].shape         # → [2, 15, 256]    ✓
```

**VideoMotionEncoder（视频帧输入）**：
```python
x = torch.randn(2, 4, 3, 256, 256)
out = VideoMotionEncoder(...)(x)
out['joint_2d'].shape           # → [2, 4, 21, 2]  ✓
out['joint_2d_heatmap'].shape   # → [2, 4, 21, 2]  ✓
out['joint_vel'].shape          # → [2, 4, 21, 3]  ✓
out['joint_accel'].shape        # → [2, 4, 21, 3]  ✓
out['feat'].shape               # → [2, 4, 256]    ✓
```

### 未交付（Phase 2）

- Phase 2 的 HTD 集成代码
- 训练脚本
- 真实视频/图像数据加载（VideoMotionEncoder 目前只能用随机输入验证）

---

## 7. 中间张量一览

### MotionEncoder（3D 坐标版）

| 模块 | 输入 | 输出 | 参数数 |
|------|------|------|--------|
| JointEmbedding | [B, F, 63] | [B, F, 256] | ~16K |
| TemporalTransformerDecoder×8 | [B, F, 256] | [B, F, 256] | ~5.3M |
| KeypointDecoder (MLP) | [B, F, 256] | [B, F, 21, 2] | ~66K |
| VelocityDecoder | [B, F, 256] | [B, F, 21, 3] | ~1.5M |
| AccelDecoder | [B, F, 256] | [B, F, 21, 3] | ~1.5M |
| **合计** | | | **~8.4M** |

### VideoMotionEncoder（视频版，ViT-S + 热图）

| 模块 | 输入 | 输出 | 参数数 |
|------|------|------|--------|
| ViTEncoder (small) | [B*F, 3, 256, 256] | [B, F, 384] + [B, F, N, 384] | ~22M |
| vit_proj | [B, F, 384] | [B, F, 256] | ~98K |
| TemporalTransformerDecoder×8 | [B, F, 256] | [B, F, 256] | ~5.3M |
| PatchHeatmapDecoder | [B, T, 256, 384] | [B, T, 21, 2] | ~9.5M |
| KeypointDecoder (MLP) | [B, F, 256] | [B, F, 21, 2] | ~66K |
| VelocityDecoder | [B, F, 256] | [B, F, 21, 3] | ~1.5M |
| AccelDecoder | [B, F, 256] | [B, F, 21, 3] | ~1.5M |
| **合计** | | | **~39M** |

---

## 8. 参考文献

| 工作 | 来源 | 参考部分 |
|------|------|---------|
| **ViTPose**: Simple Vision Transformer for Human Pose Estimation | NeurIPS 2022 | Keypoint Decoder 设计（Deconv → Heatmap → Argmax） |
| **RoFormer**: Enhanced Transformer with Rotary Position Embedding | arXiv 2021 | RoPE 实现 |
| **MotionBERT**: A Unified Pretraining Paradigm for Human Motion Analysis | CVPR 2023 | DSTFormer, MLP, DropPath, trunc_normal_ |
| **HTD-Refine**: Framework with Velocity and Acceleration Heads | CVPR 2026 Oral | 速度/加速度解码器设计理念，两阶段训练策略 |
| **SmoothNet**: A Plug-and-Play Network for Refining Human Poses | CVPR 2023 | SmoothNetLoss 参考 |
