# HTD — Hand Temporal Denoising

> **从单/多视角手势估计结果中，对 3D 手部关键点序列进行时序去噪/平滑**

## 目录

1. [项目概述](#1-项目概述)
2. [项目结构](#2-项目结构)
3. [技术方案](#3-技术方案)
4. [论文情况](#4-论文情况)
5. [依赖与环境](#5-依赖与环境)
6. [数据准备](#6-数据准备)
7. [训练](#7-训练)
8. [测试与评估](#8-测试与评估)
9. [现有结果](#9-现有结果)
10. [运行流程速查](#10-运行流程速查)

---

## 1. 项目概述

**HTD** 是一个面向**3D 手部关键点序列的时序去噪/细化**项目。它的核心任务很明确：

- **输入**：一个长度为 `T` 帧的 3D 手部关节序列（21 个关键点 × xyz），每帧由某个单帧手势估计模型（如 HaMeR、其他 off-the-shelf 方法）产出，往往带有**抖动、噪声、遮挡导致的跳变**。
- **输出**：一个同样长度的、更平滑准确的 3D 关节序列。

这是一个 **sequence-to-sequence 的时序细化（temporal refinement）** 任务，利用时序上下文信息来修正单帧估计的误差。

### 与项目其他目录的关系

```
hand_3d_reconstruction/
├── HTD/                  ← 本项目（时序去噪核心）
├── HTDrefine/            ← 调优/变体版本
├── my_hand_playground/   ← 备用的学习/实验代码
├── models/               ← 预训练模型权重
└── 总体规划.md            ← 开发时间表
```

---

## 2. 项目结构

```
HTD/
├── seq_train.py          # 核心入口：训练/测试/可视化/野榜测试
├── seq_config.py         # 所有配置参数（骨干网络、序列长度、学习率等）
│
├── model/
│   ├── FusionFormer.py   # 主模型定义（FusionModel）
│   ├── DSF.py            # DSTFormer 骨干（Dual Spatio-Temporal Transformer）
│   ├── loss.py           # SmoothNetLoss, SmoothL1Loss
│   └── drop.py           # DropPath 实现
│
├── dataset/
│   ├── seqhand.py        # SeqHand / SeqHandTest 数据集类（NPZ 高效版）
│   ├── data_aug.py       # 数据增强：序列旋转、逐帧增强等
│   ├── create_data.py    # 原始 InterHand2.6M → NPZ 预处理工具
│   ├── process_dexycb.py # DexYCB 数据预处理
│   ├── video_interhand.py / video_interhand_gen.py  # 旧版基于视频的数据加载
│   └── dataset_utils.py  # 通用工具
│
├── utils/
│   ├── config.py         # yacs 配置工具
│   ├── defaults.yaml     # yacs 默认参数
│   ├── visualize.py      # 2D 骨架绘制 + 3D 网格渲染
│   ├── video_utils.py    # 视频处理工具
│   ├── fix_seed.py       # 随机种子
│   └── logger.py         # 日志工具
│
├── data/SeqHand/
│   └── npz_30/           # 预处理后的数据集（NPZ 格式）
│       ├── InterHand_train/  # 训练集（~1.4GB gt_joint.npz）
│       ├── InterHand_test/   # 测试集
│       ├── InterHand_val/    # 验证集
│       ├── DexYCB/           # DexYCB 数据集
│       ├── ReInterHand/      # Re:InterHand 数据集
│       ├── UmeTrack_synthetic/
│       └── UmeTrack_real/
│
├── runs/                 # TensorBoard 日志
├── checkpoint/           # 模型保存目录
│   └── InterHand_train-train-MotionBERT-Seq15-Window27/
│       ├── checkpoint/latest.pth  # 最新模型
│       ├── checkpoint/best.pth    # 最佳模型（验证集最优）
│       └── log/                   # 训练日志
│
└── .gitignore
```

---

## 3. 技术方案

### 整体流程

```
输入: 抖动 3D 关键点序列 [B, V, F, J, 3]
        │
        ▼
   归一化（减均值 / 300）
        │
        ▼
   ┌─────────────────────────────┐
   │  FusionModel 时序细化网络    │
   │  ┌───────────────────────┐  │
   │  │ DSTFormer 骨干         │  │  ← 空间-时间双分支 Transformer
   │  │ (MotionBERT 架构)      │  │
   │  └───────────────────────┘  │
   │  ┌───────────────────────┐  │
   │  │ EncoderDecoder ×2      │  │  ← 全局时空注意力编码器-解码器
   │  └───────────────────────┘  │
   │  ┌───────────────────────┐  │
   │  │ 预测头 (MLP → J×3)     │  │  ← 输出每个关节的 3D 坐标
   │  └───────────────────────┘  │
   └─────────────────────────────┘
        │
        ▼
   反归一化（* 300 + 均值）
        │
        ▼
输出: 细化 3D 关键点序列 [B, V, F, J, 3]
```

### 核心模块

#### DSTFormer (Dual Spatio-Temporal Transformer)

来自 MotionBERT [Zhu et al., 2023] 的双分支 Transformer：

- **Spatial 分支**：同一帧内关节间的空间注意力
- **Temporal 分支**：同一关节跨帧的时间注意力
- **自适应融合**：可学习的权重 α 动态平衡空间和时间特征
- 输入 → Joint Embedding + Positional Embedding + Temporal Embedding

#### EncoderDecoder (FusionFormer 中的核心模块)

- **编码器**：空间-时间联合注意力（st_embed），对 `[V, F]` 维度的联合建模
- **解码器**：纯时序注意力（t_embed）+ 残差连接
- 该设计对多视角（`V` 个视角）和时序帧（`F` 帧）做统一处理

#### 损失函数

- **SmoothNetLoss**：`w_accel * L1(加速度) + w_pos * L1(位置)`，惩罚时序不平滑
- 联合位置损失 + 可选的骨骼长度损失（SmoothL1Loss）

### 支持的骨干网络

| 名称 | 代码实现 | 状态 |
|------|---------|------|
| MotionBERT | DSTFormer | ✅ 默认使用 |
| DDN | FusionModel | ✅ 可选 |
| TCN | — | 预留 |
| MotionAGFormer | — | 预留 |
| SmoothNet | — | 预留 |

---

## 4. 论文情况

**本项目没有对应的独立论文。** 它是对现有方法的组合与复现：

| 参考工作 | 来源 | 在本项目中的角色 |
|---------|------|----------------|
| **MotionBERT**[Zhu et al., 2023] | CVPR 2023 | 骨干网络 DSTFormer 的直接引用 |
| **SmoothNet**[Zeng et al., 2023] | CVPR 2023 | 时序平滑损失函数 |
| **MANO**[Romero et al., 2017] | SIGGRAPH 2017 | 手部参数化模型（相关工具代码） |
| **InterHand2.6M**[Moon et al., 2020] | ECCV 2020 | 主要训练/测试数据集 |
| **DexYCB**[Chao et al., 2021] | CVPR 2021 | 辅助数据集 |
| **Re:InterHand** | — | 辅助数据集 |
| **UmeTrack** | — | 辅助数据集 |

项目 GitHub 仓库：[https://github.com/Chenqi-miao/HTD](https://github.com/Chenqi-miao/HTD)

---

## 5. 依赖与环境

### 核心依赖

| 包 | 版本 |
|---|------|
| Python | 3.9+（根据 .pyc 文件判断） |
| PyTorch | 2.13.0 |
| numpy | 2.5.1 |
| einops | 需要安装 |
| opencv-python | 需要安装 |
| yacs | 需要安装 |
| tensorboard | 需要安装 |
| tqdm | 需要安装 |
| scipy | 1.18.0 ✅ 已安装 |
| matplotlib | 3.11.0 ✅ 已安装 |
| trimesh | 4.12.2 ✅ 已安装 |
| pyrender | 需要安装（仅可视化需要） |

### 环境安装

```bash
pip install torch numpy einops opencv-python yacs tensorboard tqdm scipy matplotlib trimesh pyrender
```

### 训练硬件参考

从训练日志看：
- 训练数据约 5580 batch/epoch，batch_size=64 → 约 35.7 万样本/epoch
- 每个 epoch 约 2-3 分钟 → 推测使用了一张较高端 GPU（如 RTX 3090/4090）
- 40 个 epoch 总训练时间约 2 小时

---

## 6. 数据准备

### 预处理数据格式

项目使用 **NPZ 格式**的预处理数据，位于 `data/SeqHand/npz_30/`，每个子数据集下包含：

| 文件 | 内容 | 形状 |
|------|------|------|
| `gt_joint.npz` | 真实 3D 关节坐标（字典结构，每个 key 对应一个序列） | [T, 21, 3] |
| `in_joint.npz` | 输入（含噪声的）3D 关节坐标 | [T, 21, 3] |
| `gt_joint_valid.npz` | 真实值有效标志 | [T, 21] |
| `in_joint_valid.npz` | 输入有效标志 | [T, 21] |
| `hand_type.npz` | 手型（0=右手, 1=左手） | [T] |
| `img_name.npz/txt` | 图像文件名 | — |

### 数据来源

输入数据来自各种单帧手势估计方法（在 `data/SeqHand/pkl/` 目录的原始版本中，每个序列下按 `method_name` 存放不同方法的估计结果）。

预处理脚本参考：
- `dataset/create_data.py` — 从原始 InterHand2.6M 创建序列数据
- `dataset/process_dexycb.py` — DexYCB 数据预处理

### 数据集统计（以 NPZ 30 帧版本为例）

**InterHand_train**（当前训练集）：
- `gt_joint.npz`: ~1.4GB
- `in_joint.npz`: ~1.2GB
- 训练样本量：约 35.7 万序列片段（每个 15 帧）

---

## 7. 训练

### 训练命令

```bash
cd HTD
python seq_train.py
```

### 配置文件 (`seq_config.py`)

```python
cfg.phase = 'train'           # 模式: train / test / vis / wild
cfg.backbone = 'MotionBERT'   # 骨干网络
cfg.data_list = ['InterHand_train']  # 训练数据集
cfg.seq_len = 15              # 序列长度 T
cfg.window_size = 27          # 全局时序窗口
cfg.view_num = 1              # 视角数量（1=单视角）
cfg.batch_size = 64
cfg.lr = 1e-4                 # 学习率
cfg.total_epoch = 40          # 总轮数
cfg.lr_scheduler = 'cosine'   # 余弦退火
cfg.eval_interval = 2         # 每2轮评估一次
```

### 关键配置项说明

| 参数 | 含义 | 当前值 |
|------|------|--------|
| `seq_len` | 输入/输出的时序窗口大小 | 15 帧 |
| `window_size` | 全局时序注意力的感受野 | 27 帧 |
| `view_num` | 多视角输入数量（1=只用单视角） | 1 |
| `global_temporal` | 是否使用全局时序建模模块 | True |
| `loader_resample` | 每轮训练后重新采样序列 | True |
| `joint_num` | 手部关键点数量 | 21 |
| `only_joint` | 是否只预测关节（不预测网格） | True |

### 数据增强

- `data_aug.py` 中的 `Augmenter`：
  - **seq_rotation**：对整个序列做随机旋转
  - **seq_part_rotation**：对序列的不同时间段做分段旋转
  - **frame_aug**：逐帧随机旋转/平移

---

## 8. 测试与评估

### 测试

```bash
# 在测试集上评估
cd HTD
# 修改 seq_config.py: cfg.phase = 'test', cfg.data_list = ['InterHand_test']
python seq_train.py

# 可视化结果（保存骨架图）
# 修改 seq_config.py: cfg.phase = 'vis'
python seq_train.py

# 野榜测试（从 seq.json 读取自定义输入）
# 修改 seq_config.py: cfg.phase = 'wild'
python seq_train.py
```

### 评估指标

- **MPJPE** (Mean Per Joint Position Error)：关节平均位置误差（毫米）
  - **Init MPJPE**：输入（原始估计）的误差
  - **Refine MPJPE**：模型输出（细化后）的误差
- 计算公式：$\text{MPJPE} = \frac{1}{N} \sum \| \text{pred} - \text{gt} \|_2$

---

## 9. 现有结果

从训练日志 `InterHand_train-train-MotionBERT-Seq15-Window27/log/` 中提取：

| 指标 | 初始值 (Epoch 0) | 最佳值 (Epoch 38) | 改善幅度 |
|------|-----------------|-------------------|---------|
| Init MPJPE | 13.64 mm | 13.64 mm | —（固定输入误差） |
| **Refine MPJPE** | **10.66 mm** | **9.398 mm** | **↓ 31%** |

Refine MPJPE 随训练轮次的变化：

```
Epoch  2: 10.11 mm
Epoch  4: 10.37 mm
Epoch  6: 10.06 mm
Epoch  8:  9.74 mm
Epoch 10:  9.83 mm
Epoch 12:  9.63 mm
Epoch 14:  9.58 mm
Epoch 16:  9.57 mm
Epoch 18:  9.48 mm
Epoch 20:  9.62 mm
Epoch 22:  9.36 mm   ← 最佳
Epoch 24:  9.50 mm
Epoch 26:  9.47 mm
Epoch 28:  9.34 mm
Epoch 30:  9.54 mm
Epoch 32:  9.41 mm
Epoch 34:  9.45 mm
Epoch 36:  9.41 mm
Epoch 38:  9.40 mm
```

→ **最佳 Refine MPJPE 约 9.34 mm**（相对初始 13.64 mm 降低了约 30%+）

---

## 10. 运行流程速查

### 第一阶段：环境与数据

```bash
# 1. 安装依赖
pip install torch einops opencv-python yacs tensorboard tqdm scipy matplotlib

# 2. 进入项目
cd /home/chenqi/workspace/hand_3d_reconstruction/HTD

# 3. 确认数据存在
ls data/SeqHand/npz_30/InterHand_train/
# 应包含: gt_joint.npz in_joint.npz 等
```

### 第二阶段：训练

```bash
# 默认训练配置: MotionBERT 骨干, seq_len=15, 40 epochs
python seq_train.py
```

> 修改配置直接编辑 `seq_config.py`，无需命令行参数。

### 第三阶段：测试

```bash
# seq_config.py 中设置:
#   cfg.phase = 'test'
#   cfg.data_list = ['InterHand_test']
#   cfg.checkpoint = 'InterHand_train-train-MotionBERT-Seq15-Window27/checkpoint/best.pth'
python seq_train.py
```

### 可视化

```bash
# seq_config.py 中设置:
#   cfg.phase = 'vis'
python seq_train.py
# 结果保存到 checkpoint/InterHand_train-train-MotionBERT-Seq15-Window27/vis/
# 输出：joint_init_%d.png, joint_pd_%d.png, joint_gt_%d.png
```

### 对自定义数据做推理（Wild 模式）

```bash
# 1. 准备 seq.json（格式：T×21×3 的关节坐标列表）
# 2. seq_config.py 中设置 cfg.phase = 'wild'
# 3. 运行
python seq_train.py
```

---

> **注意**：本项目不包含单帧手势估计（如 HaMeR）的部分。它假设你已经有了每帧的 3D 关键点估计结果（无论来自何种方法），只负责**对时序序列做去噪/细化**。
