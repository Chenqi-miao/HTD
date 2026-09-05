# HTD_PVA_s31_s1stage2 — 最终方法说明

> **任务**：InterHand 手部 21 关节 3D 关键点去噪（相机空间，输入为含噪关键点）。不仅要求位置准（MPJPE），还要求运动学保真（速度/加速度相关、幅度比）和对高抖动段的鲁棒性。
> **最终方法**：**s1 单帧空间修正 + exp2 时序精修 的级联**（即 `PVA_s31_s1stage2`，31 帧窗）。由「复现 HTDrefine 的时序精修主干（exp2）」+「新增的 s1 单帧纯空间修正」两部分组成。
> **本文以服务器代码为准**：`/home/pfren/pycharm/HTD/experiments/`。

---

## 1. 一句话与定位

```
含噪关键点 ──s1(单帧, 纯空间)──▶ p_s1 ──exp2(31帧, 时序)──▶ 精修位置
            逐帧修位置 DC            |   在干净输入上修运动
```

- **s1**：单帧、纯空间分组注意力网络（无时序层），对每一帧独立修正关节位置的系统性偏差（DC 偏移 / 位置分量）。
- **exp2**：31 帧 PVA 时序网络（关节注意力 + RoPE 时序 Transformer + 位置/速度/加速度三头），在 s1 输出的干净序列上做动力学精修。
- 级联动机：位置误差是主要误差，且注意力对整体平移（DC）不敏感 → **先让空间网络把位置修干净，再让时序网络专注修运动**。两阶段分工，位置和运动误差可分离。

对应 `experiments/` 下的命名（重命名前后均列出）：

| 现目录名（服务器） | 旧目录名 | 角色 |
|---|---|---|
| **PVA_s31_s1stage2** | exp2_s31_s1exp2 | **最终方法（s1 → exp2 级联，31 帧）** |
| PVA_space_att_grouped | exp2_s1_att_grouped | s1 单帧纯空间模型（空间修正） |
| PVA_s31_att | exp2_s30_att | exp2 时序网（复现 HTDrefine 主干，31 帧基线） |
| PVA_s15_stage2 | exp2_s15_stage2 | 同级联的 15 帧版 |
| PVA_s15_spaceDSF | exp2_s15_s1exp1 | 消融变体：s1 → exp1(位置 DSTFormer) |
| DSF_s8/s15/s31 | exp1_* | exp1 位置模型（DSTFormer）基线 |
| PVA_s15_att / PVA_s8_att | exp2_s15_att / exp2_s8_att | exp2 的 15/8 帧版 |

> ⚠️ 注意：目录名里的 `s31/s30` 均指 31 帧窗口（旧目录 `exp2_s30_att` 的 config `seq_len=31`）；`s1` 指 seq_len=1（单帧）。

---

## 2. 设计动机

来自前序分析的三个结论：

1. **位置误差是主要误差**：输入 MPJPE ~13.7 mm，而速度/加速度误差只有 ~2-4 mm（MPJVE/MPJAE）。误差大头在位置，不在运动。
2. **注意力对整体平移不敏感，修不了 DC 偏移**：单独用 exp2（纯时序）时位置几乎不提升（见第 4 节，exp2_s31 的 PA 只 +7%、高抖动桶只 +2.4%）。段级均匀平移在自注意力下没有可区分的特征。
3. **空间上下文是去噪的关键**：早期"单指/隔离"尝试（只修指尖、按运动链级联等）全部失败——指尖定位必须依赖整只手的空间上下文。

因此把"修位置"显式交给一个**单帧、空间**模型（s1），修完再让**时序**模型（exp2）做运动精修，形成"位置 DC 分工"的级联结构。

---

## 3. 方法细节

### 3.1 s1：纯空间单帧分组注意力（`model/net_sf.py`）

- 输入：单帧 21 关节位置 `[B,V,1,J,3]`（相机空间 mm）。**无速度/加速度**（单帧算不出帧间差分）。
- 前处理：以 MidMCP 为中心的中心归一化（`/300`）。
- 结构：`joint_embed: Linear(3→256)` → 8 ×（`GroupJointAttn` 组内关节注意力 + `CrossGroupAttn` 跨组注意力）→ `pos_head: Linear(256→3)`。
  - `GroupJointAttn`：按 5 指分组的组内 4 关节自注意力（空间、残差）。
  - `CrossGroupAttn`：5 指组 token + 腕共 6 个 token 间的注意力，delta 广播回关节。
  - **无任何时序层**——这是纯空间方向的"地板"（窗口=1，无法用时序注意力）。
- 损失：分指加权位置 L1（TIP×2、DIP×1.5）+ `0.2 ×` 骨长约束。
- 输出 `p_s1`（mm），逐帧独立。参数 ~4M。

### 3.2 exp2：31 帧时序精修（`model/exp2_net.py`）

- 输入：s1 输出的 31 帧位置 `p_s1`，前处理再做中心归一化并差分补全出帧间速度/加速度，拼成 **9 维**（pos+vel+acc）。
- 结构：
  - `joint_embed: Linear(9→256)`
  - `joint_attn_pre`：1 层关节维自注意力 + 关节身份嵌入（保留各关节动力学差异）
  - `temporal`：8 层 **RoPE 时序 Transformer Decoder**（按关节分开、共享权重的时序注意力）
  - `joint_attn_post`：2 层关节注意力
  - `pos_head`（位置头）+ `vel_head` / `acc_head`（速度/加速度辅助头，Conv1d 结构）
- 损失：SmoothNet 位置 L1 + 加速度 L1（`w_accel=0.1`）+ vel/acc 辅助头损失（`w_vel=0.1, w_accel=0.05`）。
- 参数 ~7.5M。

### 3.3 两阶段训练（`train_s1exp2.py`）

1. **Stage 1**：单独训练 s1（见 `PVA_space_att_grouped/`，窗口口径 Best ≈ 9.85）→ 保存 `best.pth`。
2. **Stage 2**：**冻结 s1**，在 `no_grad` 下逐帧生成 `p_s1` 作为 exp2 输入；exp2 从 exp2 基线 checkpoint 初始化并**微调**（`lr=1e-4, AdamW, cosine, 30 epochs, batch=16`）。s1 全程不更新。
3. 评估：`SeqHandTest` 31 帧、相机空间、masked MPJPE，每 5 epoch 一次，保存 `best.pth`。

> 复现注意：训练/评估脚本内仍写**旧目录名**（`exp2_s30_att` / `exp2_s1_att_grouped` / `exp2_s31_s1exp2`），当前服务器已重命名为 `PVA_*`，直接跑前需在 `experiments/` 下建软链或改脚本路径。exp2 初始化/评估统一用**原版纯 Transformer 权重 `best_orig_full.pth`**（同一目录下的 `best.pth` 曾为 LRU 混合版，结构与纯 T 不匹配，`strict=False` 加载会产生随机初始化）。

---

## 4. 效果

### 4.1 主表（`SeqHandTest`，2000 窗，相机空间 masked，逐点平均）

第一行输入 = 输入 vs GT 基线；各模型与其同窗长对比即修正量。

| 模型 | 窗长 | MPJPE | MPJVE | MPJAE | PA | corr_v | corr_a | 桶低 | 桶中 | 桶高 |
|---|---|---|---|---|---|---|---|---|---|---|
| 输入(input) | 15 | 13.62 | 4.38 | 6.74 | 9.76 | 0.415 | 0.153 | 9.80 | 11.43 | 19.96 |
| 输入(input) | 31 | 13.66 | 4.34 | 6.69 | 10.58 | 0.414 | 0.150 | 9.63 | 11.34 | 20.72 |
| **s1 单独（空间修正）** | 31 | **9.91** | 3.06 | 4.53 | 8.64 | 0.732 | 0.345 | 7.69 | 9.31 | 12.92 |
| exp1_s15 (DSTFormer, 位置) | 15 | 9.35 | 2.52 | 2.78 | 8.47 | 0.630 | 0.207 | 7.16 | 8.59 | 13.34 |
| exp1_s31 | 31 | 10.68 | 2.57 | 2.84 | 9.62 | 0.643 | 0.207 | 7.68 | 10.12 | 20.04 |
| exp2 单独_s15（时序，复现） | 15 | 9.42 | 2.33 | 2.51 | 8.57 | 0.779 | 0.377 | 7.21 | 8.70 | 13.34 |
| exp2 单独_s31（时序，复现） | 31 | 10.75 | 2.40 | 2.60 | 9.84 | 0.713 | 0.266 | 7.80 | 10.28 | 20.22 |
| **stage2 s15（级联）** | 15 | 9.41 | **2.02** | 2.31 | 7.92 | 0.868 | 0.592 | 7.46 | 8.88 | 12.08 |
| **stage2 s31（级联）★最终** | 31 | **9.08** | **1.85** | **2.11** | **7.82** | **0.880** | **0.629** | **7.12** | **8.60** | **11.96** |

指标口径：MPJPE/MPJVE/MPJAE（masked）、PA-MPJPE（Procrustes 逐窗对齐）、corr_v/corr_a（速度/加速度 Pearson 相关）、抖动三桶 = 按**输入** jerk 的 33.3/66.7 分位（低/中/高）分组后的模型 MPJPE。全程用同脚本 `eval_all_metrics.py`、同一 mask、统一非重叠逐窗协议。

### 4.2 三条关键结论

1. **stage2 s31 全面最强**：位置（9.08）、运动（corr_v/corr_a）、结构（PA 7.82）、抖动分桶全部优于 exp2 原版 s31。**高抖动桶 20.22 → 11.96，几乎减半**——此前一直困扰的"两级分化 / 高抖动桶拖垮整体"被大幅缓解。
2. **级联抹平长窗的位置劣势**：纯 exp2 时 s31(10.75) 明显差于 s15(9.42)；级联后 stage2 s31(9.08) 反超 stage2 s15(9.41)。s1 先把位置修干净，长窗的动力学优势才兑现。
3. **s1（空间修正）是增量来源**：链条（31 帧逐点）：输入 13.66 → s1 单独 9.91（-27.5%）→ stage2_s31 9.08（-33.5%）。s1 单独就把位置/高抖桶修掉大头；exp2 单独_s31 的高抖桶只 +2.4%、PA 只 +7%（几乎不修位置），接上 s1 后变成 **高抖桶 +42.3% / PA +26.1%**——这就是"位置交给 s1、运动交给 exp2"分工的直接证据。

相对输入(31 帧窗)提升：MPJPE **+33.5%**、MPJVE +57.4%、MPJAE +68.5%、PA +26.1%、corr_v +112.6%、corr_a +319.3%、桶低/中/高 +26.1 / +24.2 / **+42.3%**。

> 数字来源：`experiments/stage2_评估结果_20260825.md`（同 `eval_all_metrics.py` 协议）。当前目录 `PVA_s31_att/eval_report.json` 是全量 142930 窗的 exp2_s31 复现评估（输入 13.66 → 10.75，分桶 7.80/10.28/20.22）。

---

## 5. 复现步骤

环境：服务器 `10.112.105.35`，`source /home/pfren/anaconda3/bin/activate pytorch3d`（PyTorch 2.8 + CUDA 12.8 + pytorch3d）。数据：`data/SeqHand`（pkl 管线，TRAIN 173111 窗 / TEST 142930 窗，评估默认取 2000）。

```bash
cd /home/pfren/pycharm/HTD/experiments

# ① 训 s1（单帧纯空间）
cd PVA_space_att_grouped && CUDA_VISIBLE_DEVICES=N nohup python train_sf.py &   # Best(窗口口径) ≈ 9.85

# ② 训最终级联（冻结 s1，微调 exp2）→ 产物 checkpoint/best.pth
cd ../PVA_s31_s1stage2 && CUDA_VISIBLE_DEVICES=N nohup python train_s1exp2.py &

# ③ 评估整表（输入 / s1 / exp1 / exp2 / stage2 同协议对比）
cd .. && python eval_all_metrics.py
```

- 已训好权重：`PVA_s31_s1stage2/checkpoint/best.pth`（最终模型）、`PVA_space_att_grouped/checkpoint/best.pth`（s1）、`PVA_s31_att/checkpoint/best_orig_full.pth`（exp2 原版纯 T 基线）。
- ⚠️ `train_s1exp2.py` 与 `eval_all_metrics.py` 内仍引用旧目录名（`exp2_s30_att`/`exp2_s1_att_grouped`/`exp2_s31_s1exp2`）；服务器已统一重命名为 `PVA_*`，重跑前需建软链或改路径。exp2 侧必须用 `best_orig_full.pth`（原版纯 Transformer），不要用 `best.pth`（曾为 LRU 混合版）。

---

## 6. 口径与注意事项

1. **绝对数字带口径**：本文表格是 2000 窗**逐点平均**；历史日志的窗口平均口径（如 exp2 原版 s31 ≈ 12.77、12.18）**不可直接比较**。同协议下相对对比才公平。
2. **对比必须同窗长**：输入基线随窗长变化（15 帧 13.62 / 31 帧 13.66）。跨窗长对比会得出"无 s1 模型负优化"的假象。
3. **s31"原版"加载对 checkpoint**：纯 Transformer 结构只能加载 `best_orig_full.pth`；`best.pth` 是 LRU 混合版会输出垃圾。修复后分桶 7.80/10.28/20.22 与原版日志 Epoch25 完全一致，可作加载正确的校验。
4. **stage2 s15 的收益主要在运动**：MPJPE 几乎不动，但 corr_v/corr_a 大涨，且低/中抖动桶有轻微代价；stage2 s31 无此代价，所有桶都赢——所以最终选 s31。
