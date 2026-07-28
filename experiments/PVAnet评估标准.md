# PVA-Net 速度、加速度与 3D 关键点评估标准

> 参考 HTD-Refine 论文 (Sec 3.2, Supplementary Sec C) 的评估方法，针对手部 3D 姿态的 PVA-Net 模型。

---

## 一、输出定义

PVA-Net 同时预测三个量：

| 输出 | 符号 | 形状 | 含义 |
|------|------|------|------|
| **3D 关键点** | `K̂` ∈ ℝ^(B×V×F×J×3) | 相机坐标系 3D 关节位置 | 细化后的手部关节坐标 |
| **速度** | `V̂` ∈ ℝ^(B×V×F×J×3) | 相机坐标系 3D 关节速度 (mm/frame) | 相邻帧的关节位移 |
| **加速度** | `Â` ∈ ℝ^(B×V×F×J×3) | 相机坐标系 3D 关节加速度 (mm/frame²) | 速度的帧间差分 |

> B: batch, V: view, F: frame, J: joint (21 joints)

---

## 二、3D 关键点精度评估

### 2.1 MPJPE (Mean Per-Joint Position Error)

最核心的位置精度指标，对应论文中的 joint position error：

**MPJPE** = 1/N · Σᵢ ||K̂ᵢ − Kᵢ||₂

其中 K 为真值 (ground truth)，仅在有效帧上计算。

### 2.2 输入误差 vs 输出误差

参考实验代码，同时计算：
- **Init Error**: 输入 3D 关节到真值的 MPJPE（反映输入噪声水平）
- **Refine Error**: PVA-Net 输出 3D 关节到真值的 MPJPE（反映细化效果）

### 2.3 改进率

**Improvement** = (1 − Refine_MPJPE / Init_MPJPE) × 100%

---

## 三、速度精度评估

### 3.1 MPJVE (Mean Per-Joint Velocity Error)

参考论文 Eq. (18) 及实验代码：

**MPJVE** = 1/(T−1) · Σ_{t=2}^{T} ||(K̂ₜ − K̂ₜ₋₁) − (Kₜ − Kₜ₋₁)||₂

其中 K̂ 为预测关节位置，K 为真值关节位置。同时支持输入/输出的对比。

### 3.2 PCE@τ (Percentage of Correct Estimates for Velocity)

参考论文 Supplementary Sec C, Eq. (21)。评估直接预测的速度值 V̂ 的准确率：

**PCE@τ** = 1/N · Σᵢ 𝟙(|V̂ᵢ − Vᵢ| < τ · (r_max − r_min))

其中：
- **V̂ᵢ** / **Vᵢ**: 预测速度 / 真值速度
- **[r_min, r_max]**: 速度的动态范围，基于**真值速度分布**的 0.1 百分位和 99.9 百分位计算
- **τ**: 阈值系数，取 0.10、0.05、0.01

**三个严格度**：

| 阈值 | 含义 |
|------|------|
| PCE@0.10 | 绝对误差 < 动态范围的 10% — 宽松 |
| PCE@0.05 | 绝对误差 < 动态范围的 5% — 中等 |
| PCE@0.01 | 绝对误差 < 动态范围的 1% — 严格 |

> 注意：PCE 评估的是 PVA-Net **直接输出的速度值**（`vel_pred`），而非从关节位置差分算出的速度。

---

## 四、加速度精度评估

### 4.1 MPJAE (Mean Per-Joint Acceleration Error)

参考论文 Eq. (19) 及实验代码：

**MPJAE** = 1/(T−2) · Σ_{t=2}^{T−1} ||(Δ²K̂ₜ) − (Δ²Kₜ)||₂

其中 Δ²K̂ₜ = K̂ₜ₊₁ − 2K̂ₜ + K̂ₜ₋₁ 为加速度（二阶差分）。

### 4.2 PCE@τ for Acceleration

与速度的 PCE 完全相同的定义，但应用于直接预测的加速度值 Â：

**PCE@τ** = 1/N · Σᵢ 𝟙(|Âᵢ − Aᵢ| < τ · (r_max − r_min))

其中 [r_min, r_max] 为**真值加速度**的 0.1～99.9 百分位动态范围，τ ∈ {0.10, 0.05, 0.01}。

---

## 五、全部指标汇总

| 类别 | 指标 | 缩写 | 单位 | 评估对象 |
|------|------|------|:----:|----------|
| **位置** | Mean Per-Joint Position Error | MPJPE | mm | `pd_joint_xyz` vs `targets['joint_xyz']` |
| **速度** | Mean Per-Joint Velocity Error | MPJVE | mm/frame | 从关节差分计算的帧间速度误差 |
| **速度** | PCE @ 0.10 / 0.05 / 0.01 | PCE | % | 直接预测速度 `vel_pred` vs GT 速度 |
| **加速度** | Mean Per-Joint Acceleration Error | MPJAE | mm/frame² | 从关节差分计算的帧间加速度误差 |
| **加速度** | PCE @ 0.10 / 0.05 / 0.01 | PCE | % | 直接预测加速度 `acc_pred` vs GT 加速度 |

### 评估流程

```
输入关节 ──→ PVA-Net ──→ 3D 关键点 (pd_joint_xyz)
                            ├── 速度 (vel_pred)
                            └── 加速度 (acc_pred)
                               
MPJPE:   pd_joint_xyz  vs  targets['joint_xyz']
MPJVE:   diff(pd_joint_xyz)  vs  diff(targets['joint_xyz'])
PCEvel:  vel_pred  vs  compute_vel(targets['joint_xyz'])
MPJAE:   diff²(pd_joint_xyz)  vs  diff²(targets['joint_xyz'])
PCEacc:  acc_pred  vs  compute_acc(targets['joint_xyz'])
```

---

## 六、实验对照

为便于对比，每个实验需输出以下格式的评估结果：

```json
{
  "exp_name": "exp1_perjoint",
  "init_mpjpe": 13.40,
  "refine_mpjpe": 11.94,
  "init_mpjve": 4.41,
  "refine_mpjve": 3.06,
  "init_mpjae": 6.78,
  "refine_mpjae": 3.02,
  "vel_pce_010": 98.2,
  "vel_pce_005": 93.0,
  "vel_pce_001": 68.2,
  "acc_pce_010": 99.6,
  "acc_pce_005": 98.4,
  "acc_pce_001": 82.3
}
```

**对照目标**: pva_paper 实验 (`eval_result.json`) 提供了参考基准。

---

## 七、有效帧处理

所有指标计算时需应用 mask（来自 `meta_info`）：
- `joint_gt_val`: 关节真值是否有效
- `joint_in_val`: 输入关节是否有效（基于空间范围判定）
- `continuous_val`: 帧是否连续（帧间位移 < 30mm 视为连续）
- `data_repeat`: 测试集上因边界补齐而重复的帧需排除

**Mask 组合**: `mask = data_repeat × joint_gt_val × continuous_val × joint_in_val`
