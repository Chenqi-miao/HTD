# InterWild 数据链路说明

## 模型简介

**InterWild**（Facebook Research, 2023）是一个单帧 3D 手部网格恢复方法。输入单张 RGB 图像，输出 MANO 参数（姿态 + 形状 + 相机平移），再通过 MANO 正向计算出 3D 关节点坐标。

---

## 推理流程

```
输入: 单帧 RGB 图像 I ∈ ℝ^{H×W×3}
  │
  ↓ InterWild
  │
  ├── MANO 形状参数  β ∈ ℝ¹⁰
  ├── MANO 姿态参数  θ ∈ ℝ^{16×3}   (16 个关节点轴角，含全局旋转)
  └── 相机平移       t ∈ ℝ³
```

---

## MANO 参数 → 3D 关节（相机空间）

$$
\mathbf{J}_{\text{cam}} = \mathcal{M}(\boldsymbol{\theta}, \boldsymbol{\beta}) + \mathbf{t}
\quad\in\mathbb{R}^{21\times 3}
$$

其中 $\mathcal{M}(\cdot)$ 是 MANO 模型的正向流程：

1. **形状混合形状（Shape Blend Shapes）**：对平均模板关节 $\bar{\mathbf{J}}$ 施加形状偏移 $\mathbf{B}_S(\boldsymbol{\beta})$，得到个体化的静息手型
2. **层级旋转（Hierarchical Pose）**：沿 kinematic tree 将每个关节的轴角 $\boldsymbol{\theta}_i$ 通过 Rodrigues 公式转为旋转矩阵，逐层应用到子关节
3. **加根平移（Global Translation）**：加上 $\mathbf{t}$，从模型局部坐标系转换到**相机坐标系**

### 关键细节

每帧**独立推理**，MANO 参数 $\boldsymbol{\theta}, \boldsymbol{\beta}, \mathbf{t}$ 由单帧图像回归得到，**不包含任何时序约束**。

因此即使相邻帧画面变化很小，$\boldsymbol{\theta}$ 的微小波动也会导致 $\mathbf{J}_{\text{cam}}$ 出现可见的时序抖动。

---

## 2D 投影（几何关系）

给定相机内参矩阵 $\mathbf{K}$，3D 相机坐标可投影到像素平面：

$$
\mathbf{J}_{\text{pixel}} = \mathbf{K} \cdot \mathbf{J}_{\text{cam}}
\quad\text{其中}\quad
\mathbf{K} = \begin{bmatrix}
f_x & 0 & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix}
$$

展开后：

$$
u_j = \frac{x_j^{\text{cam}} \cdot f_x}{z_j^{\text{cam}}} + c_x, \qquad
v_j = \frac{y_j^{\text{cam}} \cdot f_y}{z_j^{\text{cam}}} + c_y
$$

$f_x, f_y, c_x, c_y$ 即数据中存储的 `focal` 和 `princpt`。有了相机参数，**任何时候都可以从 3D 坐标反算出 2D 像素坐标**。

---

## 数据预处理（原始 PKL → NPZ）

```
推理输出:
  cam_coord = J_cam(θ, β, t)  ∈ ℝ^{21×3}    相机坐标系 3D 关节点
  joint_valid                  ∈ {0,1}²¹      各关节可见性

预处理转换:
  J_world = R⁻¹ · (J_cam - T)  ∈ ℝ^{21×3}   世界坐标系 3D 关节点
  其中 R ∈ ℝ^{3×3} 为相机旋转矩阵，T ∈ ℝ³ 为相机平移向量
  (R, T 从 InterHand2.6M 官方标注获取)

最终存入 NPZ:
  in_joint    = J_world_InterWild ∈ ℝ^{T×21×3}    ← 输入（抖动）
  gt_joint    = J_world_gt        ∈ ℝ^{T×21×3}     ← 真值
```

其中：
- `in_joint` 来自 InterWild 的推理结果经 cam2world 转换
- `gt_joint` 来自 InterHand2.6M 官方的 MANO 标注（同一份 R, T）
- `joint_valid` 标记可见/被遮挡的关节

---

## 数据形态总结

| 名称 | 符号 | 维度 | 含义 |
|------|------|------|------|
| 输入 3D 坐标 | $\mathbf{X}_{\text{in}}$ | [T, 21, 3] | InterWild 估计，有噪声 |
| 真值 3D 坐标 | $\mathbf{X}_{\text{gt}}$ | [T, 21, 3] | InterHand2.6M 官方标注 |
| 输入有效标志 | $\mathbf{v}_{\text{in}}$ | [T, 21] | 输入关节是否被遮挡 |
| 真值有效标志 | $\mathbf{v}_{\text{gt}}$ | [T, 21] | 真值关节是否被遮挡 |
| 连续帧标志 | $\mathbf{c}$ | [T] | 帧间位移 > 30mm 标记为不连续 |
| 相机外参 | $\mathbf{R}, \mathbf{T}$ | [3, 3], [3] | NPZ 版已丢弃，原始 PKL 中存 |
| 相机内参 | $f_x, f_y, c_x, c_y$ | 4 | NPZ 版已丢弃，原始 PKL 中存 |

---

## 对当前工作的影响

1. **输入抖动来源**：$\mathbf{X}_{\text{in}}$ 的噪声来自 InterWild 每帧独立的 $\boldsymbol{\theta}$ 回归，不是量化/传输噪声
2. **抖动量级**：评估显示 Init Jitter（二阶差分均值）为 **7.45mm**，远大于 GT Jitter 的 **3.42mm**
3. **2D 监督可行**：只要有相机参数 $f_x, f_y, c_x, c_y$，$\mathbf{X}_{\text{gt}}$ 可投影为 2D 关键点，作为 ViTPose+ 的伪 GT
4. **速度/加速度 GT**：直接从 $\mathbf{X}_{\text{gt}}$ 差分即可获得干净的物理运动信号
