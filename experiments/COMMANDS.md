# 常用命令速查

## 环境激活

```bash
cd ~/workspace/hand_3d_reconstruction/HTD
source .venv/bin/activate
```

> `.venv` 在 `HTD/` 下，不要在其他目录运行。

---

## 训练

### 从头训练

```bash
cd ~/workspace/hand_3d_reconstruction/HTD
.venv/bin/python3 experiments/exp1_perjoint/train.py          # Exp1: DSTFormer + per-joint
.venv/bin/python3 experiments/exp2_spatialmix/train.py        # Exp2: 纯时序 + 空间混合
.venv/bin/python3 experiments/exp3_reshape/train.py           # Exp3: [B*J,F,D] reshape
```

### 后台挂起跑（推荐）

```bash
cd ~/workspace/hand_3d_reconstruction/HTD
nohup .venv/bin/python3 experiments/exp2_spatialmix/train.py \
  > experiments/exp2_spatialmix/log/train.log 2>&1 &
```

### 断点续训

```bash
cd ~/workspace/hand_3d_reconstruction/HTD
.venv/bin/python3 experiments/exp2_spatialmix/train.py \
  --resume experiments/exp2_spatialmix/checkpoint/latest.pth
```

### 看训练进度

```bash
tail -f experiments/exp2_spatialmix/log/train.log
```

### TensorBoard

```bash
cd ~/workspace/hand_3d_reconstruction/HTD
.venv/bin/python3 -m tensorboard.main --logdir experiments/exp2_spatialmix/log --port 6006
```

---

## 评估

### 评估已有 checkpoint（全量指标含 MPJVE/MPJAE）

```bash
cd ~/workspace/hand_3d_reconstruction/HTD
.venv/bin/python3 experiments/pva_paper/eval.py \
  --checkpoint experiments/exp1_perjoint/checkpoint/best.pth

# 快速验证（只跑 1000 样本）
.venv/bin/python3 experiments/pva_paper/eval.py \
  --checkpoint experiments/exp1_perjoint/checkpoint/best.pth \
  --data_num 1000
```

### 生成对比视频

```bash
cd ~/workspace/hand_3d_reconstruction/HTD
.venv/bin/python3 experiments/pva_paper/gen_video.py
```

---

## 实验目录结构

```
experiments/
├── exp1_perjoint/       DSTFormer + 3 per-joint heads  (6.8M params)
├── exp2_spatialmix/     纯时序 + SpatialMix 空间混合    (64.6M params)
├── exp3_reshape/        [B*J,F,D] 每关节独立时序        (6.8M params)
└── pva_paper/           共用评估脚本 + 优化器
```

每个实验目录内：

```
experiments/exp_X/
├── config.py           ← 改参数在这里
├── train.py            ← 训练入口
├── model/net.py        ← 模型定义
├── checkpoint/         ← 权重
│   ├── latest.pth      ← 最新
│   └── best.pth        ← 验证集最优
├── log/
│   └── train.log       ← 训练日志
└── vis/                ← 可视化
```

---

## 调参

```bash
vim experiments/exp1_perjoint/config.py
```

常用改的：

```python
w_vel = 0.1        # 速度 loss 权重
w_accel = 0.05     # 加速度 loss 权重
batch_size = 128   # batch 大小（显存不够改小）
lr = 1e-4          # 学习率
total_epoch = 40   # 总轮数
```
