# exp1_perjoint: DSTFormer + Per-Joint Heads

## 架构

**骨干**: DSTFormer (Spatio-Temporal Transformer)
- 同时建模关节间（空间）和帧间（时间）关系
- Depth=2, num_heads=8, dim_feat=256

**三个预测头**（每关节独立）：
- `PosHead`: Linear(256→3) — 直接回归 3D 关节位置
- `VelHead`: Conv1d → LayerNorm → Linear(256→3) — 逐关节速度预测
- `AccHead`: Conv1d → LayerNorm → Linear(256→3) — 逐关节加速度预测（与速度头相同架构）

## 特点

- 保留 J 维度的 DSTFormer 空间-时序双向注意力
- 每关节独立预测，无空间池化
- 参数量: **~6.8M**

## 训练配置

| 参数 | 值 |
|------|-----|
| seq_len | 15 |
| batch_size | 128 |
| lr | 1e-4 |
| total_epoch | 40 |
| transformer_depth | 2 |
| w_vel | 0.1 |
| w_accel | 0.05 |
| scheduler | cosine |

## 文件结构

```
exp1_perjoint/
├── config.py          # 实验配置
├── train.py            # 训练入口
├── model/
│   ├── net.py          # 模型定义 (Net)
│   ├── DSF.py          # DSTFormer 骨干
│   ├── loss.py         # SmoothNetLoss
│   └── drop.py         # DropPath
├── checkpoints/
│   ├── latest.pth
│   └── best.pth        # 最佳模型 (epoch=32)
├── log/
├── vis/
└── files/              # 备份的依赖代码
```

## 运行

```bash
cd experiments/exp1_perjoint
uv run --no-sync python3 train.py
```
