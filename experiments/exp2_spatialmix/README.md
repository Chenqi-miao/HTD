# exp2_spatialmix: Temporal Transformer + Spatial Mixing

## 架构

**骨干**: 纯时序 Transformer（无空间注意力）+ 双层空间混合
- `SpatialMix`: Per-frame Linear(J·D → J·D)，在时序前后各一次，提供轻量关节间信息交换
- `TemporalTransformerDecoder`: 8 层纯时序注意力 + RoPE
- Depth=8, num_heads=8, dim_feat=256

**三个预测头**（与 exp1 相同）：
- `PosHead`: Linear(256→3)
- `VelHead`: Conv1d → LayerNorm → Linear(256→3)
- `AccHead`: Conv1d → LayerNorm → Linear(256→3)

## 特点

- 纯时序注意力 + RoPE（无 DSTFormer 的空间分支）
- `SpatialMix` 提供轻量跨关节信息交换（优于简单的 mean pool）
- 输入 `[B, F, J, D]` → SpatialMix → reshape `[B*J, F, D]` → Temporal → reshape back
- 参数量: **~64.6M**

## 训练配置

| 参数 | 值 |
|------|-----|
| seq_len | 15 |
| batch_size | 128 |
| lr | 1e-4 |
| total_epoch | 40 |
| transformer_depth | 8 |
| w_vel | 0.1 |
| w_accel | 0.05 |
| scheduler | cosine |

## 文件结构

```
exp2_spatialmix/
├── config.py              # 实验配置
├── train.py                # 训练入口
├── model/
│   ├── net.py              # 模型定义 (Net)
│   ├── temporal_transformer.py  # 时序 Transformer + RoPE
│   ├── rope.py             # Rotary Position Embedding
│   ├── loss.py             # SmoothNetLoss
│   ├── DSF.py              # (未使用，保留)
│   └── drop.py             # DropPath
├── checkpoint/
│   └── latest.pth          # 最新模型
├── log/
├── vis/
└── files/                  # 备份的依赖代码
```

## 运行

```bash
cd experiments/exp2_spatialmix
uv run --no-sync python3 train.py
```
