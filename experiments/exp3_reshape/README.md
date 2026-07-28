# exp3_reshape: PVANetm — Per-Joint Reshape Temporal Transformer

## 架构

**骨干**: 纯时序 Transformer + Per-Joint Reshape（修复 mean pool 丢失关节信息的问题）
- Joint Embed → reshape `[B, F, J, D] → [B*J, F, D]` 保留 J 维度
- `TemporalTransformerDecoder`: 8 层纯时序注意力 + RoPE（在 B*J 维度上过时序）
- reshape back → `[B, F, J, D]`
- Depth=8, num_heads=8, dim_feat=256, ff_rate=4

**三个预测头**：
- `PosHead`: Linear(256→3) — 逐关节 3D 位置
- `VelHead` (MotionDecoder): Conv1d → LayerNorm → Linear(256→3)
- `AccHead` (MotionDecoder): Conv1d → LayerNorm → Linear(256→3)

## 特点

- 与 exp2 的关键区别：**无 SpatialMix**，直接用 reshape 将 J 展进 batch
- 每关节独立过时序 Transformer，避免空间混合引入的噪声
- 速度/加速度头使用更简洁的 `Linear→GELU→Linear` 结构
- 参数量: **~19.2M**（取决于具体 depth）

## 训练配置

| 参数 | 值 |
|------|-----|
| seq_len | 15 |
| batch_size | 128 |
| lr | 1e-4 |
| total_epoch | 40 |
| transformer_depth | 8 |
| w_vel | 0.05 |
| w_accel | 0.02 |
| scheduler | cosine |

## 文件结构

```
exp3_reshape/
├── config.py              # 实验配置
├── train.py                # 训练入口
├── model/
│   ├── PVANetm.py          # 模型定义 (PVANetModel) — 本地独立版本
│   ├── net.py              # 重新导出 PVANetModel
│   ├── temporal_transformer.py  # 时序 Transformer (from exp2)
│   ├── rope.py             # RoPE (from exp2)
│   ├── loss.py             # SmoothNetLoss
│   ├── DSF.py              # (未使用)
│   └── drop.py             # DropPath
├── log/
├── vis/
└── files/                  # 备份的依赖代码
```

## 运行

```bash
cd experiments/exp3_reshape
uv run --no-sync python3 train.py
```
