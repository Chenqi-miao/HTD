# experiments/tools — 数据加载 / 评估 / 可视化 / 分析工具

所有脚本通过 `__file__` 计算路径, 不依赖运行目录。

## 评估 (数据加载 + 全指标)

- **`eval_window.py`** — 窗口比较评估 (可靠版): SeqHandTest 全量测试集 (pkl 预处理),
  输入基线一致, 输出 MPJPE/PA-MPJPE/MPJVE/MPJAE/corr/ratio/PCE/highfreq/分桶。
  ```bash
  CUDA_VISIBLE_DEVICES=0 python tools/eval_window.py --exp exp2_s8_att --model exp2 --W 8
  CUDA_VISIBLE_DEVICES=0 python tools/eval_window.py --exp exp1_perjoint --model exp1 --W 15
  ```
  报告存 `experiments/<exp>/eval_W<W>.json`。

- **`eval_model.py`** — 通用模型评估 (核心指标, 增量, 可全量): 模型注册表
  (exp1/exp2/ssm/center_ssm/smoothnet), `--n-test 0` = 全量。
  ```bash
  CUDA_VISIBLE_DEVICES=0 python tools/eval_model.py --exp exp2_s15_att --model exp2 --n-test 0
  ```

## 可视化

- **`viz/viz_error_motion.py`** — 输入误差(上) vs 运动(下), 共享时间轴。发现: 输入误差基本不随运动速度变化。
  ```bash
  uv run python experiments/tools/viz/viz_error_motion.py --seq 'Capture0/ROM03_LT_No_Occlusion/InterWild/cam400262'
  ```
- **`viz/viz_filtered.py`** — 原始输入 vs 滤波后 vs GT (误差+位置)。发现: SG 滤波几乎不降位置误差。

## 分析

- **`analysis/`** — (待放) 逐关节/逐速度分桶的窗口优势分解。

## 数据

- 评估结果统一收集在 `experiments/eval_data/` (命名 `模型_窗口.json`)。

## 共享模块 (experiments/common/)

- `hand_topology.py` — 手部 21 关节拓扑/骨矩阵
- `seq_io.py` — lazy_npz_30 序列读取 (本地)
- `metrics.py` — 全指标 (mpjpe/mpjve/mpjae/pa_mpjpe/corr/ratio/pce/highfreq)
- `ssm.py` — Kalman 平滑器
- `center_ssm.py` / `bone_ssm.py` / `smoothnet.py` — 各模型
