# VLA Data Selection for Vision Cognitive Engineering

本项目对应《视觉认知工程》课程题目二：基于脑启发核心集选择的轻量级 VLA 机械臂动作预测。它面向 `lerobot/aloha_sim_transfer_cube_human` 数据集，构建了一个完整、可复现、适合课程报告撰写的实验闭环：

- 自动检查并读取 LeRobot/Hugging Face 数据集
- 提取单视角图像的冻结视觉特征
- 支持默认 14-DoF 全动作回归，也支持切换到单臂 7-DoF
- 对比 `random`、`action_change`、`visual_change`、`visual_action`、`brain_coreset`
- 用统一的 MLP 回归器评估不同数据选择策略
- 输出指标、图像和报告素材

## 目录结构

```text
.
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml
├── scripts/
├── src/
├── outputs/
└── report_assets/
```

## 环境安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果 `lerobot` 安装失败，代码会自动回退到 `datasets.load_dataset`。如果 CUDA 不可用，脚本会自动回退到 CPU。

## 数据下载与读取

默认数据集：

```text
lerobot/aloha_sim_transfer_cube_human
```

先检查数据字段：

```bash
python scripts/00_inspect_dataset.py --config configs/default.yaml
```

脚本会保存：

```text
outputs/dataset_inspection.json
```

如果自动探测字段失败，可以显式传参，例如：

```bash
python scripts/00_inspect_dataset.py \
  --config configs/default.yaml \
  --image_key observation.images.top \
  --action_key action \
  --episode_key episode_index
```

## 分步运行

1. 提取特征

```bash
python scripts/01_extract_features.py --config configs/default.yaml
```

2. 生成 episode split 与核心集选择

```bash
python scripts/02_select_coreset.py --config configs/default.yaml --method all
```

3. 训练 MLP baseline / coreset / full

```bash
python scripts/03_train_mlp.py --config configs/default.yaml --method random
python scripts/03_train_mlp.py --config configs/default.yaml --method brain_coreset
python scripts/03_train_mlp.py --config configs/default.yaml --method full
```

4. 评估并汇总结果

```bash
python scripts/04_evaluate.py --config configs/default.yaml --method all
```

5. 生成课程报告图像与摘要

```bash
python scripts/05_make_plots.py --config configs/default.yaml
```

## 一键运行

```bash
bash scripts/run_all.sh
```

## 14D Ablation

针对 14 维双臂动作，项目额外提供了 4 组对照配置，分别隔离“按维标准化”“仅联合聚类”“降低协同项权重”“降低动作多样性权重”四种改动：

- `configs/ablations/14d_norm_only.yaml`
- `configs/ablations/14d_joint_diversity_only.yaml`
- `configs/ablations/14d_low_coordination.yaml`
- `configs/ablations/14d_low_action_diversity.yaml`

如果已经有 `outputs/features/features_resnet18.npz` 这份 14 维特征缓存，可以直接批量运行：

```bash
bash scripts/run_14d_ablations.sh
```

每组实验会把 `selection/checkpoint/metrics/figures/report_assets` 输出到各自独立的 `outputs/ablations/...` 和 `report_assets/ablations/...` 目录，避免覆盖默认实验结果。

## 各脚本功能

- `scripts/00_inspect_dataset.py`: 检查数据集长度、字段结构和候选 key
- `scripts/01_extract_features.py`: 提取 ResNet-18 图像特征并缓存到 `.npz`
- `scripts/02_select_coreset.py`: 生成 episode split、分数和采样索引
- `scripts/03_train_mlp.py`: 按指定方法训练 MLP 动作回归器
- `scripts/04_evaluate.py`: 在 val/test 上评估并保存结果
- `scripts/05_make_plots.py`: 生成柱状图、PCA、时间轴图和实验摘要
- `scripts/run_all.sh`: 串起整套流程

## 实验方法说明

### 输入输出

- 输入：`[image_feature, language_feature]`
- 输出：默认 14 维全动作；也支持通过 `dataset.arm=left/right` 切换到 7 维单臂动作

### 特征提取

- 默认视觉模型：`ResNet-18`
- 图像特征维度：`512`
- 语言模式默认：`constant`

### 数据划分

- 默认 `episode-level 80/10/10`
- 核心集选择只在 train split 内完成

### 核心集方法

- `random`
- `action_change`
- `visual_change`
- `visual_action`
- `brain_coreset`

`brain_coreset` 使用两阶段策略：

1. 用视觉变化和动作变化构建 utility 分数；在 14 维动作下，会先做按维标准化，并分别建模左臂变化、右臂变化和双臂协同变化
2. 在高 utility 候选池内执行联合 KMeans 多样性选择，聚类空间同时考虑视觉特征和动作特征

## 输出文件说明

主要输出包括：

```text
outputs/dataset_inspection.json
outputs/features/features_resnet18.npz
outputs/selections/split_seed42.json
outputs/selections/random_10_seed42.npy
outputs/selections/brain_coreset_10_seed42.npy
outputs/metrics/results.csv
outputs/metrics/results.json
outputs/figures/mse_bar.png
outputs/figures/per_dim_mse.png
outputs/figures/score_distribution.png
outputs/figures/feature_pca.png
report_assets/experiment_summary.md
report_assets/tables/results_latex.txt
```

## 课程报告素材位置

- 图像：`report_assets/figures/`
- 表格：`report_assets/tables/`
- 文字摘要：`report_assets/experiment_summary.md`

## 常见错误与解决方式

- 数据字段不匹配：先运行 `00_inspect_dataset.py`，再显式指定 `--image_key/--action_key/--episode_key`
- `Feature cache not found`：先运行 `01_extract_features.py`
- `Selection file not found`：先运行 `02_select_coreset.py`
- 没有 CUDA：代码会自动回退到 CPU，运行速度会慢一些
- action 维度不是 7 或 14：需要检查数据字段是否选错
- 需要 14 维全动作训练：确认 `configs/default.yaml` 中 `dataset.arm` 为 `both`，并重新执行特征提取与后续流程
- KMeans 聚类数量过大：代码会自动将 cluster 数量截断到候选样本数

## 与题目要求的对应关系

- 题目要求的数据集读取、冻结视觉特征、MLP baseline、随机 10% 和脑启发 10% 对比，已经被完整纳入当前 pipeline
- 该实现重点是“跑通实验闭环并产出可写入报告的材料”，不追求训练完整 VLA 大模型
