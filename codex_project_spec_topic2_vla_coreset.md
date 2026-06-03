# Codex 项目构建说明：基于脑启发核心集选择的轻量级 VLA 机械臂动作预测

## 0. 项目定位

本项目用于完成《视觉认知工程》课程考查题目二：**基于脑启发核心集选择的轻量级 VLA 机械臂动作预测**。

目标不是训练完整 VLA 大模型，而是在指定 ALOHA 仿真操作数据集上构建一个轻量级、可复现、可写入课程报告的实验闭环：

1. 读取 `lerobot/aloha_sim_transfer_cube_human` 数据集。
2. 选择一个相机视角和单臂 7 自由度动作标签。
3. 使用冻结的轻量视觉模型提取图像特征。
4. 构建 `[视觉特征 + 语言指令/任务编码] -> [7-DoF action]` 的动作回归数据集。
5. 训练 MLP baseline，并报告 MSE。
6. 设计脑启发核心集选择算法，从全量数据中筛选 10% 高价值样本。
7. 使用同样 MLP 训练核心集版本，并与随机 10% baseline 对比。
8. 输出表格、图像、日志和可用于课程报告的分析材料。

> 课程要求强调：跑通代码、完成规定任务、讨论实验结果；不要求达到当前最佳性能。

---

## 1. 项目核心实验问题

本项目需要回答的问题：

> 在相同 10% 数据比例下，基于视觉变化、动作变化和分布覆盖的脑启发核心集选择，是否比随机抽样更适合训练轻量级 VLA 动作预测模型？

核心对比：

| 方法 | 数据比例 | 数据选择方式 | 特征提取器 | 回归模型 | 指标 |
|---|---:|---|---|---|---|
| Random-10% | 10% | 随机抽样 | ResNet-18 或 CLIP | MLP | Action MSE |
| Coreset-10% | 10% | 脑启发核心集选择 | 同上 | 同上 | Action MSE |
| Full-100% | 100% | 全量数据，可选 | 同上 | 同上 | Action MSE |

必须保证不同方法只改变训练数据选择方式，其余训练设置尽可能一致。

---

## 2. 数据集要求

### 2.1 指定数据集

使用 Hugging Face / LeRobot 数据集：

```text
lerobot/aloha_sim_transfer_cube_human
```

任务：ALOHA Sim Transfer Cube Human Demonstrations。

课程题目中说明该数据集包含：

- 50 个成功的人类演示 trajectories / episodes；
- 多视角相机图像；
- 末端执行器状态；
- 14 自由度联合动作标签；
- 可简化为单视角图像 + 单臂 7 自由度动作预测任务。

### 2.2 数据读取策略

Codex 实现时需要优先支持 LeRobot 官方读取方式，但要写成兼容版本，避免因为 LeRobot 版本差异导致字段名变化。

建议实现 `dataset_utils.py`，提供以下能力：

1. 自动下载或加载本地缓存数据集。
2. 打印并保存 dataset sample 的 key 结构，便于确认字段。
3. 自动探测图像字段候选：
   - `observation.images.top`
   - `observation.images.cam_high`
   - `observation.image`
   - `image`
   - 其他包含 `image` / `images` 的字段。
4. 自动探测动作字段候选：
   - `action`
   - 其他包含 `action` 的字段。
5. 自动探测 episode index 字段候选：
   - `episode_index`
   - `episode_idx`
   - `episode.id`
   - 如果没有，则根据 `timestamp` 或 frame 顺序回退。
6. 支持用户通过命令行参数显式指定字段名。

命令行参数建议：

```bash
--repo_id lerobot/aloha_sim_transfer_cube_human
--image_key observation.images.top
--action_key action
--episode_key episode_index
--arm left
--cache_dir ./data/cache
```

如果无法自动确定字段，需要报出清晰错误，并打印可用 keys。

### 2.3 动作标签处理

数据集中动作可能为 14 维双臂动作。课程任务要求可降维为单臂 7 自由度动作。

实现规则：

```python
if action_dim == 14:
    if arm == "left":
        y = action[:7]
    elif arm == "right":
        y = action[7:14]
elif action_dim == 7:
    y = action
else:
    raise ValueError("Unsupported action dimension")
```

需要在日志中记录：

- 原始 action 维度；
- 最终使用的 action 维度；
- 选择的是 left arm 还是 right arm。

---

## 3. 项目目录结构

请 Codex 按以下结构构建项目：

```text
vla_coreset_assignment/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml
├── scripts/
│   ├── 00_inspect_dataset.py
│   ├── 01_extract_features.py
│   ├── 02_select_coreset.py
│   ├── 03_train_mlp.py
│   ├── 04_evaluate.py
│   ├── 05_make_plots.py
│   └── run_all.sh
├── src/
│   ├── __init__.py
│   ├── dataset_utils.py
│   ├── feature_extractor.py
│   ├── coreset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── plot_utils.py
│   └── utils.py
├── outputs/
│   ├── features/
│   ├── selections/
│   ├── checkpoints/
│   ├── metrics/
│   └── figures/
└── report_assets/
    ├── tables/
    └── figures/
```

所有脚本都应支持命令行运行，并尽量不依赖 notebook。

---

## 4. 依赖要求

`requirements.txt` 建议包含：

```text
torch
torchvision
transformers
datasets
huggingface_hub
numpy
pandas
scikit-learn
matplotlib
tqdm
pyyaml
pillow
```

可选：

```text
lerobot
open_clip_torch
```

注意：

1. 如果 `lerobot` 安装失败，需要回退到 `datasets.load_dataset` 或本地 parquet 读取。
2. 项目默认优先使用 ResNet-18，原因是安装稳定、速度快、足够完成课程任务。
3. CLIP 作为可选特征提取器，不作为唯一依赖。

---

## 5. 配置文件要求

`configs/default.yaml` 建议内容：

```yaml
seed: 42

paths:
  output_dir: outputs
  feature_dir: outputs/features
  selection_dir: outputs/selections
  checkpoint_dir: outputs/checkpoints
  metrics_dir: outputs/metrics
  figure_dir: outputs/figures
  report_asset_dir: report_assets

dataset:
  repo_id: lerobot/aloha_sim_transfer_cube_human
  cache_dir: data/cache
  image_key: null
  action_key: action
  episode_key: null
  camera: auto
  arm: left
  max_frames: null
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  split_by_episode: true

features:
  extractor: resnet18
  batch_size: 64
  device: cuda
  normalize: true
  save_format: npz

selection:
  ratio: 0.10
  methods:
    - random
    - action_change
    - visual_change
    - visual_action
    - brain_coreset
  lambda_visual: 1.0
  lambda_action: 1.0
  lambda_diversity: 0.5
  use_diversity: true
  diversity_method: kmeans
  per_episode_quota: true

training:
  hidden_dims: [512, 256]
  dropout: 0.1
  lr: 0.001
  weight_decay: 0.0001
  batch_size: 256
  epochs: 100
  patience: 15
  device: cuda
  loss: mse

logging:
  save_predictions: true
  save_plots: true
```

所有命令行参数应能覆盖 YAML 配置。

---

## 6. 特征提取模块

### 6.1 默认特征提取器

默认使用 torchvision 的 ResNet-18：

- 加载 ImageNet 预训练权重；
- 冻结参数；
- 去掉最后分类层；
- 输出 512 维图像特征。

### 6.2 图像预处理

ResNet-18 预处理：

```python
Resize(224)
CenterCrop(224)
ToTensor()
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

### 6.3 语言指令处理

由于 ALOHA Transfer Cube 通常任务指令变化有限，本项目允许简化语言处理。

实现以下两种模式：

1. `language_mode=constant`：使用一个固定 one-hot 或全 1 标量表示任务指令。
2. `language_mode=text_hash`：如果数据中存在 language / instruction 字段，则将文本 hash 到固定维度 embedding。

为了课程任务可控，默认使用 `constant`。

最终输入特征：

```text
x = concat(image_feature, language_feature)
y = 7-DoF action
```

如果 `language_feature` 是常量，也需要在报告中说明：本实验主要验证视觉-动作数据核心集筛选，语言指令在单任务数据集中变化较小，因此采用固定任务编码。

### 6.4 特征缓存

特征提取较耗时，需要保存到本地：

```text
outputs/features/features_resnet18.npz
```

保存内容：

```python
{
  "features": float32 array, shape [N, D],
  "actions": float32 array, shape [N, 7],
  "episode_indices": int array, shape [N],
  "frame_indices": int array, shape [N],
  "timestamps": optional float array, shape [N],
  "image_keys": optional list,
  "metadata": json string
}
```

---

## 7. 数据划分要求

必须避免 train/test 泄露。

优先采用 episode-level split：

```text
train episodes: 80%
val episodes: 10%
test episodes: 10%
```

所有数据选择方法只能在 train split 内筛选训练样本。

验证集和测试集保持一致，不参与核心集选择。

需要保存 split 文件：

```text
outputs/selections/split_seed42.json
```

包含：

```json
{
  "train_episodes": [...],
  "val_episodes": [...],
  "test_episodes": [...],
  "seed": 42
}
```

---

## 8. 核心集选择算法设计

需要实现多个选择方法，用于 baseline、主方法和消融实验。

### 8.1 Random Sampling

随机选择 train split 中 10% 样本。

要求：

- 支持固定随机种子；
- 支持 per-episode quota，避免只选到少数 episode。

输出：

```text
outputs/selections/random_10_seed42.npy
```

### 8.2 Action-Change Sampling

基于相邻帧动作变化定义价值：

```text
action_change_t = ||a_t - a_{t-1}||_2
```

在每个 episode 内计算，然后选择 action_change 最高的 10% 样本。

边界处理：每个 episode 第一帧的 action_change 设为 0 或与后一帧比较。

认知解释：动作变化大代表操作者意图或控制状态发生变化，对动作预测更有训练价值。

### 8.3 Visual-Change Sampling

基于相邻帧视觉特征变化定义价值：

```text
visual_change_t = ||f_t - f_{t-1}||_2
```

在每个 episode 内计算，然后选择 visual_change 最高的 10% 样本。

认知解释：视觉状态变化大对应预测编码中的 prediction error，说明该帧携带新信息。

### 8.4 Visual + Action Sampling

综合视觉变化和动作变化：

```text
score_t = lambda_visual * zscore(visual_change_t) + lambda_action * zscore(action_change_t)
```

注意：

- z-score 需在 train split 内计算；
- 如果标准差接近 0，需要加 eps；
- 保存每个样本的 score 供可视化。

### 8.5 Brain-inspired Coreset：主方法

主方法应综合三个因素：

1. 视觉预测误差：`visual_change_t`
2. 动作变化 / 控制效用：`action_change_t`
3. 分布多样性：`diversity_t`

建议实现为两阶段选择：

#### Stage 1：高效用候选池

先计算：

```text
utility_t = lambda_visual * zscore(visual_change_t)
          + lambda_action * zscore(action_change_t)
```

选出 top `candidate_ratio`，例如 30% 样本作为候选池。

#### Stage 2：多样性约束选择

在候选池中进行多样性选择，最终得到 10% 样本。

可选实现 A：KMeans Medoid-like Selection

1. 对候选池特征做 KMeans，cluster 数量等于目标样本数。
2. 每个 cluster 选择距离 cluster center 最近、utility 较高的样本。
3. 若样本数不足，则从剩余高 utility 样本中补齐。

可选实现 B：Farthest Point Sampling

1. 初始样本选择 utility 最高样本。
2. 后续每次选择与已选集合距离最大且 utility 较高的样本。
3. 为避免复杂度过高，可只在候选池内执行。

默认建议实现 KMeans，速度更快。

最终主方法 score 可记为：

```text
brain_score_t = utility_t + lambda_diversity * diversity_bonus_t
```

但代码实现不必拘泥于一个公式，只要清晰保存选择依据即可。

---

## 9. MLP 动作预测模型

### 9.1 输入输出

输入：

```text
x: [image_feature, language_feature]
```

输出：

```text
y_hat: 7-DoF action
```

### 9.2 模型结构

建议：

```python
MLP(
  Linear(input_dim, 512),
  ReLU,
  Dropout(0.1),
  Linear(512, 256),
  ReLU,
  Dropout(0.1),
  Linear(256, 7)
)
```

### 9.3 训练设置

- loss: MSELoss
- optimizer: AdamW
- lr: 1e-3
- weight_decay: 1e-4
- epochs: 100
- early stopping patience: 15
- batch size: 256
- seed: 42

### 9.4 归一化

必须对 action 做 train split 统计归一化：

```python
y_norm = (y - mean_train) / std_train
```

评估时再还原或直接在归一化空间报告 MSE。建议同时报告：

1. `mse_norm`：归一化 action 空间 MSE；
2. `mse_raw`：原始 action 空间 MSE。

特征也建议标准化：

```python
x_norm = (x - mean_train) / std_train
```

标准化参数只能用训练集统计，不能用 val/test。

---

## 10. 评估指标

主指标：

```text
MSE = mean((pred_action - gt_action)^2)
```

需要输出：

1. overall MSE；
2. 每个自由度的 MSE；
3. val MSE；
4. test MSE；
5. 多随机种子均值和标准差，可选但推荐。

保存路径：

```text
outputs/metrics/results.csv
outputs/metrics/results.json
```

`results.csv` 示例：

```csv
method,ratio,seed,n_train,mse_norm_val,mse_norm_test,mse_raw_val,mse_raw_test
random,0.1,42,xxx,xxx,xxx,xxx,xxx
action_change,0.1,42,xxx,xxx,xxx,xxx,xxx
visual_change,0.1,42,xxx,xxx,xxx,xxx,xxx
visual_action,0.1,42,xxx,xxx,xxx,xxx,xxx
brain_coreset,0.1,42,xxx,xxx,xxx,xxx,xxx
full,1.0,42,xxx,xxx,xxx,xxx,xxx
```

---

## 11. 可视化输出要求

为了方便写课程报告，需要生成以下图像。

### 11.1 主结果柱状图

文件：

```text
outputs/figures/mse_bar.png
report_assets/figures/mse_bar.png
```

内容：不同数据选择方法的 test MSE 对比。

### 11.2 样本选择时间轴图

文件：

```text
outputs/figures/selection_timeline_episode_{id}.png
```

内容：

- x 轴：frame index；
- y 轴：action_change 或 visual_change；
- 标出被 random / brain_coreset 选中的帧。

用途：说明核心集倾向于选择动作变化大、视觉状态变化明显的关键帧。

### 11.3 特征空间分布图

文件：

```text
outputs/figures/feature_tsne_or_pca.png
```

内容：

- 对 train features 做 PCA 到 2D；
- 显示全量训练样本、random 10%、brain_coreset 10%；
- 用不同 marker 表示，不要求复杂美化。

用途：说明核心集覆盖了更广或更关键的分布区域。

### 11.4 动作维度 MSE 图

文件：

```text
outputs/figures/per_dim_mse.png
```

内容：每个 action dimension 的 MSE 对比。

用途：分析核心集方法主要改善哪些动作维度。

### 11.5 选择分数分布图

文件：

```text
outputs/figures/score_distribution.png
```

内容：visual_change、action_change、utility_score 的分布。

用途：支撑“冗余定义”和样本价值分析。

---

## 12. 脚本功能说明

### 12.1 `scripts/00_inspect_dataset.py`

功能：

1. 加载数据集；
2. 打印 dataset 长度；
3. 打印第一个 sample 的所有 keys；
4. 打印 image/action/episode 字段候选；
5. 保存检查结果到 `outputs/dataset_inspection.json`。

运行：

```bash
python scripts/00_inspect_dataset.py --config configs/default.yaml
```

### 12.2 `scripts/01_extract_features.py`

功能：

1. 加载数据集；
2. 提取图像和 action；
3. 使用 ResNet-18 提取特征；
4. 保存 `.npz` 特征缓存。

运行：

```bash
python scripts/01_extract_features.py --config configs/default.yaml
```

### 12.3 `scripts/02_select_coreset.py`

功能：

1. 加载 features；
2. 生成 episode-level split；
3. 在 train split 内执行不同数据选择方法；
4. 保存每种方法的 selected indices；
5. 保存每个样本的 visual_change、action_change、utility score。

运行：

```bash
python scripts/02_select_coreset.py --config configs/default.yaml --method brain_coreset
```

也支持：

```bash
python scripts/02_select_coreset.py --config configs/default.yaml --method all
```

### 12.4 `scripts/03_train_mlp.py`

功能：

1. 加载 features；
2. 根据 selection 文件构建训练集；
3. 训练 MLP；
4. 保存 checkpoint；
5. 保存训练日志。

运行：

```bash
python scripts/03_train_mlp.py --config configs/default.yaml --method random
python scripts/03_train_mlp.py --config configs/default.yaml --method brain_coreset
python scripts/03_train_mlp.py --config configs/default.yaml --method full
```

### 12.5 `scripts/04_evaluate.py`

功能：

1. 加载 checkpoint；
2. 在 val/test 上评估；
3. 保存预测结果和 MSE。

运行：

```bash
python scripts/04_evaluate.py --config configs/default.yaml --method brain_coreset
```

### 12.6 `scripts/05_make_plots.py`

功能：

1. 读取 metrics 和 selections；
2. 生成所有报告图像；
3. 复制到 `report_assets/figures/`。

运行：

```bash
python scripts/05_make_plots.py --config configs/default.yaml
```

### 12.7 `scripts/run_all.sh`

一键运行：

```bash
bash scripts/run_all.sh
```

内容应包括：

```bash
python scripts/00_inspect_dataset.py --config configs/default.yaml
python scripts/01_extract_features.py --config configs/default.yaml
python scripts/02_select_coreset.py --config configs/default.yaml --method all
python scripts/03_train_mlp.py --config configs/default.yaml --method random
python scripts/03_train_mlp.py --config configs/default.yaml --method action_change
python scripts/03_train_mlp.py --config configs/default.yaml --method visual_change
python scripts/03_train_mlp.py --config configs/default.yaml --method visual_action
python scripts/03_train_mlp.py --config configs/default.yaml --method brain_coreset
python scripts/03_train_mlp.py --config configs/default.yaml --method full
python scripts/04_evaluate.py --config configs/default.yaml --method all
python scripts/05_make_plots.py --config configs/default.yaml
```

---

## 13. 必须保存的输出文件

项目完成后至少应生成：

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
```

可选但推荐：

```text
outputs/figures/selection_timeline_episode_*.png
outputs/checkpoints/mlp_random.pt
outputs/checkpoints/mlp_brain_coreset.pt
outputs/checkpoints/mlp_full.pt
```

---

## 14. README 需要包含的内容

请 Codex 自动生成 `README.md`，内容包括：

1. 项目简介；
2. 课程题目对应关系；
3. 环境安装；
4. 数据下载与读取；
5. 一键运行命令；
6. 各脚本说明；
7. 输出文件说明；
8. 实验方法说明；
9. 报告中可使用的结果表格位置；
10. 常见错误与解决方式。

示例命令：

```bash
pip install -r requirements.txt
bash scripts/run_all.sh
```

如果 CUDA 不可用，需要自动回退 CPU。

---

## 15. 报告写作所需结果模板

Codex 需要额外生成一个 Markdown 文件：

```text
report_assets/experiment_summary.md
```

内容自动填充实验结果，格式如下：

```markdown
# 实验结果摘要

## 数据设置

- 数据集：lerobot/aloha_sim_transfer_cube_human
- 图像特征提取器：ResNet-18
- 动作维度：7
- 数据划分：episode-level 80/10/10
- 核心集比例：10%

## 主结果

| 方法 | 训练样本数 | Val MSE | Test MSE |
|---|---:|---:|---:|
| Random 10% | | | |
| Brain-inspired Coreset 10% | | | |
| Full 100% | | | |

## 结论草稿

在相同 10% 数据比例下，脑启发核心集选择方法相比随机采样取得了 [更低/相近/更高] 的动作预测 MSE。该结果说明 [需要根据实际结果自动写一句保守解释]。
```

注意：结论要保守，不要在结果不支持时强行声称显著提升。

---

## 16. 实现细节要求

### 16.1 随机种子

所有随机过程需要固定 seed：

```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

### 16.2 日志

所有脚本需要打印关键日志：

- 当前配置；
- 数据样本数；
- train/val/test 数量；
- 选择样本数量；
- 模型参数量；
- 每 epoch train/val loss；
- final test MSE。

### 16.3 错误处理

必须处理：

1. 数据字段名不匹配；
2. 没有 CUDA；
3. action 维度不是 7 或 14；
4. selection 文件不存在；
5. feature 文件不存在；
6. 某个 episode 样本数过少；
7. KMeans cluster 数量大于候选样本数量。

### 16.4 代码风格

要求：

- 函数命名清晰；
- 每个核心函数有 docstring；
- 不写过度复杂框架；
- 优先可跑通；
- 所有路径从 config 或命令行读取；
- 不在代码中硬编码绝对路径。

---

## 17. 预期实现顺序

请 Codex 按以下顺序实现，避免一次性生成不可调试的大工程：

1. 创建项目目录结构。
2. 写 `requirements.txt`、`configs/default.yaml`、`README.md`。
3. 实现 `dataset_utils.py` 和 `00_inspect_dataset.py`。
4. 确认能读取数据和字段。
5. 实现 ResNet-18 特征提取和 `.npz` 缓存。
6. 实现 split 和 random selection。
7. 实现 MLP 训练与评估。
8. 跑通 random 10% baseline。
9. 实现 action_change、visual_change、visual_action。
10. 实现 brain_coreset 主方法。
11. 跑通所有方法。
12. 实现可视化和 `experiment_summary.md`。
13. 最后清理 README 和注释。

---

## 18. 最低验收标准

项目最低验收标准：

1. `bash scripts/run_all.sh` 能从头运行到结束，或 README 中给出清晰分步命令。
2. 能读取 ALOHA 数据集。
3. 能提取图像特征。
4. 能训练 MLP。
5. 至少包含 `random` 和 `brain_coreset` 两种数据选择方法。
6. 能输出两种方法的 MSE 对比。
7. 能生成至少一张 MSE 对比图。
8. 代码有基本注释。
9. 报告素材保存在 `report_assets/`。

---

## 19. 推荐增强项

如果基础版本跑通，再实现：

1. 多随机种子实验：`seed=0,1,2`。
2. 消融实验：
   - Random
   - Action Change
   - Visual Change
   - Visual + Action
   - Brain Coreset
   - Full Data
3. 每个 episode 的选择时间轴可视化。
4. PCA 特征分布可视化。
5. 每个 action dimension 的 MSE 分析。
6. 自动生成 LaTeX 表格：

```text
report_assets/tables/results_latex.txt
```

---

## 20. 课程报告中的方法解释口径

实现的主方法在报告中可以命名为：

```text
Brain-inspired Utility-Diversity Coreset Selection, BUD-CS
```

解释：

该方法将人脑预测编码机制和任务效用过滤机制转化为可计算的数据选择准则。预测编码对应视觉特征变化，视觉变化越大说明当前状态越难由上一帧预测，信息增量越高；任务效用过滤对应动作变化，动作变化越大说明该帧更可能处于操作决策关键阶段；分布多样性约束用于避免核心集集中在少数高分但相似的状态上，从而提高 10% 子集对全任务状态空间的覆盖能力。

报告中不要声称该方法真正模拟大脑机制，只能说“受脑启发”或“类脑动机”。

---

## 21. 需要避免的问题

1. 不要训练完整 ACT / Diffusion Policy / OpenVLA，这超出课程题目要求。
2. 不要用测试集参与核心集选择。
3. 不要只报告训练 MSE，必须报告 val/test MSE。
4. 不要随机每帧划分 train/test，否则相邻帧泄露会导致结果虚高。
5. 不要让 coreset 方法和 random 方法使用不同 MLP 结构或训练超参数。
6. 不要在结果不提升时伪造结论；如果不提升，应分析原因：数据太小、特征过弱、动作标签噪声、选择策略偏向突变帧导致平稳控制样本不足等。

---

## 22. 最终交付物

Codex 完成后，项目应包含：

```text
vla_coreset_assignment.zip 或完整项目文件夹
```

其中应至少包括：

```text
README.md
requirements.txt
configs/default.yaml
scripts/*.py
src/*.py
outputs/metrics/results.csv
outputs/figures/*.png
report_assets/experiment_summary.md
report_assets/figures/*.png
```

最终课程提交时，可以将代码作为附录或压缩包，研究报告单独导出为 PDF。
