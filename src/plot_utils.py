from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from .utils import ensure_dir, load_json, ratio_to_percent


def _copy_to_report(fig_path: Path, report_dir: Path) -> None:
    ensure_dir(report_dir)
    shutil.copy2(fig_path, report_dir / fig_path.name)


def plot_mse_bar(results: dict[str, Any], figure_dir: str | Path, report_dir: str | Path) -> Path:
    methods = list(results.keys())
    values = [results[method]["mse_raw_test"] for method in methods]
    labels = [method.replace("_", "\n") for method in methods]

    figure_dir = ensure_dir(figure_dir)
    report_dir = ensure_dir(report_dir)
    path = figure_dir / "mse_bar.png"
    plt.figure(figsize=(9, 5))
    plt.bar(labels, values, color=["#6baed6", "#9ecae1", "#c6dbef", "#fdd0a2", "#fc9272", "#74c476"][: len(labels)])
    plt.ylabel("Test MSE (raw action space)")
    plt.title("Method Comparison on Action Prediction")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    _copy_to_report(path, report_dir)
    return path


def plot_per_dim_mse(results: dict[str, Any], figure_dir: str | Path, report_dir: str | Path) -> Path:
    figure_dir = ensure_dir(figure_dir)
    report_dir = ensure_dir(report_dir)
    path = figure_dir / "per_dim_mse.png"
    plt.figure(figsize=(10, 5))
    for method, payload in results.items():
        plt.plot(payload["per_dim_mse_raw_test"], marker="o", label=method)
    plt.xlabel("Action dimension")
    plt.ylabel("Per-dim Test MSE")
    plt.title("Per-dimension Action Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    _copy_to_report(path, report_dir)
    return path


def plot_score_distribution(score_path: str | Path, figure_dir: str | Path, report_dir: str | Path) -> Path:
    data = np.load(score_path)
    figure_dir = ensure_dir(figure_dir)
    report_dir = ensure_dir(report_dir)
    path = figure_dir / "score_distribution.png"
    plt.figure(figsize=(10, 6))
    for key, color in [("visual_change", "#3182bd"), ("action_change", "#31a354"), ("utility", "#e6550d")]:
        plt.hist(data[key], bins=50, alpha=0.45, label=key, color=color)
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.title("Selection Score Distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    _copy_to_report(path, report_dir)
    return path


def plot_feature_pca(
    features: np.ndarray,
    split_indices: dict[str, np.ndarray],
    random_indices: np.ndarray,
    brain_indices: np.ndarray,
    figure_dir: str | Path,
    report_dir: str | Path,
) -> Path:
    figure_dir = ensure_dir(figure_dir)
    report_dir = ensure_dir(report_dir)
    path = figure_dir / "feature_pca.png"
    train_idx = split_indices["train"]
    pca = PCA(n_components=2, random_state=0)
    projected = pca.fit_transform(features[train_idx])

    random_mask = np.isin(train_idx, random_indices)
    brain_mask = np.isin(train_idx, brain_indices)

    plt.figure(figsize=(8, 7))
    plt.scatter(projected[:, 0], projected[:, 1], s=10, alpha=0.15, label="train full", color="#636363")
    plt.scatter(projected[random_mask, 0], projected[random_mask, 1], s=18, alpha=0.7, label="random 10%", color="#3182bd")
    plt.scatter(projected[brain_mask, 0], projected[brain_mask, 1], s=18, alpha=0.7, label="brain coreset 10%", color="#de2d26")
    plt.title("PCA of Train Features")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    _copy_to_report(path, report_dir)
    return path


def plot_selection_timeline(
    score_path: str | Path,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    random_indices: np.ndarray,
    brain_indices: np.ndarray,
    figure_dir: str | Path,
) -> list[Path]:
    data = np.load(score_path)
    utility = data["utility"]
    figure_dir = ensure_dir(figure_dir)
    paths: list[Path] = []

    for episode in np.unique(episode_indices)[:3]:
        ep_idx = np.where(episode_indices == episode)[0]
        order = ep_idx[np.argsort(frame_indices[ep_idx])]
        path = figure_dir / f"selection_timeline_episode_{episode}.png"
        plt.figure(figsize=(10, 4))
        plt.plot(frame_indices[order], utility[order], color="#525252", label="utility")
        random_points = order[np.isin(order, random_indices)]
        brain_points = order[np.isin(order, brain_indices)]
        plt.scatter(frame_indices[random_points], utility[random_points], color="#3182bd", label="random", s=30)
        plt.scatter(frame_indices[brain_points], utility[brain_points], color="#de2d26", label="brain", s=30)
        plt.xlabel("Frame index")
        plt.ylabel("Utility score")
        plt.title(f"Episode {episode} selection timeline")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()
        paths.append(path)
    return paths


def write_experiment_summary(
    results: dict[str, Any],
    output_path: str | Path,
    dataset_name: str,
    ratio: float,
) -> Path:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    random_mse = results.get("random", {}).get("mse_raw_test")
    brain_mse = results.get("brain_coreset", {}).get("mse_raw_test")
    full_mse = results.get("full", {}).get("mse_raw_test")

    conclusion = "结果尚未完整生成。"
    if random_mse is not None and brain_mse is not None:
        delta = brain_mse - random_mse
        if abs(delta) < max(1e-6, 0.02 * random_mse):
            relation = "相近"
            explanation = "说明当前核心集策略在该设置下表现稳定，但优势还不够明显，可能需要更强特征或更多随机种子验证。"
        elif delta < 0:
            relation = "更低"
            explanation = "说明受脑启发的效用-多样性筛选在相同比例下保留了更有训练价值的视觉-动作样本。"
        else:
            relation = "更高"
            explanation = "说明当前核心集策略可能过度偏向突变帧，后续可继续平衡关键帧与平稳控制样本。"
        conclusion = f"在相同 {ratio_to_percent(ratio)}% 数据比例下，脑启发核心集选择方法相比随机采样取得了{relation}的动作预测 MSE。该结果{explanation}"

    table_rows = [
        ("Random 10%", results.get("random")),
        ("Brain-inspired Coreset 10%", results.get("brain_coreset")),
        ("Full 100%", results.get("full")),
    ]
    lines = [
        "# 实验结果摘要",
        "",
        "## 数据设置",
        "",
        f"- 数据集：{dataset_name}",
        "- 图像特征提取器：ResNet-18",
        "- 动作维度：7",
        "- 数据划分：episode-level 80/10/10",
        f"- 核心集比例：{ratio_to_percent(ratio)}%",
        "",
        "## 主结果",
        "",
        "| 方法 | 训练样本数 | Val MSE | Test MSE |",
        "|---|---:|---:|---:|",
    ]
    for name, payload in table_rows:
        if payload is None:
            lines.append(f"| {name} |  |  |  |")
        else:
            lines.append(
                f"| {name} | {payload['n_train']} | {payload['mse_raw_val']:.6f} | {payload['mse_raw_test']:.6f} |"
            )
    lines.extend(["", "## 结论草稿", "", conclusion, ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def write_latex_table(results: dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    lines = [
        "\\begin{tabular}{lccc}",
        "\\hline",
        "Method & Train Samples & Val MSE & Test MSE \\\\",
        "\\hline",
    ]
    for method, payload in results.items():
        lines.append(
            f"{method.replace('_', ' ')} & {payload['n_train']} & {payload['mse_raw_val']:.6f} & {payload['mse_raw_test']:.6f} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path

