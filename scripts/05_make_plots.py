#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.coreset import indices_from_split
from src.feature_extractor import load_feature_cache
from src.plot_utils import plot_feature_pca, plot_mse_bar, plot_per_dim_mse, plot_score_distribution, plot_selection_timeline, write_experiment_summary, write_latex_table
from src.utils import ensure_output_dirs, load_config, load_json, ratio_to_percent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create report figures and summary files.")
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_output_dirs(config)

    metrics_path = Path(config["paths"]["metrics_dir"]) / "results.json"
    score_path = Path(config["paths"]["selection_dir"]) / f"scores_seed{config['seed']}.npz"
    split_path = Path(config["paths"]["selection_dir"]) / f"split_seed{config['seed']}.json"
    feature_path = Path(config["paths"]["feature_dir"]) / f"features_{config['features']['extractor']}.npz"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    results = load_json(metrics_path)
    figure_dir = Path(config["paths"]["figure_dir"])
    report_figure_dir = Path(config["paths"]["report_asset_dir"]) / "figures"

    plot_mse_bar(results, figure_dir, report_figure_dir)
    plot_per_dim_mse(results, figure_dir, report_figure_dir)
    if score_path.exists():
        plot_score_distribution(score_path, figure_dir, report_figure_dir)

    if feature_path.exists() and split_path.exists():
        payload = load_feature_cache(feature_path)
        split_indices = indices_from_split(payload["episode_indices"], load_json(split_path))
        percent = ratio_to_percent(float(config["selection"]["ratio"]))
        random_indices = np.load(Path(config["paths"]["selection_dir"]) / f"random_{percent}_seed{config['seed']}.npy")
        brain_indices = np.load(Path(config["paths"]["selection_dir"]) / f"brain_coreset_{percent}_seed{config['seed']}.npy")
        plot_feature_pca(payload["features"], split_indices, random_indices, brain_indices, figure_dir, report_figure_dir)
        if score_path.exists():
            plot_selection_timeline(
                score_path=score_path,
                episode_indices=payload["episode_indices"],
                frame_indices=payload["frame_indices"],
                random_indices=random_indices,
                brain_indices=brain_indices,
                figure_dir=figure_dir,
            )

    write_experiment_summary(
        results=results,
        output_path=Path(config["paths"]["report_asset_dir"]) / "experiment_summary.md",
        dataset_name=config["dataset"]["repo_id"],
        ratio=float(config["selection"]["ratio"]),
    )
    write_latex_table(results, Path(config["paths"]["report_asset_dir"]) / "tables" / "results_latex.txt")
    print(f"Saved plots to {figure_dir} and report assets to {config['paths']['report_asset_dir']}")


if __name__ == "__main__":
    main()
