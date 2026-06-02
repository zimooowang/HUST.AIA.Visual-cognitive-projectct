#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.coreset import indices_from_split
from src.evaluate import evaluate_checkpoint, update_results_files
from src.feature_extractor import load_feature_cache
from src.utils import ensure_output_dirs, load_config, load_json, ratio_to_percent, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved checkpoints on val/test splits.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--method", default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_output_dirs(config)
    seed_everything(int(config["seed"]))

    feature_path = Path(config["paths"]["feature_dir"]) / f"features_{config['features']['extractor']}.npz"
    split_path = Path(config["paths"]["selection_dir"]) / f"split_seed{config['seed']}.json"
    payload = load_feature_cache(feature_path)
    split_payload = load_json(split_path)
    split_indices = indices_from_split(payload["episode_indices"], split_payload)

    default_methods = [*config["selection"]["methods"], "full"]
    methods = default_methods if args.method == "all" else [args.method]
    all_metrics = []

    for method in methods:
        checkpoint_path = Path(config["paths"]["checkpoint_dir"]) / f"mlp_{method}_seed{config['seed']}.pt"
        if not checkpoint_path.exists():
            print(f"Skipping {method}: checkpoint not found at {checkpoint_path}")
            continue

        evaluation = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            features=payload["features"],
            actions=payload["actions"],
            split_indices=split_indices,
            device_preference=config["training"].get("device", "cpu"),
        )
        metrics = evaluation["metrics"]
        ratio = 1.0 if method == "full" else float(config["selection"]["ratio"])
        update_results_files(config["paths"]["metrics_dir"], metrics, ratio=ratio)

        if config["logging"].get("save_predictions", True):
            pred_path = Path(config["paths"]["metrics_dir"]) / f"predictions_{method}_seed{config['seed']}.npz"
            np.savez_compressed(pred_path, **evaluation["predictions"])
        all_metrics.append(metrics)

    print(json.dumps(all_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
