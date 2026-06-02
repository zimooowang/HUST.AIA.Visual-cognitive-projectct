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
from src.feature_extractor import load_feature_cache
from src.train import train_mlp
from src.utils import ensure_output_dirs, load_config, load_json, ratio_to_percent, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MLP action predictor.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--method", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_output_dirs(config)
    seed_everything(int(config["seed"]))

    feature_path = Path(config["paths"]["feature_dir"]) / f"features_{config['features']['extractor']}.npz"
    split_path = Path(config["paths"]["selection_dir"]) / f"split_seed{config['seed']}.json"
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature cache not found: {feature_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    payload = load_feature_cache(feature_path)
    split_payload = load_json(split_path)
    split_indices = indices_from_split(payload["episode_indices"], split_payload)

    if args.method == "full":
        train_indices = split_indices["train"]
    else:
        percent = ratio_to_percent(float(config["selection"]["ratio"]))
        selection_path = Path(config["paths"]["selection_dir"]) / f"{args.method}_{percent}_seed{config['seed']}.npy"
        if not selection_path.exists():
            raise FileNotFoundError(f"Selection file not found: {selection_path}")
        train_indices = np.load(selection_path)

    result = train_mlp(
        features=payload["features"],
        actions=payload["actions"],
        split_indices=split_indices,
        train_indices=train_indices,
        method=args.method,
        training_cfg=config["training"],
        checkpoint_dir=config["paths"]["checkpoint_dir"],
        seed=int(config["seed"]),
    )
    history_path = Path(config["paths"]["checkpoint_dir"]) / f"mlp_{args.method}_seed{config['seed']}_history.json"
    save_json(history_path, {"history": result.history, "param_count": result.param_count, "n_train": int(len(train_indices))})
    print(
        json.dumps(
            {
                "method": args.method,
                "n_train": int(len(train_indices)),
                "param_count": result.param_count,
                "checkpoint_path": str(result.checkpoint_path),
                "history_path": str(history_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
