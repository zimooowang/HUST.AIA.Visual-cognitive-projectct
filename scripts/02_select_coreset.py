#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.coreset import build_selections, generate_episode_split, indices_from_split, save_scores, save_selection_files, save_split
from src.feature_extractor import load_feature_cache
from src.utils import apply_overrides, ensure_output_dirs, load_config, ratio_to_percent, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create episode split and selection indices.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--method", default="all")
    parser.add_argument("--ratio", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = apply_overrides(config, args, {"ratio": ("selection", "ratio")})
    ensure_output_dirs(config)
    seed_everything(int(config["seed"]))

    feature_path = Path(config["paths"]["feature_dir"]) / f"features_{config['features']['extractor']}.npz"
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature cache not found: {feature_path}")

    feature_payload = load_feature_cache(feature_path)
    split_payload = generate_episode_split(
        feature_payload["episode_indices"],
        train_ratio=float(config["dataset"]["train_ratio"]),
        val_ratio=float(config["dataset"]["val_ratio"]),
        test_ratio=float(config["dataset"]["test_ratio"]),
        seed=int(config["seed"]),
    )
    split_path = Path(config["paths"]["selection_dir"]) / f"split_seed{config['seed']}.json"
    save_split(split_path, split_payload)

    split_indices = indices_from_split(feature_payload["episode_indices"], split_payload)
    selections, scores = build_selections(
        features=feature_payload["features"],
        actions=feature_payload["actions"],
        episode_indices=feature_payload["episode_indices"],
        frame_indices=feature_payload["frame_indices"],
        split_indices=split_indices,
        selection_cfg=config["selection"],
        seed=int(config["seed"]),
    )
    score_path = Path(config["paths"]["selection_dir"]) / f"scores_seed{config['seed']}.npz"
    save_scores(score_path, scores)
    saved_paths = save_selection_files(
        selection_dir=config["paths"]["selection_dir"],
        selections=selections,
        ratio=float(config["selection"]["ratio"]),
        seed=int(config["seed"]),
    )

    if args.method != "all" and args.method not in saved_paths:
        raise ValueError(f"Unknown method: {args.method}")

    print(
        json.dumps(
            {
                "split_path": str(split_path),
                "score_path": str(score_path),
                "saved_methods": list(saved_paths.keys()) if args.method == "all" else [args.method],
                "ratio_percent": ratio_to_percent(float(config["selection"]["ratio"])),
                "train_samples": len(split_indices["train"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
