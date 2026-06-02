#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_utils import inspect_dataset, load_dataset_compat
from src.utils import apply_overrides, ensure_output_dirs, load_config, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect LeRobot/HF dataset structure.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--repo_id")
    parser.add_argument("--cache_dir")
    parser.add_argument("--split")
    parser.add_argument("--image_key")
    parser.add_argument("--action_key")
    parser.add_argument("--episode_key")
    parser.add_argument("--instruction_key")
    parser.add_argument("--timestamp_key")
    parser.add_argument("--video_backend")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = apply_overrides(
        config,
        args,
        {
            "repo_id": ("dataset", "repo_id"),
            "cache_dir": ("dataset", "cache_dir"),
            "split": ("dataset", "split"),
            "image_key": ("dataset", "image_key"),
            "action_key": ("dataset", "action_key"),
            "episode_key": ("dataset", "episode_key"),
            "instruction_key": ("dataset", "instruction_key"),
            "timestamp_key": ("dataset", "timestamp_key"),
            "video_backend": ("dataset", "video_backend"),
        },
    )
    ensure_output_dirs(config)
    seed_everything(int(config["seed"]))

    dataset_cfg = config["dataset"]
    dataset, loader_name = load_dataset_compat(
        repo_id=dataset_cfg["repo_id"],
        cache_dir=dataset_cfg.get("cache_dir"),
        split=dataset_cfg.get("split", "train"),
        download_videos=False,
        video_backend=dataset_cfg.get("video_backend", "pyav"),
    )
    info = inspect_dataset(
        dataset,
        explicit_image_key=dataset_cfg.get("image_key"),
        explicit_action_key=dataset_cfg.get("action_key"),
        explicit_episode_key=dataset_cfg.get("episode_key"),
        explicit_instruction_key=dataset_cfg.get("instruction_key"),
        explicit_timestamp_key=dataset_cfg.get("timestamp_key"),
    )
    info["loader"] = loader_name
    output_path = Path(config["paths"]["output_dir"]) / "dataset_inspection.json"
    save_json(output_path, info)

    print(json.dumps({"loader": loader_name, "length": info["length"]}, ensure_ascii=False, indent=2))
    print(f"Selected keys: {info['selected_keys']}")
    print(f"Saved inspection to {output_path}")


if __name__ == "__main__":
    main()
