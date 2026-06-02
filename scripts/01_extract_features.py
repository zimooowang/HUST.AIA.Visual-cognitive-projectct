#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_utils import build_records, inspect_dataset, load_dataset_compat
from src.feature_extractor import extract_feature_cache, save_feature_cache
from src.utils import apply_overrides, ensure_output_dirs, load_config, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract image features and cache them.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--repo_id")
    parser.add_argument("--cache_dir")
    parser.add_argument("--image_key")
    parser.add_argument("--action_key")
    parser.add_argument("--episode_key")
    parser.add_argument("--instruction_key")
    parser.add_argument("--timestamp_key")
    parser.add_argument("--arm")
    parser.add_argument("--max_frames", type=int)
    parser.add_argument("--video_backend")
    parser.add_argument("--extractor")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--language_mode")
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
            "image_key": ("dataset", "image_key"),
            "action_key": ("dataset", "action_key"),
            "episode_key": ("dataset", "episode_key"),
            "instruction_key": ("dataset", "instruction_key"),
            "timestamp_key": ("dataset", "timestamp_key"),
            "arm": ("dataset", "arm"),
            "max_frames": ("dataset", "max_frames"),
            "video_backend": ("dataset", "video_backend"),
            "extractor": ("features", "extractor"),
            "batch_size": ("features", "batch_size"),
            "device": ("features", "device"),
            "language_mode": ("features", "language_mode"),
        },
    )
    ensure_output_dirs(config)
    seed_everything(int(config["seed"]))

    dataset_cfg = config["dataset"]
    feature_cfg = config["features"]
    dataset, loader_name = load_dataset_compat(
        repo_id=dataset_cfg["repo_id"],
        cache_dir=dataset_cfg.get("cache_dir"),
        split=dataset_cfg.get("split", "train"),
        download_videos=True,
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
    selected = info["selected_keys"]
    if not selected["image_key"] or not selected["action_key"]:
        raise ValueError(f"Unable to determine image/action keys. Available keys: {info['sample_keys']}")

    records, record_meta = build_records(
        dataset=dataset,
        image_key=selected["image_key"],
        action_key=selected["action_key"],
        arm=dataset_cfg.get("arm", "left"),
        episode_key=selected.get("episode_key"),
        instruction_key=selected.get("instruction_key"),
        timestamp_key=selected.get("timestamp_key"),
        max_frames=dataset_cfg.get("max_frames"),
    )
    cache = extract_feature_cache(
        records=records,
        extractor_name=feature_cfg.get("extractor", "resnet18"),
        batch_size=int(feature_cfg.get("batch_size", 64)),
        device_preference=feature_cfg.get("device", "cpu"),
        language_mode=feature_cfg.get("language_mode", "constant"),
        language_dim=int(feature_cfg.get("language_dim", 1)),
        text_hash_dim=int(feature_cfg.get("text_hash_dim", 16)),
    )
    cache["metadata"].update(record_meta)
    cache["metadata"]["loader"] = loader_name
    cache["metadata"]["selected_keys"] = selected
    cache["metadata"]["repo_id"] = dataset_cfg["repo_id"]

    output_path = Path(config["paths"]["feature_dir"]) / f"features_{feature_cfg.get('extractor', 'resnet18')}.npz"
    save_feature_cache(output_path, cache)
    print(json.dumps(cache["metadata"], ensure_ascii=False, indent=2))
    print(f"Saved feature cache to {output_path}")


if __name__ == "__main__":
    main()
