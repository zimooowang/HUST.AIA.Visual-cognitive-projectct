from __future__ import annotations

import inspect
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from .utils import find_candidate_keys, flatten_sample, get_nested_value


@dataclass
class FrameRecord:
    image: Image.Image
    action: np.ndarray
    episode_index: int
    frame_index: int
    instruction: str
    timestamp: float | None


def _try_load_lerobot(repo_id: str, cache_dir: str | None, download_videos: bool = False, video_backend: str = "pyav") -> tuple[Any, str] | None:
    try:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except Exception:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except Exception:
        return None

    try:
        if cache_dir:
            cache_path = Path(cache_dir)
            os.environ.setdefault("HF_HOME", str(cache_path / "hf_home"))
            os.environ.setdefault("XDG_CACHE_HOME", str(cache_path / "xdg_cache"))
        signature = inspect.signature(LeRobotDataset)
        kwargs: dict[str, Any] = {}
        if "repo_id" in signature.parameters:
            kwargs["repo_id"] = repo_id
        elif "dataset_id" in signature.parameters:
            kwargs["dataset_id"] = repo_id
        else:
            return None
        if cache_dir and "root" in signature.parameters:
            kwargs["root"] = cache_dir
        if "download_videos" in signature.parameters:
            kwargs["download_videos"] = download_videos
        if "video_backend" in signature.parameters:
            kwargs["video_backend"] = video_backend
        dataset = LeRobotDataset(**kwargs)
        return dataset, "lerobot"
    except Exception:
        return None


def _try_load_hf(repo_id: str, cache_dir: str | None, split: str | None) -> tuple[Any, str]:
    from datasets import load_dataset

    if split is not None:
        try:
            dataset = load_dataset(repo_id, split=split, cache_dir=cache_dir)
            return dataset, "datasets"
        except Exception:
            pass

    dataset = load_dataset(repo_id, cache_dir=cache_dir)
    if hasattr(dataset, "keys"):
        keys = list(dataset.keys())
        if split and split in dataset:
            return dataset[split], "datasets"
        if "train" in dataset:
            return dataset["train"], "datasets"
        return dataset[keys[0]], "datasets"
    return dataset, "datasets"


def load_dataset_compat(repo_id: str, cache_dir: str | None = None, split: str | None = "train", download_videos: bool = False, video_backend: str = "pyav") -> tuple[Any, str]:
    dataset = _try_load_lerobot(repo_id, cache_dir, download_videos=download_videos, video_backend=video_backend)
    if dataset is not None:
        return dataset
    return _try_load_hf(repo_id, cache_dir, split)


def inspect_dataset(dataset: Any, explicit_image_key: str | None = None, explicit_action_key: str | None = None, explicit_episode_key: str | None = None, explicit_instruction_key: str | None = None, explicit_timestamp_key: str | None = None) -> dict[str, Any]:
    length = len(dataset)
    flat_sample: dict[str, Any] = {}
    sample_preview_types: dict[str, str] = {}

    if hasattr(dataset, "features") and isinstance(dataset.features, dict):
        flat_keys = sorted(dataset.features.keys())
        sample_preview_types = {key: str(type(value)) for key, value in dataset.features.items()}
        try:
            sample = dataset[0]
            flat_sample = flatten_sample(sample)
            flat_keys = sorted(set(flat_keys).union(flat_sample.keys()))
            sample_preview_types.update({key: str(type(value)) for key, value in flat_sample.items()})
        except Exception:
            pass
    else:
        sample = dataset[0]
        flat_sample = flatten_sample(sample)
        flat_keys = sorted(flat_sample.keys())
        sample_preview_types = {key: str(type(value)) for key, value in flat_sample.items()}

    image_candidates = []
    preferred_images = [
        "observation.images.top",
        "observation.images.cam_high",
        "observation.image",
        "image",
    ]
    for key in preferred_images:
        if key in flat_keys:
            image_candidates.append(key)
    for key in find_candidate_keys(flat_keys, ["image", "images"]):
        if key not in image_candidates:
            image_candidates.append(key)

    action_candidates = []
    if "action" in flat_keys:
        action_candidates.append("action")
    for key in find_candidate_keys(flat_keys, ["action"]):
        if key not in action_candidates:
            action_candidates.append(key)

    episode_candidates = []
    preferred_episode = ["episode_index", "episode_idx", "episode.id"]
    for key in preferred_episode:
        if key in flat_keys:
            episode_candidates.append(key)
    for key in find_candidate_keys(flat_keys, ["episode"]):
        if key not in episode_candidates:
            episode_candidates.append(key)

    instruction_candidates = []
    for key in find_candidate_keys(flat_keys, ["instruction", "language", "task", "prompt"]):
        if key not in instruction_candidates:
            instruction_candidates.append(key)

    timestamp_candidates = []
    for key in find_candidate_keys(flat_keys, ["timestamp", "time", "frame_index", "index"]):
        if key not in timestamp_candidates:
            timestamp_candidates.append(key)

    return {
        "length": length,
        "sample_keys": flat_keys,
        "image_candidates": image_candidates,
        "action_candidates": action_candidates,
        "episode_candidates": episode_candidates,
        "instruction_candidates": instruction_candidates,
        "timestamp_candidates": timestamp_candidates,
        "selected_keys": {
            "image_key": explicit_image_key or (image_candidates[0] if image_candidates else None),
            "action_key": explicit_action_key or (action_candidates[0] if action_candidates else None),
            "episode_key": explicit_episode_key or (episode_candidates[0] if episode_candidates else None),
            "instruction_key": explicit_instruction_key or (instruction_candidates[0] if instruction_candidates else None),
            "timestamp_key": explicit_timestamp_key or (timestamp_candidates[0] if timestamp_candidates else None),
        },
        "sample_preview_types": sample_preview_types,
    }


def _image_from_value(value: Any) -> Image.Image:
    try:
        import torch
    except Exception:
        torch = None

    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if torch is not None and isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
            tensor = tensor.permute(1, 2, 0)
        array = tensor.numpy()
        if array.dtype != np.uint8:
            if array.max() <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array).convert("RGB")
    if isinstance(value, np.ndarray):
        array = value
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array).convert("RGB")
    if isinstance(value, dict):
        if "bytes" in value and value["bytes"] is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if "path" in value and value["path"]:
            return Image.open(value["path"]).convert("RGB")
    raise TypeError(f"Unsupported image value type: {type(value)}")


def _action_from_value(value: Any, arm: str) -> np.ndarray:
    arm = str(arm).lower()
    action = np.asarray(value, dtype=np.float32).reshape(-1)
    if action.size == 14:
        if arm == "left":
            return action[:7]
        if arm == "right":
            return action[7:14]
        if arm in {"both", "all", "full"}:
            return action
        raise ValueError(f"Unsupported arm selection: {arm}")
    if action.size == 7:
        return action
    raise ValueError(f"Unsupported action dimension: {action.size}")


def build_records(dataset: Any, image_key: str, action_key: str, arm: str, episode_key: str | None = None, instruction_key: str | None = None, timestamp_key: str | None = None, max_frames: int | None = None) -> tuple[list[FrameRecord], dict[str, Any]]:
    records: list[FrameRecord] = []
    action_dims: list[int] = []
    fallback_episode = 0
    last_episode_value: Any = None
    iterator = range(len(dataset))
    if max_frames is not None:
        iterator = range(min(len(dataset), int(max_frames)))

    for frame_index in tqdm(iterator, desc="Building records"):
        sample = dataset[frame_index]
        image_value = get_nested_value(sample, image_key)
        action_value = get_nested_value(sample, action_key)
        if image_value is None:
            raise KeyError(f"Image key '{image_key}' not found in sample.")
        if action_value is None:
            raise KeyError(f"Action key '{action_key}' not found in sample.")

        episode_value = get_nested_value(sample, episode_key) if episode_key else None
        if episode_value is None:
            timestamp_value = get_nested_value(sample, timestamp_key) if timestamp_key else None
            if frame_index == 0:
                fallback_episode = 0
            elif timestamp_value is not None and last_episode_value is not None and timestamp_value < last_episode_value:
                fallback_episode += 1
            episode_idx = fallback_episode
            last_episode_value = timestamp_value
        else:
            if isinstance(episode_value, (list, tuple, np.ndarray)):
                episode_idx = int(np.asarray(episode_value).reshape(-1)[0])
            else:
                episode_idx = int(episode_value)

        instruction_value = get_nested_value(sample, instruction_key) if instruction_key else ""
        if instruction_value is None:
            instruction_value = ""

        timestamp_value = get_nested_value(sample, timestamp_key) if timestamp_key else None
        if isinstance(timestamp_value, (list, tuple, np.ndarray)):
            timestamp_value = float(np.asarray(timestamp_value).reshape(-1)[0])
        elif timestamp_value is not None:
            timestamp_value = float(timestamp_value)

        image = _image_from_value(image_value)
        action = _action_from_value(action_value, arm)
        action_dims.append(np.asarray(action_value).reshape(-1).size)
        records.append(
            FrameRecord(
                image=image,
                action=action,
                episode_index=episode_idx,
                frame_index=frame_index,
                instruction=str(instruction_value),
                timestamp=timestamp_value,
            )
        )

    metadata = {
        "n_records": len(records),
        "raw_action_dims": sorted(set(action_dims)),
        "final_action_dim": int(records[0].action.shape[0]) if records else 0,
        "arm": arm,
    }
    return records, metadata
