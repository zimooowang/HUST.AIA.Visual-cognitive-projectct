from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy objects into plain Python types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_output_dirs(config: dict[str, Any]) -> None:
    paths = config.get("paths", {})
    ensure_dir(paths.get("output_dir", "outputs"))
    ensure_dir(paths.get("feature_dir", "outputs/features"))
    ensure_dir(paths.get("selection_dir", "outputs/selections"))
    ensure_dir(paths.get("checkpoint_dir", "outputs/checkpoints"))
    ensure_dir(paths.get("metrics_dir", "outputs/metrics"))
    ensure_dir(paths.get("figure_dir", "outputs/figures"))
    ensure_dir(paths.get("report_asset_dir", "report_assets"))
    ensure_dir(Path(paths.get("report_asset_dir", "report_assets")) / "figures")
    ensure_dir(Path(paths.get("report_asset_dir", "report_assets")) / "tables")


def deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def apply_overrides(config: dict[str, Any], args: Any, mapping: dict[str, tuple[str, str]]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for arg_name, (section, key) in mapping.items():
        value = getattr(args, arg_name, None)
        if value is None:
            continue
        overrides.setdefault(section, {})[key] = value
    return deep_update(config, overrides)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def flatten_sample(sample: Any, prefix: str = "") -> dict[str, Any]:
    items: dict[str, Any] = {}
    if isinstance(sample, dict):
        for key, value in sample.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            items.update(flatten_sample(value, next_prefix))
    else:
        items[prefix] = sample
    return items


def get_nested_value(sample: Any, dotted_key: str | None, default: Any = None) -> Any:
    if dotted_key is None:
        return default
    if isinstance(sample, dict) and dotted_key in sample:
        return sample[dotted_key]
    current = sample
    for part in dotted_key.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    return current


def find_candidate_keys(flat_keys: list[str], keywords: list[str]) -> list[str]:
    candidates = []
    lower_keywords = [keyword.lower() for keyword in keywords]
    for key in flat_keys:
        key_lower = key.lower()
        if any(keyword in key_lower for keyword in lower_keywords):
            candidates.append(key)
    return candidates


def get_device(preferred: str | None = None) -> str:
    preferred = preferred or "cpu"
    try:
        import torch

        if preferred.startswith("cuda") and torch.cuda.is_available():
            return preferred
    except Exception:
        return "cpu"
    return "cpu"


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, cls=NumpyEncoder)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def zscore(values: np.ndarray, mask: np.ndarray | None = None, eps: float = 1e-8) -> np.ndarray:
    if mask is None:
        subset = values
    else:
        subset = values[mask]
    mean = subset.mean() if len(subset) else 0.0
    std = subset.std() if len(subset) else 1.0
    std = max(float(std), eps)
    return (values - mean) / std


def ratio_to_percent(ratio: float) -> int:
    return int(round(ratio * 100))


def format_float(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "nan"
    return f"{value:.6f}"
