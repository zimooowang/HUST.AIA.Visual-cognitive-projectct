from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model import MLPRegressor
from .utils import ensure_dir, format_float, get_device, load_json, save_json


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def _normalize(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((features - mean) / std).astype(np.float32)


def _predict(model: MLPRegressor, features: np.ndarray, device: str, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start : start + batch_size]).to(device)
            outputs.append(model(batch).cpu().numpy())
    return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, model.network[-1].out_features), dtype=np.float32)


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    features: np.ndarray,
    actions: np.ndarray,
    split_indices: dict[str, np.ndarray],
    device_preference: str,
) -> dict[str, Any]:
    checkpoint = load_checkpoint(checkpoint_path)
    device = get_device(device_preference)

    model = MLPRegressor(
        input_dim=int(checkpoint["input_dim"]),
        output_dim=int(checkpoint["output_dim"]),
        hidden_dims=checkpoint["hidden_dims"],
        dropout=float(checkpoint["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    x_mean = checkpoint["x_mean"].astype(np.float32)
    x_std = checkpoint["x_std"].astype(np.float32)
    y_mean = checkpoint["y_mean"].astype(np.float32)
    y_std = checkpoint["y_std"].astype(np.float32)
    normalized_features = _normalize(features, x_mean, x_std)

    metrics: dict[str, Any] = {
        "method": checkpoint["method"],
        "seed": int(checkpoint["seed"]),
        "n_train": int(len(checkpoint["train_indices"])),
        "checkpoint_path": str(checkpoint_path),
    }
    predictions_payload: dict[str, np.ndarray] = {}

    for split_name in ["val", "test"]:
        indices = split_indices[split_name]
        pred_norm = _predict(model, normalized_features[indices], device)
        gt_norm = ((actions[indices] - y_mean) / y_std).astype(np.float32)
        pred_raw = pred_norm * y_std + y_mean
        gt_raw = actions[indices]

        mse_norm = np.mean((pred_norm - gt_norm) ** 2, axis=0)
        mse_raw = np.mean((pred_raw - gt_raw) ** 2, axis=0)

        metrics[f"mse_norm_{split_name}"] = float(mse_norm.mean())
        metrics[f"mse_raw_{split_name}"] = float(mse_raw.mean())
        metrics[f"per_dim_mse_norm_{split_name}"] = mse_norm.tolist()
        metrics[f"per_dim_mse_raw_{split_name}"] = mse_raw.tolist()
        predictions_payload[f"{split_name}_pred"] = pred_raw.astype(np.float32)
        predictions_payload[f"{split_name}_gt"] = gt_raw.astype(np.float32)

    return {"metrics": metrics, "predictions": predictions_payload}


def update_results_files(metrics_dir: str | Path, metrics: dict[str, Any], ratio: float) -> None:
    metrics_dir = ensure_dir(metrics_dir)
    csv_path = metrics_dir / "results.csv"
    json_path = metrics_dir / "results.json"
    method_path = metrics_dir / f"{metrics['method']}_seed{metrics['seed']}.json"

    row = {
        "method": metrics["method"],
        "ratio": ratio,
        "seed": metrics["seed"],
        "n_train": metrics["n_train"],
        "mse_norm_val": metrics["mse_norm_val"],
        "mse_norm_test": metrics["mse_norm_test"],
        "mse_raw_val": metrics["mse_raw_val"],
        "mse_raw_test": metrics["mse_raw_test"],
    }

    existing_rows: list[dict[str, Any]] = []
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            existing_rows = list(csv.DictReader(handle))
    existing_rows = [
        current
        for current in existing_rows
        if not (current["method"] == row["method"] and int(current["seed"]) == int(row["seed"]))
    ]
    existing_rows.append({key: format_float(value) if isinstance(value, float) else value for key, value in row.items()})
    existing_rows = sorted(existing_rows, key=lambda item: item["method"])

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerows(existing_rows)

    aggregate = load_json(json_path) if json_path.exists() else {}
    aggregate[metrics["method"]] = metrics
    save_json(json_path, aggregate)
    save_json(method_path, metrics)
