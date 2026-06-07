from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .model import MLPRegressor, count_parameters
from .utils import ensure_dir, get_device


@dataclass
class TrainResult:
    """Container for training outputs used by later evaluation scripts."""

    model: MLPRegressor
    history: list[dict[str, float]]
    checkpoint_path: Path
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    param_count: int


def _standardize(train_array: np.ndarray, full_array: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize data using statistics computed only from the training set.

    This prevents validation/test information from leaking into training while
    still allowing the whole dataset to be transformed consistently.
    """

    mean = train_array.mean(axis=0, keepdims=True).astype(np.float32)
    std = train_array.std(axis=0, keepdims=True).astype(np.float32)
    # Avoid division by zero for nearly constant dimensions.
    std = np.maximum(std, eps)
    normalized = (full_array - mean) / std
    return normalized.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def prepare_datasets(
    features: np.ndarray,
    actions: np.ndarray,
    split_indices: dict[str, np.ndarray],
    train_indices: np.ndarray,
    batch_size: int,
) -> tuple[dict[str, DataLoader], dict[str, np.ndarray]]:
    """Prepare normalized arrays and DataLoaders for train/val/test splits."""

    train_x_raw = features[train_indices]
    train_y_raw = actions[train_indices]
    all_x_norm, x_mean, x_std = _standardize(train_x_raw, features)
    all_y_norm, y_mean, y_std = _standardize(train_y_raw, actions)

    # The normalization statistics are saved so evaluation can reuse the exact
    # same feature and target scaling as training.
    arrays = {
        "x": all_x_norm,
        "y": all_y_norm,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }
    loaders: dict[str, DataLoader] = {}
    for split_name, indices in [("train", train_indices), ("val", split_indices["val"]), ("test", split_indices["test"])]:
        dataset = TensorDataset(
            torch.from_numpy(all_x_norm[indices]),
            torch.from_numpy(all_y_norm[indices]),
        )
        # Shuffle only the training set; validation and test should stay fixed.
        shuffle = split_name == "train"
        loaders[split_name] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loaders, arrays


def _eval_loss(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: str) -> float:
    """Compute mean loss on a full split without gradient updates."""

    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            prediction = model(batch_x)
            losses.append(float(loss_fn(prediction, batch_y).item()))
    return float(np.mean(losses)) if losses else float("nan")


def train_mlp(
    features: np.ndarray,
    actions: np.ndarray,
    split_indices: dict[str, np.ndarray],
    train_indices: np.ndarray,
    method: str,
    training_cfg: dict[str, Any],
    checkpoint_dir: str | Path,
    seed: int,
) -> TrainResult:
    """Train the MLP predictor and save the best validation checkpoint.

    The same routine is used for full-data training and for each coreset
    baseline, so only ``train_indices`` changes across experimental settings.
    """

    device = get_device(training_cfg.get("device", "cpu"))
    batch_size = int(training_cfg.get("batch_size", 256))
    loaders, arrays = prepare_datasets(features, actions, split_indices, train_indices, batch_size)

    # The predictor receives the concatenated feature vector and outputs the
    # normalized 7-DoF action target.
    model = MLPRegressor(
        input_dim=features.shape[1],
        output_dim=actions.shape[1],
        hidden_dims=training_cfg.get("hidden_dims", [512, 256]),
        dropout=float(training_cfg.get("dropout", 0.1)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("lr", 1e-3)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )
    loss_fn = nn.MSELoss()
    epochs = int(training_cfg.get("epochs", 100))
    patience = int(training_cfg.get("patience", 15))
    best_val = float("inf")
    best_state: dict[str, Any] | None = None
    patience_count = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses: list[float] = []
        for batch_x, batch_y in loaders["train"]:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            prediction = model(batch_x)
            loss = loss_fn(prediction, batch_y)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        val_loss = _eval_loss(model, loaders["val"], loss_fn, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        # Early stopping keeps the parameters from the epoch with the best
        # validation loss instead of blindly using the final epoch.
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint_dir = ensure_dir(checkpoint_dir)
    checkpoint_path = checkpoint_dir / f"mlp_{method}_seed{seed}.pt"
    # Save both model weights and preprocessing statistics so evaluation can be
    # reproduced exactly from the checkpoint alone.
    torch.save(
        {
            "model_state": model.state_dict(),
            "method": method,
            "seed": seed,
            "input_dim": features.shape[1],
            "output_dim": actions.shape[1],
            "hidden_dims": training_cfg.get("hidden_dims", [512, 256]),
            "dropout": float(training_cfg.get("dropout", 0.1)),
            "x_mean": arrays["x_mean"],
            "x_std": arrays["x_std"],
            "y_mean": arrays["y_mean"],
            "y_std": arrays["y_std"],
            "history": history,
            "train_indices": train_indices,
            "split_indices": split_indices,
        },
        checkpoint_path,
    )

    return TrainResult(
        model=model,
        history=history,
        checkpoint_path=checkpoint_path,
        x_mean=arrays["x_mean"],
        x_std=arrays["x_std"],
        y_mean=arrays["y_mean"],
        y_std=arrays["y_std"],
        param_count=count_parameters(model),
    )
