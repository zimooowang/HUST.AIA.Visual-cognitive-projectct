from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from tqdm import tqdm

from .dataset_utils import FrameRecord
from .utils import ensure_dir, get_device


class FrozenResNet18(nn.Module):
    """ResNet-18 feature extractor that outputs 512-d embeddings."""

    def __init__(self) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        backbone = models.resnet18(weights=weights)
        self.preprocess = weights.transforms()
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        return features.flatten(1)


def build_feature_extractor(name: str, device: str) -> tuple[nn.Module, transforms.Compose]:
    if name != "resnet18":
        raise ValueError(f"Unsupported extractor: {name}")
    model = FrozenResNet18().to(device)
    model.eval()
    return model, model.preprocess


def _language_feature(text: str, mode: str, language_dim: int, text_hash_dim: int) -> np.ndarray:
    if mode == "constant":
        return np.ones((language_dim,), dtype=np.float32)
    if mode == "text_hash":
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        return rng.standard_normal(text_hash_dim).astype(np.float32)
    raise ValueError(f"Unsupported language mode: {mode}")


def extract_feature_cache(records: list[FrameRecord], extractor_name: str, batch_size: int, device_preference: str, language_mode: str, language_dim: int, text_hash_dim: int) -> dict[str, Any]:
    device = get_device(device_preference)
    model, preprocess = build_feature_extractor(extractor_name, device)
    image_features: list[np.ndarray] = []

    with torch.no_grad():
        for start in tqdm(range(0, len(records), batch_size), desc="Extracting features"):
            batch_records = records[start : start + batch_size]
            batch_images = torch.stack([preprocess(record.image) for record in batch_records], dim=0).to(device)
            batch_features = model(batch_images).cpu().numpy().astype(np.float32)
            image_features.append(batch_features)

    image_features_array = np.concatenate(image_features, axis=0) if image_features else np.zeros((0, 512), dtype=np.float32)
    language_features = np.stack(
        [_language_feature(record.instruction, language_mode, language_dim, text_hash_dim) for record in records],
        axis=0,
    ).astype(np.float32)

    features = np.concatenate([image_features_array, language_features], axis=1).astype(np.float32)
    actions = np.stack([record.action for record in records], axis=0).astype(np.float32)
    episode_indices = np.asarray([record.episode_index for record in records], dtype=np.int64)
    frame_indices = np.asarray([record.frame_index for record in records], dtype=np.int64)
    timestamps = np.asarray(
        [np.nan if record.timestamp is None else record.timestamp for record in records],
        dtype=np.float32,
    )

    metadata = {
        "extractor": extractor_name,
        "device_used": device,
        "feature_dim": int(features.shape[1]),
        "image_feature_dim": int(image_features_array.shape[1]) if image_features_array.size else 0,
        "language_mode": language_mode,
        "language_dim": int(language_features.shape[1]) if language_features.size else 0,
        "n_samples": int(features.shape[0]),
    }
    return {
        "features": features,
        "actions": actions,
        "episode_indices": episode_indices,
        "frame_indices": frame_indices,
        "timestamps": timestamps,
        "metadata": metadata,
    }


def save_feature_cache(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    np.savez_compressed(
        path,
        features=payload["features"],
        actions=payload["actions"],
        episode_indices=payload["episode_indices"],
        frame_indices=payload["frame_indices"],
        timestamps=payload["timestamps"],
        metadata=json.dumps(payload["metadata"], ensure_ascii=False),
    )


def load_feature_cache(path: str | Path) -> dict[str, Any]:
    data = np.load(Path(path), allow_pickle=True)
    metadata = json.loads(str(data["metadata"]))
    return {
        "features": data["features"].astype(np.float32),
        "actions": data["actions"].astype(np.float32),
        "episode_indices": data["episode_indices"].astype(np.int64),
        "frame_indices": data["frame_indices"].astype(np.int64),
        "timestamps": data["timestamps"].astype(np.float32),
        "metadata": metadata,
    }
