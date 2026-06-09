from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans

from .utils import ratio_to_percent, save_json, zscore


@dataclass
class SelectionArtifacts:
    """Paths to artifacts generated during coreset construction."""

    split_path: Path
    score_path: Path
    selection_paths: dict[str, Path]


def generate_episode_split(
    episode_indices: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, Any]:
    """Split data at the episode level to avoid temporal leakage across sets."""

    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    unique_episodes = np.unique(episode_indices)
    rng = np.random.default_rng(seed)
    shuffled = unique_episodes.copy()
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = max(1, int(round(n_total * train_ratio)))
    n_val = max(1, int(round(n_total * val_ratio))) if n_total >= 3 else 0
    n_train = min(n_train, n_total - n_val)
    n_test = max(0, n_total - n_train - n_val)
    if n_test == 0 and n_total >= 3:
        n_test = 1
        n_train = max(1, n_train - 1)

    train_eps = shuffled[:n_train]
    val_eps = shuffled[n_train : n_train + n_val]
    test_eps = shuffled[n_train + n_val :]

    # The split file records episode ids rather than frame ids so that later
    # scripts can reconstruct frame-level indices from cached features.
    return {
        "train_episodes": train_eps.tolist(),
        "val_episodes": val_eps.tolist(),
        "test_episodes": test_eps.tolist(),
        "seed": seed,
    }


def indices_from_split(episode_indices: np.ndarray, split_payload: dict[str, Any]) -> dict[str, np.ndarray]:
    """Convert episode-level split metadata back into frame-level indices."""

    train_idx = np.where(np.isin(episode_indices, split_payload["train_episodes"]))[0]
    val_idx = np.where(np.isin(episode_indices, split_payload["val_episodes"]))[0]
    test_idx = np.where(np.isin(episode_indices, split_payload["test_episodes"]))[0]
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def compute_change_scores(values: np.ndarray, episode_indices: np.ndarray, frame_indices: np.ndarray) -> np.ndarray:
    """Measure frame-to-frame change within each episode.

    For visual features this approximates perceptual novelty, and for actions it
    approximates motor change. The first frame of every episode is assigned zero
    because there is no previous frame to compare against.
    """

    scores = np.zeros((len(values),), dtype=np.float32)
    for episode in np.unique(episode_indices):
        episode_mask = np.where(episode_indices == episode)[0]
        ordered = episode_mask[np.argsort(frame_indices[episode_mask])]
        if len(ordered) == 0:
            continue
        scores[ordered[0]] = 0.0
        for previous, current in zip(ordered[:-1], ordered[1:]):
            scores[current] = float(np.linalg.norm(values[current] - values[previous]))
    return scores


def _standardize_with_reference(values: np.ndarray, reference_indices: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Standardize each dimension using statistics from reference indices only."""

    if len(values) == 0:
        return values.astype(np.float32)
    reference = values[reference_indices] if len(reference_indices) else values
    mean = reference.mean(axis=0, keepdims=True).astype(np.float32)
    std = reference.std(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(std, eps)
    return ((values - mean) / std).astype(np.float32)


def _coordination_change(left_change: np.ndarray, right_change: np.ndarray, mode: str) -> np.ndarray:
    """Summarize how strongly the two arms move together."""

    mode = str(mode).lower()
    if mode == "min":
        return np.minimum(left_change, right_change).astype(np.float32)
    if mode == "mean":
        return (0.5 * (left_change + right_change)).astype(np.float32)
    if mode in {"geom", "geometric_mean", "geomean"}:
        return np.sqrt(np.maximum(left_change, 0.0) * np.maximum(right_change, 0.0)).astype(np.float32)
    raise ValueError(f"Unsupported coordination mode: {mode}")


def _weighted_average(components: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Combine score components while preserving the original score scale."""

    if not components:
        return np.zeros((0,), dtype=np.float32)
    total_weight = 0.0
    combined = np.zeros_like(next(iter(components.values())), dtype=np.float32)
    for name, values in components.items():
        weight = float(weights.get(name, 0.0))
        if weight <= 0.0:
            continue
        combined += weight * values.astype(np.float32)
        total_weight += weight
    if total_weight <= 0.0:
        return np.zeros_like(combined, dtype=np.float32)
    return (combined / total_weight).astype(np.float32)


def _weighted_zscore_average(
    components: dict[str, np.ndarray],
    weights: dict[str, float],
    train_mask: np.ndarray,
) -> np.ndarray:
    """Combine multiple change scores after per-component train-split z-scoring."""

    if not components:
        return np.zeros((0,), dtype=np.float32)
    total_weight = 0.0
    combined = np.zeros_like(next(iter(components.values())), dtype=np.float32)
    for name, values in components.items():
        weight = float(weights.get(name, 0.0))
        if weight <= 0.0:
            continue
        combined += weight * zscore(values.astype(np.float32), train_mask)
        total_weight += weight
    if total_weight <= 0.0:
        return np.zeros_like(combined, dtype=np.float32)
    return (combined / total_weight).astype(np.float32)


def _compute_action_scores(
    actions: np.ndarray,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    train_indices: np.ndarray,
    train_mask: np.ndarray,
    selection_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Build action novelty scores, with a bimanual-aware path for 14D actions."""

    normalize_actions = bool(selection_cfg.get("normalize_action_for_scores", True))
    action_values = _standardize_with_reference(actions, train_indices) if normalize_actions else actions.astype(np.float32)
    action_dim = action_values.shape[1] if action_values.ndim == 2 else 0
    bimanual_action_scoring = bool(selection_cfg.get("bimanual_action_scoring", True))

    if action_dim == 14 and bimanual_action_scoring:
        left_change = compute_change_scores(action_values[:, :7], episode_indices, frame_indices)
        right_change = compute_change_scores(action_values[:, 7:14], episode_indices, frame_indices)
        coordination = _coordination_change(
            left_change,
            right_change,
            selection_cfg.get("coordination_mode", "min"),
        )
        components = {
            "action_left_change": left_change.astype(np.float32),
            "action_right_change": right_change.astype(np.float32),
            "coordination_change": coordination.astype(np.float32),
        }
        weights = {
            "action_left_change": float(selection_cfg.get("lambda_action_left", 1.0)),
            "action_right_change": float(selection_cfg.get("lambda_action_right", 1.0)),
            "coordination_change": float(selection_cfg.get("lambda_action_coordination", 0.5)),
        }
        action_change = _weighted_average(components, weights)
        action_z = _weighted_zscore_average(components, weights, train_mask)
        return action_change, action_z, components

    action_change = compute_change_scores(action_values, episode_indices, frame_indices).astype(np.float32)
    return action_change, zscore(action_change, train_mask).astype(np.float32), {}


def _episode_quotas(indices: np.ndarray, episode_indices: np.ndarray, target_count: int) -> dict[int, int]:
    """Allocate how many samples each episode may contribute.

    The quota is roughly proportional to episode length, which prevents long
    episodes from being ignored and short episodes from dominating by chance.
    """

    unique_eps, counts = np.unique(episode_indices[indices], return_counts=True)
    if target_count <= 0:
        return {int(ep): 0 for ep in unique_eps}

    raw = counts / counts.sum() * target_count
    base = np.floor(raw).astype(int)
    remaining = target_count - int(base.sum())
    fractions = raw - base
    order = np.argsort(-fractions)
    for idx in order[:remaining]:
        base[idx] += 1
    return {int(ep): int(count) for ep, count in zip(unique_eps, base)}


def _select_top_per_episode(indices: np.ndarray, scores: np.ndarray, episode_indices: np.ndarray, target_count: int) -> np.ndarray:
    """Select the highest-scoring frames under episode-wise quotas."""

    quotas = _episode_quotas(indices, episode_indices, target_count)
    selected: list[int] = []
    for episode, quota in quotas.items():
        if quota <= 0:
            continue
        episode_idx = indices[episode_indices[indices] == episode]
        ranked = episode_idx[np.argsort(-scores[episode_idx], kind="mergesort")]
        selected.extend(ranked[:quota].tolist())
    return np.asarray(selected, dtype=np.int64)


def _random_per_episode(indices: np.ndarray, episode_indices: np.ndarray, target_count: int, seed: int) -> np.ndarray:
    """Episode-balanced random baseline used for comparison."""

    quotas = _episode_quotas(indices, episode_indices, target_count)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for episode, quota in quotas.items():
        if quota <= 0:
            continue
        episode_idx = indices[episode_indices[indices] == episode]
        chosen = rng.choice(episode_idx, size=min(quota, len(episode_idx)), replace=False)
        selected.extend(chosen.tolist())
    return np.asarray(sorted(set(selected)), dtype=np.int64)


def _kmeans_diverse_selection(
    diversity_vectors: np.ndarray,
    candidate_indices: np.ndarray,
    utility_scores: np.ndarray,
    target_count: int,
    seed: int,
) -> np.ndarray:
    """Choose a diverse subset from high-utility candidates with KMeans.

    The algorithm first clusters candidate samples in feature space, then picks
    one representative from each cluster. Within a cluster, we prefer samples
    that are close to the centroid and also have high utility.
    """

    if len(candidate_indices) <= target_count:
        return candidate_indices

    n_clusters = min(target_count, len(candidate_indices))
    if n_clusters <= 0:
        return np.asarray([], dtype=np.int64)

    candidate_vectors = diversity_vectors[candidate_indices]
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(candidate_vectors)
    centers = kmeans.cluster_centers_

    selected: list[int] = []
    for cluster_id in range(n_clusters):
        cluster_positions = np.where(labels == cluster_id)[0]
        if len(cluster_positions) == 0:
            continue
        cluster_global = candidate_indices[cluster_positions]
        distances = np.linalg.norm(candidate_vectors[cluster_positions] - centers[cluster_id], axis=1)
        utility = utility_scores[cluster_global]
        ranking = np.lexsort((-utility, distances))
        selected.append(int(cluster_global[ranking[0]]))

    if len(selected) < target_count:
        remaining = [idx for idx in candidate_indices.tolist() if idx not in selected]
        remaining = sorted(remaining, key=lambda idx: utility_scores[idx], reverse=True)
        selected.extend(remaining[: target_count - len(selected)])

    return np.asarray(sorted(set(selected))[:target_count], dtype=np.int64)


def _build_diversity_vectors(
    features: np.ndarray,
    actions: np.ndarray,
    train_indices: np.ndarray,
    selection_cfg: dict[str, Any],
) -> np.ndarray:
    """Build clustering vectors that balance visual and action structure."""

    diversity_space = str(selection_cfg.get("diversity_space", "visual_action")).lower()
    visual_vectors = _standardize_with_reference(features, train_indices)
    action_vectors = _standardize_with_reference(actions, train_indices)

    visual_block = visual_vectors / np.sqrt(max(1, visual_vectors.shape[1]))
    action_block = action_vectors / np.sqrt(max(1, action_vectors.shape[1]))
    visual_weight = float(selection_cfg.get("diversity_visual_weight", 1.0))
    action_weight = float(selection_cfg.get("diversity_action_weight", 1.0))

    if diversity_space == "visual":
        return (visual_weight * visual_block).astype(np.float32)
    if diversity_space == "action":
        return (action_weight * action_block).astype(np.float32)
    if diversity_space in {"visual_action", "joint"}:
        return np.concatenate(
            [
                visual_weight * visual_block,
                action_weight * action_block,
            ],
            axis=1,
        ).astype(np.float32)
    raise ValueError(f"Unsupported diversity space: {diversity_space}")


def build_selections(
    features: np.ndarray,
    actions: np.ndarray,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    split_indices: dict[str, np.ndarray],
    selection_cfg: dict[str, Any],
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Construct all data-selection baselines used in the experiments.

    The main idea is:
    1. Compute visual-change and action-change scores on the training split.
    2. Combine them into a utility score.
    3. Compare simple top-k selection, random selection, and a diversity-aware
       brain-inspired coreset strategy.
    """

    train_indices = split_indices["train"]
    train_mask = np.zeros((len(features),), dtype=bool)
    train_mask[train_indices] = True
    target_count = max(1, int(round(len(train_indices) * float(selection_cfg["ratio"]))))

    # Visual novelty is measured in feature space; action novelty is measured in
    # label space. Both are computed as frame-to-frame changes within episodes.
    visual_change = compute_change_scores(features, episode_indices, frame_indices)
    action_change, action_z, action_components = _compute_action_scores(
        actions=actions,
        episode_indices=episode_indices,
        frame_indices=frame_indices,
        train_indices=train_indices,
        train_mask=train_mask,
        selection_cfg=selection_cfg,
    )

    # Z-score normalization makes the two score sources comparable in scale.
    visual_z = zscore(visual_change, train_mask)
    utility = (
        float(selection_cfg.get("lambda_visual", 1.0)) * visual_z
        + float(selection_cfg.get("lambda_action", 1.0)) * action_z
    ).astype(np.float32)

    selections: dict[str, np.ndarray] = {}
    selections["random"] = _random_per_episode(train_indices, episode_indices, target_count, seed)
    selections["action_change"] = _select_top_per_episode(train_indices, action_change, episode_indices, target_count)
    selections["visual_change"] = _select_top_per_episode(train_indices, visual_change, episode_indices, target_count)
    selections["visual_action"] = _select_top_per_episode(train_indices, utility, episode_indices, target_count)

    # The brain-inspired variant first keeps only a high-utility candidate pool,
    # then adds diversity by selecting representatives from KMeans clusters.
    candidate_ratio = float(selection_cfg.get("candidate_ratio", 0.3))
    candidate_count = max(target_count, int(round(len(train_indices) * candidate_ratio)))
    utility_ranked = train_indices[np.argsort(-utility[train_indices], kind="mergesort")]
    candidate_indices = utility_ranked[:candidate_count]
    if selection_cfg.get("use_diversity", True):
        diversity_vectors = _build_diversity_vectors(features, actions, train_indices, selection_cfg)
        brain = _kmeans_diverse_selection(diversity_vectors, candidate_indices, utility, target_count, seed)
    else:
        brain = candidate_indices[:target_count]
    selections["brain_coreset"] = np.asarray(brain, dtype=np.int64)

    scores = {
        "visual_change": visual_change.astype(np.float32),
        "action_change": action_change.astype(np.float32),
        "utility": utility.astype(np.float32),
    }
    scores.update({name: values.astype(np.float32) for name, values in action_components.items()})
    return selections, scores


def save_split(path: str | Path, split_payload: dict[str, Any]) -> None:
    """Save episode split metadata to JSON."""
    save_json(path, split_payload)


def save_scores(path: str | Path, scores: dict[str, np.ndarray]) -> None:
    """Persist per-frame selection scores for later visualization."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **scores)


def save_selection_files(selection_dir: str | Path, selections: dict[str, np.ndarray], ratio: float, seed: int) -> dict[str, Path]:
    """Save sampled frame indices for every selection method."""

    selection_dir = Path(selection_dir)
    selection_dir.mkdir(parents=True, exist_ok=True)
    percent = ratio_to_percent(ratio)
    paths: dict[str, Path] = {}
    for method, indices in selections.items():
        path = selection_dir / f"{method}_{percent}_seed{seed}.npy"
        np.save(path, indices.astype(np.int64))
        paths[method] = path
    return paths
