"""Train-only quantile statistics for SE(3) relative pose10d training."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from lerobot.processor.end_effector_pose_processor import relative_pose10d


POSE_FEATURE_NAMES = [
    "x",
    "y",
    "z",
    "rot6d_0",
    "rot6d_1",
    "rot6d_2",
    "rot6d_3",
    "rot6d_4",
    "rot6d_5",
    "gripper",
]
POSE_DIMENSION = len(POSE_FEATURE_NAMES)
SCALED_POSE_INDICES = [0, 1, 2, 9]
STATE_STATS_FILENAME = "relative_pose_state_q01_q99.json"
MANIFEST_FILENAME = "relative_pose_stats_manifest.json"


@dataclass(frozen=True)
class PoseQuantileStats:
    q01: np.ndarray
    q99: np.ndarray
    count: int


@dataclass(frozen=True)
class RelativePoseStatsBundle:
    state: PoseQuantileStats
    actions: dict[int, PoseQuantileStats]
    feature_names: list[str]
    scaled_indices: list[int]


def _episode_array(episode: np.ndarray | torch.Tensor, index: int) -> np.ndarray:
    values = episode.detach().cpu().numpy() if isinstance(episode, torch.Tensor) else np.asarray(episode)
    if values.ndim != 2 or values.shape[1] != POSE_DIMENSION or values.shape[0] == 0:
        raise ValueError(f"episode {index} must have shape [T, {POSE_DIMENSION}] with T > 0, got {values.shape}")
    values = values.astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"episode {index} contains non-finite pose values")
    return values


def _relative(anchor: np.ndarray, target: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return relative_pose10d(torch.from_numpy(anchor), torch.from_numpy(target)).numpy()


def _identity_relative(current: np.ndarray) -> np.ndarray:
    result = np.zeros_like(current)
    result[..., 3] = 1.0
    result[..., 7] = 1.0
    result[..., 9] = current[..., 9]
    return result


def _quantile_stats(values: np.ndarray) -> PoseQuantileStats:
    quantiles = np.quantile(values, [0.01, 0.99], axis=0)
    return PoseQuantileStats(q01=quantiles[0], q99=quantiles[1], count=len(values))


def compute_relative_pose_stats_from_episodes(
    episodes: Sequence[np.ndarray | torch.Tensor], *, horizons: Sequence[int]
) -> RelativePoseStatsBundle:
    """Compute exact q01/q99 from absolute pose10d train episodes only."""
    if not episodes:
        raise ValueError("episodes must not be empty")
    if not horizons or any(type(horizon) is not int or horizon <= 0 for horizon in horizons):
        raise ValueError("horizons must be nonempty positive integers")
    if len(set(horizons)) != len(horizons):
        raise ValueError("horizons must be unique")
    episode_arrays = [_episode_array(episode, index) for index, episode in enumerate(episodes)]
    state_values = []
    for episode in episode_arrays:
        relative_state = np.empty_like(episode)
        relative_state[0] = _identity_relative(episode[:1])[0]
        if len(episode) > 1:
            relative_state[1:] = _relative(episode[:-1], episode[1:])
        state_values.append(relative_state)
    actions: dict[int, PoseQuantileStats] = {}
    for horizon in horizons:
        targets = []
        for episode in episode_arrays:
            for offset in range(1, min(horizon, len(episode) - 1) + 1):
                targets.append(_relative(episode[:-offset], episode[offset:]))
        if not targets:
            raise ValueError("episodes must contain at least one action target")
        actions[horizon] = _quantile_stats(np.concatenate(targets, axis=0))
    return RelativePoseStatsBundle(
        state=_quantile_stats(np.concatenate(state_values, axis=0)),
        actions=actions,
        feature_names=POSE_FEATURE_NAMES.copy(),
        scaled_indices=SCALED_POSE_INDICES.copy(),
    )


def _action_filename(horizon: int) -> str:
    return f"relative_pose_action_chunk{horizon}_q01_q99.json"


def _stats_payload(stats: PoseQuantileStats) -> dict[str, object]:
    return {"q01": stats.q01.tolist(), "q99": stats.q99.tolist(), "count": stats.count}


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(payload, temporary_file, indent=2, sort_keys=True)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def save_relative_pose_stats(bundle: RelativePoseStatsBundle, output_dir: str | Path) -> None:
    """Persist one state statistic set and every requested action horizon atomically."""
    output_dir = Path(output_dir)
    _atomic_write_json(output_dir / STATE_STATS_FILENAME, _stats_payload(bundle.state))
    action_files: dict[str, str] = {}
    for horizon, stats in sorted(bundle.actions.items()):
        filename = _action_filename(horizon)
        action_files[str(horizon)] = filename
        payload = _stats_payload(stats)
        payload["horizon"] = horizon
        _atomic_write_json(output_dir / filename, payload)
    _atomic_write_json(
        output_dir / MANIFEST_FILENAME,
        {
            "format_version": 1,
            "formula_version": "relative_end_effector_pose_v1",
            "feature_names": bundle.feature_names,
            "scaled_indices": bundle.scaled_indices,
            "state_file": STATE_STATS_FILENAME,
            "action_files": action_files,
        },
    )


def _load_stats(path: Path) -> PoseQuantileStats:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        q01 = np.asarray(payload["q01"], dtype=np.float64)
        q99 = np.asarray(payload["q99"], dtype=np.float64)
        count = int(payload["count"])
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid relative pose stats file {path}: {exc}") from exc
    if q01.shape != (POSE_DIMENSION,) or q99.shape != (POSE_DIMENSION,) or count <= 0:
        raise ValueError(f"invalid relative pose statistics dimensions in {path}")
    return PoseQuantileStats(q01=q01, q99=q99, count=count)


def load_relative_pose_stats_paths(
    state_path: str | Path, action_path: str | Path, *, expected_horizon: int
) -> RelativePoseStatsBundle:
    """Load a verified state/action pair for one pose action horizon."""
    state_path = Path(state_path)
    action_path = Path(action_path)
    if state_path.name != STATE_STATS_FILENAME:
        raise ValueError(f"relative pose state stats must be named {STATE_STATS_FILENAME}")
    if action_path.name != _action_filename(expected_horizon):
        raise ValueError(f"relative pose action stats must be named {_action_filename(expected_horizon)}")
    if state_path.parent.resolve() != action_path.parent.resolve():
        raise ValueError("relative pose state and action stats must be in the same directory")
    try:
        manifest = json.loads((state_path.parent / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid relative pose stats manifest: {exc}") from exc
    if (
        manifest.get("format_version") != 1
        or manifest.get("formula_version") != "relative_end_effector_pose_v1"
        or manifest.get("feature_names") != POSE_FEATURE_NAMES
        or manifest.get("scaled_indices") != SCALED_POSE_INDICES
        or manifest.get("state_file") != STATE_STATS_FILENAME
        or manifest.get("action_files", {}).get(str(expected_horizon)) != _action_filename(expected_horizon)
    ):
        raise ValueError("relative pose stats manifest does not match the requested pose contract")
    action_payload = json.loads(action_path.read_text(encoding="utf-8"))
    if action_payload.get("horizon") != expected_horizon:
        raise ValueError("relative pose action stats horizon does not match the requested horizon")
    return RelativePoseStatsBundle(
        state=_load_stats(state_path),
        actions={expected_horizon: _load_stats(action_path)},
        feature_names=POSE_FEATURE_NAMES.copy(),
        scaled_indices=SCALED_POSE_INDICES.copy(),
    )
