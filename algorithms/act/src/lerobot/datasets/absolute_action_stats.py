"""Train-only q01/q99 statistics for absolute state and future action chunks."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


STATE_STATS_FILENAME = "absolute_state_q01_q99.json"
MANIFEST_FILENAME = "absolute_stats_manifest.json"


@dataclass(frozen=True)
class AbsoluteQuantileStats:
    q01: np.ndarray
    q99: np.ndarray
    count: int

    def __post_init__(self) -> None:
        q01 = np.asarray(self.q01, dtype=np.float64).copy()
        q99 = np.asarray(self.q99, dtype=np.float64).copy()
        if q01.ndim != 1 or q01.size == 0 or q99.shape != q01.shape:
            raise ValueError("q01 and q99 must be nonempty one-dimensional arrays with matching shapes")
        if not np.isfinite(q01).all() or not np.isfinite(q99).all() or np.any(q01 > q99):
            raise ValueError("quantiles must be finite and satisfy q01 <= q99")
        if type(self.count) is not int or self.count <= 0:
            raise ValueError("count must be a positive integer")
        q01.setflags(write=False)
        q99.setflags(write=False)
        object.__setattr__(self, "q01", q01)
        object.__setattr__(self, "q99", q99)


@dataclass(frozen=True)
class AbsoluteActionStatsBundle:
    state: AbsoluteQuantileStats
    actions: dict[int, AbsoluteQuantileStats]
    feature_names: list[str]
    scaled_indices: list[int]

    def __post_init__(self) -> None:
        dimension = self.state.q01.size
        if len(self.feature_names) != dimension or any(not isinstance(name, str) or not name for name in self.feature_names):
            raise ValueError("feature_names must contain one nonempty name per dimension")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be unique")
        if not self.actions:
            raise ValueError("actions must contain at least one horizon")
        for horizon, stats in self.actions.items():
            if type(horizon) is not int or horizon <= 0 or stats.q01.shape != (dimension,):
                raise ValueError("action statistics must have positive horizons and matching dimensions")
        if not self.scaled_indices or len(set(self.scaled_indices)) != len(self.scaled_indices):
            raise ValueError("scaled_indices must be a nonempty unique index list")
        if any(type(index) is not int or index < 0 or index >= dimension for index in self.scaled_indices):
            raise ValueError("scaled_indices must be valid feature indices")


def _episode_array(episode: np.ndarray | torch.Tensor, index: int, dimension: int | None) -> np.ndarray:
    values = episode.detach().cpu().numpy() if isinstance(episode, torch.Tensor) else np.asarray(episode)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"episode {index} must have shape [T, D] with T > 0, got {values.shape}")
    if dimension is not None and values.shape[1] != dimension:
        raise ValueError(f"episode {index} has dimension {values.shape[1]}, expected {dimension}")
    values = values.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"episode {index} contains non-finite values")
    return values


def _quantile_stats(values: np.ndarray) -> AbsoluteQuantileStats:
    q01, q99 = np.quantile(values, [0.01, 0.99], axis=0)
    return AbsoluteQuantileStats(q01=q01, q99=q99, count=int(values.shape[0]))


def compute_absolute_action_stats_from_episodes(
    episodes: Sequence[np.ndarray | torch.Tensor],
    *,
    horizons: Sequence[int],
    feature_names: Sequence[str],
    scaled_indices: Sequence[int],
) -> AbsoluteActionStatsBundle:
    """Compute separate absolute state and future action quantiles from train episodes."""
    if not episodes:
        raise ValueError("episodes must not be empty")
    if not horizons or any(type(horizon) is not int or horizon <= 0 for horizon in horizons):
        raise ValueError("horizons must be nonempty positive integers")
    if len(set(horizons)) != len(horizons):
        raise ValueError("horizons must be unique")
    dimension = len(feature_names)
    if dimension == 0:
        raise ValueError("feature_names must not be empty")
    episode_arrays = [_episode_array(episode, index, dimension) for index, episode in enumerate(episodes)]
    actions: dict[int, AbsoluteQuantileStats] = {}
    for horizon in horizons:
        future_targets = [
            episode[offset:]
            for episode in episode_arrays
            for offset in range(1, min(horizon, len(episode) - 1) + 1)
        ]
        if not future_targets:
            raise ValueError("episodes must contain at least one valid future action target")
        actions[horizon] = _quantile_stats(np.concatenate(future_targets, axis=0))
    return AbsoluteActionStatsBundle(
        state=_quantile_stats(np.concatenate(episode_arrays, axis=0)),
        actions=actions,
        feature_names=list(feature_names),
        scaled_indices=list(scaled_indices),
    )


def _action_filename(horizon: int) -> str:
    return f"absolute_action_chunk{horizon}_q01_q99.json"


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


def _stats_payload(stats: AbsoluteQuantileStats) -> dict[str, object]:
    return {"q01": stats.q01.tolist(), "q99": stats.q99.tolist(), "count": stats.count}


def save_absolute_action_stats(bundle: AbsoluteActionStatsBundle, output_dir: str | Path) -> None:
    """Atomically persist one state statistic set and each requested action horizon."""
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
            "formula_version": "absolute_future_action_v1",
            "feature_names": bundle.feature_names,
            "scaled_indices": bundle.scaled_indices,
            "state_file": STATE_STATS_FILENAME,
            "action_files": action_files,
        },
    )


def _load_stats(path: Path, dimension: int) -> AbsoluteQuantileStats:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        q01 = np.asarray(payload["q01"], dtype=np.float64)
        q99 = np.asarray(payload["q99"], dtype=np.float64)
        count = int(payload["count"])
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid absolute statistic file {path}: {error}") from error
    if q01.shape != (dimension,) or q99.shape != (dimension,):
        raise ValueError(f"absolute statistic file {path} must have dimension {dimension}")
    return AbsoluteQuantileStats(q01=q01, q99=q99, count=count)


def load_absolute_action_stats_paths(
    state_path: str | Path,
    action_path: str | Path,
    *,
    expected_horizon: int,
    feature_names: Sequence[str],
    scaled_indices: Sequence[int],
) -> AbsoluteActionStatsBundle:
    """Load a verified state/action absolute-stat pair for one action horizon."""
    state_path = Path(state_path)
    action_path = Path(action_path)
    if state_path.name != STATE_STATS_FILENAME:
        raise ValueError(f"absolute state stats must be named {STATE_STATS_FILENAME}")
    if action_path.name != _action_filename(expected_horizon):
        raise ValueError(f"absolute action stats must be named {_action_filename(expected_horizon)}")
    if state_path.parent.resolve() != action_path.parent.resolve():
        raise ValueError("absolute state and action stats must be in the same directory")
    try:
        manifest = json.loads((state_path.parent / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid absolute statistics manifest: {error}") from error
    if (
        manifest.get("format_version") != 1
        or manifest.get("formula_version") != "absolute_future_action_v1"
        or manifest.get("feature_names") != list(feature_names)
        or manifest.get("scaled_indices") != list(scaled_indices)
        or manifest.get("state_file") != STATE_STATS_FILENAME
        or manifest.get("action_files", {}).get(str(expected_horizon)) != _action_filename(expected_horizon)
    ):
        raise ValueError("absolute statistics manifest does not match the requested representation")
    try:
        action_payload = json.loads(action_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid absolute action statistics file: {error}") from error
    if action_payload.get("horizon") != expected_horizon:
        raise ValueError("absolute action statistics horizon does not match the requested horizon")
    dimension = len(feature_names)
    return AbsoluteActionStatsBundle(
        state=_load_stats(state_path, dimension),
        actions={expected_horizon: _load_stats(action_path, dimension)},
        feature_names=list(feature_names),
        scaled_indices=list(scaled_indices),
    )
