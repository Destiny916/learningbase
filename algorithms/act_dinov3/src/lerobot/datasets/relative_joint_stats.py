#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


STATE_STATS_FILENAME = "relative_state_q01_q99.json"
MANIFEST_FILENAME = "relative_stats_manifest.json"


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


@dataclass(frozen=True)
class QuantileStats:
    q01: np.ndarray
    q99: np.ndarray
    count: int

    def __post_init__(self) -> None:
        q01 = np.asarray(self.q01, dtype=np.float64).copy()
        q99 = np.asarray(self.q99, dtype=np.float64).copy()
        if q01.ndim != 1 or q01.size == 0 or q99.shape != q01.shape:
            raise ValueError("q01 and q99 must be nonempty one-dimensional arrays with matching shapes")
        if not np.all(np.isfinite(q01)) or not np.all(np.isfinite(q99)):
            raise ValueError("quantiles must contain only finite values")
        if np.any(q01 > q99):
            raise ValueError("q01 must be less than or equal to q99 in every dimension")
        if not isinstance(self.count, int) or isinstance(self.count, bool) or self.count <= 0:
            raise ValueError("count must be a positive integer")
        q01.setflags(write=False)
        q99.setflags(write=False)
        object.__setattr__(self, "q01", q01)
        object.__setattr__(self, "q99", q99)


@dataclass(frozen=True)
class RelativeJointStatsBundle:
    state: QuantileStats
    actions: dict[int, QuantileStats]
    feature_names: list[str]
    gripper_indices: list[int]
    source_manifest_sha256: str
    source_dataset_root: str = ""
    state_feature_names: list[str] | None = None
    state_gripper_indices: list[int] | None = None
    state_absolute_indices: list[int] | None = None

    def __post_init__(self) -> None:
        state_dimension = self.state.q01.shape[0]
        action_dimension = len(self.feature_names)
        feature_names = list(self.feature_names)
        gripper_indices = list(self.gripper_indices)
        state_feature_names = list(self.state_feature_names or feature_names)
        state_gripper_indices = list(self.state_gripper_indices or gripper_indices)
        state_absolute_indices = list(self.state_absolute_indices or [])
        actions = dict(self.actions)
        if len(feature_names) != action_dimension or any(
            not isinstance(name, str) or not name for name in feature_names
        ):
            raise ValueError(f"action feature names must contain one nonempty name for each of the {action_dimension} dimensions")
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("action feature names must be unique")
        if len(state_feature_names) != state_dimension or any(
            not isinstance(name, str) or not name for name in state_feature_names
        ):
            raise ValueError(
                f"state feature names must contain one nonempty name for each of the {state_dimension} dimensions"
            )
        if len(set(state_feature_names)) != len(state_feature_names):
            raise ValueError("state feature names must be unique")
        _validate_gripper_indices(gripper_indices, action_dimension)
        _validate_gripper_indices(state_gripper_indices, state_dimension)
        _validate_indices(state_absolute_indices, state_dimension, "state absolute indices")
        if not actions:
            raise ValueError("actions must contain at least one horizon")
        for horizon, stats in actions.items():
            if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
                raise ValueError("action horizons must be positive integers")
            if stats.q01.shape != (action_dimension,):
                raise ValueError(
                    f"action horizon {horizon} has dimension {stats.q01.shape}, expected {(action_dimension,)}"
                )
        sha256 = self.source_manifest_sha256
        if sha256 and not _is_sha256(sha256):
            raise ValueError("source manifest SHA256 must be a lowercase 64-character hexadecimal digest")
        source_dataset_root = self.source_dataset_root
        if not isinstance(source_dataset_root, str):
            raise ValueError("source dataset root must be a string")
        if source_dataset_root and str(Path(source_dataset_root).expanduser().resolve()) != source_dataset_root:
            raise ValueError("source dataset root must be an absolute canonical path")
        object.__setattr__(self, "feature_names", feature_names)
        object.__setattr__(self, "gripper_indices", gripper_indices)
        object.__setattr__(self, "state_feature_names", state_feature_names)
        object.__setattr__(self, "state_gripper_indices", state_gripper_indices)
        object.__setattr__(self, "state_absolute_indices", state_absolute_indices)
        object.__setattr__(self, "actions", actions)

    @property
    def action_feature_names(self) -> list[str]:
        return self.feature_names

    @property
    def action_gripper_indices(self) -> list[int]:
        return self.gripper_indices


def _validate_gripper_indices(gripper_indices: Sequence[int], dimension: int) -> None:
    if not gripper_indices:
        raise ValueError("gripper indices must not be empty")
    if any(not isinstance(index, int) or isinstance(index, bool) for index in gripper_indices):
        raise ValueError("gripper indices must be integers")
    if len(set(gripper_indices)) != len(gripper_indices):
        raise ValueError("gripper indices must be unique")
    if any(index < 0 or index >= dimension for index in gripper_indices):
        raise ValueError(f"gripper indices must be within [0, {dimension})")


def _validate_indices(indices: Sequence[int], dimension: int, name: str) -> None:
    if any(not isinstance(index, int) or isinstance(index, bool) for index in indices):
        raise ValueError(f"{name} must be integers")
    if len(set(indices)) != len(indices):
        raise ValueError(f"{name} must be unique")
    if any(index < 0 or index >= dimension for index in indices):
        raise ValueError(f"{name} must be within [0, {dimension})")


def _as_episode_array(episode: torch.Tensor | np.ndarray, index: int, dimension: int) -> np.ndarray:
    if isinstance(episode, torch.Tensor):
        values = episode.detach().cpu().numpy()
    else:
        values = np.asarray(episode)
    if values.ndim != 2 or values.shape[1] != dimension:
        raise ValueError(f"episode {index} must have shape [T, {dimension}], got {values.shape}")
    if values.shape[0] == 0:
        raise ValueError(f"episode {index} must contain at least one frame")
    values = values.astype(np.float64, copy=False)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"episode {index} contains non-finite joint values")
    return values


def _quantile_stats(values: np.ndarray) -> QuantileStats:
    quantiles = np.quantile(values, [0.01, 0.99], axis=0)
    return QuantileStats(q01=quantiles[0], q99=quantiles[1], count=values.shape[0])


def compute_relative_joint_stats_from_episodes(
    episodes: Sequence[torch.Tensor | np.ndarray],
    gripper_indices: Sequence[int] | None = None,
    horizons: Sequence[int] | None = None,
    feature_names: Sequence[str] | None = None,
    source_manifest_sha256: str = "",
    source_dataset_root: str = "",
    action_episodes: Sequence[torch.Tensor | np.ndarray] | None = None,
    state_feature_names: Sequence[str] | None = None,
    action_feature_names: Sequence[str] | None = None,
    state_gripper_indices: Sequence[int] | None = None,
    action_gripper_indices: Sequence[int] | None = None,
    action_state_indices: Sequence[int] | None = None,
    state_absolute_indices: Sequence[int] | None = None,
) -> RelativeJointStatsBundle:
    """Compute exact train-only relative state and action quantiles.

    ``action_episodes`` is optional for backwards compatibility. When omitted, action
    targets are read from the state episodes using the name mapping. When provided,
    each action row is treated as an absolute target with the same temporal contract
    as the legacy state-derived action rows.
    """
    if not episodes:
        raise ValueError("episodes must not be empty")
    if not horizons:
        raise ValueError("horizons must not be empty")
    if any(not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0 for horizon in horizons):
        raise ValueError("horizons must contain only positive integers")
    if len(set(horizons)) != len(horizons):
        raise ValueError("horizons must be unique")
    first_episode = np.asarray(episodes[0])
    if first_episode.ndim != 2 or first_episode.shape[1] <= 0:
        raise ValueError(f"episode 0 must have a nonempty [T, D] shape, got {first_episode.shape}")
    state_dimension = first_episode.shape[1]
    state_names = list(state_feature_names or feature_names or [f"joint_{index}" for index in range(state_dimension)])
    if len(state_names) != state_dimension:
        raise ValueError(f"state feature names must contain {state_dimension} names")
    if state_gripper_indices is None and gripper_indices is None:
        raise ValueError("state gripper indices must be provided")
    state_grippers = list(state_gripper_indices if state_gripper_indices is not None else gripper_indices or [])
    _validate_gripper_indices(state_grippers, state_dimension)
    state_absolute = list(state_absolute_indices or [])
    _validate_indices(state_absolute, state_dimension, "state absolute indices")
    episode_arrays = [_as_episode_array(episode, index, state_dimension) for index, episode in enumerate(episodes)]

    if action_feature_names is None:
        action_names = list(feature_names or state_names)
    else:
        action_names = list(action_feature_names)
    if not action_names:
        raise ValueError("action feature names must not be empty")
    if len(set(action_names)) != len(action_names) or any(not isinstance(name, str) or not name for name in action_names):
        raise ValueError("action feature names must be unique nonempty strings")
    state_name_to_index = {name: index for index, name in enumerate(state_names)}
    if action_state_indices is None:
        try:
            action_to_state = [state_name_to_index[name] for name in action_names]
        except KeyError as error:
            raise ValueError(f"action feature name {error.args[0]!r} is missing from state feature names") from error
    else:
        action_to_state = list(action_state_indices)
        if len(action_to_state) != len(action_names):
            raise ValueError("action_state_indices must contain one state index per action feature")
        if any(not isinstance(index, int) or isinstance(index, bool) or index < 0 or index >= state_dimension for index in action_to_state):
            raise ValueError("action_state_indices must contain valid unique state indices")
        if len(set(action_to_state)) != len(action_to_state):
            raise ValueError("action_state_indices must be unique")
        expected_names = [state_names[index] for index in action_to_state]
        if expected_names != action_names:
            raise ValueError("action feature names do not match action_state_indices")
    action_dimension = len(action_names)
    if action_gripper_indices is None and gripper_indices is None:
        raise ValueError("action gripper indices must be provided")
    action_grippers = list(action_gripper_indices if action_gripper_indices is not None else gripper_indices or [])
    _validate_gripper_indices(action_grippers, action_dimension)
    action_arrays = (
        [
            _as_episode_array(episode, index, action_dimension)
            for index, episode in enumerate(action_episodes)
        ]
        if action_episodes is not None
        else [episode[:, action_to_state] for episode in episode_arrays]
    )
    if len(action_arrays) != len(episode_arrays):
        raise ValueError("action_episodes must contain the same number of episodes as state episodes")

    relative_states = []
    for episode in episode_arrays:
        relative = episode.copy()
        state_non_relative = set(state_grippers) | set(state_absolute)
        state_arm_indices = [index for index in range(state_dimension) if index not in state_non_relative]
        relative[0, state_arm_indices] = 0.0
        relative[1:, state_arm_indices] = episode[1:, state_arm_indices] - episode[:-1, state_arm_indices]
        relative_states.append(relative)
    state_stats = _quantile_stats(np.concatenate(relative_states, axis=0))

    action_stats = {}
    for horizon in horizons:
        relative_targets = []
        for state_episode, action_episode in zip(episode_arrays, action_arrays, strict=True):
            for offset in range(1, min(horizon, len(state_episode) - 1) + 1):
                target = action_episode[offset:].copy()
                action_arm_indices = [index for index in range(action_dimension) if index not in action_grippers]
                target[:, action_arm_indices] = (
                    action_episode[offset:, action_arm_indices]
                    - state_episode[:-offset, np.asarray(action_to_state)[action_arm_indices]]
                )
                relative_targets.append(target)
        if not relative_targets:
            raise ValueError("episodes must contain at least one valid action target")
        action_stats[horizon] = _quantile_stats(np.concatenate(relative_targets, axis=0))

    return RelativeJointStatsBundle(
        state=state_stats,
        actions=action_stats,
        feature_names=action_names,
        gripper_indices=action_grippers,
        source_manifest_sha256=source_manifest_sha256,
        source_dataset_root=source_dataset_root,
        state_feature_names=state_names,
        state_gripper_indices=state_grippers,
        state_absolute_indices=state_absolute,
    )


def _action_stats_filename(horizon: int) -> str:
    return f"relative_action_chunk{horizon}_q01_q99.json"


def _stats_payload(stats: QuantileStats) -> dict:
    return {"q01": stats.q01.tolist(), "q99": stats.q99.tolist(), "count": stats.count}


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
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


def save_relative_joint_stats(
    bundle: RelativeJointStatsBundle,
    output_dir: str | Path,
    generation_command: str = "",
) -> None:
    output_dir = Path(output_dir)
    _atomic_write_json(output_dir / STATE_STATS_FILENAME, _stats_payload(bundle.state))
    action_files = {}
    for horizon in sorted(bundle.actions):
        filename = _action_stats_filename(horizon)
        action_files[str(horizon)] = filename
        action_payload = _stats_payload(bundle.actions[horizon])
        action_payload["horizon"] = horizon
        _atomic_write_json(output_dir / filename, action_payload)
    is_legacy = (
        bundle.state_feature_names == bundle.feature_names
        and bundle.state_gripper_indices == bundle.gripper_indices
        and not bundle.state_absolute_indices
    )
    manifest = {
        "format_version": 1 if is_legacy else (3 if bundle.state_absolute_indices else 2),
        "formula_version": "relative_joint_v1"
        if is_legacy
        else ("relative_joint_v3" if bundle.state_absolute_indices else "relative_joint_v2"),
        "generation_command": generation_command,
        "feature_names": bundle.feature_names,
        "gripper_indices": bundle.gripper_indices,
        "state_feature_names": bundle.state_feature_names,
        "state_gripper_indices": bundle.state_gripper_indices,
        "state_absolute_indices": bundle.state_absolute_indices,
        "source_manifest_sha256": bundle.source_manifest_sha256,
        "source_dataset_root": bundle.source_dataset_root,
        "state_file": STATE_STATS_FILENAME,
        "horizons": sorted(bundle.actions),
        "action_files": action_files,
    }
    _atomic_write_json(output_dir / MANIFEST_FILENAME, manifest)


def _load_json(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as json_file:
            payload = json.load(json_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to load relative joint stats file {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"relative joint stats file {path} must contain a JSON object")
    return payload


def _stats_from_payload(payload: dict, path: Path) -> QuantileStats:
    try:
        return QuantileStats(q01=payload["q01"], q99=payload["q99"], count=payload["count"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid quantile stats in {path}: {error}") from error


def load_relative_joint_stats(
    output_dir: str | Path,
    requested_horizon: int,
    expected_feature_names: Sequence[str],
    expected_gripper_indices: Sequence[int],
    expected_source_manifest_sha256: str,
    expected_source_dataset_root: str | Path | None = None,
    expected_state_feature_names: Sequence[str] | None = None,
    expected_state_gripper_indices: Sequence[int] | None = None,
    expected_state_absolute_indices: Sequence[int] | None = None,
) -> RelativeJointStatsBundle:
    output_dir = Path(output_dir)
    manifest_path = output_dir / MANIFEST_FILENAME
    manifest = _load_json(manifest_path)
    if manifest.get("format_version") not in {1, 2, 3}:
        raise ValueError("relative stats manifest must have format_version 1, 2, or 3")
    expected_formula = {
        1: "relative_joint_v1",
        2: "relative_joint_v2",
        3: "relative_joint_v3",
    }[manifest.get("format_version")]
    if manifest.get("formula_version") != expected_formula:
        raise ValueError(f"relative stats manifest must have formula_version {expected_formula}")
    feature_names = manifest.get("feature_names")
    gripper_indices = manifest.get("gripper_indices")
    state_feature_names = manifest.get("state_feature_names", feature_names)
    state_gripper_indices = manifest.get("state_gripper_indices", gripper_indices)
    state_absolute_indices = manifest.get("state_absolute_indices", [])
    source_sha256 = manifest.get("source_manifest_sha256")
    source_dataset_root = manifest.get("source_dataset_root")
    if feature_names != list(expected_feature_names):
        raise ValueError("relative stats action feature names do not match the expected feature names")
    if expected_state_feature_names is not None and state_feature_names != list(expected_state_feature_names):
        raise ValueError("relative stats state feature names do not match the expected feature names")
    if gripper_indices != list(expected_gripper_indices):
        raise ValueError("relative stats action gripper indices do not match the expected gripper indices")
    if expected_state_gripper_indices is not None and state_gripper_indices != list(expected_state_gripper_indices):
        raise ValueError("relative stats state gripper indices do not match the expected gripper indices")
    if expected_state_absolute_indices is not None and state_absolute_indices != list(expected_state_absolute_indices):
        raise ValueError("relative stats state absolute indices do not match the expected indices")
    if source_sha256 != expected_source_manifest_sha256:
        raise ValueError("relative stats source manifest SHA256 does not match the expected SHA256")
    if not isinstance(source_dataset_root, str):
        raise ValueError("relative stats source dataset root must be a string")
    if expected_source_dataset_root is not None:
        expected_root = str(Path(expected_source_dataset_root).expanduser().resolve())
        if source_dataset_root != expected_root:
            raise ValueError("relative stats source dataset root does not match the expected source dataset root")
    if not isinstance(requested_horizon, int) or isinstance(requested_horizon, bool) or requested_horizon <= 0:
        raise ValueError("requested horizon must be a positive integer")
    action_files = manifest.get("action_files")
    horizons = manifest.get("horizons")
    expected_action_filename = _action_stats_filename(requested_horizon)
    if (
        not isinstance(horizons, list)
        or requested_horizon not in horizons
        or not isinstance(action_files, dict)
        or action_files.get(str(requested_horizon)) != expected_action_filename
    ):
        raise ValueError(f"requested horizon {requested_horizon} is not present in the relative stats manifest")
    if manifest.get("state_file") != STATE_STATS_FILENAME:
        raise ValueError("relative stats manifest has an invalid state filename")

    state_path = output_dir / STATE_STATS_FILENAME
    action_path = output_dir / expected_action_filename
    state = _stats_from_payload(_load_json(state_path), state_path)
    action_payload = _load_json(action_path)
    if action_payload.get("horizon") != requested_horizon:
        raise ValueError(
            f"relative action stats horizon must equal requested horizon {requested_horizon}"
        )
    action = _stats_from_payload(action_payload, action_path)
    return RelativeJointStatsBundle(
        state=state,
        actions={requested_horizon: action},
        feature_names=feature_names,
        gripper_indices=gripper_indices,
        source_manifest_sha256=source_sha256,
        source_dataset_root=source_dataset_root,
        state_feature_names=state_feature_names,
        state_gripper_indices=state_gripper_indices,
        state_absolute_indices=state_absolute_indices,
    )


def load_relative_joint_stats_paths(
    state_path: str | Path,
    action_path: str | Path,
    expected_horizon: int,
    expected_feature_names: Sequence[str],
    expected_gripper_indices: Sequence[int],
    expected_state_feature_names: Sequence[str] | None = None,
    expected_state_gripper_indices: Sequence[int] | None = None,
    expected_state_absolute_indices: Sequence[int] | None = None,
) -> RelativeJointStatsBundle:
    """Load a matched state/action stats pair through its canonical manifest."""
    state_path = Path(state_path)
    action_path = Path(action_path)
    for path in (state_path, action_path):
        if not path.is_file():
            raise ValueError(f"relative joint stats path does not exist or is not a file: {path}")

    if state_path.parent.resolve() != action_path.parent.resolve():
        raise ValueError("relative state and action stats paths must be in the same directory")
    if state_path.name != STATE_STATS_FILENAME:
        raise ValueError(f"relative state stats path must use fixed filename {STATE_STATS_FILENAME}")
    expected_action_filename = _action_stats_filename(expected_horizon)
    if action_path.name != expected_action_filename:
        raise ValueError(
            f"relative action stats path must use horizon {expected_horizon} filename {expected_action_filename}"
        )

    manifest = _load_json(state_path.parent / MANIFEST_FILENAME)
    source_manifest_sha256 = manifest.get("source_manifest_sha256")
    if not _is_sha256(source_manifest_sha256):
        raise ValueError(
            "relative stats source manifest SHA256 must be a lowercase 64-character hexadecimal digest"
        )
    return load_relative_joint_stats(
        state_path.parent,
        requested_horizon=expected_horizon,
        expected_feature_names=expected_feature_names,
        expected_gripper_indices=expected_gripper_indices,
        expected_source_manifest_sha256=source_manifest_sha256,
        expected_state_feature_names=expected_state_feature_names,
        expected_state_gripper_indices=expected_state_gripper_indices,
        expected_state_absolute_indices=expected_state_absolute_indices,
    )
