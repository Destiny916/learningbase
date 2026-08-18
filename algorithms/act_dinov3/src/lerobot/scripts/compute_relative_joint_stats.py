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

"""Compute train-only relative joint q01/q99 statistics from LeRobot v3 parquet data."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pyarrow.parquet as pq
import torch

from lerobot.datasets.relative_joint_stats import (
    RelativeJointStatsBundle,
    compute_relative_joint_stats_from_episodes,
    save_relative_joint_stats,
)


REQUIRED_HORIZONS = {16, 50}


def _parse_int_list(value: str) -> list[int]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError("expected a JSON list of integers, for example '[50,16]'") from error
    if (
        not isinstance(parsed, list)
        or not parsed
        or any(not isinstance(item, int) or isinstance(item, bool) for item in parsed)
    ):
        raise argparse.ArgumentTypeError("expected a nonempty JSON list of integers")
    return parsed


def _split_root(manifest: dict, split: str, manifest_path: Path) -> Path:
    try:
        root_value = manifest["splits"][split]["root"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"split manifest must define splits.{split}.root") from error
    if not isinstance(root_value, str) or not root_value:
        raise ValueError(f"split manifest splits.{split}.root must be a nonempty path string")
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        root = manifest_path.parent / root
    return root.resolve()


def _load_and_validate_manifest(dataset_root: Path, manifest_path: Path) -> str:
    manifest_bytes = manifest_path.read_bytes()
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as error:
        raise ValueError(f"split manifest is not valid JSON: {manifest_path}") from error
    if not isinstance(manifest, dict):
        raise ValueError("split manifest must contain a JSON object")
    resolved_dataset_root = dataset_root.expanduser().resolve()
    if resolved_dataset_root.name.lower() == "test":
        raise ValueError(
            f"refusing to compute train-only stats from a dataset root named test: {resolved_dataset_root}"
        )
    manifest_test_root = _split_root(manifest, "test", manifest_path)
    if resolved_dataset_root == manifest_test_root:
        raise ValueError(f"refusing to compute train-only stats from manifest test root: {resolved_dataset_root}")
    manifest_train_root = _split_root(manifest, "train", manifest_path)
    if resolved_dataset_root != manifest_train_root:
        raise ValueError(
            f"dataset root must resolve to manifest train root {manifest_train_root}, got {resolved_dataset_root}"
        )
    return hashlib.sha256(manifest_bytes).hexdigest()


def _load_feature_names(dataset_root: Path) -> tuple[list[str], list[str], list[int]]:
    info_path = dataset_root / "meta" / "info.json"
    try:
        with info_path.open(encoding="utf-8") as info_file:
            info = json.load(info_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to read LeRobot metadata {info_path}: {error}") from error
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"dataset must be LeRobot v3.0, got {info.get('codebase_version')!r}")
    try:
        state_feature = info["features"]["observation.state"]
        action_feature = info["features"]["action"]
    except (KeyError, TypeError) as error:
        raise ValueError("dataset metadata must define observation.state and action features") from error
    state_shape = state_feature.get("shape")
    action_shape = action_feature.get("shape")
    if (
        not isinstance(state_shape, list)
        or not isinstance(action_shape, list)
        or len(state_shape) != 1
        or len(action_shape) != 1
        or state_shape[0] <= 0
        or action_shape[0] <= 0
    ):
        raise ValueError("observation.state and action must both have nonempty one-dimensional shapes [D]")
    state_dimension = state_shape[0]
    action_dimension = action_shape[0]
    state_names = state_feature.get("names")
    action_names = action_feature.get("names")
    if (
        not isinstance(state_names, list)
        or len(state_names) != state_dimension
        or any(not isinstance(name, str) or not name for name in state_names)
    ):
        raise ValueError("observation.state must define nonempty feature names")
    if (
        not isinstance(action_names, list)
        or len(action_names) != action_dimension
        or any(not isinstance(name, str) or not name for name in action_names)
    ):
        raise ValueError("action must define nonempty feature names")
    if len(set(state_names)) != len(state_names) or len(set(action_names)) != len(action_names):
        raise ValueError("state and action feature names must be unique")
    state_name_to_index = {name: index for index, name in enumerate(state_names)}
    try:
        action_state_indices = [state_name_to_index[name] for name in action_names]
    except KeyError as error:
        raise ValueError(f"action feature name {error.args[0]!r} is missing from observation.state") from error
    if state_dimension == 7 and state_names != [f"joint_{index}" for index in range(6)] + ["gripper"]:
        raise ValueError("feature names: seven-dimensional dataset must use canonical joint_0..joint_5,gripper names")
    return state_names, action_names, action_state_indices


def _load_episodes_from_parquet(
    dataset_root: Path,
    state_dimension: int,
    action_dimension: int,
    action_state_indices: Sequence[int],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    parquet_paths = sorted((dataset_root / "data").glob("**/*.parquet"))
    if not parquet_paths:
        raise ValueError(f"no train parquet data found under {dataset_root / 'data'}")
    grouped_frames: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    for parquet_path in parquet_paths:
        try:
            table = pq.read_table(
                parquet_path,
                columns=["observation.state", "action", "episode_index", "frame_index"],
            )
        except Exception as error:
            raise ValueError(f"failed to read train parquet file {parquet_path}: {error}") from error
        states = table.column("observation.state").to_pylist()
        actions = table.column("action").to_pylist()
        episode_indices = table.column("episode_index").to_pylist()
        frame_indices = table.column("frame_index").to_pylist()
        for state, action, episode_index, frame_index in zip(
            states,
            actions,
            episode_indices,
            frame_indices,
            strict=True,
        ):
            state_values = np.asarray(state, dtype=np.float64)
            action_values = np.asarray(action, dtype=np.float64)
            if state_values.shape != (state_dimension,):
                raise ValueError(
                    f"observation.state in {parquet_path} must have shape [{state_dimension}], got {state_values.shape}"
                )
            if action_values.shape != (action_dimension,):
                raise ValueError(f"action in {parquet_path} must have shape [{action_dimension}], got {action_values.shape}")
            grouped_frames.setdefault(int(episode_index), []).append(
                (int(frame_index), state_values, action_values)
            )
    if not grouped_frames:
        raise ValueError("train parquet data contains no frames")

    episodes = []
    for episode_index in sorted(grouped_frames):
        frames = sorted(grouped_frames[episode_index], key=lambda item: item[0])
        frame_indices = [frame_index for frame_index, _, _ in frames]
        if len(set(frame_indices)) != len(frame_indices):
            raise ValueError(f"episode {episode_index} contains duplicate frame_index values")
        if frame_indices != list(range(len(frames))):
            raise ValueError(
                f"episode {episode_index} frame_index values must be contiguous from zero, got {frame_indices}"
            )
        states = np.stack([state for _, state, _ in frames])
        actions = np.stack([action for _, _, action in frames])
        if len(frames) > 1 and not np.allclose(
            actions[:-1],
            states[1:, action_state_indices],
            rtol=1e-5,
            atol=1e-6,
        ):
            max_error = np.max(np.abs(actions[:-1] - states[1:, action_state_indices]))
            raise ValueError(
                f"episode {episode_index} violates action contract action[t] == state[t+1] "
                f"for non-tail frames (max absolute error {max_error:g})"
            )
        episodes.append(torch.from_numpy(states))
    return episodes, [torch.from_numpy(np.stack([action for _, _, action in sorted(grouped_frames[index], key=lambda item: item[0])])) for index in sorted(grouped_frames)]


def compute_relative_joint_stats_for_dataset(
    dataset_root: str | Path,
    split_manifest: str | Path,
    output_dir: str | Path,
    horizons: Sequence[int],
    gripper_indices: Sequence[int],
    state_absolute_indices: Sequence[int] = (),
    generation_command: str = "",
) -> RelativeJointStatsBundle:
    if len(horizons) != len(REQUIRED_HORIZONS) or set(horizons) != REQUIRED_HORIZONS:
        raise ValueError("horizons must contain exactly 16 and 50")
    dataset_root = Path(dataset_root)
    manifest_path = Path(split_manifest).expanduser().resolve()
    manifest_sha256 = _load_and_validate_manifest(dataset_root, manifest_path)
    resolved_dataset_root = dataset_root.expanduser().resolve()
    state_feature_names, action_feature_names, action_state_indices = _load_feature_names(resolved_dataset_root)
    if any(
        not isinstance(index, int)
        or index < 0
        or index >= len(action_feature_names)
        or "gripper" not in action_feature_names[index].lower()
        for index in gripper_indices
    ):
        raise ValueError("gripper indices must point to action features whose names contain 'gripper'")
    state_gripper_indices = [action_state_indices[index] for index in gripper_indices]
    episodes, action_episodes = _load_episodes_from_parquet(
        resolved_dataset_root,
        len(state_feature_names),
        len(action_feature_names),
        action_state_indices,
    )
    bundle = compute_relative_joint_stats_from_episodes(
        episodes,
        gripper_indices=gripper_indices,
        horizons=horizons,
        feature_names=action_feature_names,
        state_feature_names=state_feature_names,
        action_feature_names=action_feature_names,
        state_gripper_indices=state_gripper_indices,
        action_gripper_indices=gripper_indices,
        action_state_indices=action_state_indices,
        action_episodes=action_episodes,
        state_absolute_indices=state_absolute_indices,
        source_manifest_sha256=manifest_sha256,
        source_dataset_root=str(resolved_dataset_root),
    )
    save_relative_joint_stats(bundle, output_dir, generation_command=generation_command)
    return bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizons", type=_parse_int_list, required=True)
    parser.add_argument("--gripper-indices", type=_parse_int_list, required=True)
    parser.add_argument("--state-absolute-indices", type=_parse_int_list, default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generation_command = shlex.join(
        [
            sys.executable,
            "-m",
            "lerobot.scripts.compute_relative_joint_stats",
            f"--dataset-root={args.dataset_root}",
            f"--split-manifest={args.split_manifest}",
            f"--output-dir={args.output_dir}",
            f"--horizons={json.dumps(args.horizons, separators=(',', ':'))}",
            f"--gripper-indices={json.dumps(args.gripper_indices, separators=(',', ':'))}",
            f"--state-absolute-indices={json.dumps(args.state_absolute_indices, separators=(',', ':'))}",
        ]
    )
    compute_relative_joint_stats_for_dataset(
        dataset_root=args.dataset_root,
        split_manifest=args.split_manifest,
        output_dir=args.output_dir,
        horizons=args.horizons,
        gripper_indices=args.gripper_indices,
        state_absolute_indices=args.state_absolute_indices,
        generation_command=generation_command,
    )


if __name__ == "__main__":
    main()
