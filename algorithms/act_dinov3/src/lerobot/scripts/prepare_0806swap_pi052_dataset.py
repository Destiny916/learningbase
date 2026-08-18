"""Prepare the converted 0806 dataset and PI052 q01/q99 bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

TASK = "Pick up the bread with the right gripper, transfer it to the left gripper, and place it in the bowl."
STATE_NAMES = (
    [f"left_joint_{i}" for i in range(6)]
    + ["left_endpoint_x", "left_endpoint_y", "left_endpoint_z", "left_gripper"]
    + [f"right_joint_{i}" for i in range(6)]
    + ["right_endpoint_x", "right_endpoint_y", "right_endpoint_z", "right_gripper"]
)
ACTION_NAMES = [
    *[f"left_joint_{i}" for i in range(6)],
    "left_gripper",
    *[f"right_joint_{i}" for i in range(6)],
    "right_gripper",
]
ACTION_TO_STATE = [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 19]
STATE_GRIPPERS = [9, 19]
ACTION_GRIPPERS = [6, 13]
STATE_ABSOLUTE_RELATIVE = [6, 7, 8, 9, 16, 17, 18, 19]
STATE_JOINTS = [i for i in range(20) if i not in STATE_ABSOLUTE_RELATIVE]
HORIZON = 50
CAMERA_KEY_RENAMES = (
    ("observation.images.wrist_left", "observation.images.gripper_left"),
    ("observation.images.wrist_right", "observation.images.gripper_right"),
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _q(values: np.ndarray) -> dict:
    return {
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
        "count": int(values.shape[0]),
    }


def _rewrite_metadata(root: Path) -> None:
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info["features"]
    for old, new in CAMERA_KEY_RENAMES:
        if old in features:
            features[new] = features.pop(old)
    for feature in features.values():
        if feature["dtype"] in {"image", "video"}:
            feature.setdefault("names", None)
    info["features"] = features
    info.pop("task_description", None)
    _write_json(info_path, info)

    tasks_path = root / "meta/tasks.parquet"
    tasks = pq.read_table(tasks_path)
    tasks = tasks.set_column(tasks.schema.get_field_index("task"), "task", pa.array([TASK]))
    pq.write_table(tasks, tasks_path)

    stats_path = root / "meta/stats.json"
    metadata_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    for old, new in CAMERA_KEY_RENAMES:
        if old in metadata_stats:
            metadata_stats[new] = metadata_stats.pop(old)
    _write_json(stats_path, metadata_stats)

    for episodes_path in sorted((root / "meta/episodes").glob("chunk-*/file-*.parquet")):
        episodes = pq.read_table(episodes_path)
        renamed_columns = []
        for name in episodes.schema.names:
            for old, new in CAMERA_KEY_RENAMES:
                name = name.replace(old, new)
            renamed_columns.append(name)
        pq.write_table(episodes.rename_columns(renamed_columns), episodes_path)


def _rewrite_data(root: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    state_episodes: dict[int, list[list[float]]] = {}
    action_episodes: dict[int, list[list[float]]] = {}
    for path in sorted((root / "data").glob("chunk-*/file-*.parquet")):
        table = pq.read_table(path)
        states = table["observation.state"].to_pylist()
        actions = table["action"].to_pylist()
        episode_ids = table["episode_index"].to_pylist()
        updated_states = []
        updated_actions = []
        for state, episode in zip(states, episode_ids, strict=True):
            values = np.asarray(state, dtype=np.float32)
            if values.shape != (20,) or not np.isfinite(values).all():
                raise ValueError(f"invalid state row in {path}: {values.shape}")
            values = values.copy()
            values[7] += 0.49
            updated_states.append(values.tolist())
            state_episodes.setdefault(int(episode), []).append(values.tolist())
        for action, episode in zip(actions, episode_ids, strict=True):
            values = np.asarray(action, dtype=np.float32)
            if values.shape != (14,) or not np.isfinite(values).all():
                raise ValueError(f"invalid action row in {path}: {values.shape}")
            values = values.copy()
            updated_actions.append(values.tolist())
            action_episodes.setdefault(int(episode), []).append(values.tolist())
        state_idx = table.schema.get_field_index("observation.state")
        table = table.set_column(state_idx, "observation.state", pa.array(updated_states, type=table["observation.state"].type))
        action_idx = table.schema.get_field_index("action")
        table = table.set_column(action_idx, "action", pa.array(updated_actions, type=table["action"].type))
        metadata = table.schema.metadata
        if metadata:
            metadata = {key: value.replace(b"observation.images.wrist_left", b"observation.images.gripper_left").replace(b"observation.images.wrist_right", b"observation.images.gripper_right") for key, value in metadata.items()}
            table = table.replace_schema_metadata(metadata)
        pq.write_table(table, path)
    return (
        [np.asarray(state_episodes[index], dtype=np.float64) for index in sorted(state_episodes)],
        [np.asarray(action_episodes[index], dtype=np.float64) for index in sorted(action_episodes)],
    )


def _relative_stats(states: list[np.ndarray], actions: list[np.ndarray], state_absolute: list[int]) -> tuple[dict, dict]:
    relative_states = []
    relative_actions = []
    action_arm = [i for i in range(14) if i not in ACTION_GRIPPERS]
    for state, action in zip(states, actions, strict=True):
        relative = state.copy()
        relative[0, STATE_JOINTS] = 0.0
        relative[1:, STATE_JOINTS] = state[1:, STATE_JOINTS] - state[:-1, STATE_JOINTS]
        relative_states.append(relative)
        targets = []
        for offset in range(1, min(HORIZON, len(state) - 1) + 1):
            # action[t] already stores the next state target q[t + 1]. For a
            # horizon offset k, align action[t + k - 1] with state[t].
            target = action[offset - 1 : -1].copy()
            target[:, action_arm] = target[:, action_arm] - state[:-offset, np.asarray(ACTION_TO_STATE)[action_arm]]
            targets.append(target)
        relative_actions.append(np.concatenate(targets, axis=0))
    return _q(np.concatenate(relative_states, axis=0)), _q(np.concatenate(relative_actions, axis=0))


def _write_stats(root: Path, states: list[np.ndarray], actions: list[np.ndarray]) -> None:
    relative_state, relative_action = _relative_stats(states, actions, STATE_ABSOLUTE_RELATIVE)
    absolute_state = _q(np.concatenate(states, axis=0))
    absolute_action_values = []
    for action in actions:
        for offset in range(1, min(HORIZON, len(action) - 1) + 1):
            absolute_action_values.append(action[offset:])
    absolute_action = _q(np.concatenate(absolute_action_values, axis=0))
    metadata_stats_path = root / "meta/stats.json"
    metadata_stats = json.loads(metadata_stats_path.read_text(encoding="utf-8"))
    for key, values in (
        ("observation.state", np.concatenate(states, axis=0)),
        ("action", np.concatenate(actions, axis=0)),
    ):
        entry = metadata_stats.setdefault(key, {})
        entry.update(
            {
                "min": np.min(values, axis=0).tolist(),
                "max": np.max(values, axis=0).tolist(),
                "mean": np.mean(values, axis=0).tolist(),
                "std": np.std(values, axis=0).tolist(),
                "q01": np.quantile(values, 0.01, axis=0).tolist(),
                "q99": np.quantile(values, 0.99, axis=0).tolist(),
                "count": [int(values.shape[0])],
            }
        )
    _write_json(metadata_stats_path, metadata_stats)
    source = root / "meta/info.json"
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    state_relative_manifest = {
        "format_version": 3,
        "formula_version": "relative_joint_v3",
        "feature_names": ACTION_NAMES,
        "gripper_indices": ACTION_GRIPPERS,
        "state_feature_names": list(STATE_NAMES),
        "state_gripper_indices": STATE_GRIPPERS,
        "state_absolute_indices": STATE_ABSOLUTE_RELATIVE,
        "source_manifest_sha256": source_sha,
        "source_dataset_root": str(root.resolve()),
        "state_file": "relative_state_q01_q99.json",
        "horizons": [HORIZON],
        "action_files": {str(HORIZON): f"relative_action_chunk{HORIZON}_q01_q99.json"},
    }
    relative_root = root / "normalization_relative"
    _write_json(relative_root / "relative_state_q01_q99.json", relative_state)
    relative_action["horizon"] = HORIZON
    _write_json(relative_root / f"relative_action_chunk{HORIZON}_q01_q99.json", relative_action)
    _write_json(relative_root / "relative_stats_manifest.json", state_relative_manifest)

    mixed_manifest = dict(state_relative_manifest)
    mixed_manifest["state_absolute_indices"] = list(range(20))
    mixed_root = root / "normalization_absolute_state_relative_action"
    _write_json(mixed_root / "relative_state_q01_q99.json", absolute_state)
    relative_action_with_horizon = dict(relative_action)
    _write_json(mixed_root / f"relative_action_chunk{HORIZON}_q01_q99.json", relative_action_with_horizon)
    _write_json(mixed_root / "relative_stats_manifest.json", mixed_manifest)
    _write_json(mixed_root / "absolute_state_q01_q99.json", absolute_state)

    absolute_root = root / "normalization_absolute"
    absolute_manifest = {
        "format_version": 1,
        "formula_version": "absolute_future_action_v1",
        "feature_names": ACTION_NAMES,
        "scaled_indices": list(range(14)),
        "state_file": "absolute_state_q01_q99.json",
        "action_files": {str(HORIZON): f"absolute_action_chunk{HORIZON}_q01_q99.json"},
    }
    _write_json(absolute_root / "absolute_state_q01_q99.json", absolute_state)
    absolute_action["horizon"] = HORIZON
    _write_json(absolute_root / f"absolute_action_chunk{HORIZON}_q01_q99.json", absolute_action)
    _write_json(absolute_root / "absolute_stats_manifest.json", absolute_manifest)


def prepare(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    shutil.copytree(source, output)
    for old, new in CAMERA_KEY_RENAMES:
        old_path, new_path = output / "videos" / old, output / "videos" / new
        old_path.rename(new_path)
    _rewrite_metadata(output)
    states, actions = _rewrite_data(output)
    _write_stats(output, states, actions)
    _write_json(output / "conversion_summary.json", {"source": str(source), "task": TASK, "state_dim": 20, "action_dim": 14, "left_endpoint_y_offset_m": 0.49, "gripper_values_preserved_from_source": True, "video_frames_changed": False, "episodes": len(states), "frames": int(sum(len(x) for x in states))})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)
