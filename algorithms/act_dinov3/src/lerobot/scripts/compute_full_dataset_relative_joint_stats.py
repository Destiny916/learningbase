#!/usr/bin/env python

"""Compute relative-joint statistics from every episode in one LeRobot v3 dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path

from lerobot.datasets.relative_joint_stats import (
    compute_relative_joint_stats_from_episodes,
    save_relative_joint_stats,
)
from lerobot.scripts.compute_relative_joint_stats import _load_episodes_from_parquet, _load_feature_names


def parse_int_list(value: str) -> list[int]:
    try:
        values = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError("expected a JSON list of integers") from error
    if not isinstance(values, list) or not values or any(type(item) is not int for item in values):
        raise argparse.ArgumentTypeError("expected a nonempty JSON list of integers")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizons", type=parse_int_list, required=True)
    parser.add_argument("--gripper-indices", type=parse_int_list, required=True)
    parser.add_argument(
        "--state-absolute-indices",
        type=parse_int_list,
        default=None,
        help="State dimensions to retain as absolute values. Defaults to the mapped gripper dimensions.",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    info_path = dataset_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    state_feature_names, action_feature_names, action_state_indices = _load_feature_names(dataset_root)
    if any(
        not isinstance(index, int)
        or index < 0
        or index >= len(action_feature_names)
        or "gripper" not in action_feature_names[index].lower()
        for index in args.gripper_indices
    ):
        raise ValueError("gripper indices must point to action features whose names contain 'gripper'")
    default_state_absolute_indices = [action_state_indices[index] for index in args.gripper_indices]
    state_absolute_indices = (
        default_state_absolute_indices
        if args.state_absolute_indices is None
        else list(args.state_absolute_indices)
    )
    if len(set(state_absolute_indices)) != len(state_absolute_indices) or any(
        index < 0 or index >= len(state_feature_names) for index in state_absolute_indices
    ):
        raise ValueError("state absolute indices must be unique valid observation.state dimensions")
    episodes, action_episodes = _load_episodes_from_parquet(
        dataset_root,
        len(state_feature_names),
        len(action_feature_names),
        action_state_indices,
    )

    full_manifest = {
        "format_version": 1,
        "mode": "all_dataset_episodes_no_holdout",
        "dataset_root": str(dataset_root),
        "info_sha256": hashlib.sha256(info_path.read_bytes()).hexdigest(),
        "total_episodes": int(info["total_episodes"]),
        "total_frames": int(info["total_frames"]),
        "episode_indices": list(range(len(episodes))),
    }
    manifest_bytes = (json.dumps(full_manifest, indent=2, sort_keys=True) + "\n").encode()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "full_dataset_manifest.json").write_bytes(manifest_bytes)

    command = shlex.join(
        [
            sys.executable,
            "-m",
            "lerobot.scripts.compute_full_dataset_relative_joint_stats",
            f"--dataset-root={dataset_root}",
            f"--output-dir={output_dir}",
            f"--horizons={json.dumps(args.horizons, separators=(',', ':'))}",
            f"--gripper-indices={json.dumps(args.gripper_indices, separators=(',', ':'))}",
            f"--state-absolute-indices={json.dumps(state_absolute_indices, separators=(',', ':'))}",
        ]
    )
    stats = compute_relative_joint_stats_from_episodes(
        episodes,
        gripper_indices=args.gripper_indices,
        horizons=args.horizons,
        feature_names=action_feature_names,
        state_feature_names=state_feature_names,
        action_feature_names=action_feature_names,
        state_gripper_indices=default_state_absolute_indices,
        action_gripper_indices=args.gripper_indices,
        action_state_indices=action_state_indices,
        action_episodes=action_episodes,
        state_absolute_indices=state_absolute_indices,
        source_manifest_sha256=manifest_sha256,
        source_dataset_root=str(dataset_root),
    )
    save_relative_joint_stats(stats, output_dir, generation_command=command)
    print(
        f"FULL_DATASET_RELATIVE_STATS_OK episodes={len(episodes)} frames={sum(len(ep) for ep in episodes)} "
        f"manifest_sha256={manifest_sha256}"
    )


if __name__ == "__main__":
    main()
