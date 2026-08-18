"""Compute train-only relative pose10d q01/q99 statistics from a LeRobot v3 split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lerobot.datasets.end_effector_pose_stats import (
    compute_relative_pose_stats_from_episodes,
    save_relative_pose_stats,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _parse_horizons(value: str) -> list[int]:
    try:
        horizons = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("horizons must be a JSON array such as '[16, 50]'") from exc
    if not isinstance(horizons, list) or any(type(horizon) is not int or horizon <= 0 for horizon in horizons):
        raise argparse.ArgumentTypeError("horizons must be a nonempty JSON array of positive integers")
    return horizons


def load_train_episodes(dataset_root: Path, repo_id: str) -> list[np.ndarray]:
    """Read only the absolute state column and restore episode boundaries from v3 metadata."""
    dataset_root = dataset_root.resolve()
    if dataset_root.name.lower() == "test":
        raise ValueError(f"refusing to compute train-only statistics from test root {dataset_root}")
    dataset = LeRobotDataset(repo_id, root=dataset_root)
    states = np.asarray(dataset.hf_dataset["observation.state"], dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != 10:
        raise ValueError(f"expected pose10d observation.state with shape [N, 10], got {states.shape}")
    episodes = []
    for episode in dataset.meta.episodes:
        start = int(episode["dataset_from_index"])
        end = int(episode["dataset_to_index"])
        episodes.append(states[start:end])
    if not episodes:
        raise ValueError("dataset has no episodes")
    return episodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--horizons", required=True, type=_parse_horizons)
    args = parser.parse_args()
    bundle = compute_relative_pose_stats_from_episodes(
        load_train_episodes(args.dataset_root, args.repo_id), horizons=args.horizons
    )
    save_relative_pose_stats(bundle, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "horizons": args.horizons}, indent=2))


if __name__ == "__main__":
    main()
