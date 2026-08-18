"""Compute train-only absolute state and future-action q01/q99 statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lerobot.datasets.absolute_action_stats import compute_absolute_action_stats_from_episodes, save_absolute_action_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _parse_indices(value: str) -> list[int]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError("value must be a JSON integer list") from error
    if not isinstance(parsed, list) or not parsed or any(type(item) is not int for item in parsed):
        raise argparse.ArgumentTypeError("value must be a nonempty JSON integer list")
    return parsed


def _load_train_episodes(dataset_root: Path, repo_id: str) -> tuple[list[np.ndarray], list[str]]:
    dataset_root = dataset_root.resolve()
    if dataset_root.name.lower() == "test":
        raise ValueError(f"refusing to compute train-only statistics from test root {dataset_root}")
    dataset = LeRobotDataset(repo_id, root=dataset_root)
    states = np.asarray(dataset.hf_dataset["observation.state"], dtype=np.float32)
    actions = np.asarray(dataset.hf_dataset["action"], dtype=np.float32)
    if states.ndim != 2 or actions.shape != states.shape:
        raise ValueError("observation.state and action must be matching two-dimensional arrays")
    try:
        feature_names = dataset.meta.info.features["observation.state"]["names"]
    except (KeyError, TypeError) as error:
        raise ValueError("dataset observation.state metadata must define feature names") from error
    if not isinstance(feature_names, list) or len(feature_names) != states.shape[1]:
        raise ValueError("observation.state feature names must match the state dimension")
    episodes = []
    for episode in dataset.meta.episodes:
        start = int(episode["dataset_from_index"])
        end = int(episode["dataset_to_index"])
        state_episode = states[start:end]
        action_episode = actions[start:end]
        if len(state_episode) > 1 and not np.allclose(action_episode[:-1], state_episode[1:], rtol=1e-5, atol=1e-6):
            raise ValueError("dataset violates action[t] == state[t+1] for a non-tail frame")
        episodes.append(state_episode)
    if not episodes:
        raise ValueError("dataset contains no episodes")
    return episodes, feature_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizons", type=_parse_indices, required=True)
    parser.add_argument("--scaled-indices", type=_parse_indices, required=True)
    args = parser.parse_args()
    episodes, feature_names = _load_train_episodes(args.dataset_root, args.repo_id)
    bundle = compute_absolute_action_stats_from_episodes(
        episodes,
        horizons=args.horizons,
        feature_names=feature_names,
        scaled_indices=args.scaled_indices,
    )
    save_absolute_action_stats(bundle, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "horizons": args.horizons}, indent=2))


if __name__ == "__main__":
    main()
