"""Rebuild pose10d train/test splits using a joint7d split's source episodes."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.dataset_tools import split_dataset
from lerobot.datasets.end_effector_pose_stats import compute_relative_pose_stats_from_episodes, save_relative_pose_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file {path} must contain an object")
    return payload


def _source_episode_by_joint_index(summary: dict[str, Any]) -> dict[int, str]:
    rows = summary.get("episodes")
    if not isinstance(rows, list):
        raise ValueError("joint conversion summary must contain an episodes list")
    result: dict[int, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("joint conversion summary episode rows must be objects")
        index = row.get("output_episode_index")
        source_episode = row.get("source_episode")
        if not isinstance(index, int) or not isinstance(source_episode, str) or not source_episode:
            raise ValueError("joint conversion summary rows require output_episode_index and source_episode")
        if index in result:
            raise ValueError(f"duplicate joint output episode index: {index}")
        result[index] = source_episode
    return result


def build_pose_split_indices(
    joint_split_manifest: dict[str, Any],
    joint_conversion_summary: dict[str, Any],
    pose_conversion_summaries: list[dict[str, Any]],
) -> dict[str, list[int]]:
    """Return aggregate pose episode indices with the joint split membership."""
    joint_sources = _source_episode_by_joint_index(joint_conversion_summary)
    split_payload = joint_split_manifest.get("splits")
    if not isinstance(split_payload, dict):
        raise ValueError("joint split manifest must contain splits")

    requested: dict[str, set[str]] = {}
    for split_name in ("train", "test"):
        split = split_payload.get(split_name)
        if not isinstance(split, dict) or not isinstance(split.get("source_episode_indices"), list):
            raise ValueError(f"joint split manifest split '{split_name}' is invalid")
        try:
            requested[split_name] = {joint_sources[index] for index in split["source_episode_indices"]}
        except KeyError as exc:
            raise ValueError(f"joint split references an unknown converted episode: {exc.args[0]}") from exc

    if requested["train"] & requested["test"]:
        raise ValueError("joint train and test source episodes overlap")

    pose_indices: dict[str, int] = {}
    aggregate_offset = 0
    for summary in pose_conversion_summaries:
        rows = summary.get("episodes")
        if not isinstance(rows, list):
            raise ValueError("pose conversion summary must contain an episodes list")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("pose conversion summary episode rows must be objects")
            index = row.get("output_episode_index")
            source_episode = row.get("source_episode")
            if not isinstance(index, int) or not isinstance(source_episode, str) or not source_episode:
                raise ValueError("pose conversion summary rows require output_episode_index and source_episode")
            aggregate_index = aggregate_offset + index
            if source_episode in pose_indices:
                raise ValueError(f"duplicate pose source episode: {source_episode}")
            pose_indices[source_episode] = aggregate_index
        aggregate_offset += len(rows)

    requested_all = requested["train"] | requested["test"]
    missing = requested_all - set(pose_indices)
    if missing:
        raise ValueError(f"joint split episodes missing from pose datasets: {sorted(missing)}")
    if set(pose_indices) != requested_all:
        unexpected = set(pose_indices) - requested_all
        raise ValueError(f"pose datasets contain episodes absent from joint split: {sorted(unexpected)}")

    return {split_name: sorted(pose_indices[source] for source in requested[split_name]) for split_name in requested}


def _load_pose_train_episodes(dataset_root: Path, repo_id: str):
    dataset = LeRobotDataset(repo_id, root=dataset_root)
    states = dataset.hf_dataset["observation.state"]
    episodes = []
    for episode in dataset.meta.episodes:
        start = int(episode["dataset_from_index"])
        end = int(episode["dataset_to_index"])
        episodes.append(states[start:end])
    return episodes


def rebuild_pose_split(
    *,
    joint_split_manifest: Path,
    joint_conversion_summary: Path,
    pose_train_root: Path,
    pose_test_root: Path,
    pose_train_summary: Path,
    pose_test_summary: Path,
    output_root: Path,
    train_repo_id: str,
    test_repo_id: str,
) -> dict[str, list[int]]:
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output root: {output_root}")
    split_indices = build_pose_split_indices(
        _load_json(joint_split_manifest),
        _load_json(joint_conversion_summary),
        [_load_json(pose_train_summary), _load_json(pose_test_summary)],
    )
    build_root = output_root.parent / f".{output_root.name}.building"
    if build_root.exists():
        raise FileExistsError(f"Stale build root exists: {build_root}")

    try:
        aggregate_root = build_root / "aggregate"
        aggregate_datasets(
            repo_ids=["local/pose_source_train", "local/pose_source_test"],
            aggr_repo_id="local/pose_source_aggregate",
            roots=[pose_train_root, pose_test_root],
            aggr_root=aggregate_root,
            concatenate_data=False,
            concatenate_videos=False,
        )
        aggregate = LeRobotDataset("local/pose_source_aggregate", root=aggregate_root)
        split_dataset(aggregate, split_indices, output_dir=build_root)

        train_root = build_root / "train"
        stats = compute_relative_pose_stats_from_episodes(
            _load_pose_train_episodes(train_root, train_repo_id), horizons=[16, 50]
        )
        save_relative_pose_stats(stats, build_root / "normalization")
        (build_root / "split_manifest.json").write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "split_source": "joint7d split manifest",
                    "joint_split_manifest": str(joint_split_manifest.resolve()),
                    "joint_conversion_summary": str(joint_conversion_summary.resolve()),
                    "pose_source_roots": [str(pose_train_root.resolve()), str(pose_test_root.resolve())],
                    "aggregate_episode_indices": split_indices,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        shutil.rmtree(aggregate_root)
        build_root.rename(output_root)
        return split_indices
    except BaseException:
        shutil.rmtree(build_root, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-split-manifest", required=True, type=Path)
    parser.add_argument("--joint-conversion-summary", required=True, type=Path)
    parser.add_argument("--pose-train-root", required=True, type=Path)
    parser.add_argument("--pose-test-root", required=True, type=Path)
    parser.add_argument("--pose-train-summary", required=True, type=Path)
    parser.add_argument("--pose-test-summary", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--train-repo-id", required=True)
    parser.add_argument("--test-repo-id", required=True)
    args = parser.parse_args()
    split_indices = rebuild_pose_split(
        joint_split_manifest=args.joint_split_manifest,
        joint_conversion_summary=args.joint_conversion_summary,
        pose_train_root=args.pose_train_root,
        pose_test_root=args.pose_test_root,
        pose_train_summary=args.pose_train_summary,
        pose_test_summary=args.pose_test_summary,
        output_root=args.output_root,
        train_repo_id=args.train_repo_id,
        test_repo_id=args.test_repo_id,
    )
    print(json.dumps(split_indices, indent=2))


if __name__ == "__main__":
    main()
