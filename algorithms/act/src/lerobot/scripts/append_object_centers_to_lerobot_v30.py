#!/usr/bin/env python

"""Create a LeRobot v3 copy whose state appends top-stereo bread/bowl XYZ."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from copy import deepcopy
from pathlib import Path

import datasets
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.compute_stats import get_feature_stats
from lerobot.datasets.feature_utils import get_hf_features_from_features
from lerobot.datasets.io_utils import write_stats, write_table_one_row_group_per_episode
from lerobot.scripts.preprocess_relative_chunk_dataset import (
    OBJECT_CENTER_STATE_NAMES,
    _features_for_hf,
    _load_json,
    _stats_as_lists,
    _write_json,
    make_state_with_object_centers_features,
)


DEFAULT_REPO_ID = "local/730_subtask_doubletop_rgb_with_object_xyz"


def _read_source(source: Path) -> tuple[dict, pa.Table, list[dict], np.ndarray]:
    info = _load_json(source / "meta/info.json")
    data_path = source / "data/chunk-000/file-000.parquet"
    episode_path = source / "meta/episodes/chunk-000/file-000.parquet"
    table = pq.read_table(data_path)
    episodes = pq.read_table(episode_path).to_pylist()
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    if states.shape != (int(info["total_frames"]), 14):
        raise ValueError(f"Expected source 14D state rows, got {states.shape}")
    if len(episodes) != int(info["total_episodes"]):
        raise ValueError("Episode metadata count disagrees with meta/info.json")
    return info, table, episodes, states


def _read_centers(path: Path, expected_frames: int) -> np.ndarray:
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != expected_frames:
        raise ValueError(f"Center CSV has {len(rows)} rows; expected {expected_frames}")
    rows_by_index = {int(row["dataset_index"]): row for row in rows}
    if len(rows_by_index) != expected_frames or set(rows_by_index) != set(range(expected_frames)):
        raise ValueError("Center CSV dataset_index must contain every source frame exactly once")
    values = np.empty((expected_frames, 6), dtype=np.float32)
    for index in range(expected_frames):
        row = rows_by_index[index]
        for column, name in enumerate(OBJECT_CENTER_STATE_NAMES):
            value = row.get(name.replace("_m", "_smooth_m"))
            if value in (None, ""):
                raise ValueError(f"Center CSV row {index} is missing {name}")
            values[index, column] = float(value)
    if not np.isfinite(values).all():
        raise ValueError("Center CSV contains non-finite XYZ values")
    return values


def _rewrite_data(build_root: Path, table: pa.Table, features: dict, states: np.ndarray) -> None:
    columns = {name: table[name].to_pylist() for name in table.column_names}
    columns["observation.state"] = states.tolist()
    hf_dataset = datasets.Dataset.from_dict(
        columns,
        features=get_hf_features_from_features(_features_for_hf(features)),
        split="train",
    )
    output_table = hf_dataset.with_format("arrow")[:]
    data_path = build_root / "data/chunk-000/file-000.parquet"
    data_path.unlink()
    write_table_one_row_group_per_episode(output_table, data_path)


def _rewrite_metadata(build_root: Path, info: dict, features: dict, episodes: list[dict], states: np.ndarray) -> None:
    output_info = deepcopy(info)
    output_info["features"] = features
    _write_json(build_root / "meta/info.json", output_info)

    stats = _load_json(build_root / "meta/stats.json")
    stats["observation.state"] = _stats_as_lists(get_feature_stats(states, axis=0, keepdims=False))
    write_stats(stats, build_root)

    rewritten_episodes = deepcopy(episodes)
    for episode in rewritten_episodes:
        start = int(episode["dataset_from_index"])
        stop = int(episode["dataset_to_index"])
        state_stats = get_feature_stats(states[start:stop], axis=0, keepdims=False)
        for stat_name, value in _stats_as_lists(state_stats).items():
            episode[f"stats/observation.state/{stat_name}"] = value
    episode_path = build_root / "meta/episodes/chunk-000/file-000.parquet"
    episode_path.unlink()
    pq.write_table(pa.Table.from_pylist(rewritten_episodes), episode_path, compression="snappy", use_dictionary=True)


def _copy_html_audits(centers_root: Path, build_root: Path, episode_count: int) -> None:
    visualization_root = build_root / "object_xyz_visualizations"
    visualization_root.mkdir()
    for episode in range(episode_count):
        source = centers_root / "episodes" / f"episode_{episode:03d}" / f"episode_{episode:03d}_object_xyz.html"
        if not source.is_file():
            raise FileNotFoundError(f"Missing interactive trajectory HTML: {source}")
        shutil.copy2(source, visualization_root / f"episode_{episode:03d}.html")


def convert_dataset(source: Path, centers_root: Path, output: Path, repo_id: str = DEFAULT_REPO_ID) -> None:
    source, centers_root, output = source.resolve(), centers_root.resolve(), output.resolve()
    if not source.is_dir() or not centers_root.is_dir():
        raise FileNotFoundError("source and centers-root must both exist")
    if source == output:
        raise ValueError("source and output must differ")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    build_root = output.parent / f".{output.name}.building"
    if build_root.exists():
        raise FileExistsError(f"Stale build directory exists: {build_root}")

    info, table, episodes, states = _read_source(source)
    centers = _read_centers(centers_root / "all_centers_smoothed.csv", len(states))
    features = make_state_with_object_centers_features(info["features"])
    augmented_states = np.concatenate((states, centers), axis=1)

    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, build_root, copy_function=shutil.copy2)
        _rewrite_data(build_root, table, features, augmented_states)
        _rewrite_metadata(build_root, info, features, episodes, augmented_states)
        _copy_html_audits(centers_root, build_root, len(episodes))
        _write_json(
            build_root / "object_xyz_conversion_summary.json",
            {
                "source": str(source),
                "output": str(output),
                "repo_id": repo_id,
                "state": "[left_arm_7d, right_arm_7d, bread_xyz_m, bowl_xyz_m]",
                "action": "preserved source 14D action without modification",
                "coordinate_frame": "rectified_left_camera",
                "units": "meters",
                "total_episodes": len(episodes),
                "total_frames": len(states),
                "interactive_html": "object_xyz_visualizations/episode_XXX.html",
            },
        )
        build_root.rename(output)
    except BaseException:
        shutil.rmtree(build_root, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--centers-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_dataset(args.source, args.centers_root, args.output, args.repo_id)


if __name__ == "__main__":
    main()
