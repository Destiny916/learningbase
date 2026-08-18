from __future__ import annotations

import csv
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

from lerobot.scripts.append_object_centers_to_lerobot_v30 import convert_dataset


def _write_source_dataset(root: Path) -> None:
    names = [f"left_joint_{index}" for index in range(6)] + ["left_gripper"]
    names += [f"right_joint_{index}" for index in range(6)] + ["right_gripper"]
    info = {
        "codebase_version": "v3.0",
        "fps": 25,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [14], "names": names},
            "action": {"dtype": "float32", "shape": [14], "names": names},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
        "total_episodes": 1,
        "total_frames": 2,
    }
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/stats.json").write_text(json.dumps({"observation.state": {}, "action": {}}))
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"observation.state": [float(value) for value in range(14)], "action": [1.0] * 14, "timestamp": 0.0, "frame_index": 0, "episode_index": 0, "index": 0, "task_index": 0},
                {"observation.state": [float(value + 14) for value in range(14)], "action": [2.0] * 14, "timestamp": 0.04, "frame_index": 1, "episode_index": 0, "index": 1, "task_index": 0},
            ]
        ),
        root / "data/chunk-000/file-000.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [{"episode_index": 0, "length": 2, "dataset_from_index": 0, "dataset_to_index": 2}],
        ),
        root / "meta/episodes/chunk-000/file-000.parquet",
    )


def _write_centers(root: Path) -> None:
    episode_dir = root / "episodes/episode_000"
    episode_dir.mkdir(parents=True)
    (episode_dir / "episode_000_object_xyz.html").write_text("<html>trajectory</html>")
    with (root / "all_centers_smoothed.csv").open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["dataset_index"]
            + [f"{object_name}_{axis}_smooth_m" for object_name in ("bread", "bowl") for axis in "xyz"],
        )
        writer.writeheader()
        writer.writerow({"dataset_index": 0, "bread_x_smooth_m": 0.1, "bread_y_smooth_m": 0.2, "bread_z_smooth_m": 0.3, "bowl_x_smooth_m": 0.4, "bowl_y_smooth_m": 0.5, "bowl_z_smooth_m": 0.6})
        writer.writerow({"dataset_index": 1, "bread_x_smooth_m": 0.7, "bread_y_smooth_m": 0.8, "bread_z_smooth_m": 0.9, "bowl_x_smooth_m": 1.0, "bowl_y_smooth_m": 1.1, "bowl_z_smooth_m": 1.2})


def test_appends_object_centers_to_state_and_preserves_actions_and_html(tmp_path: Path) -> None:
    source, centers, output = tmp_path / "source", tmp_path / "centers", tmp_path / "output"
    _write_source_dataset(source)
    _write_centers(centers)

    convert_dataset(source, centers, output)

    table = pq.read_table(output / "data/chunk-000/file-000.parquet")
    np.testing.assert_allclose(
        table["observation.state"].to_pylist(),
        [
            [float(value) for value in range(14)] + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [float(value + 14) for value in range(14)] + [0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        ],
    )
    assert table["action"].to_pylist() == [[1.0] * 14, [2.0] * 14]
    info = json.loads((output / "meta/info.json").read_text())
    assert info["features"]["observation.state"]["shape"] == [20]
    assert info["features"]["observation.state"]["names"][-6:] == [
        "bread_x_m",
        "bread_y_m",
        "bread_z_m",
        "bowl_x_m",
        "bowl_y_m",
        "bowl_z_m",
    ]
    assert (output / "object_xyz_visualizations/episode_000.html").read_text() == "<html>trajectory</html>"
