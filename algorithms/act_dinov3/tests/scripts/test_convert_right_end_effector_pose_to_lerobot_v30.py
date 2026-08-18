"""Tests for the standalone right end-effector pose v3 converter."""

from __future__ import annotations

import importlib
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from lerobot.scripts.convert_right_end_effector_pose_to_lerobot_v30 import (
    absolute_pose10d,
    align_end_pose_and_gripper_to_camera,
    build_next_actions,
    convert_capture_episodes,
    convert_capture_roots_with_split,
    filter_convertible_episode_dirs,
    discover_episode_dirs,
    euler_rpy_to_matrix,
    matrix_to_rot6d,
    pose10d_from_end_pose,
    relative_pose10d,
    rot6d_to_matrix,
    split_episode_dirs_by_source,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _write_rgb(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=(1, 2, 3)).save(path)


def _require_h264() -> None:
    pytest.importorskip("av", reason="av is required for H.264 conversion tests")
    from lerobot.datasets.pyav_utils import get_codec

    if get_codec("h264") is None:
        pytest.skip("H.264 encoder is unavailable")


def _make_pose_episode(root: Path) -> None:
    pose_dir = root / "arm/endPose/puppetRight"
    joint_dir = root / "arm/jointState/puppetRight"
    camera_dir = root / "camera/color/pikaGripperFisheyeCamera_r"
    for index, timestamp in enumerate((1.000, 1.033, 1.066)):
        _write_json(
            pose_dir / f"{timestamp:.3f}.json",
            {
                "x": 0.1 + 0.01 * index,
                "y": -0.2,
                "z": 0.3,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.1 * index,
            },
        )
        _write_json(joint_dir / f"{timestamp:.3f}.json", {"position": [0, 0, 0, 0, 0, 0, 0.02 * index]})
        _write_rgb(camera_dir / f"{timestamp:.3f}.jpg")


def test_euler_rpy_uses_piper_rz_ry_rx_order() -> None:
    matrix = euler_rpy_to_matrix(roll=0.0, pitch=0.0, yaw=np.pi / 2)

    np.testing.assert_allclose(
        matrix,
        np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        atol=1e-7,
    )


def test_pose10d_keeps_absolute_gripper_and_round_trips_rot6d() -> None:
    pose = pose10d_from_end_pose(
        {"x": 0.2, "y": -0.1, "z": 0.3, "roll": 0.2, "pitch": -0.3, "yaw": 0.4},
        gripper=0.07,
    )

    assert pose.shape == (10,)
    np.testing.assert_allclose(pose[:3], [0.2, -0.1, 0.3])
    assert pose[9] == 0.07
    np.testing.assert_allclose(rot6d_to_matrix(pose[3:9]), euler_rpy_to_matrix(0.2, -0.3, 0.4), atol=1e-6)
    np.testing.assert_allclose(matrix_to_rot6d(rot6d_to_matrix(pose[3:9])), pose[3:9], atol=1e-6)


def test_relative_pose_uses_transform_composition_and_keeps_target_gripper() -> None:
    anchor = pose10d_from_end_pose(
        {"x": 0.3, "y": -0.2, "z": 0.1, "roll": 0.0, "pitch": 0.0, "yaw": np.pi / 2},
        gripper=0.02,
    )
    target = pose10d_from_end_pose(
        {"x": 0.3, "y": -0.1, "z": 0.1, "roll": 0.1, "pitch": 0.0, "yaw": np.pi / 2},
        gripper=0.09,
    )

    relative = relative_pose10d(anchor, target)
    restored = absolute_pose10d(anchor, relative)

    np.testing.assert_allclose(relative[:3], [0.1, 0.0, 0.0], atol=1e-6)
    assert relative[9] == 0.09
    np.testing.assert_allclose(restored, target, atol=1e-6)


def test_build_next_actions_repeats_terminal_absolute_pose() -> None:
    states = np.arange(30, dtype=np.float32).reshape(3, 10)

    actions = build_next_actions(states)

    np.testing.assert_array_equal(actions, np.vstack((states[1:], states[-1:])))


def test_alignment_requires_nearby_end_pose_and_gripper_samples(tmp_path: Path) -> None:
    pose_dir = tmp_path / "arm/endPose/puppetRight"
    joint_dir = tmp_path / "arm/jointState/puppetRight"
    camera_dir = tmp_path / "camera/color/pikaGripperFisheyeCamera_r"
    _write_json(
        pose_dir / "1.000.json",
        {"x": 0.1, "y": 0.2, "z": 0.3, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
    )
    _write_json(
        pose_dir / "1.030.json",
        {"x": 0.4, "y": 0.5, "z": 0.6, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
    )
    _write_json(joint_dir / "1.001.json", {"position": [0, 0, 0, 0, 0, 0, 0.04]})
    _write_rgb(camera_dir / "1.002.jpg")
    _write_rgb(camera_dir / "1.020.jpg")

    aligned = align_end_pose_and_gripper_to_camera(
        pose_dir,
        joint_dir,
        camera_dir,
        max_alignment_delta_sec=0.01,
    )

    assert [frame.camera_path.name for frame in aligned] == ["1.002.jpg"]
    assert aligned[0].end_pose_path.name == "1.000.json"
    assert aligned[0].gripper_path.name == "1.001.json"


def test_even_episode_split_is_stratified_and_reproducible(tmp_path: Path) -> None:
    source_a = tmp_path / "normal"
    source_b = tmp_path / "wrongplace"
    for root, count in ((source_a, 10), (source_b, 5)):
        for index in range(count):
            (root / f"episode{index}" / "arm/endPose/puppetRight").mkdir(parents=True)

    by_source = {
        source_a: discover_episode_dirs(source_a, episode_index_divisor=2),
        source_b: discover_episode_dirs(source_b, episode_index_divisor=2),
    }
    first = split_episode_dirs_by_source(by_source, test_fraction=0.2, seed=42)
    second = split_episode_dirs_by_source(by_source, test_fraction=0.2, seed=42)

    assert [path.name for path in by_source[source_a]] == ["episode0", "episode2", "episode4", "episode6", "episode8"]
    assert [path.name for path in by_source[source_b]] == ["episode0", "episode2", "episode4"]
    assert first == second
    assert len(first.train[source_a]) == 4
    assert len(first.test[source_a]) == 1
    assert len(first.train[source_b]) == 2
    assert len(first.test[source_b]) == 1
    assert set(first.train[source_a]).isdisjoint(first.test[source_a])


def test_conversion_writes_10d_absolute_pose_and_next_action(tmp_path: Path) -> None:
    _require_h264()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    source_root = tmp_path / "normal"
    episode = source_root / "episode0"
    _make_pose_episode(episode)
    output_root = tmp_path / "converted"

    report = convert_capture_episodes(
        {source_root: [episode]},
        output_root,
        repo_id="local/right-end-pose-test",
        encoder_threads=1,
    )

    dataset = LeRobotDataset("local/right-end-pose-test", root=output_root)
    assert report.total_output_episodes == 1
    assert report.total_kept_frames == 3
    assert dataset.meta.features["observation.state"]["shape"] == (10,)
    first = dataset[0]["observation.state"].numpy()
    second = dataset[1]["observation.state"].numpy()
    action = dataset[0]["action"].numpy()
    assert first[9] == pytest.approx(0.0)
    assert second[9] == pytest.approx(0.02)
    np.testing.assert_allclose(action, second, atol=1e-6)
    np.testing.assert_allclose(dataset[2]["action"].numpy(), dataset[2]["observation.state"].numpy(), atol=1e-6)


def test_encoder_queue_capacity_covers_the_longest_selected_episode(tmp_path: Path) -> None:
    module = importlib.import_module("lerobot.scripts.convert_right_end_effector_pose_to_lerobot_v30")
    source_root = tmp_path / "normal"
    episode = source_root / "episode0"
    camera_dir = episode / "camera/color/pikaGripperFisheyeCamera_r"
    camera_dir.mkdir(parents=True)
    for index in range(72):
        (camera_dir / f"{index / 30:.6f}.jpg").touch()

    queue_size = module.encoder_queue_maxsize_for_episodes([(source_root, episode)])

    assert queue_size == 80


def test_split_conversion_writes_separate_train_test_roots_and_manifest(tmp_path: Path) -> None:
    _require_h264()
    source_a = tmp_path / "normal"
    source_b = tmp_path / "differentplace"
    for source in (source_a, source_b):
        for index in range(4):
            _make_pose_episode(source / f"episode{index}")

    output_root = tmp_path / "pose_split"
    result = convert_capture_roots_with_split(
        [source_a, source_b],
        output_root,
        train_repo_id="local/pose-train-test",
        test_repo_id="local/pose-test-test",
        episode_index_divisor=2,
        test_fraction=0.2,
        split_seed=42,
        encoder_threads=1,
    )

    assert result.train.total_output_episodes == 2
    assert result.test.total_output_episodes == 2
    manifest = json.loads((output_root / "split_manifest.json").read_text())
    assert manifest["episode_index_divisor"] == 2
    assert manifest["split_seed"] == 42
    assert set(manifest["splits"]) == {"train", "test"}
    assert (output_root / "train/meta/info.json").is_file()
    assert (output_root / "test/meta/info.json").is_file()


def test_convertible_filter_removes_incomplete_episode_before_split(tmp_path: Path) -> None:
    source = tmp_path / "differentplace"
    _make_pose_episode(source / "episode0")
    _make_pose_episode(source / "episode2")
    (source / "episode4/arm/endPose/puppetRight").mkdir(parents=True)

    usable, rejected = filter_convertible_episode_dirs(
        discover_episode_dirs(source, episode_index_divisor=2),
        max_alignment_delta_sec=0.01,
    )

    assert [episode.name for episode in usable] == ["episode0", "episode2"]
    assert list(rejected) == ["episode4"]
    assert "No timestamped .json" in rejected["episode4"]


def test_module_entrypoint_runs_after_pose_helpers_are_defined(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _require_h264()
    source = tmp_path / "normal"
    for index in (0, 2, 4):
        _make_pose_episode(source / f"episode{index}")
    output_root = tmp_path / "entrypoint-output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert-pose",
            "--input-root",
            str(source),
            "--output-root",
            str(output_root),
            "--train-repo-id",
            "local/entrypoint-train",
            "--test-repo-id",
            "local/entrypoint-test",
            "--encoder-threads",
            "1",
        ],
    )

    runpy.run_module("lerobot.scripts.convert_right_end_effector_pose_to_lerobot_v30", run_name="__main__")

    assert (output_root / "train/meta/info.json").is_file()
