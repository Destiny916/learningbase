import importlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def test_trim_window_is_inclusive_and_regenerates_terminal_action() -> None:
    module = importlib.import_module("lerobot.scripts.convert_dual_arm_stereo_rgbd_to_lerobot_v30")
    states = np.arange(6 * 14, dtype=np.float32).reshape(6, 14)

    trimmed_states, trimmed_actions = module.trim_states_and_actions(states, start=2, stop=4)

    np.testing.assert_array_equal(trimmed_states, states[2:5])
    np.testing.assert_array_equal(trimmed_actions[:-1], states[3:5])
    np.testing.assert_array_equal(trimmed_actions[-1], states[4])


def test_depth_clipping_uses_camera_specific_physical_limits() -> None:
    module = importlib.import_module("lerobot.scripts.convert_dual_arm_stereo_rgbd_to_lerobot_v30")
    depth_m = np.array([[0.01, 0.07, 0.60, 0.90, 1.20]], dtype=np.float32)

    right = module.clip_depth_m(depth_m, camera="right")
    left = module.clip_depth_m(depth_m, camera="left")

    np.testing.assert_allclose(right, [[0.07, 0.07, 0.60, 0.60, 0.60]])
    np.testing.assert_allclose(left, [[0.07, 0.07, 0.60, 0.90, 0.90]])


def test_depth_encoding_stays_lossless_with_one_x265_worker_pool() -> None:
    module = importlib.import_module("lerobot.scripts.convert_dual_arm_stereo_rgbd_to_lerobot_v30")

    assert module.DEPTH_ENCODER_EXTRA_OPTIONS == {"x265-params": "lossless=1:pools=1"}


def test_all_configured_episode_windows_total_5547_frames() -> None:
    module = importlib.import_module("lerobot.scripts.convert_dual_arm_stereo_rgbd_to_lerobot_v30")

    assert len(module.EPISODE_WINDOWS) == 39
    assert sum(stop - start + 1 for start, stop in module.EPISODE_WINDOWS.values()) == 5547


def test_sparse_raw_episode_numbers_preserve_numeric_order_for_logical_windows(tmp_path: Path) -> None:
    module = importlib.import_module("lerobot.scripts.convert_dual_arm_stereo_rgbd_to_lerobot_v30")
    episodes = []
    for raw_index in (0, 1, 3, 5):
        episode = tmp_path / f"episode{raw_index}"
        episode.mkdir()
        episodes.append(episode)

    planned = module.ordered_episode_windows(episodes, {0: (10, 11), 1: (20, 21), 2: (30, 31), 3: (40, 41)})

    assert [(index, episode.name, window) for index, episode, window in planned] == [
        (0, "episode0", (10, 11)),
        (1, "episode1", (20, 21)),
        (2, "episode3", (30, 31)),
        (3, "episode5", (40, 41)),
    ]


def test_select_window_keeps_raw_stereo_indexes_before_async_modality_validation() -> None:
    module = importlib.import_module("lerobot.scripts.convert_dual_arm_stereo_rgbd_to_lerobot_v30")
    frame = lambda index, delta: module.AlignedFrame(  # noqa: E731
        source_index=index,
        top_left_path=Path(f"left-{index}"),
        top_right_path=Path(f"right-{index}"),
        gripper_left_rgb_path=Path(f"left-rgb-{index}"),
        gripper_right_rgb_path=Path(f"right-rgb-{index}"),
        gripper_left_depth_path=Path(f"left-depth-{index}"),
        gripper_right_depth_path=Path(f"right-depth-{index}"),
        left_joint_path=Path(f"left-joint-{index}"),
        right_joint_path=Path(f"right-joint-{index}"),
        max_alignment_delta_sec=delta,
    )
    aligned = [frame(0, 0.30), frame(1, 0.02), frame(2, 0.04)]

    selected = module.select_window(aligned, start=1, stop=2, max_alignment_delta_sec=0.05)

    assert [item.source_index for item in selected] == [1, 2]


def test_converter_writes_and_recovers_rgbd_episode(tmp_path: Path) -> None:
    module = importlib.import_module("lerobot.scripts.convert_dual_arm_stereo_rgbd_to_lerobot_v30")
    raw_episode = tmp_path / "raw" / "episode0"
    rgb_paths = (
        module.TOP_LEFT_PATH,
        module.TOP_RIGHT_PATH,
        module.GRIPPER_LEFT_RGB_PATH,
        module.GRIPPER_RIGHT_RGB_PATH,
    )
    depth_paths = (module.GRIPPER_LEFT_DEPTH_PATH, module.GRIPPER_RIGHT_DEPTH_PATH)
    for frame_index in range(3):
        timestamp = f"{1000.0 + frame_index * 0.04:.6f}"
        for relative in rgb_paths:
            destination = raw_episode / relative / f"{timestamp}.jpg"
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.full((64, 64, 3), frame_index * 50, dtype=np.uint8)).save(destination)
        for relative in depth_paths:
            destination = raw_episode / relative / f"{timestamp}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.full((64, 64), 10 + frame_index * 1000, dtype=np.uint16)).save(destination)
        for relative, offset in ((module.LEFT_JOINT_PATH, 0.0), (module.RIGHT_JOINT_PATH, 10.0)):
            destination = raw_episode / relative / f"{timestamp}.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps({"position": [offset + frame_index] * 7}), encoding="utf-8")

    output_root = tmp_path / "converted"
    report = module.convert_capture_root(
        raw_episode.parent,
        output_root,
        repo_id="local/test_stereo_rgbd",
        encoder_threads=1,
        episode_windows={0: (0, 2)},
    )

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset("local/test_stereo_rgbd", root=output_root, depth_output_unit="m")
    sample = dataset[0]
    assert report.total_output_episodes == 1
    assert report.total_kept_frames == 3
    assert set(dataset.meta.depth_keys) == {
        "observation.images.gripper_left_depth",
        "observation.images.gripper_right_depth",
    }
    np.testing.assert_allclose(
        sample["action"].numpy(), np.concatenate((np.full(7, 1.0), np.full(7, 11.0))).astype(np.float32)
    )
    np.testing.assert_allclose(sample["observation.images.gripper_left_depth"].numpy(), 0.07)
    np.testing.assert_allclose(sample["observation.images.gripper_right_depth"].numpy(), 0.07)
