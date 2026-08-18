import importlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _require_h264():
    pytest.importorskip("av", reason="av is required for H.264 conversion tests")
    from lerobot.datasets.pyav_utils import get_codec

    if get_codec("h264") is None:
        pytest.skip("H.264 encoder is unavailable")


def _write_joint(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"position": values}), encoding="utf-8")


def _write_image(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(1, 2, 3)).save(path)


def test_converts_three_cameras_and_left_then_right_joint_state(tmp_path: Path) -> None:
    _require_h264()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    module = importlib.import_module("lerobot.scripts.convert_dual_arm_three_camera_to_lerobot_v30")
    source = tmp_path / "raw" / "episode0"
    timestamps = (1.000, 1.033)
    cameras = {
        "stereoRight": (48, 33),
        "pikaGripperDepthCamera_r": (32, 32),
        "pikaGripperDepthCamera_l": (32, 32),
    }
    for index, timestamp in enumerate(timestamps):
        _write_joint(source / "arm/jointState/puppetLeft" / f"{timestamp:.3f}.json", [index] * 7)
        _write_joint(source / "arm/jointState/puppetRight" / f"{timestamp:.3f}.json", [10 + index] * 7)
        for camera, size in cameras.items():
            _write_image(source / "camera/color" / camera / f"{timestamp:.3f}.jpg", size)

    output = tmp_path / "converted"
    report = module.convert_capture_root(
        tmp_path / "raw", output, repo_id="local/dual-arm-three-camera-test", encoder_threads=1
    )

    dataset = LeRobotDataset("local/dual-arm-three-camera-test", root=output)
    assert report.total_output_episodes == 1
    assert len(dataset) == 2
    assert dataset.meta.features["observation.images.top"]["shape"] == (33, 48, 3)
    assert dataset.meta.features["observation.images.top"]["info"]["video.pix_fmt"] == "yuv444p"
    assert dataset.meta.features["observation.images.gripper_right"]["shape"] == (32, 32, 3)
    assert dataset.meta.features["observation.images.gripper_left"]["shape"] == (32, 32, 3)
    np.testing.assert_array_equal(dataset[0]["observation.state"].numpy(), np.array([0] * 7 + [10] * 7))
    np.testing.assert_array_equal(dataset[0]["action"].numpy(), np.array([1] * 7 + [11] * 7))
    np.testing.assert_array_equal(dataset[1]["action"].numpy(), np.array([1] * 7 + [11] * 7))


def test_resamples_stereo_anchors_to_fixed_rate(tmp_path: Path) -> None:
    module = importlib.import_module("lerobot.scripts.convert_dual_arm_three_camera_to_lerobot_v30")
    source = tmp_path / "raw" / "episode0"
    timestamps = (1.000, 1.020, 1.040, 1.060, 1.080)
    for timestamp in timestamps:
        _write_joint(source / "arm/jointState/puppetLeft" / f"{timestamp:.3f}.json", [0] * 7)
        _write_joint(source / "arm/jointState/puppetRight" / f"{timestamp:.3f}.json", [0] * 7)
        for camera in ("stereoRight", "pikaGripperDepthCamera_r", "pikaGripperDepthCamera_l"):
            _write_image(source / "camera/color" / camera / f"{timestamp:.3f}.jpg", (32, 32))

    frames = module.align_episode(source, max_alignment_delta_sec=0.01, resample_to_fps=25)

    assert [float(frame.stereo_path.stem) for frame in frames] == [1.000, 1.040, 1.080]
