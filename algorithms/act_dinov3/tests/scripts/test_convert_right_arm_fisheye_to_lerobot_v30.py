import importlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _require_h264():
    av = pytest.importorskip("av", reason="av is required for H.264 conversion tests")
    from lerobot.datasets.pyav_utils import get_codec

    if get_codec("h264") is None:
        pytest.skip("H.264 encoder is unavailable")
    return av


def _write_joint(path: Path, position: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"position": position}))


def _write_rgb(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=(1, 2, 3)).save(path)


def test_alignment_uses_nearest_joint_and_discards_large_residuals(tmp_path: Path) -> None:
    module = importlib.import_module("lerobot.scripts.convert_right_arm_fisheye_to_lerobot_v30")
    joint_dir = tmp_path / "arm/jointState/puppetRight"
    camera_dir = tmp_path / "camera/color/pikaGripperFisheyeCamera_r"
    _write_joint(joint_dir / "1.000.json", [1, 2, 3, 4, 5, 6, 7])
    _write_joint(joint_dir / "1.030.json", [8, 9, 10, 11, 12, 13, 14])
    _write_rgb(camera_dir / "1.002.jpg")
    _write_rgb(camera_dir / "1.010.jpg")
    _write_rgb(camera_dir / "1.047.jpg")

    aligned = module.align_right_arm_to_camera(
        joint_dir,
        camera_dir,
        max_alignment_delta_sec=0.01,
    )

    assert [item.camera_path.name for item in aligned] == ["1.002.jpg", "1.010.jpg"]
    assert aligned[0].joint_path.name == "1.000.json"
    assert aligned[0].alignment_delta_sec == pytest.approx(0.002)
    assert aligned[1].joint_path.name == "1.000.json"
    assert aligned[1].alignment_delta_sec == pytest.approx(0.01)


def test_next_actions_shift_and_clamp_the_terminal_state() -> None:
    module = importlib.import_module("lerobot.scripts.convert_right_arm_fisheye_to_lerobot_v30")
    states = np.asarray([[1] * 7, [2] * 7], dtype=np.float32)

    actions = module.build_next_actions(states)

    np.testing.assert_array_equal(actions, np.asarray([[2] * 7, [2] * 7], dtype=np.float32))


def _make_episode(root: Path, *, position_offset: int = 0) -> None:
    joint_dir = root / "arm/jointState/puppetRight"
    camera_dir = root / "camera/color/pikaGripperFisheyeCamera_r"
    for index, timestamp in enumerate((1.000, 1.033, 1.066)):
        _write_joint(
            joint_dir / f"{timestamp:.3f}.json",
            [position_offset + index * 10 + value for value in range(7)],
        )
        _write_rgb(camera_dir / f"{timestamp:.3f}.jpg")


def test_convert_root_writes_h264_and_next_joint_actions(tmp_path: Path) -> None:
    av = _require_h264()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    module = importlib.import_module("lerobot.scripts.convert_right_arm_fisheye_to_lerobot_v30")
    source_root = tmp_path / "raw"
    _make_episode(source_root / "episode0")
    output_root = tmp_path / "converted"

    report = module.convert_capture_roots(
        [source_root],
        output_root,
        repo_id="local/right-fisheye-test",
        fps=30,
        max_alignment_delta_sec=0.01,
        encoder_threads=1,
    )

    dataset = LeRobotDataset("local/right-fisheye-test", root=output_root)
    assert report.total_kept_frames == 3
    assert dataset.meta.total_episodes == 1
    assert len(dataset) == 3
    np.testing.assert_array_equal(dataset[0]["observation.state"].numpy(), np.arange(7, dtype=np.float32))
    np.testing.assert_array_equal(dataset[0]["action"].numpy(), np.arange(7, dtype=np.float32) + 10)
    np.testing.assert_array_equal(dataset[2]["action"].numpy(), np.arange(7, dtype=np.float32) + 20)

    video_path = next((output_root / "videos").rglob("*.mp4"))
    with av.open(str(video_path)) as container:
        assert container.streams.video[0].codec_context.name == "h264"
        assert sum(1 for _ in container.decode(video=0)) == 3


def test_convert_root_accepts_custom_joint_and_camera_modalities(tmp_path: Path) -> None:
    _require_h264()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    module = importlib.import_module("lerobot.scripts.convert_right_arm_fisheye_to_lerobot_v30")
    source_root = tmp_path / "raw"
    joint_path = Path("arm/jointState/puppet")
    camera_path = Path("camera/color/pikaGripperDepthCamera")
    for index, timestamp in enumerate((1.000, 1.033, 1.066)):
        _write_joint(source_root / "episode0" / joint_path / f"{timestamp:.3f}.json", [index] * 7)
        _write_rgb(source_root / "episode0" / camera_path / f"{timestamp:.3f}.jpg")

    output_root = tmp_path / "converted"
    module.convert_capture_roots(
        [source_root],
        output_root,
        repo_id="local/depth-camera-test",
        joint_relative_path=joint_path,
        camera_relative_path=camera_path,
        camera_key="observation.images.gripper_depth_color",
        encoder_threads=1,
    )

    dataset = LeRobotDataset("local/depth-camera-test", root=output_root)
    assert dataset.meta.features["observation.images.gripper_depth_color"]["dtype"] == "video"
    np.testing.assert_array_equal(dataset[0]["observation.state"].numpy(), np.zeros(7, dtype=np.float32))


def test_convert_root_skips_episodes_without_a_right_camera_frame(tmp_path: Path) -> None:
    _require_h264()
    module = importlib.import_module("lerobot.scripts.convert_right_arm_fisheye_to_lerobot_v30")
    source_root = tmp_path / "raw"
    _make_episode(source_root / "episode0")
    _write_joint(
        source_root / "episode1/arm/jointState/puppetRight/2.000.json",
        list(range(7)),
    )

    report = module.convert_capture_roots(
        [source_root],
        tmp_path / "converted",
        repo_id="local/right-fisheye-skip-test",
        encoder_threads=1,
    )

    assert report.total_source_episodes == 2
    assert report.total_output_episodes == 1
    assert len(report.skipped_episodes) == 1
    assert report.skipped_episodes[0].source_episode.endswith("episode1")
    assert report.skipped_episodes[0].reason == "missing right fisheye JPEG frames"


def test_convert_two_roots_keeps_root_then_episode_order(tmp_path: Path) -> None:
    _require_h264()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    module = importlib.import_module("lerobot.scripts.convert_right_arm_fisheye_to_lerobot_v30")
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    _make_episode(first_root / "episode0")
    _make_episode(second_root / "episode0", position_offset=100)

    output_root = tmp_path / "combined"
    report = module.convert_capture_roots(
        [first_root, second_root],
        output_root,
        repo_id="local/right-fisheye-two-roots-test",
        encoder_threads=1,
    )

    dataset = LeRobotDataset("local/right-fisheye-two-roots-test", root=output_root)
    assert [episode.output_episode_index for episode in report.episodes] == [0, 1]
    assert [Path(episode.source_root).name for episode in report.episodes] == ["first", "second"]
    assert dataset.meta.total_episodes == 2
    np.testing.assert_array_equal(dataset[3]["observation.state"].numpy(), np.arange(7, dtype=np.float32) + 100)


def test_convert_root_filters_episode_indices_by_divisor(tmp_path: Path) -> None:
    _require_h264()
    module = importlib.import_module("lerobot.scripts.convert_right_arm_fisheye_to_lerobot_v30")
    source_root = tmp_path / "raw"
    _make_episode(source_root / "episode0")
    _make_episode(source_root / "episode1", position_offset=100)
    _make_episode(source_root / "episode2", position_offset=200)
    _make_episode(source_root / "episode_invalid", position_offset=300)

    report = module.convert_capture_roots(
        [source_root],
        tmp_path / "converted",
        repo_id="local/right-fisheye-filter-test",
        episode_index_divisor=2,
        encoder_threads=1,
    )

    assert report.episode_index_divisor == 2
    assert report.total_source_episodes == 2
    assert report.total_output_episodes == 2
    assert [Path(episode.source_episode).name for episode in report.episodes] == ["episode0", "episode2"]


def test_convert_root_rejects_non_positive_episode_index_divisor(tmp_path: Path) -> None:
    module = importlib.import_module("lerobot.scripts.convert_right_arm_fisheye_to_lerobot_v30")

    with pytest.raises(ValueError, match="episode_index_divisor must be positive"):
        module.convert_capture_roots(
            [tmp_path / "raw"],
            tmp_path / "converted",
            repo_id="local/right-fisheye-filter-test",
            episode_index_divisor=0,
        )


@pytest.mark.parametrize("invalid_divisor", [True, 2.5, "2"])
def test_convert_root_rejects_non_integer_episode_index_divisor(
    tmp_path: Path, invalid_divisor: object
) -> None:
    module = importlib.import_module("lerobot.scripts.convert_right_arm_fisheye_to_lerobot_v30")

    with pytest.raises(ValueError, match="positive integer"):
        module.convert_capture_roots(
            [tmp_path / "raw"],
            tmp_path / "converted",
            repo_id="local/right-fisheye-filter-test",
            episode_index_divisor=invalid_divisor,
        )


@pytest.mark.parametrize("episode_name", ["episode1", "episode_invalid"])
def test_discover_direct_episode_root_applies_index_divisor(tmp_path: Path, episode_name: str) -> None:
    module = importlib.import_module("lerobot.scripts.convert_right_arm_fisheye_to_lerobot_v30")
    episode_root = tmp_path / episode_name
    _make_episode(episode_root)

    with pytest.raises(ValueError, match="does not match episode index divisor 2"):
        module.discover_episode_dirs(episode_root, episode_index_divisor=2)


def test_convert_multiple_roots_skips_collection_without_matching_indices(tmp_path: Path) -> None:
    _require_h264()
    module = importlib.import_module("lerobot.scripts.convert_right_arm_fisheye_to_lerobot_v30")
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    _make_episode(first_root / "episode1", position_offset=100)
    _make_episode(second_root / "episode2", position_offset=200)

    report = module.convert_capture_roots(
        [first_root, second_root],
        tmp_path / "converted",
        repo_id="local/right-fisheye-multi-root-filter-test",
        episode_index_divisor=2,
        encoder_threads=1,
    )

    assert report.total_source_episodes == 1
    assert report.total_output_episodes == 1
    assert [Path(episode.source_episode).name for episode in report.episodes] == ["episode2"]


def test_parse_args_accepts_episode_index_divisor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = importlib.import_module("lerobot.scripts.convert_right_arm_fisheye_to_lerobot_v30")
    monkeypatch.setattr(
        "sys.argv",
        [
            "convert-right-fisheye",
            "--input-root",
            str(tmp_path / "raw"),
            "--output-root",
            str(tmp_path / "converted"),
            "--repo-id",
            "local/right-fisheye-filter-test",
            "--episode-index-divisor",
            "2",
        ],
    )

    args = module._parse_args()

    assert args.episode_index_divisor == 2


def test_all_invalid_episodes_write_a_rejection_report(tmp_path: Path) -> None:
    module = importlib.import_module("lerobot.scripts.convert_right_arm_fisheye_to_lerobot_v30")
    source_root = tmp_path / "raw"
    _write_joint(
        source_root / "episode0/arm/jointState/puppetRight/1.000.json",
        list(range(7)),
    )
    output_root = tmp_path / "converted"

    with pytest.raises(ValueError, match="No episodes contain both"):
        module.convert_capture_roots(
            [source_root],
            output_root,
            repo_id="local/right-fisheye-all-invalid-test",
        )

    rejection = json.loads((tmp_path / "converted.rejected_episodes.json").read_text())
    assert rejection["total_source_episodes"] == 1
    assert rejection["skipped_episodes"][0]["reason"] == "missing right fisheye JPEG frames"
