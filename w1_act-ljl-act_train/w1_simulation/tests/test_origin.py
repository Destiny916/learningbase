from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from w1_simulation.replay.origin import InitialPoseLoader, OriginReplay
from w1_simulation.w1_profile import ACT_IMAGE_KEYS

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def replay(origin_root):
    return OriginReplay(origin_root, max_camera_skew_ms=50.0)


def test_origin_replay_exposes_all_1391_synchronized_triplets(replay) -> None:
    assert len(replay.frames) == 1391
    assert replay.dropped_for_skew == 0
    assert all(frame.camera_skew_ms <= 50.0 for frame in replay.frames)


def test_origin_replay_uses_head_left_timestamp_as_triplet_timestamp(replay) -> None:
    for frame in replay.frames:
        anchor = frame.records["observation.images.cam_high_left"]
        assert frame.timestamp == anchor.timestamp


def test_origin_frame_loads_three_rgb_uint8_images(replay) -> None:
    images, _, _ = replay.frames[0].load_images()

    assert tuple(images) == ACT_IMAGE_KEYS
    for image in images.values():
        assert image.shape == (360, 640, 3)
        assert image.dtype == np.uint8


def test_origin_frame_hashes_source_files_and_policy_inputs(replay) -> None:
    frame = replay.frames[0]
    images, source_hashes, input_hashes = frame.load_images()

    for key in ACT_IMAGE_KEYS:
        assert source_hashes[key] == hashlib.sha256(frame.records[key].path.read_bytes()).hexdigest()
        assert input_hashes[key] == hashlib.sha256(images[key].tobytes()).hexdigest()


def test_initial_pose_can_only_be_read_once(origin_root, replay) -> None:
    loader = InitialPoseLoader(origin_root)
    pose, match = loader.read_nearest(replay.frames[0].timestamp)

    assert pose
    assert match["delta_ms"] >= 0.0
    with pytest.raises(RuntimeError, match="only be read once"):
        loader.read_nearest(replay.frames[0].timestamp)


def test_origin_metadata_hash_matches_recording_file(origin_root, replay) -> None:
    expected = hashlib.sha256((origin_root / "metadata.jsonl").read_bytes()).hexdigest()

    assert replay.metadata_sha256 == expected


def test_pose_match_selects_a_recorded_frame(origin_root, replay) -> None:
    loader = InitialPoseLoader(origin_root)
    _, match = loader.read_nearest(replay.frames[0].timestamp)
    payload = json.loads(loader.path.read_text(encoding="utf-8"))

    assert match["frame_id"] in {int(frame["frame_id"]) for frame in payload["frames"]}
