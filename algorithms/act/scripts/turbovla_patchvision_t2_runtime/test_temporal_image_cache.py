from __future__ import annotations

import numpy as np
import pytest

from pi052_reference_client import ActualStateSample, DualActHardware, TopStereoRightRaw
from temporal_image_cache import CameraProducer, TemporalImageCache


CAMERAS = ("top", "gripper_left", "gripper_right")


def image(marker: int) -> np.ndarray:
    return np.full((2, 3, 3), marker, dtype=np.uint8)


def add_triplet(cache: TemporalImageCache, timestamp: float, marker: int, skew_s: float = 0.002) -> None:
    for index, camera in enumerate(CAMERAS):
        cache.add_frame(camera, image(marker + index), timestamp + index * skew_s / 2.0)


def test_latest_pair_uses_adjacent_synchronized_samples() -> None:
    cache = TemporalImageCache(
        max_camera_skew_s=0.05,
        min_pair_interval_s=0.015,
        max_pair_interval_s=0.060,
    )
    add_triplet(cache, 1.000, marker=10)
    add_triplet(cache, 1.033, marker=20)

    pair = cache.latest_pair(timeout_s=0.01)

    assert (pair.previous.sequence, pair.current.sequence) == (0, 1)
    assert pair.interval_s == pytest.approx(0.033)
    assert [frame.camera for frame in pair.current.frames] == list(CAMERAS)
    assert [int(frame.image[0, 0, 0]) for frame in pair.current.frames] == [20, 21, 22]


def test_excessive_camera_skew_does_not_emit_sample() -> None:
    cache = TemporalImageCache(max_camera_skew_s=0.01)
    cache.add_frame("top", image(1), 1.000)
    cache.add_frame("gripper_left", image(2), 1.005)
    cache.add_frame("gripper_right", image(3), 1.030)

    with pytest.raises(TimeoutError, match="synchronized temporal image pair"):
        cache.latest_pair(timeout_s=0.01)


@pytest.mark.parametrize("second_timestamp", [1.010, 1.080])
def test_invalid_pair_interval_is_rejected(second_timestamp: float) -> None:
    cache = TemporalImageCache(
        max_camera_skew_s=0.01,
        min_pair_interval_s=0.015,
        max_pair_interval_s=0.060,
    )
    add_triplet(cache, 1.000, marker=1, skew_s=0.001)
    add_triplet(cache, second_timestamp, marker=4, skew_s=0.001)

    with pytest.raises(TimeoutError, match="synchronized temporal image pair"):
        cache.latest_pair(timeout_s=0.01)


def test_camera_frame_is_not_reused_for_next_sample() -> None:
    cache = TemporalImageCache(max_camera_skew_s=0.01)
    add_triplet(cache, 1.000, marker=1, skew_s=0.001)
    cache.add_frame("top", image(4), 1.033)
    cache.add_frame("gripper_left", image(5), 1.034)

    assert cache.sample_count == 1
    with pytest.raises(TimeoutError, match="synchronized temporal image pair"):
        cache.latest_pair(timeout_s=0.01)


def test_rejects_unknown_camera_and_non_uint8_image() -> None:
    cache = TemporalImageCache()
    with pytest.raises(ValueError, match="unknown camera"):
        cache.add_frame("middle", image(1), 1.0)
    with pytest.raises(ValueError, match="uint8 HWC RGB"):
        cache.add_frame("top", image(1).astype(np.float32), 1.0)


class SequenceReader:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = iter(frames)

    def __call__(self) -> np.ndarray:
        return next(self._frames)


class SequenceClock:
    def __init__(self, timestamps: list[float]) -> None:
        self._timestamps = iter(timestamps)

    def __call__(self) -> float:
        return next(self._timestamps)


def test_camera_producer_adds_each_returned_frame_once() -> None:
    cache = TemporalImageCache()
    producer = CameraProducer(
        "top",
        SequenceReader([image(1), image(2)]),
        cache,
        clock=SequenceClock([1.000, 1.033]),
    )

    producer.start()
    producer.join(timeout_s=1.0)

    assert not producer.is_alive
    assert cache.camera_sequence("top") == 1


def test_camera_producer_preserves_reader_capture_timestamp() -> None:
    cache = TemporalImageCache(max_camera_skew_s=0.01)
    producers = [
        CameraProducer(camera, SequenceReader([(1.0, image(index))]), cache)
        for index, camera in enumerate(CAMERAS)
    ]
    for producer in producers:
        producer.start()
    for producer in producers:
        producer.join(timeout_s=1.0)

    assert cache.sample_count == 1


def test_camera_producer_error_reaches_pair_waiter() -> None:
    cache = TemporalImageCache()

    def broken_reader() -> np.ndarray:
        raise RuntimeError("camera disconnected")

    producer = CameraProducer("top", broken_reader, cache)
    producer.start()
    producer.join(timeout_s=1.0)

    with pytest.raises(RuntimeError, match="top camera producer failed"):
        cache.latest_pair(timeout_s=0.1)


def test_top_camera_waits_for_actual_new_frame_sequence() -> None:
    camera = TopStereoRightRaw("/dev/null")
    top = np.zeros((405, 720, 3), dtype=np.uint8)
    camera._publish_right_frame(top, timestamp=1.000)

    sequence0, timestamp0, frame0 = camera.read_right_after(-1, timeout_s=0.01)
    camera._publish_right_frame(top + 1, timestamp=1.033)
    sequence1, timestamp1, frame1 = camera.read_right_after(sequence0, timeout_s=0.01)

    assert (sequence0, timestamp0, int(frame0[0, 0, 0])) == (0, 1.000, 0)
    assert (sequence1, timestamp1, int(frame1[0, 0, 0])) == (1, 1.033, 1)


class FakeStateSampler:
    def snapshot(self):
        return (
            ActualStateSample(10, 1.0, np.arange(20, dtype=np.float64)),
            ActualStateSample(11, 1.033, np.arange(20, dtype=np.float64) + 1),
        )


def test_state_only_observation_does_not_read_cameras() -> None:
    hardware = DualActHardware.__new__(DualActHardware)
    hardware._connected = True
    hardware.left_arm = object()
    hardware.right_arm = object()
    hardware.state_sampler = FakeStateSampler()

    observation = hardware.get_state_observation()

    assert "top" not in observation
    assert "gripper_left" not in observation
    assert "gripper_right" not in observation
    assert observation["__relative_joint_previous_sequence"] == 10
    assert observation["__relative_joint_current_sequence"] == 11
