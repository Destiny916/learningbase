"""Thread-safe temporal synchronization for PatchVision T2 camera inputs."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np


CAMERAS = ("top", "gripper_left", "gripper_right")


@dataclass(frozen=True)
class TimestampedFrame:
    camera: str
    sequence: int
    timestamp: float
    image: np.ndarray


@dataclass(frozen=True)
class TemporalViewSample:
    sequence: int
    timestamp: float
    frames: tuple[TimestampedFrame, TimestampedFrame, TimestampedFrame]
    max_camera_skew_s: float


@dataclass(frozen=True)
class TemporalPair:
    previous: TemporalViewSample
    current: TemporalViewSample
    interval_s: float


class TemporalImageCache:
    def __init__(
        self,
        *,
        max_camera_skew_s: float = 0.05,
        min_pair_interval_s: float = 0.015,
        max_pair_interval_s: float = 0.060,
        sample_capacity: int = 16,
    ) -> None:
        if not 0 <= max_camera_skew_s:
            raise ValueError("max_camera_skew_s must be non-negative")
        if not 0 <= min_pair_interval_s <= max_pair_interval_s:
            raise ValueError("pair interval limits are invalid")
        if sample_capacity < 2:
            raise ValueError("sample_capacity must be at least 2")
        self.max_camera_skew_s = float(max_camera_skew_s)
        self.min_pair_interval_s = float(min_pair_interval_s)
        self.max_pair_interval_s = float(max_pair_interval_s)
        self._condition = threading.Condition()
        self._latest: dict[str, TimestampedFrame] = {}
        self._camera_sequences = {camera: -1 for camera in CAMERAS}
        self._last_emitted_sequences = {camera: -1 for camera in CAMERAS}
        self._samples: deque[TemporalViewSample] = deque(maxlen=sample_capacity)
        self._sample_sequence = 0
        self._error: tuple[str, BaseException] | None = None

    @property
    def sample_count(self) -> int:
        with self._condition:
            return len(self._samples)

    def camera_sequence(self, camera: str) -> int:
        with self._condition:
            self._validate_camera(camera)
            return self._camera_sequences[camera]

    @staticmethod
    def _validate_camera(camera: str) -> None:
        if camera not in CAMERAS:
            raise ValueError(f"unknown camera {camera!r}; expected one of {CAMERAS}")

    def add_frame(self, camera: str, image: np.ndarray, timestamp: float) -> None:
        self._validate_camera(camera)
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"{camera} frame must be uint8 HWC RGB, got {image.shape} {image.dtype}")
        timestamp = float(timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("frame timestamp must be finite")
        with self._condition:
            prior = self._latest.get(camera)
            if prior is not None and timestamp <= prior.timestamp:
                raise ValueError(f"{camera} timestamps must increase monotonically")
            sequence = self._camera_sequences[camera] + 1
            self._camera_sequences[camera] = sequence
            self._latest[camera] = TimestampedFrame(
                camera=camera,
                sequence=sequence,
                timestamp=timestamp,
                image=image.copy(),
            )
            self._try_emit_sample()
            self._condition.notify_all()

    def report_error(self, camera: str, error: BaseException) -> None:
        self._validate_camera(camera)
        with self._condition:
            if self._error is None:
                self._error = (camera, error)
            self._condition.notify_all()

    def _try_emit_sample(self) -> None:
        if any(camera not in self._latest for camera in CAMERAS):
            return
        frames = tuple(self._latest[camera] for camera in CAMERAS)
        if any(
            frame.sequence <= self._last_emitted_sequences[frame.camera]
            for frame in frames
        ):
            return
        timestamps = [frame.timestamp for frame in frames]
        skew = max(timestamps) - min(timestamps)
        if skew > self.max_camera_skew_s:
            return
        sample = TemporalViewSample(
            sequence=self._sample_sequence,
            timestamp=max(timestamps),
            frames=frames,
            max_camera_skew_s=skew,
        )
        self._samples.append(sample)
        self._sample_sequence += 1
        for frame in frames:
            self._last_emitted_sequences[frame.camera] = frame.sequence

    def latest_pair(self, timeout_s: float = 1.0) -> TemporalPair:
        deadline = time.monotonic() + float(timeout_s)
        with self._condition:
            while True:
                if self._error is not None:
                    camera, error = self._error
                    raise RuntimeError(f"{camera} camera producer failed") from error
                if len(self._samples) >= 2:
                    previous, current = self._samples[-2], self._samples[-1]
                    interval = current.timestamp - previous.timestamp
                    if (
                        current.sequence == previous.sequence + 1
                        and self.min_pair_interval_s <= interval <= self.max_pair_interval_s
                    ):
                        return TemporalPair(previous=previous, current=current, interval_s=interval)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for a synchronized temporal image pair")
                self._condition.wait(remaining)


class CameraProducer:
    def __init__(
        self,
        camera: str,
        reader: Callable[[], np.ndarray | tuple[float, np.ndarray]],
        cache: TemporalImageCache,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        cache._validate_camera(camera)
        self.camera = camera
        self.reader = reader
        self.cache = cache
        self.clock = clock
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_alive:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"patchvision-{self.camera}",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                captured = self.reader()
                if isinstance(captured, tuple):
                    timestamp, image = captured
                else:
                    timestamp, image = self.clock(), captured
                self.cache.add_frame(self.camera, image, timestamp)
        except StopIteration:
            return
        except BaseException as error:
            if not self._stop.is_set():
                self.cache.report_error(self.camera, error)

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout_s: float = 2.0) -> None:
        if self._thread is not None:
            self._thread.join(timeout_s)
