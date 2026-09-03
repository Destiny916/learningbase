"""Pure PC1 contracts for the isolated ACT-DINOv3 160000 client."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import os


class Client160000Error(RuntimeError):
    """Raised when PC1 cannot construct a safe protocol-v2 request."""


ACTION_HORIZON = int(os.environ.get("XWIZ_ACTION_HORIZON", "16"))


def latest_fresh_image(
    buffer: Any,
    name: str,
    *,
    now: float,
    max_age_seconds: float = 1.0,
) -> np.ndarray:
    if not buffer:
        raise Client160000Error(f"missing {name} image")
    received_at, image = buffer[-1]
    age = float(now) - float(received_at)
    if not np.isfinite(age) or age < 0.0:
        raise Client160000Error(f"{name} image has invalid age: {age}")
    if age > float(max_age_seconds):
        raise Client160000Error(
            f"{name} image is stale: age={age:.3f}s "
            f"limit={float(max_age_seconds):.3f}s"
        )
    return np.asarray(image)


def assemble_absolute_state(
    robot_positions: Any,
    left_gripper: float,
    right_gripper: float,
) -> np.ndarray:
    positions = np.asarray(robot_positions, dtype=np.float32).reshape(-1)
    grippers = np.asarray((left_gripper, right_gripper), dtype=np.float32)
    if positions.shape != (20,):
        raise Client160000Error(
            f"robot feedback must have shape (20,), got {positions.shape}"
        )
    if not np.isfinite(positions).all() or not np.isfinite(grippers).all():
        raise Client160000Error("robot and gripper feedback must be finite")
    return np.concatenate(
        (
            positions[3:4],
            positions[6:13],
            positions[4:6],
            positions[13:20],
            grippers,
        )
    ).astype(np.float32, copy=False)


def hand_openness_from_feedback(
    positions: Any,
    closed: Any,
    opened: Any,
) -> float:
    values = np.asarray(positions, dtype=np.float64)
    closed_values = np.asarray(closed, dtype=np.float64)
    opened_values = np.asarray(opened, dtype=np.float64)
    if values.shape != (6,) or closed_values.shape != (6,) or opened_values.shape != (6,):
        raise Client160000Error("hand feedback and endpoints must each have six values")
    if not np.isfinite(values).all():
        raise Client160000Error("hand feedback must be finite")
    if np.any(values < 0.0) or np.any(values > 100.0):
        raise Client160000Error("hand feedback must use the PC1 percentage scale 0..100")
    direction = opened_values - closed_values
    denominator = float(np.dot(direction, direction))
    if denominator <= 0.0:
        raise Client160000Error("hand endpoints must differ")
    fraction = np.dot(values - closed_values, direction) / denominator
    return float(np.clip(fraction, 0.0, 1.0) * 100.0)


def hand_command_from_openness(
    scalar: float,
    closed: Any,
    opened: Any,
) -> tuple[float, ...]:
    value = float(scalar)
    if not np.isfinite(value):
        raise Client160000Error("hand openness must be finite")
    closed_values = np.asarray(closed, dtype=np.float64)
    opened_values = np.asarray(opened, dtype=np.float64)
    if closed_values.shape != (6,) or opened_values.shape != (6,):
        raise Client160000Error("hand endpoints must each have six values")
    fraction = float(np.clip(value, 0.0, 100.0)) / 100.0
    command = closed_values * (1.0 - fraction) + opened_values * fraction
    return tuple(float(item) for item in command)


class AdjacentFeedbackBuffer:
    """Keep the latest two consecutive snapshots from the real feedback callback."""

    def __init__(self) -> None:
        self._frames: deque[dict[str, Any]] = deque(maxlen=2)
        self._next_sequence = 0
        self._last_taken_sequence = -1

    @property
    def newest_sequence(self) -> int:
        return int(self._frames[-1]["sequence"]) if self._frames else -1

    def append(self, state: Any, timestamp: float, received_at: float | None = None) -> None:
        values = np.asarray(state, dtype=np.float32).reshape(-1).copy()
        source_timestamp = float(timestamp)
        receive_time = source_timestamp if received_at is None else float(received_at)
        if values.shape != (19,):
            raise Client160000Error(
                f"real feedback state must have shape (19,), got {values.shape}"
            )
        if not np.isfinite(values).all() or not np.isfinite(source_timestamp) or not np.isfinite(receive_time):
            raise Client160000Error("real feedback state and timestamp must be finite")
        if self._frames and source_timestamp <= float(self._frames[-1]["timestamp"]):
            raise Client160000Error("real feedback timestamp must be strictly increasing")
        self._frames.append(
            {
                "state": values,
                "sequence": self._next_sequence,
                "timestamp": source_timestamp,
                "received_at": receive_time,
            }
        )
        self._next_sequence += 1

    def take_new_pair(
        self,
        *,
        now: float | None = None,
        max_age_seconds: float = 1.0,
    ) -> dict[str, Any]:
        if len(self._frames) != 2:
            raise Client160000Error("two real feedback frames are required")
        previous, current = self._frames
        if int(current["sequence"]) != int(previous["sequence"]) + 1:
            raise Client160000Error("real feedback pair is not consecutive")
        if int(current["sequence"]) <= self._last_taken_sequence:
            raise Client160000Error("no new adjacent real feedback pair is available")
        current_time = float(current["received_at"] if now is None else now)
        ages = (
            current_time - float(previous["received_at"]),
            current_time - float(current["received_at"]),
        )
        if any(age < 0.0 or age > float(max_age_seconds) for age in ages):
            raise Client160000Error(
                f"real feedback pair is stale: ages={ages} limit={float(max_age_seconds)}"
            )
        self._last_taken_sequence = int(current["sequence"])
        return {
            "protocol_version": 2,
            "previous_feedback": {
                "state": previous["state"].copy(),
                "sequence": int(previous["sequence"]),
                "timestamp": float(previous["timestamp"]),
                "age_seconds": float(ages[0]),
            },
            "current_feedback": {
                "state": current["state"].copy(),
                "sequence": int(current["sequence"]),
                "timestamp": float(current["timestamp"]),
                "age_seconds": float(ages[1]),
            },
        }


@dataclass(frozen=True)
class ChunkProgress:
    global_frame: int
    frame_in_chunk: int
    chunk_complete: bool


class Chunk16Gate:
    def __init__(self) -> None:
        self.published = 0
        self.last_requested_feedback_sequence = -1
        self.chunk_completed_feedback_sequence = -1

    def mark_published(self) -> ChunkProgress:
        self.published += 1
        frame = (self.published - 1) % ACTION_HORIZON + 1
        return ChunkProgress(self.published, frame, frame == ACTION_HORIZON)

    def mark_requested(self, current_feedback_sequence: int) -> None:
        self.last_requested_feedback_sequence = int(current_feedback_sequence)

    def mark_chunk_completed(self, newest_feedback_sequence: int) -> None:
        if self.published == 0 or self.published % ACTION_HORIZON != 0:
            raise Client160000Error(
                f"chunk completion must occur after exactly {ACTION_HORIZON} frames"
            )
        self.chunk_completed_feedback_sequence = int(newest_feedback_sequence)

    def can_request(self, *, queue_size: int, newest_feedback_sequence: int) -> bool:
        if queue_size != 0:
            return False
        if self.published == 0:
            return newest_feedback_sequence >= 1
        return (
            self.published % ACTION_HORIZON == 0
            and newest_feedback_sequence > self.last_requested_feedback_sequence
            and newest_feedback_sequence > self.chunk_completed_feedback_sequence
        )


def _validate_rgb(image: Any, shape: tuple[int, int, int], name: str) -> np.ndarray:
    array = np.asarray(image)
    if array.shape != shape or array.dtype != np.uint8:
        raise Client160000Error(
            f"{name} image must have shape {shape} and dtype uint8, got {array.shape} {array.dtype}"
        )
    return array


def preprocess_images(
    head: Any,
    physical_left: Any,
    physical_right: Any,
    *,
    resize: Callable[[np.ndarray, tuple[int, int]], np.ndarray],
) -> dict[str, np.ndarray]:
    head_rgb = _validate_rgb(head, (540, 960, 3), "head")
    left_rgb = _validate_rgb(physical_left, (360, 640, 3), "physical left")
    right_rgb = _validate_rgb(physical_right, (480, 640, 3), "physical right")

    head_square = np.zeros((960, 960, 3), dtype=np.uint8)
    head_square[210:750] = head_rgb
    left_square = resize(left_rgb, (360, 360))
    right_square = resize(right_rgb, (480, 480))
    converted = {
        "cam_high_right": resize(head_square, (224, 224)),
        "cam_left_wrist": resize(left_square, (224, 224)),
        "cam_right_wrist": resize(right_square, (224, 224)),
    }
    for key, image in converted.items():
        if np.asarray(image).shape != (224, 224, 3):
            raise Client160000Error(f"{key} conversion did not produce 224x224 RGB")
    return converted
