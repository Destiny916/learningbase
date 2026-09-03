"""Adapters between the vendor W1 wire format and the LeRobot ACT contract."""

from __future__ import annotations

from collections.abc import Mapping
import os
from typing import Any

import numpy as np


IMAGE_WIDTH = 640
IMAGE_HEIGHT = 360
ACTION_SHAPE = (int(os.environ.get("XWIZ_ACTION_HORIZON", "16")), 19)

STATE_GROUPS = (
    ("waistqpos", 1),
    ("left_armqpos", 7),
    ("headqpos", 2),
    ("right_armqpos", 7),
    ("left_eefgripper", 1),
    ("right_eefgripper", 1),
)

IMAGE_MAPPING = {
    "cam_left_wrist": "observation.images.cam_hand_left",
    "cam_right_wrist": "observation.images.cam_hand_right",
}


class ContractError(ValueError):
    """Raised when a request violates the trained model contract."""


def assemble_state(states: Mapping[str, Any]) -> np.ndarray:
    values = []
    for name, width in STATE_GROUPS:
        if name not in states:
            raise ContractError(f"missing state group: {name}")
        array = np.asarray(states[name], dtype=np.float32).reshape(-1)
        if array.shape != (width,):
            raise ContractError(
                f"state group {name} must have shape ({width},), got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ContractError(f"state group {name} must contain finite values")
        values.append(array)
    state = np.concatenate(values).astype(np.float32, copy=False)
    if state.shape != (19,):
        raise ContractError(f"assembled state must have shape (19,), got {state.shape}")
    return state


def decode_bgr_image(data: bytes, target_size: tuple[int, int], key: str) -> np.ndarray:
    size = tuple(int(value) for value in target_size)
    if size[0] < 1 or size[1] < 1:
        raise ContractError(
            f"{key} target size must be {(IMAGE_WIDTH, IMAGE_HEIGHT)}, got {size}"
        )
    expected_bytes = size[0] * size[1] * 3
    if not isinstance(data, (bytes, bytearray, memoryview)) or len(data) != expected_bytes:
        actual = len(data) if hasattr(data, "__len__") else None
        raise ContractError(
            f"{key} must contain {expected_bytes} BGR bytes for {size}, got {actual}"
        )
    bgr = np.frombuffer(data, dtype=np.uint8).reshape(size[1], size[0], 3)
    return bgr[..., ::-1].copy()


def decode_observation(request: Mapping[str, Any]) -> dict[str, np.ndarray]:
    states = request.get("states")
    if not isinstance(states, Mapping):
        raise ContractError("states must be a mapping")
    observation = {"observation.state": assemble_state(states)}
    # outputs/500000 was trained from the physical right head camera.  The
    # client sends both head views; retain the old cam_high fallback for peers
    # that only send one view.
    head_key = "cam_high_r" if "cam_high_r" in request else "cam_high"
    head_training_key = "observation.images.cam_high_right" if head_key == "cam_high_r" else "observation.images.cam_high_left"
    if head_key not in request:
        raise ContractError(f"missing image: {head_key}")
    observation[head_training_key] = decode_bgr_image(
        request[head_key], request.get("head_target_size", (IMAGE_WIDTH, IMAGE_HEIGHT)), head_key
    )
    for legacy_key, training_key in IMAGE_MAPPING.items():
        if legacy_key not in request:
            raise ContractError(f"missing image: {legacy_key}")
        size_key = "hand_left_target_size" if legacy_key == "cam_left_wrist" else "hand_right_target_size"
        size = request.get(size_key, request.get("hand_target_size", (IMAGE_WIDTH, IMAGE_HEIGHT)))
        observation[training_key] = decode_bgr_image(request[legacy_key], size, legacy_key)
    return observation


def group_action_chunk(actions: Any) -> dict[str, np.ndarray]:
    array = np.asarray(actions, dtype=np.float32)
    if array.shape != ACTION_SHAPE:
        raise ContractError(f"action chunk must have shape {ACTION_SHAPE}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ContractError("action chunk must contain finite values")
    grouped = {}
    start = 0
    for name, width in STATE_GROUPS:
        grouped[name] = array[:, start : start + width].copy()
        start += width
    return grouped
