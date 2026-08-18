"""Pure deployment contract for the 0806swap 20D-state TurboVLA policy."""

from __future__ import annotations

import numpy as np

STATE_DIM = 20
ACTION_DIM = 14
LEFT_JOINTS = slice(0, 6)
LEFT_ENDPOINT_AND_GRIPPER = slice(6, 10)
RIGHT_JOINTS = slice(10, 16)
RIGHT_ENDPOINT_AND_GRIPPER = slice(16, 20)
ACTION_LEFT_JOINTS = slice(0, 6)
ACTION_RIGHT_JOINTS = slice(7, 13)
ACTION_GRIPPER_INDICES = (6, 13)


def _vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    if vector.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {vector.shape}")
    return vector


def build_model_state(previous_actual: np.ndarray, current_actual: np.ndarray) -> np.ndarray:
    """Encode adjacent 20D feedback as training-time state: joint delta plus absolute TCP/gripper."""
    previous = _vector(previous_actual, STATE_DIM, "previous_actual")
    current = _vector(current_actual, STATE_DIM, "current_actual")
    encoded = current.copy()
    encoded[LEFT_JOINTS] = current[LEFT_JOINTS] - previous[LEFT_JOINTS]
    encoded[RIGHT_JOINTS] = current[RIGHT_JOINTS] - previous[RIGHT_JOINTS]
    return encoded


def q99_normalize(values: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    low = np.asarray(q01, dtype=np.float32)
    high = np.asarray(q99, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    if low.shape != high.shape or values.shape[-1] != low.shape[0]:
        raise ValueError("incompatible q99 statistics")
    result = 2.0 * (values - low) / (high - low) - 1.0
    return np.clip(result, -1.0, 1.0)


def q99_unnormalize(normalized: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    low = np.asarray(q01, dtype=np.float32)
    high = np.asarray(q99, dtype=np.float32)
    normalized = np.asarray(normalized, dtype=np.float32)
    if low.shape != high.shape or normalized.shape[-1] != low.shape[0]:
        raise ValueError("incompatible q99 statistics")
    return 0.5 * (np.clip(normalized, -1.0, 1.0) + 1.0) * (high - low) + low


def unnormalize_and_anchor_action(
    normalized_actions: np.ndarray,
    current_actual_state: np.ndarray,
    action_q01: np.ndarray,
    action_q99: np.ndarray,
) -> np.ndarray:
    """Decode a full rel-action chunk to absolute Piper/Pika commands.

    Training defines every joint action as q(t+k)-q(t), so every row is anchored
    to the same measured chunk-boundary state. Gripper values are absolute widths.
    """
    current = _vector(current_actual_state, STATE_DIM, "current_actual_state")
    normalized = np.asarray(normalized_actions, dtype=np.float32)
    if normalized.ndim != 2 or normalized.shape[1] != ACTION_DIM:
        raise ValueError(f"normalized_actions must have shape [N,{ACTION_DIM}], got {normalized.shape}")
    absolute = q99_unnormalize(normalized, action_q01, action_q99)
    absolute[:, ACTION_LEFT_JOINTS] += current[LEFT_JOINTS]
    absolute[:, ACTION_RIGHT_JOINTS] += current[RIGHT_JOINTS]
    return absolute


def model_images_from_observation(observation: dict) -> list:
    """Return raw RGB frames in the checkpoint's fixed top, left, right order."""
    return [observation["top"], observation["gripper_left"], observation["gripper_right"]]
