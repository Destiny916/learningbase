"""Pure contracts shared by the XWiz W1 simulation and real-robot adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


class RuntimeContractError(RuntimeError):
    """Raised before publication when a real-robot contract is violated."""


EXECUTION_SINGLE = "single"
EXECUTION_CONTINUOUS = "continuous"
# Vendor worker loops require a finite numeric max_steps value.
CONTINUOUS_MAX_STEPS = 9_007_199_254_740_991


BODY_ORDER = (
    "WAIST",
    "LEFT_J1", "LEFT_J2", "LEFT_J3", "LEFT_J4", "LEFT_J5", "LEFT_J6", "LEFT_J7",
    "NECK1", "NECK2",
    "RIGHT_J1", "RIGHT_J2", "RIGHT_J3", "RIGHT_J4", "RIGHT_J5", "RIGHT_J6", "RIGHT_J7",
)

ROBOT_ORDER = (
    "ANKLE", "KNEE", "BUTTOCK", "WAIST", "NECK1", "NECK2",
    "LEFT_J1", "LEFT_J2", "LEFT_J3", "LEFT_J4", "LEFT_J5", "LEFT_J6", "LEFT_J7",
    "RIGHT_J1", "RIGHT_J2", "RIGHT_J3", "RIGHT_J4", "RIGHT_J5", "RIGHT_J6", "RIGHT_J7",
)

ACT_DEFAULT_20 = (
    0.6642, -1.3265, 0.6632, 0.0, 0.0, 0.15,
    0.172, -0.928, 0.342, -1.859, -0.581, -0.268, 0.204,
    -0.228, 0.929, -0.363, 2.025, 0.399, 0.199, -0.423,
)

HAND_JOINT_NAMES = (
    "T_CMC_YAW", "T_MCP", "IF_MCP_PITCH", "MF_MCP_PITCH", "RF_MCP_PITCH", "LF_MCP_PITCH",
)

# The checkpoint scalar is openness: 0 is the task-specific closed grasp and
# 100 is the open hand. Values are Linker L6 command percentages.
LEFT_CLOSED = (0.0, 100.0, 35.0, 45.0, 47.0, 37.0)
LEFT_OPEN = (0.0, 70.0, 0.0, 0.0, 0.0, 0.0)
RIGHT_CLOSED = (65.0, 100.0, 70.0, 75.0, 100.0, 100.0)
RIGHT_OPEN = (0.0, 70.0, 0.0, 0.0, 0.0, 0.0)

BODY_LIMITS = {
    "WAIST": (-2.9670597284, 2.9670597284),
    "LEFT_J1": (-2.9670597284, 2.9670597284),
    "LEFT_J2": (-2.0943951024, 1.5707963268),
    "LEFT_J3": (-2.9670597284, 2.9670597284),
    "LEFT_J4": (-2.3561944902, 1.5707963268),
    "LEFT_J5": (-2.9670597284, 2.9670597284),
    "LEFT_J6": (-0.7853981634, 0.7853981634),
    "LEFT_J7": (-1.5707963268, 1.0471975512),
    "NECK1": (-1.5707963268, 1.5707963268),
    "NECK2": (-0.7853981634, 0.4363323130),
    "RIGHT_J1": (-2.9670597284, 2.9670597284),
    "RIGHT_J2": (-1.5707963268, 2.0943951024),
    "RIGHT_J3": (-2.9670597284, 2.9670597284),
    "RIGHT_J4": (-1.5707963268, 2.3561944902),
    "RIGHT_J5": (-2.9670597284, 2.9670597284),
    "RIGHT_J6": (-0.7853981634, 0.7853981634),
    "RIGHT_J7": (-1.0471975512, 1.5707963268),
}


@dataclass(frozen=True)
class ActionCommands:
    body_names: tuple[str, ...]
    body_positions: tuple[float, ...]
    left_hand: tuple[float, ...]
    right_hand: tuple[float, ...]


def hand_command_from_openness(
    scalar: float,
    closed: Sequence[float],
    opened: Sequence[float],
) -> tuple[float, ...]:
    value = float(scalar)
    if not np.isfinite(value):
        raise RuntimeContractError("gripper openness must be finite")
    closed_array = np.asarray(closed, dtype=np.float64)
    opened_array = np.asarray(opened, dtype=np.float64)
    if closed_array.shape != (6,) or opened_array.shape != (6,):
        raise RuntimeContractError("hand endpoints must each contain six values")
    fraction = float(np.clip(value, 0.0, 100.0)) / 100.0
    command = closed_array * (1.0 - fraction) + opened_array * fraction
    return tuple(float(item) for item in command)


def scalar_from_hand_command(
    positions: Sequence[float],
    closed: Sequence[float],
    opened: Sequence[float],
) -> float:
    values = np.asarray(positions, dtype=np.float64)
    closed_array = np.asarray(closed, dtype=np.float64)
    opened_array = np.asarray(opened, dtype=np.float64)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise RuntimeContractError("hand feedback must contain six finite values")
    direction = opened_array - closed_array
    denominator = float(np.dot(direction, direction))
    if denominator <= 0.0:
        raise RuntimeContractError("open and closed hand endpoints must differ")
    fraction = float(np.dot(values - closed_array, direction) / denominator)
    return float(np.clip(fraction, 0.0, 1.0) * 100.0)


def normalize_hand_feedback(positions: Sequence[float]) -> tuple[float, ...]:
    """Convert Linker L6 feedback to the ACT command scale (0..100).

    W1 firmware versions report either normalized ratios (0..1) or percentage
    values (0..100).  The v0.4.6 Linker feedback observed on this robot is the
    latter, so accept both wire representations without changing the model
    contract.
    """
    values = np.asarray(positions, dtype=np.float64)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise RuntimeContractError("hand feedback must contain six finite values")
    if np.any(values < 0.0) or np.any(values > 100.0):
        raise RuntimeContractError("Linker L6 feedback positions must be in 0..100")
    if np.any(values > 1.0):
        return tuple(float(value) for value in values)
    return tuple(float(value) for value in values * 100.0)


def feedback_positions_by_name(joint_states: Sequence[object]) -> tuple[float, ...]:
    values = {
        str(getattr(state, "name", "")): float(getattr(state, "position", np.nan))
        for state in joint_states
    }
    missing = [name for name in HAND_JOINT_NAMES if name not in values]
    if missing:
        raise RuntimeContractError(f"hand feedback is missing joints: {', '.join(missing)}")
    ordered = np.asarray([values[name] for name in HAND_JOINT_NAMES], dtype=np.float64)
    if not np.isfinite(ordered).all():
        raise RuntimeContractError("hand feedback positions must be finite")
    return tuple(float(value) for value in ordered)


def gripper_scalars_from_feedback(
    left_positions: Sequence[float],
    right_positions: Sequence[float],
) -> tuple[float, float]:
    return (
        scalar_from_hand_command(normalize_hand_feedback(left_positions), LEFT_CLOSED, LEFT_OPEN),
        scalar_from_hand_command(normalize_hand_feedback(right_positions), RIGHT_CLOSED, RIGHT_OPEN),
    )


def mode_topics(mode: int) -> dict[str, str]:
    if mode == 1:
        return {
            "body": "/mj_sim/control/joint_position",
            "left_hand": "/mj_sim/control/ee/left",
            "right_hand": "/mj_sim/control/ee/right",
        }
    if mode == 2:
        return {
            "body": "/control/joint_position",
            "left_hand": "/control/ee/left",
            "right_hand": "/control/ee/right",
        }
    raise RuntimeContractError(f"mode must be 1 or 2, got {mode}")


def validate_observation_buffers(
    buffers: Mapping[str, Sequence[object]],
    *,
    use_wrist_images: bool,
) -> None:
    required = ["head_left", "head_right", "joint_state", "left_hand", "right_hand"]
    if use_wrist_images:
        required.extend(("wrist_left", "wrist_right"))
    missing = [name for name in required if not buffers.get(name)]
    if missing:
        raise RuntimeContractError(f"observation buffers are not ready: {', '.join(missing)}")
    if len(buffers["joint_state"]) != 20:
        raise RuntimeContractError("joint_state buffer must contain 20 positions")


def validate_timed_actions(timed_actions: Sequence[object]) -> np.ndarray:
    values = [action.get_action() for action in timed_actions]
    return validate_action_chunk(values)


def validate_action_chunk(actions: object) -> np.ndarray:
    array = np.asarray(actions, dtype=np.float64)
    if array.shape != (100, 19):
        raise RuntimeContractError(f"action chunk must have shape (100, 19), got {array.shape}")
    if not np.isfinite(array).all():
        raise RuntimeContractError("action chunk must contain only finite values")
    return array


def action_to_commands(action: object) -> ActionCommands:
    values = np.asarray(action, dtype=np.float64)
    if values.shape != (19,) or not np.isfinite(values).all():
        raise RuntimeContractError("action frame must contain 19 finite values")
    body = tuple(
        float(np.clip(value, *BODY_LIMITS[name]))
        for name, value in zip(BODY_ORDER, values[:17], strict=True)
    )
    return ActionCommands(
        body_names=BODY_ORDER,
        body_positions=body,
        left_hand=hand_command_from_openness(values[17], LEFT_CLOSED, LEFT_OPEN),
        right_hand=hand_command_from_openness(values[18], RIGHT_CLOSED, RIGHT_OPEN),
    )


def validate_robot_health(
    payload: Mapping[str, object],
    *,
    allowed_status: Sequence[str] = ("Idle",),
) -> None:
    if payload.get("status") not in allowed_status:
        raise RuntimeContractError(
            f"robot status must be one of {tuple(allowed_status)}, got {payload.get('status')!r}"
        )
    motor_status = list(payload.get("motor_status", ()))
    if len(motor_status) != 20 or any(status != "OP" for status in motor_status):
        raise RuntimeContractError("all 20 motors must be OP")
    motor_errors = list(payload.get("motor_error_code", ()))
    if len(motor_errors) != 20 or any(int(code) != 0 for code in motor_errors):
        raise RuntimeContractError("all motor error codes must be zero")
    server_errors = list(payload.get("server_error_code", ()))
    if len(server_errors) != 20 or any(str(code) != "None" for code in server_errors):
        raise RuntimeContractError("all server error codes must be None")


def validate_robot_ready(payload: Mapping[str, object], tolerance_rad: float = 0.05) -> float:
    validate_robot_health(payload, allowed_status=("Idle",))
    positions = np.asarray(payload.get("joint_position", ()), dtype=np.float64)
    if positions.shape != (20,) or not np.isfinite(positions).all():
        raise RuntimeContractError("robot feedback must contain 20 finite joint positions")
    error = np.abs(positions - np.asarray(ACT_DEFAULT_20, dtype=np.float64))
    max_error = float(error.max())
    if max_error > float(tolerance_rad):
        joint = ROBOT_ORDER[int(error.argmax())]
        raise RuntimeContractError(
            f"robot is not at ACT default pose: {joint} error={max_error:.6f} rad"
        )
    return max_error


def prepare_client_config(config: Mapping[str, object], mode: int) -> dict[str, object]:
    if mode not in (1, 2):
        raise RuntimeContractError(f"mode must be 1 or 2, got {mode}")
    execution_mode = str(config.get("execution_mode", EXECUTION_SINGLE))
    if execution_mode not in (EXECUTION_SINGLE, EXECUTION_CONTINUOUS):
        raise RuntimeContractError(
            f"execution_mode must be {EXECUTION_SINGLE!r} or {EXECUTION_CONTINUOUS!r}"
        )
    if execution_mode == EXECUTION_CONTINUOUS and mode != 2:
        raise RuntimeContractError("continuous execution is only available in real mode")
    prepared = dict(config)
    prepared.update(
        mode=mode,
        execution_mode=execution_mode,
        action_horizon=100,
        max_steps=(
            CONTINUOUS_MAX_STEPS
            if execution_mode == EXECUTION_CONTINUOUS
            else 100
        ),
        sample_factor=1.0,
        chunk_size_threshold=0.0,
        home_position="",
    )
    return prepared


@dataclass(frozen=True)
class ActionProgress:
    global_frame: int
    chunk_index: int
    frame_in_chunk: int
    chunk_complete: bool
    session_complete: bool


class ChunkExecutionGate:
    def __init__(self, execution_mode: str, chunk_size: int = 100):
        if execution_mode not in (EXECUTION_SINGLE, EXECUTION_CONTINUOUS):
            raise ValueError(f"unsupported execution mode: {execution_mode}")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.execution_mode = execution_mode
        self.chunk_size = int(chunk_size)
        self.published = 0

    def mark_published(self) -> ActionProgress:
        if self.execution_mode == EXECUTION_SINGLE and self.published >= self.chunk_size:
            raise RuntimeContractError("single action chunk is already complete")
        self.published += 1
        chunk_index = (self.published - 1) // self.chunk_size + 1
        frame_in_chunk = (self.published - 1) % self.chunk_size + 1
        chunk_complete = frame_in_chunk == self.chunk_size
        return ActionProgress(
            global_frame=self.published,
            chunk_index=chunk_index,
            frame_in_chunk=frame_in_chunk,
            chunk_complete=chunk_complete,
            session_complete=(
                self.execution_mode == EXECUTION_SINGLE and chunk_complete
            ),
        )


def should_request_next_chunk(
    *,
    execution_mode: str,
    queue_size: int,
    frame_in_chunk: int,
    chunk_completed_at: float,
    feedback_received_at: float,
) -> bool:
    if execution_mode != EXECUTION_CONTINUOUS:
        return queue_size == 0
    return (
        queue_size == 0
        and frame_in_chunk == 100
        and chunk_completed_at > 0.0
        and feedback_received_at > chunk_completed_at
    )


def validate_feedback_freshness(
    received_at: float,
    *,
    now: float,
    timeout_seconds: float,
) -> float:
    if received_at <= 0.0:
        raise RuntimeContractError("real execution has no robot state timestamp")
    age = float(now) - float(received_at)
    if age < 0.0 or age > float(timeout_seconds):
        raise RuntimeContractError(
            f"robot state feedback is stale: age={age:.3f}s "
            f"limit={float(timeout_seconds):.3f}s"
        )
    return age


class SingleChunkGate:
    def __init__(self, limit: int = 100):
        if limit <= 0:
            raise ValueError("limit must be positive")
        self.limit = int(limit)
        self.published = 0

    def mark_published(self) -> bool:
        if self.published >= self.limit:
            raise RuntimeContractError("single action chunk is already complete")
        self.published += 1
        return self.published == self.limit
