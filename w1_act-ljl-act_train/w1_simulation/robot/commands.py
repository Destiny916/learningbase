from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from w1_simulation.robot.joints import BODY_JOINTS, HAND_POSITION_JOINTS

BODY_POSITION_TOPIC = "/control/joint_position"
LEFT_HAND_POSITION_TOPIC = "/control/hand/left"
RIGHT_HAND_POSITION_TOPIC = "/control/hand/right"


def _validated_vector(names: tuple[str, ...], values: np.ndarray, label: str) -> np.ndarray:
    if len(names) != len(set(names)):
        raise ValueError(f"{label} contains duplicate joint names")
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (len(names),):
        raise ValueError(f"{label} must have shape ({len(names)},), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain only finite values")
    result = array.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class BodyPositionCommand:
    name: tuple[str, ...]
    position: np.ndarray

    def __post_init__(self) -> None:
        names = tuple(self.name)
        if not names or any(name not in BODY_JOINTS for name in names):
            raise ValueError("Body command contains unsupported W1 body joints")
        object.__setattr__(self, "name", names)
        object.__setattr__(
            self,
            "position",
            _validated_vector(names, self.position, "body position command"),
        )


@dataclass(frozen=True)
class HandPositionCommand:
    name: tuple[str, ...]
    value: np.ndarray
    mode: str = "POSITION"

    def __post_init__(self) -> None:
        names = tuple(self.name)
        if self.mode != "POSITION":
            raise ValueError(f"Unsupported hand command mode: {self.mode}")
        if names != HAND_POSITION_JOINTS:
            raise ValueError("Hand command does not use the W1 hardware joint order")
        values = _validated_vector(names, self.value, "hand position command")
        if np.any(values < 0.0) or np.any(values > 100.0):
            raise ValueError("Hand position values must stay in [0, 100]")
        object.__setattr__(self, "name", names)
        object.__setattr__(self, "value", values)


@dataclass(frozen=True)
class W1PositionCommand:
    body: BodyPositionCommand
    left_hand: HandPositionCommand
    right_hand: HandPositionCommand


@dataclass(frozen=True)
class W1ControlEndpoints:
    body: str = BODY_POSITION_TOPIC
    left_hand: str = LEFT_HAND_POSITION_TOPIC
    right_hand: str = RIGHT_HAND_POSITION_TOPIC

    def __post_init__(self) -> None:
        topics = (self.body, self.left_hand, self.right_hand)
        if any(not topic.startswith("/") for topic in topics) or len(set(topics)) != len(topics):
            raise ValueError("W1 control endpoints must be distinct absolute ROS topic names")


def position_command_contract(
    endpoints: W1ControlEndpoints | None = None,
    body_joint_names: tuple[str, ...] = BODY_JOINTS,
) -> dict[str, object]:
    selected = W1ControlEndpoints() if endpoints is None else endpoints
    body_names = tuple(body_joint_names)
    if not body_names or len(set(body_names)) != len(body_names):
        raise ValueError("Body command contract requires unique joint names")
    if any(name not in BODY_JOINTS for name in body_names):
        raise ValueError("Body command contract contains unsupported W1 body joints")
    hands = {
        "ros_type": "end_effector_interfaces/msg/EEJointControl",
        "fields": ["mode", "name", "value"],
        "mode": "POSITION",
        "joint_names": list(HAND_POSITION_JOINTS),
        "dimension": len(HAND_POSITION_JOINTS),
        "unit": "percent",
        "range": [0.0, 100.0],
    }
    return {
        "type": "w1_position",
        "simulation_consumption": "atomic_body_left_hand_right_hand",
        "ros_publication": "three_endpoint_messages",
        "body": {
            "topic": selected.body,
            "ros_type": "joint_interfaces/msg/JointPositionControl",
            "fields": ["name", "position"],
            "joint_names": list(body_names),
            "dimension": len(body_names),
            "unit": "rad",
        },
        "left_hand": {"topic": selected.left_hand, **hands},
        "right_hand": {"topic": selected.right_hand, **hands},
    }
