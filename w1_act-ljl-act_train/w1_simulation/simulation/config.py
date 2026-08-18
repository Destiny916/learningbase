from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from w1_simulation.robot.joints import (
    ACTIVE_JOINTS as ACTIVE_JOINTS,
)
from w1_simulation.robot.joints import (
    BODY_JOINTS as BODY_JOINTS,
)
from w1_simulation.robot.joints import (
    CONTROLLED_JOINTS as CONTROLLED_JOINTS,
)
from w1_simulation.robot.joints import (
    LEFT_HAND_JOINTS as LEFT_HAND_JOINTS,
)
from w1_simulation.robot.joints import (
    RIGHT_HAND_JOINTS as RIGHT_HAND_JOINTS,
)
from w1_simulation.w1_profile import DEFAULT_PROFILE

SOURCE_URDF = Path(os.environ.get("W1_SIMULATION_URDF", DEFAULT_PROFILE.urdf)).expanduser().resolve()

HAND_MIMIC_JOINTS = {
    "LEFT_IF_DIP": ("LEFT_IF_MCP_PITCH", 1.2, 0.19373154697137057),
    "LEFT_MF_DIP": ("LEFT_MF_MCP_PITCH", 1.2, 0.1832595714594046),
    "LEFT_RF_DIP": ("LEFT_RF_MCP_PITCH", 1.2, 0.16580627893946132),
    "LEFT_LF_DIP": ("LEFT_LF_MCP_PITCH", 1.2, 0.20245819323134223),
    "RIGHT_IF_DIP": ("RIGHT_IF_MCP_PITCH", 1.2, 0.19373154697137057),
    "RIGHT_MF_DIP": ("RIGHT_MF_MCP_PITCH", 1.2, 0.1832595714594046),
    "RIGHT_RF_DIP": ("RIGHT_RF_MCP_PITCH", 1.2, 0.16580627893946132),
    "RIGHT_LF_DIP": ("RIGHT_LF_MCP_PITCH", 1.2, 0.20245819323134223),
}

SELF_COLLISION_EXCLUDES = (
    ("buttock", "waist"),
    ("left_j5", "left_j7"),
    ("right_j5", "right_j7"),
    ("left_hand_base_link", "left_t_mcp_link"),
    ("right_hand_base_link", "right_t_mcp_link"),
)

DEFAULT_LOCKED_JOINT_VALUES = DEFAULT_PROFILE.locked_joint_values


@dataclass(frozen=True)
class ActSimulationConfig:
    control_hz: float = float(DEFAULT_PROFILE.simulation["control_hz"])
    timestep: float = float(DEFAULT_PROFILE.simulation["timestep"])
    frame_skip: int = int(DEFAULT_PROFILE.simulation["frame_skip"])
    replan_interval: int = 10
    max_camera_skew_ms: float = 50.0
    body_kp: float = 120.0
    hand_kp: float = 24.0
    control_mode: str = str(DEFAULT_PROFILE.simulation["control_mode"])

    def __post_init__(self) -> None:
        if self.control_hz <= 0:
            raise ValueError("control_hz must be positive")
        if self.frame_skip <= 0 or self.replan_interval <= 0:
            raise ValueError("frame_skip and replan_interval must be positive")
        if self.control_mode not in {"kinematic", "dynamic"}:
            raise ValueError(f"Unknown ACT simulation control mode: {self.control_mode}")
        actual_hz = 1.0 / (self.timestep * self.frame_skip)
        if abs(actual_hz - self.control_hz) > 1e-9:
            raise ValueError(f"Simulation timebase is {actual_hz} Hz, expected {self.control_hz} Hz")


@dataclass(frozen=True)
class SimulationConfig:
    timestep: float = 0.002
    frame_skip: int = 10
    body_kp: float = 120.0
    hand_kp: float = 24.0
    locked_joint_values: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_LOCKED_JOINT_VALUES))

    def __post_init__(self) -> None:
        if self.timestep <= 0.0:
            raise ValueError("timestep must be positive")
        if self.frame_skip < 1:
            raise ValueError("frame_skip must be at least one")
        if self.body_kp <= 0.0 or self.hand_kp <= 0.0:
            raise ValueError("controller gains must be positive")
        object.__setattr__(self, "locked_joint_values", dict(self.locked_joint_values))
