from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from w1_simulation.robot.commands import BodyPositionCommand, HandPositionCommand, W1PositionCommand
from w1_simulation.robot.joints import ACT_STATE_JOINTS, BODY_JOINTS, HAND_POSITION_JOINTS


@dataclass(frozen=True)
class ActHandGestureConfig:
    # Public/PC1 order: T_MCP, T_CMC_YAW, IF, MF, RF, LF.
    left_gripper_0: tuple[float, ...] = (0.0, 100.0, 35.0, 45.0, 47.0, 37.0)
    left_gripper_100: tuple[float, ...] = (0.0, 70.0, 0.0, 0.0, 0.0, 0.0)
    right_gripper_0: tuple[float, ...] = (65.0, 100.0, 70.0, 75.0, 100.0, 100.0)
    right_gripper_100: tuple[float, ...] = (0.0, 70.0, 0.0, 0.0, 0.0, 0.0)
    mapping: str = "hand_command_thumb1_mcp_thumb2_cmc_to_urdf_range"
    scalar_semantics: str = "piecewise_linear_between_gripper_0_and_100"

    def __post_init__(self) -> None:
        for name in (
            "left_gripper_0",
            "left_gripper_100",
            "right_gripper_0",
            "right_gripper_100",
        ):
            values = np.asarray(getattr(self, name), dtype=np.float64)
            if values.shape != (6,) or not np.isfinite(values).all():
                raise ValueError(f"{name} must contain exactly six finite values")
            if np.any(values < 0.0) or np.any(values > 100.0):
                raise ValueError(f"{name} values must stay in [0, 100]")
        if self.left_gripper_0 == self.left_gripper_100:
            raise ValueError("Left gripper endpoints must differ")
        if self.right_gripper_0 == self.right_gripper_100:
            raise ValueError("Right gripper endpoints must differ")

    @staticmethod
    def _endpoint(payload: Mapping[str, object], side: str, scalar: str) -> tuple[float, ...]:
        try:
            values = payload[side]
            if not isinstance(values, Mapping):
                raise TypeError
            endpoint = values[scalar]
            if not isinstance(endpoint, list):
                raise TypeError
            return tuple(float(value) for value in endpoint)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Hand mapping requires {side}.{scalar} as a numeric list") from exc

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ActHandGestureConfig:
        hands = payload.get("hands")
        if isinstance(hands, Mapping):
            payload = hands
        return cls(
            left_gripper_0=cls._endpoint(payload, "left", "0"),
            left_gripper_100=cls._endpoint(payload, "left", "100"),
            right_gripper_0=cls._endpoint(payload, "right", "0"),
            right_gripper_100=cls._endpoint(payload, "right", "100"),
        )

    @classmethod
    def from_json(cls, path: Path) -> ActHandGestureConfig:
        with Path(path).resolve().open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, Mapping):
            raise ValueError("Hand mapping JSON root must be an object")
        return cls.from_dict(payload)

    def as_dict(self) -> dict[str, object]:
        return {
            "mapping": self.mapping,
            "scalar_semantics": self.scalar_semantics,
            "left": {"0": list(self.left_gripper_0), "100": list(self.left_gripper_100)},
            "right": {"0": list(self.right_gripper_0), "100": list(self.right_gripper_100)},
        }


class W1ActAdapter:
    def __init__(
        self,
        body_lower: Mapping[str, float],
        body_upper: Mapping[str, float],
        gestures: ActHandGestureConfig | None = None,
        selected_body_names: tuple[str, ...] = BODY_JOINTS,
    ) -> None:
        if set(body_lower) != set(BODY_JOINTS) or set(body_upper) != set(BODY_JOINTS):
            raise ValueError("ACT adapter body limits must cover the 17 active W1 body joints")
        self.body_lower = np.asarray([body_lower[name] for name in BODY_JOINTS], dtype=np.float64)
        self.body_upper = np.asarray([body_upper[name] for name in BODY_JOINTS], dtype=np.float64)
        if np.isnan(self.body_lower).any() or np.isnan(self.body_upper).any():
            raise ValueError("ACT adapter body limits must not contain NaN")
        if np.any(self.body_lower > self.body_upper):
            raise ValueError("ACT adapter body lower limits must not exceed upper limits")
        selected = tuple(selected_body_names)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("Selected ACT body joints must be a non-empty unique sequence")
        if any(name not in BODY_JOINTS for name in selected):
            raise ValueError("Selected ACT body joints must be a subset of the model body order")
        body_index = {name: index for index, name in enumerate(BODY_JOINTS)}
        self.selected_body_names = selected
        self.selected_body_indices = np.asarray([body_index[name] for name in selected], dtype=np.int64)
        self.gestures = ActHandGestureConfig() if gestures is None else gestures

    @staticmethod
    def gesture_percent(scalar: float, closed: tuple[float, ...], opened: tuple[float, ...]) -> np.ndarray:
        value = float(scalar)
        if not np.isfinite(value):
            raise ValueError("gripper openness must be finite")
        open_fraction = float(np.clip(value, 0.0, 100.0)) / 100.0
        return (
            np.asarray(closed, dtype=np.float64) * (1.0 - open_fraction)
            + np.asarray(opened, dtype=np.float64) * open_fraction
        )

    def action_to_command(
        self,
        action: np.ndarray,
    ) -> W1PositionCommand:
        values = np.asarray(action, dtype=np.float64)
        if values.shape != (len(ACT_STATE_JOINTS),):
            raise ValueError(f"Expected a {len(ACT_STATE_JOINTS)}D ACT action, got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("ACT action contains non-finite values")
        clipped_body = np.clip(values[: len(BODY_JOINTS)], self.body_lower, self.body_upper)
        left_hand = self.gesture_percent(
            0.0 if values[-2] < 95.0 else values[-2],
            self.gestures.left_gripper_0, self.gestures.left_gripper_100
        )
        right_hand = self.gesture_percent(
            0.0 if values[-1] < 95.0 else values[-1],
            self.gestures.right_gripper_0, self.gestures.right_gripper_100
        )
        return W1PositionCommand(
            body=BodyPositionCommand(
                name=self.selected_body_names,
                position=clipped_body[self.selected_body_indices],
            ),
            left_hand=HandPositionCommand(name=HAND_POSITION_JOINTS, value=left_hand),
            right_hand=HandPositionCommand(name=HAND_POSITION_JOINTS, value=right_hand),
        )
