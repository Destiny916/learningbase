from __future__ import annotations

import mujoco
import numpy as np

from w1_simulation.robot.act_adapter import ActHandGestureConfig, W1ActAdapter
from w1_simulation.robot.commands import W1PositionCommand
from w1_simulation.robot.joints import (
    ACT_STATE_JOINTS,
    BODY_JOINTS,
    CONTROLLED_JOINTS,
    LEFT_HAND_JOINTS,
    RIGHT_HAND_JOINTS,
)

LEFT_ORIGIN_HAND_FIELDS = (
    "LEFT_HAND_THUMB1",
    "LEFT_HAND_THUMB2",
    "LEFT_HAND_INDEX",
    "LEFT_HAND_MIDDLE",
    "LEFT_HAND_RING",
    "LEFT_HAND_PINKY",
)
RIGHT_ORIGIN_HAND_FIELDS = (
    "RIGHT_HAND_THUMB1",
    "RIGHT_HAND_THUMB2",
    "RIGHT_HAND_INDEX",
    "RIGHT_HAND_MIDDLE",
    "RIGHT_HAND_RING",
    "RIGHT_HAND_PINKY",
)
HAND_PROTOCOL_TO_URDF_ORDER = (1, 0, 2, 3, 4, 5)


class ActJointMapper:
    def __init__(
        self,
        model: mujoco.MjModel,
        gestures: ActHandGestureConfig | None = None,
        selected_body_names: tuple[str, ...] = BODY_JOINTS,
    ) -> None:
        self.model = model
        self.gestures = ActHandGestureConfig() if gestures is None else gestures
        self.joint_ids = np.asarray(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in CONTROLLED_JOINTS]
        )
        if np.any(self.joint_ids < 0):
            missing = [
                name for name, joint_id in zip(CONTROLLED_JOINTS, self.joint_ids, strict=True) if joint_id < 0
            ]
            raise ValueError(f"MuJoCo model is missing controlled joints: {missing}")
        actuator_joints = tuple(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(joint_id))
            for joint_id in model.actuator_trnid[:, 0]
        )
        if actuator_joints != CONTROLLED_JOINTS:
            raise ValueError(f"Actuator order mismatch: {actuator_joints}")
        self.qpos_ids = model.jnt_qposadr[self.joint_ids]
        self.lower = model.jnt_range[self.joint_ids, 0].astype(np.float64)
        self.upper = model.jnt_range[self.joint_ids, 1].astype(np.float64)
        self.left_slice = slice(len(BODY_JOINTS), len(BODY_JOINTS) + len(LEFT_HAND_JOINTS))
        self.right_slice = slice(self.left_slice.stop, self.left_slice.stop + len(RIGHT_HAND_JOINTS))
        self.act_adapter = W1ActAdapter(
            dict(zip(BODY_JOINTS, self.lower[: len(BODY_JOINTS)], strict=True)),
            dict(zip(BODY_JOINTS, self.upper[: len(BODY_JOINTS)], strict=True)),
            self.gestures,
            selected_body_names,
        )

    @staticmethod
    def _gesture_percent(scalar: float, closed: tuple[float, ...], opened: tuple[float, ...]) -> np.ndarray:
        return W1ActAdapter.gesture_percent(scalar, closed, opened)

    @staticmethod
    def _scalar_from_percent(
        percent: np.ndarray,
        closed: tuple[float, ...],
        opened: tuple[float, ...],
    ) -> float:
        closed_array = np.asarray(closed, dtype=np.float64)
        direction = np.asarray(opened, dtype=np.float64) - closed_array
        denominator = float(np.dot(direction, direction))
        if denominator <= 0.0:
            raise ValueError("Open and closed hand gestures must differ")
        open_fraction = float(np.dot(percent - closed_array, direction) / denominator)
        return float(np.clip(open_fraction, 0.0, 1.0) * 100.0)

    @staticmethod
    def _hand_protocol_to_urdf_order(values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (6,):
            raise ValueError(f"Expected a 6D hand command, got {array.shape}")
        return array[list(HAND_PROTOCOL_TO_URDF_ORDER)]

    @staticmethod
    def _urdf_to_hand_protocol_order(values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (6,):
            raise ValueError(f"Expected 6D hand joint positions, got {array.shape}")
        return array[list(HAND_PROTOCOL_TO_URDF_ORDER)]

    def _percent_to_radians(self, percent: np.ndarray, joint_slice: slice) -> np.ndarray:
        clipped = np.clip(np.asarray(percent, dtype=np.float64), 0.0, 100.0)
        return self.lower[joint_slice] + clipped / 100.0 * (self.upper[joint_slice] - self.lower[joint_slice])

    def _radians_to_percent(self, radians: np.ndarray, joint_slice: slice) -> np.ndarray:
        span = np.maximum(self.upper[joint_slice] - self.lower[joint_slice], 1e-12)
        return np.clip((radians - self.lower[joint_slice]) / span * 100.0, 0.0, 100.0)

    def act_action_to_target(self, action: np.ndarray) -> np.ndarray:
        values = np.asarray(action, dtype=np.float64)
        if values.shape != (len(ACT_STATE_JOINTS),):
            raise ValueError(f"Expected a {len(ACT_STATE_JOINTS)}D ACT action, got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("ACT action contains non-finite values")
        target = np.empty(len(CONTROLLED_JOINTS), dtype=np.float64)
        target[: len(BODY_JOINTS)] = values[: len(BODY_JOINTS)]
        left_percent = self._gesture_percent(
            values[-2], self.gestures.left_gripper_0, self.gestures.left_gripper_100
        )
        right_percent = self._gesture_percent(
            values[-1], self.gestures.right_gripper_0, self.gestures.right_gripper_100
        )
        target[self.left_slice] = self._percent_to_radians(
            self._hand_protocol_to_urdf_order(left_percent), self.left_slice
        )
        target[self.right_slice] = self._percent_to_radians(
            self._hand_protocol_to_urdf_order(right_percent), self.right_slice
        )
        return np.clip(target, self.lower, self.upper)

    def act_action_to_command(
        self,
        action: np.ndarray,
    ) -> W1PositionCommand:
        return self.act_adapter.action_to_command(action)

    def initial_target_from_pose(self, pose: dict[str, float]) -> np.ndarray:
        required = set(BODY_JOINTS + LEFT_ORIGIN_HAND_FIELDS + RIGHT_ORIGIN_HAND_FIELDS)
        missing = sorted(required - pose.keys())
        if missing:
            raise ValueError(f"Origin pose is missing simulator joints: {missing}")
        target = np.empty(len(CONTROLLED_JOINTS), dtype=np.float64)
        target[: len(BODY_JOINTS)] = [pose[name] for name in BODY_JOINTS]
        target[self.left_slice] = self._percent_to_radians(
            self._hand_protocol_to_urdf_order(np.asarray([pose[name] for name in LEFT_ORIGIN_HAND_FIELDS])),
            self.left_slice,
        )
        target[self.right_slice] = self._percent_to_radians(
            self._hand_protocol_to_urdf_order(np.asarray([pose[name] for name in RIGHT_ORIGIN_HAND_FIELDS])),
            self.right_slice,
        )
        return np.clip(target, self.lower, self.upper)

    def act_state_from_sim(self, controlled_qpos: np.ndarray) -> np.ndarray:
        positions = np.asarray(controlled_qpos, dtype=np.float64)
        if positions.shape != (len(CONTROLLED_JOINTS),):
            raise ValueError(f"Expected {len(CONTROLLED_JOINTS)}D simulator qpos, got {positions.shape}")
        if not np.isfinite(positions).all():
            raise ValueError("Simulator qpos contains non-finite values")
        left_percent = self._urdf_to_hand_protocol_order(
            self._radians_to_percent(positions[self.left_slice], self.left_slice)
        )
        right_percent = self._urdf_to_hand_protocol_order(
            self._radians_to_percent(positions[self.right_slice], self.right_slice)
        )
        left_scalar = self._scalar_from_percent(
            left_percent, self.gestures.left_gripper_0, self.gestures.left_gripper_100
        )
        right_scalar = self._scalar_from_percent(
            right_percent, self.gestures.right_gripper_0, self.gestures.right_gripper_100
        )
        return np.concatenate((positions[: len(BODY_JOINTS)], [left_scalar, right_scalar])).astype(np.float32)
