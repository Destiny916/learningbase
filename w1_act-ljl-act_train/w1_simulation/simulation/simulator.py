from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

from w1_simulation.robot.commands import W1PositionCommand
from w1_simulation.robot.joints import (
    BODY_JOINTS,
    HAND_POSITION_JOINTS_IN_URDF_ORDER,
    LEFT_HAND_JOINTS,
    RIGHT_HAND_JOINTS,
)
from w1_simulation.simulation.config import ACTIVE_JOINTS, HAND_MIMIC_JOINTS, SimulationConfig


class W1Simulator:
    def __init__(
        self,
        model_path: str | Path,
        config: SimulationConfig | None = None,
    ) -> None:
        import mujoco

        self.config = SimulationConfig() if config is None else config
        self.model_path = Path(model_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.joint_ids = np.asarray(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ACTIVE_JOINTS],
            dtype=np.int32,
        )
        if np.any(self.joint_ids < 0):
            missing = [
                name for name, joint_id in zip(ACTIVE_JOINTS, self.joint_ids, strict=True) if joint_id < 0
            ]
            raise ValueError(f"model missing active joints: {missing}")
        if self.model.nu != len(ACTIVE_JOINTS):
            raise ValueError(f"expected {len(ACTIVE_JOINTS)} actuators, got {self.model.nu}")
        self.qpos_ids = self.model.jnt_qposadr[self.joint_ids]
        self.dof_ids = self.model.jnt_dofadr[self.joint_ids]
        self.lower = self.model.jnt_range[self.joint_ids, 0].astype(np.float64)
        self.upper = self.model.jnt_range[self.joint_ids, 1].astype(np.float64)
        self.left_hand_slice = slice(len(BODY_JOINTS), len(BODY_JOINTS) + len(LEFT_HAND_JOINTS))
        self.right_hand_slice = slice(
            self.left_hand_slice.stop, self.left_hand_slice.stop + len(RIGHT_HAND_JOINTS)
        )
        self.target = np.clip((self.lower + self.upper) * 0.5, self.lower, self.upper)
        self._mimic_ids: list[tuple[int, int, float, float]] = []
        for dependent, (source, multiplier, offset) in HAND_MIMIC_JOINTS.items():
            dependent_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, dependent)
            source_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, source)
            if dependent_id < 0 or source_id < 0:
                raise ValueError(f"model missing mimic pair: {dependent} <- {source}")
            self._mimic_ids.append((dependent_id, source_id, multiplier, offset))
        self.reset()

    @property
    def control_dt(self) -> float:
        return float(self.model.opt.timestep * self.config.frame_skip)

    @property
    def joint_limits(self) -> np.ndarray:
        return np.column_stack((self.lower, self.upper)).copy()

    @property
    def limits(self) -> tuple[np.ndarray, np.ndarray]:
        return self.lower.copy(), self.upper.copy()

    @property
    def current_qpos(self) -> np.ndarray:
        return np.asarray(self.data.qpos[self.qpos_ids], dtype=np.float64).copy()

    @property
    def current_mimic_qpos(self) -> dict[str, float]:
        return {
            name: float(self.data.qpos[self.model.jnt_qposadr[dependent_id]])
            for name, (dependent_id, _, _, _) in zip(HAND_MIMIC_JOINTS, self._mimic_ids, strict=True)
        }

    def _raw_vector(self, values: npt.ArrayLike, label: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (len(ACTIVE_JOINTS),):
            raise ValueError(f"{label} must have shape ({len(ACTIVE_JOINTS)},), got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{label} must contain only finite values")
        return array

    def _set_mimic_qpos(self) -> None:
        for dependent_id, source_id, multiplier, offset in self._mimic_ids:
            dependent_qpos = self.model.jnt_qposadr[dependent_id]
            source_qpos = self.model.jnt_qposadr[source_id]
            value = offset + multiplier * self.data.qpos[source_qpos]
            lower, upper = self.model.jnt_range[dependent_id]
            self.data.qpos[dependent_qpos] = np.clip(value, lower, upper)

    def set_target(self, target: npt.ArrayLike) -> np.ndarray:
        raw = self._raw_vector(target, "target")
        self.target = np.clip(raw, self.lower, self.upper)
        self.data.ctrl[:] = self.target
        return self.target.copy()

    @staticmethod
    def _named_values(names: tuple[str, ...], values: np.ndarray) -> dict[str, float]:
        return {name: float(value) for name, value in zip(names, values, strict=True)}

    def _hand_target(self, command_values: dict[str, float], joint_slice: slice) -> np.ndarray:
        percent = np.asarray(
            [command_values[name] for name in HAND_POSITION_JOINTS_IN_URDF_ORDER],
            dtype=np.float64,
        )
        lower = self.lower[joint_slice]
        upper = self.upper[joint_slice]
        return lower + percent / 100.0 * (upper - lower)

    def target_from_command(self, command: W1PositionCommand) -> np.ndarray:
        if not isinstance(command, W1PositionCommand):
            raise TypeError(f"Expected W1PositionCommand, got {type(command).__name__}")
        body = self._named_values(command.body.name, command.body.position)
        target = self.target.copy()
        body_index = {name: index for index, name in enumerate(BODY_JOINTS)}
        for name, value in body.items():
            target[body_index[name]] = value
        target[self.left_hand_slice] = self._hand_target(
            self._named_values(command.left_hand.name, command.left_hand.value),
            self.left_hand_slice,
        )
        target[self.right_hand_slice] = self._hand_target(
            self._named_values(command.right_hand.name, command.right_hand.value),
            self.right_hand_slice,
        )
        return np.clip(target, self.lower, self.upper)

    def reset(
        self,
        qpos: npt.ArrayLike | None = None,
        target: npt.ArrayLike | None = None,
    ) -> np.ndarray:
        import mujoco

        mujoco.mj_resetData(self.model, self.data)
        positions = self.target if qpos is None else self._raw_vector(qpos, "qpos")
        positions = np.clip(positions, self.lower, self.upper)
        self.data.qpos[self.qpos_ids] = positions
        self.data.qvel[self.dof_ids] = 0.0
        self._set_mimic_qpos()
        self.set_target(positions if target is None else target)
        mujoco.mj_forward(self.model, self.data)
        return self.current_qpos

    def step(self, command: W1PositionCommand) -> np.ndarray:
        import mujoco

        self.set_target(self.target_from_command(command))
        for _ in range(self.config.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self._set_mimic_qpos()
        mujoco.mj_forward(self.model, self.data)
        return self.current_qpos

    def step_kinematic(self, command: W1PositionCommand) -> np.ndarray:
        import mujoco

        positions = self.set_target(self.target_from_command(command))
        self.data.qpos[self.qpos_ids] = positions
        self.data.qvel[self.dof_ids] = 0.0
        self._set_mimic_qpos()
        mujoco.mj_forward(self.model, self.data)
        return self.current_qpos

    def close(self) -> None:
        return None
