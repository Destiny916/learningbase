from __future__ import annotations

import json
import queue
import threading
from pathlib import Path

import mujoco
import numpy as np

from w1_simulation.artifacts import sha256_file
from w1_simulation.robot.joints import ACT_STATE_JOINTS, BODY_JOINTS, CONTROLLED_JOINTS
from w1_simulation.robot.mapping import ActJointMapper

QUALITY_METRICS = ("pose", "end_effector", "motion_direction", "amplitude")
QUALITY_WEIGHTS = {
    "pose": 0.40,
    "end_effector": 0.30,
    "motion_direction": 0.20,
    "amplitude": 0.10,
}


def validate_quality_metrics(metrics: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    selected = set(metrics)
    unknown = sorted(selected - set(QUALITY_METRICS))
    if unknown:
        raise ValueError(f"Unknown quality metrics: {unknown}; expected a subset of {QUALITY_METRICS}")
    return tuple(name for name in QUALITY_METRICS if name in selected)


def _nearest_delta_ms(reference_timestamps: np.ndarray, query_timestamps: np.ndarray) -> float:
    positions = np.searchsorted(reference_timestamps, query_timestamps)
    left = np.clip(positions - 1, 0, len(reference_timestamps) - 1)
    right = np.clip(positions, 0, len(reference_timestamps) - 1)
    nearest = np.minimum(
        np.abs(query_timestamps - reference_timestamps[left]),
        np.abs(query_timestamps - reference_timestamps[right]),
    )
    return float(np.max(nearest) * 1000.0)


def load_reference_states(origin_root: Path, timestamps: np.ndarray) -> tuple[np.ndarray, Path, float]:
    matches = sorted(Path(origin_root).resolve().glob("pose_record_*.json"))
    if len(matches) != 1:
        raise ValueError(f"Expected one origin pose record, found {len(matches)}")
    payload = json.loads(matches[0].read_text(encoding="utf-8"))
    frames = payload.get("frames", [])
    if not frames:
        raise ValueError(f"Origin pose record is empty: {matches[0]}")
    reference_timestamps = np.asarray([float(frame["timestamp"]) for frame in frames], dtype=np.float64)
    if np.any(np.diff(reference_timestamps) <= 0.0):
        raise ValueError("Origin pose timestamps must be strictly increasing")
    reference_rows = np.asarray(
        [[float(frame["data"][name]) for name in ACT_STATE_JOINTS] for frame in frames],
        dtype=np.float64,
    )
    query = np.asarray(timestamps, dtype=np.float64)
    if query.ndim != 1 or len(query) == 0 or not np.isfinite(query).all():
        raise ValueError("Quality evaluation timestamps must be a non-empty finite vector")
    interpolated = np.column_stack(
        [
            np.interp(query, reference_timestamps, reference_rows[:, index])
            for index in range(len(ACT_STATE_JOINTS))
        ]
    )
    return interpolated, matches[0], _nearest_delta_ms(reference_timestamps, query)


def _rotation_error_degrees(actual: np.ndarray, reference: np.ndarray) -> float:
    relative = actual.T @ reference
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


class MotionQualityEvaluator:
    def __init__(
        self,
        origin_root: Path,
        timestamps: np.ndarray,
        mapper: ActJointMapper,
        initial_qpos: np.ndarray,
        metrics: tuple[str, ...] | list[str],
        execution: str = "offline_post_rollout",
    ) -> None:
        self.metrics = validate_quality_metrics(metrics)
        if not self.metrics:
            raise ValueError("MotionQualityEvaluator requires at least one metric")
        self.mapper = mapper
        self.execution = execution
        self.model = mapper.model
        self.reference_states, self.reference_file, self.max_reference_delta_ms = load_reference_states(
            origin_root, timestamps
        )
        self.reference_sha256 = sha256_file(self.reference_file)
        initial = np.asarray(initial_qpos, dtype=np.float64)
        if initial.shape != (len(CONTROLLED_JOINTS),) or not np.isfinite(initial).all():
            raise ValueError(f"Initial simulator qpos must have shape ({len(CONTROLLED_JOINTS)},)")
        self.initial_qpos = initial.copy()
        self.initial_state = mapper.act_state_from_sim(initial).astype(np.float64)
        self.state_span = np.concatenate(
            (
                np.maximum(mapper.upper[: len(BODY_JOINTS)] - mapper.lower[: len(BODY_JOINTS)], 1e-9),
                np.asarray([100.0, 100.0]),
            )
        )
        self.actual_min = self.initial_state.copy()
        self.actual_max = self.initial_state.copy()
        self.reference_min = self.reference_states[0].copy()
        self.reference_max = self.reference_states[0].copy()
        self.previous_actual = self.initial_state.copy()
        self.previous_reference = self.reference_states[0].copy()
        self.history: dict[str, list[float]] = {name: [] for name in self.metric_names}
        self._kinematics_data = mujoco.MjData(self.model) if "end_effector" in self.metrics else None
        self._end_effector_ids = (
            {
                side: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ee")
                for side in ("left", "right")
            }
            if "end_effector" in self.metrics
            else {}
        )
        if any(body_id < 0 for body_id in self._end_effector_ids.values()):
            raise ValueError("MuJoCo model must contain left_ee and right_ee bodies")
        self._hold_end_effectors = (
            self._end_effector_poses(self.initial_qpos) if "end_effector" in self.metrics else {}
        )

    @property
    def metric_names(self) -> tuple[str, ...]:
        names = ["score", "hold_score", "score_gain_over_hold"]
        if "pose" in self.metrics:
            names.extend(("pose_score", "pose_nrmse", "body_rmse_rad", "gripper_mae"))
        if "end_effector" in self.metrics:
            names.extend(
                (
                    "end_effector_score",
                    "left_ee_position_cm",
                    "right_ee_position_cm",
                    "mean_ee_position_cm",
                    "mean_ee_orientation_deg",
                )
            )
        if "motion_direction" in self.metrics:
            names.extend(("motion_direction_score", "motion_direction_agreement"))
        if "amplitude" in self.metrics:
            names.extend(("amplitude_score", "amplitude_coverage", "waist_amplitude_coverage"))
        return tuple(names)

    def _end_effector_poses(self, controlled_qpos: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        if self._kinematics_data is None:
            raise RuntimeError("End-effector kinematics are disabled")
        data = self._kinematics_data
        data.qpos[:] = self.model.qpos0
        data.qpos[self.mapper.qpos_ids] = controlled_qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, data)
        return {
            side: (data.xpos[body_id].copy(), data.xmat[body_id].reshape(3, 3).copy())
            for side, body_id in self._end_effector_ids.items()
        }

    def _pose_values(self, actual: np.ndarray, reference: np.ndarray) -> tuple[dict[str, float], float]:
        normalized = (actual - reference) / self.state_span
        pose_nrmse = float(np.sqrt(np.mean(np.square(normalized))))
        values = {
            "pose_nrmse": pose_nrmse,
            "body_rmse_rad": float(
                np.sqrt(np.mean(np.square(actual[: len(BODY_JOINTS)] - reference[: len(BODY_JOINTS)])))
            ),
            "gripper_mae": float(np.mean(np.abs(actual[-2:] - reference[-2:]))),
        }
        return values, float(np.clip(1.0 - pose_nrmse / 0.25, 0.0, 1.0))

    def _end_effector_values(
        self,
        actual_qpos: np.ndarray,
        reference_state: np.ndarray,
        actual_poses: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> tuple[dict[str, float], float]:
        actual_poses = actual_poses or self._end_effector_poses(actual_qpos)
        reference_poses = self._end_effector_poses(self.mapper.act_action_to_target(reference_state))
        position_errors = {
            side: float(np.linalg.norm(actual_poses[side][0] - reference_poses[side][0]) * 100.0)
            for side in ("left", "right")
        }
        orientation_errors = [
            _rotation_error_degrees(actual_poses[side][1], reference_poses[side][1])
            for side in ("left", "right")
        ]
        mean_position = float(np.mean(list(position_errors.values())))
        mean_orientation = float(np.mean(orientation_errors))
        position_score = float(np.clip(1.0 - mean_position / 30.0, 0.0, 1.0))
        orientation_score = float(np.clip(1.0 - mean_orientation / 90.0, 0.0, 1.0))
        return {
            "left_ee_position_cm": position_errors["left"],
            "right_ee_position_cm": position_errors["right"],
            "mean_ee_position_cm": mean_position,
            "mean_ee_orientation_deg": mean_orientation,
        }, 0.75 * position_score + 0.25 * orientation_score

    def _direction_score(self, actual_delta: np.ndarray, reference_delta: np.ndarray) -> float:
        normalized_actual = actual_delta / self.state_span
        normalized_reference = reference_delta / self.state_span
        active = np.abs(normalized_reference) >= 1e-4
        if not np.any(active):
            return float(np.max(np.abs(normalized_actual)) < 1e-4)
        same_direction = np.sign(normalized_actual[active]) == np.sign(normalized_reference[active])
        moving = np.abs(normalized_actual[active]) >= 1e-5
        return float(np.mean(same_direction & moving))

    def _amplitude_values(
        self,
        actual_min: np.ndarray,
        actual_max: np.ndarray,
        reference_min: np.ndarray,
        reference_max: np.ndarray,
    ) -> tuple[dict[str, float], float]:
        actual_amplitude = actual_max - actual_min
        reference_amplitude = reference_max - reference_min
        active = reference_amplitude >= self.state_span * 0.01
        if np.any(active):
            ratio = actual_amplitude[active] / reference_amplitude[active]
            coverage = float(np.median(np.clip(ratio, 0.0, 2.0)))
        else:
            coverage = 1.0
        waist_reference = float(reference_amplitude[0])
        waist_coverage = (
            float(np.clip(actual_amplitude[0] / waist_reference, 0.0, 2.0))
            if waist_reference >= self.state_span[0] * 0.01
            else 1.0
        )
        return {
            "amplitude_coverage": coverage,
            "waist_amplitude_coverage": waist_coverage,
        }, float(np.clip(1.0 - abs(coverage - 1.0), 0.0, 1.0))

    def _weighted_score(self, components: dict[str, float]) -> float:
        total_weight = sum(QUALITY_WEIGHTS[name] for name in self.metrics)
        return float(
            100.0 * sum(QUALITY_WEIGHTS[name] * components[name] for name in self.metrics) / total_weight
        )

    def evaluate(
        self,
        step: int,
        controlled_qpos: np.ndarray,
        reference_state: np.ndarray | None = None,
    ) -> dict[str, float]:
        if not 0 <= step < len(self.reference_states):
            raise IndexError(f"Quality step {step} is outside [0, {len(self.reference_states) - 1}]")
        qpos = np.asarray(controlled_qpos, dtype=np.float64)
        if qpos.shape != (len(CONTROLLED_JOINTS),) or not np.isfinite(qpos).all():
            raise ValueError(f"Quality qpos must have shape ({len(CONTROLLED_JOINTS)},)")
        actual = self.mapper.act_state_from_sim(qpos).astype(np.float64)
        reference = (
            self.reference_states[step]
            if reference_state is None
            else np.asarray(reference_state, dtype=np.float64)
        )
        if reference.shape != (len(ACT_STATE_JOINTS),) or not np.isfinite(reference).all():
            raise ValueError(f"Quality reference state must have shape ({len(ACT_STATE_JOINTS)},)")
        self.actual_min = np.minimum(self.actual_min, actual)
        self.actual_max = np.maximum(self.actual_max, actual)
        self.reference_min = np.minimum(self.reference_min, reference)
        self.reference_max = np.maximum(self.reference_max, reference)
        values: dict[str, float] = {}
        actual_components: dict[str, float] = {}
        hold_components: dict[str, float] = {}

        if "pose" in self.metrics:
            pose_values, actual_components["pose"] = self._pose_values(actual, reference)
            _, hold_components["pose"] = self._pose_values(self.initial_state, reference)
            values.update(pose_values)
            values["pose_score"] = actual_components["pose"]
        if "end_effector" in self.metrics:
            ee_values, actual_components["end_effector"] = self._end_effector_values(qpos, reference)
            _, hold_components["end_effector"] = self._end_effector_values(
                self.initial_qpos, reference, self._hold_end_effectors
            )
            values.update(ee_values)
            values["end_effector_score"] = actual_components["end_effector"]
        if "motion_direction" in self.metrics:
            if step == 0:
                actual_components["motion_direction"] = 1.0
                hold_components["motion_direction"] = 1.0
            else:
                reference_delta = reference - self.previous_reference
                actual_components["motion_direction"] = self._direction_score(
                    actual - self.previous_actual, reference_delta
                )
                hold_components["motion_direction"] = self._direction_score(
                    np.zeros_like(actual), reference_delta
                )
            values["motion_direction_score"] = actual_components["motion_direction"]
            values["motion_direction_agreement"] = actual_components["motion_direction"]
        if "amplitude" in self.metrics:
            amplitude_values, actual_components["amplitude"] = self._amplitude_values(
                self.actual_min, self.actual_max, self.reference_min, self.reference_max
            )
            _, hold_components["amplitude"] = self._amplitude_values(
                self.initial_state,
                self.initial_state,
                self.reference_min,
                self.reference_max,
            )
            values.update(amplitude_values)
            values["amplitude_score"] = actual_components["amplitude"]

        values["score"] = self._weighted_score(actual_components)
        values["hold_score"] = self._weighted_score(hold_components)
        values["score_gain_over_hold"] = values["score"] - values["hold_score"]
        self.previous_actual = actual
        self.previous_reference = reference.copy()
        for name in self.metric_names:
            self.history[name].append(float(values[name]))
        return {name: float(values[name]) for name in self.metric_names}

    def trajectory_arrays(self) -> dict[str, np.ndarray]:
        return {
            f"quality_{name}": np.asarray(self.history[name], dtype=np.float32) for name in self.metric_names
        }

    def evaluate_trajectory(self, controlled_qpos: np.ndarray) -> dict[str, np.ndarray]:
        trajectory = np.asarray(controlled_qpos, dtype=np.float64)
        if trajectory.shape != (len(self.reference_states), len(CONTROLLED_JOINTS)):
            raise ValueError(
                f"Quality trajectory must have shape ({len(self.reference_states)}, {len(CONTROLLED_JOINTS)})"
            )
        for step, qpos in enumerate(trajectory):
            self.evaluate(step, qpos)
        return self.trajectory_arrays()

    def summary(self) -> dict[str, object]:
        if not self.history["score"]:
            raise RuntimeError("Quality summary requires at least one evaluated step")
        return {
            "enabled": True,
            "metrics": list(self.metrics),
            "weights": {name: QUALITY_WEIGHTS[name] for name in self.metrics},
            "reference_file": str(self.reference_file),
            "reference_sha256": self.reference_sha256,
            "reference_use": "evaluation_only_not_policy_input",
            "execution": self.execution,
            "max_reference_alignment_delta_ms": self.max_reference_delta_ms,
            "mean": {name: float(np.mean(values)) for name, values in self.history.items()},
            "final": {name: float(values[-1]) for name, values in self.history.items()},
            "minimum_score": float(np.min(self.history["score"])),
            "tensorboard_tags": [f"quality/{name}" for name in self.metric_names],
            "score_contract": {
                "range": [0.0, 100.0],
                "selected_weights_are_renormalized": True,
                "pose_zero_score_nrmse": 0.25,
                "end_effector_zero_score_position_cm": 30.0,
                "end_effector_zero_score_orientation_deg": 90.0,
                "hold_baseline": "simulator_initial_pose",
            },
        }

    def terminal_fragment(self) -> str:
        if not self.history["score"]:
            return ""
        parts = [
            f"quality={self.history['score'][-1]:.1f}",
            f"gain={self.history['score_gain_over_hold'][-1]:+.1f}",
        ]
        if "pose" in self.metrics:
            parts.append(f"pose={self.history['pose_nrmse'][-1]:.3f}")
        if "end_effector" in self.metrics:
            parts.append(f"ee={self.history['mean_ee_position_cm'][-1]:.1f}cm")
        if "motion_direction" in self.metrics:
            parts.append(f"dir={self.history['motion_direction_agreement'][-1] * 100.0:.0f}%")
        if "amplitude" in self.metrics:
            parts.append(f"amp={self.history['amplitude_coverage'][-1] * 100.0:.0f}%")
        return " ".join(parts)


class AsyncMotionQualityEvaluator:
    def __init__(self, evaluator: MotionQualityEvaluator) -> None:
        self.evaluator = evaluator
        self._queue: queue.Queue[tuple[int, np.ndarray, np.ndarray | None] | None] = queue.Queue()
        self._lock = threading.Lock()
        self._latest_step = -1
        self._latest_fragment = ""
        self._error: BaseException | None = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._worker,
            name="w1-simulation-quality",
            daemon=True,
        )
        self._thread.start()

    def _worker(self) -> None:
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    return
                step, qpos, reference_state = item
                self.evaluator.evaluate(step, qpos, reference_state)
                fragment = self.evaluator.terminal_fragment()
                with self._lock:
                    self._latest_step = step
                    self._latest_fragment = fragment
        except BaseException as exc:
            self._error = exc

    def submit(
        self,
        step: int,
        controlled_qpos: np.ndarray,
        reference_state: np.ndarray | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("Async quality evaluator is closed")
        if self._error is not None:
            raise RuntimeError("Async quality evaluator failed") from self._error
        reference = None if reference_state is None else np.asarray(reference_state, dtype=np.float64).copy()
        self._queue.put((step, np.asarray(controlled_qpos, dtype=np.float64).copy(), reference))

    def terminal_fragment(self) -> str:
        with self._lock:
            latest_step = self._latest_step
            fragment = self._latest_fragment
        return f"{fragment} score_step={latest_step}" if fragment else "quality=pending"

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join()
        if self._error is not None:
            raise RuntimeError("Async quality evaluator failed") from self._error

    def trajectory_arrays(self) -> dict[str, np.ndarray]:
        if not self._closed:
            raise RuntimeError("Close async quality evaluator before reading its trajectory")
        return self.evaluator.trajectory_arrays()

    def summary(self) -> dict[str, object]:
        if not self._closed:
            raise RuntimeError("Close async quality evaluator before reading its summary")
        return self.evaluator.summary()
