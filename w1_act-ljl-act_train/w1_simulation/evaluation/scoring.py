from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

RUN_SCORE_COMPONENTS = ("motion_reproduction", "smoothness", "amplitude", "realtime")
RUN_SCORE_WEIGHTS = {
    "motion_reproduction": 0.70,
    "smoothness": 0.10,
    "amplitude": 0.10,
    "realtime": 0.10,
}
MOTION_REPRODUCTION_WEIGHTS = {
    "quality_pose_score": 0.40,
    "quality_end_effector_score": 0.30,
    "quality_motion_direction_score": 0.20,
}
SMOOTHNESS_VELOCITY_ZERO_SCORE_NRMSE = 0.05
SMOOTHNESS_ACCELERATION_ZERO_SCORE_NRMSE = 0.02


@dataclass(frozen=True)
class RunScoreResult:
    summary: dict[str, object]
    trajectory_arrays: dict[str, np.ndarray]


def _finite_vector(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a non-empty finite vector")
    return array


def _finite_states(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 19 or not len(array) or not np.isfinite(array).all():
        raise ValueError(f"{name} must have shape (steps, 19) and contain only finite values")
    return array


def _motion_reproduction_series(quality_arrays: Mapping[str, np.ndarray]) -> np.ndarray | None:
    selected = {
        name: _finite_vector(quality_arrays[name], name)
        for name in MOTION_REPRODUCTION_WEIGHTS
        if name in quality_arrays
    }
    if not selected:
        return None
    lengths = {len(values) for values in selected.values()}
    if len(lengths) != 1:
        raise ValueError("Motion-reproduction quality arrays must have the same length")
    total_weight = sum(MOTION_REPRODUCTION_WEIGHTS[name] for name in selected)
    score = sum(MOTION_REPRODUCTION_WEIGHTS[name] * values for name, values in selected.items())
    return np.clip(score / total_weight * 100.0, 0.0, 100.0)


def _smoothness_series(
    actual_states: np.ndarray,
    reference_states: np.ndarray,
    state_span: np.ndarray,
) -> np.ndarray:
    actual = _finite_states(actual_states, "actual_states")
    reference = _finite_states(reference_states, "reference_states")
    if actual.shape != reference.shape:
        raise ValueError("Actual and reference states must have identical shapes")
    span = _finite_vector(state_span, "state_span")
    if span.shape != (19,) or np.any(span <= 0.0):
        raise ValueError("state_span must contain 19 positive values")

    actual = actual / span
    reference = reference / span
    score = np.full(len(actual), 100.0, dtype=np.float64)
    if len(actual) >= 2:
        velocity_error = np.sqrt(
            np.mean(np.square(np.diff(actual, axis=0) - np.diff(reference, axis=0)), axis=1)
        )
        velocity_score = np.clip(
            1.0 - velocity_error / SMOOTHNESS_VELOCITY_ZERO_SCORE_NRMSE,
            0.0,
            1.0,
        )
        score[1:] = velocity_score * 100.0
    if len(actual) >= 3:
        acceleration_error = np.sqrt(
            np.mean(
                np.square(np.diff(actual, n=2, axis=0) - np.diff(reference, n=2, axis=0)),
                axis=1,
            )
        )
        acceleration_score = np.clip(
            1.0 - acceleration_error / SMOOTHNESS_ACCELERATION_ZERO_SCORE_NRMSE,
            0.0,
            1.0,
        )
        score[2:] = (0.4 * velocity_score[1:] + 0.6 * acceleration_score) * 100.0
    return score


def _realtime_score(timing: Mapping[str, object], control_hz: float, frames: int) -> float:
    if control_hz <= 0.0 or frames <= 0:
        raise ValueError("control_hz and frames must be positive")
    effective_fps = float(timing["effective_fps"])
    p95_cycle_ms = float(timing["p95_cycle_ms"])
    telemetry_flush_seconds = float(timing["telemetry_flush_seconds"])
    deadline_misses = int(timing["deadline_misses"])
    values = np.asarray(
        [effective_fps, p95_cycle_ms, telemetry_flush_seconds, float(deadline_misses)],
        dtype=np.float64,
    )
    if not np.isfinite(values).all() or np.any(values < 0.0) or deadline_misses > frames:
        raise ValueError("Realtime timing values are invalid")
    target_cycle_ms = 1000.0 / control_hz
    fps_score = np.clip(effective_fps / control_hz, 0.0, 1.0)
    cycle_score = np.clip(target_cycle_ms / max(p95_cycle_ms, target_cycle_ms), 0.0, 1.0)
    deadline_score = np.clip(1.0 - deadline_misses / frames, 0.0, 1.0)
    flush_score = np.clip(1.0 - telemetry_flush_seconds / 5.0, 0.0, 1.0)
    return float(100.0 * (0.45 * fps_score + 0.30 * cycle_score + 0.20 * deadline_score + 0.05 * flush_score))


def compute_run_score(
    *,
    quality_arrays: Mapping[str, np.ndarray],
    actual_states: np.ndarray,
    reference_states: np.ndarray,
    state_span: np.ndarray,
    timing: Mapping[str, object],
    control_hz: float,
    frames: int,
    joint_limit_violations: int,
    target_step_errors: np.ndarray,
    action_pipeline: str,
    enable_smoothness: bool = True,
    enable_realtime: bool = True,
) -> RunScoreResult:
    if action_pipeline not in {"raw", "bridge"}:
        raise ValueError(f"Unknown action pipeline: {action_pipeline}")
    target_errors = np.asarray(target_step_errors)
    if target_errors.shape != (frames,) or not np.isfinite(target_errors).all():
        raise ValueError("target_step_errors must be a finite vector matching frames")
    aligned = action_pipeline == "raw" or bool(np.all(np.isin(target_errors, (-1, 0))))
    safety_pass = joint_limit_violations == 0 and aligned

    arrays: dict[str, np.ndarray] = {}
    component_values: dict[str, float | None] = dict.fromkeys(RUN_SCORE_COMPONENTS)
    disabled_reasons: dict[str, str] = {}

    motion = _motion_reproduction_series(quality_arrays)
    if motion is not None:
        if len(motion) != frames:
            raise ValueError("Motion-reproduction series must match frames")
        arrays["score_motion_reproduction"] = motion.astype(np.float32)
        component_values["motion_reproduction"] = float(np.mean(motion))
    else:
        disabled_reasons["motion_reproduction"] = "no pose, end-effector, or direction metric selected"

    if enable_smoothness:
        smoothness = _smoothness_series(actual_states, reference_states, state_span)
        if len(smoothness) != frames:
            raise ValueError("Smoothness series must match frames")
        arrays["score_smoothness"] = smoothness.astype(np.float32)
        component_values["smoothness"] = float(np.mean(smoothness))
    else:
        disabled_reasons["smoothness"] = "disabled by configuration"

    if "quality_amplitude_score" in quality_arrays:
        amplitude = np.clip(
            _finite_vector(quality_arrays["quality_amplitude_score"], "quality_amplitude_score")
            * 100.0,
            0.0,
            100.0,
        )
        if len(amplitude) != frames:
            raise ValueError("Amplitude series must match frames")
        arrays["score_amplitude"] = amplitude.astype(np.float32)
        component_values["amplitude"] = float(np.mean(amplitude))
    else:
        disabled_reasons["amplitude"] = "amplitude quality metric not selected"

    realtime_requested = bool(timing.get("realtime_requested", False))
    if enable_realtime and realtime_requested:
        realtime_value = _realtime_score(timing, control_hz, frames)
        arrays["score_realtime"] = np.asarray([realtime_value], dtype=np.float32)
        component_values["realtime"] = realtime_value
    elif not enable_realtime:
        disabled_reasons["realtime"] = "disabled by configuration"
    else:
        disabled_reasons["realtime"] = "rollout was not requested in realtime mode"

    selected = tuple(name for name in RUN_SCORE_COMPONENTS if component_values[name] is not None)
    selected_weight = sum(RUN_SCORE_WEIGHTS[name] for name in selected)
    normalized_weights = {
        name: RUN_SCORE_WEIGHTS[name] / selected_weight for name in selected
    } if selected_weight else {}
    average_score = (
        float(sum(normalized_weights[name] * float(component_values[name]) for name in selected))
        if selected and safety_pass
        else None
    )
    tensorboard_tags = [f"score/component/{name}" for name in selected]
    tensorboard_tags.append("score/safety_pass")
    if average_score is not None:
        tensorboard_tags.append("score/average")

    return RunScoreResult(
        summary={
            "enabled": bool(selected),
            "valid": average_score is not None,
            "average_score": average_score,
            "components": component_values,
            "base_weights": dict(RUN_SCORE_WEIGHTS),
            "normalized_weights": normalized_weights,
            "selected_components": list(selected),
            "disabled_reasons": disabled_reasons,
            "requested": {
                "smoothness": enable_smoothness,
                "realtime": enable_realtime,
            },
            "comparable": safety_pass and set(selected) == set(RUN_SCORE_COMPONENTS),
            "safety_status": "passed" if safety_pass else "failed",
            "safety_checks": {
                "joint_limit_violations": int(joint_limit_violations),
                "timestamp_alignment_passed": aligned,
            },
            "execution": "offline_post_rollout",
            "tensorboard_tags": tensorboard_tags,
            "contract": {
                "range": [0.0, 100.0],
                "formula": "motion_reproduction*0.70 + smoothness*0.10 + amplitude*0.10 + realtime*0.10",
                "disabled_component_weights_are_renormalized": True,
                "motion_reproduction_sources": dict(MOTION_REPRODUCTION_WEIGHTS),
                "smoothness_velocity_zero_score_nrmse": SMOOTHNESS_VELOCITY_ZERO_SCORE_NRMSE,
                "smoothness_acceleration_zero_score_nrmse": SMOOTHNESS_ACCELERATION_ZERO_SCORE_NRMSE,
                "realtime_sources": {
                    "effective_fps": 0.45,
                    "p95_cycle_ms": 0.30,
                    "deadline_misses": 0.20,
                    "telemetry_flush_seconds": 0.05,
                },
                "official_score_requires_safety_pass": True,
            },
        },
        trajectory_arrays=arrays,
    )
