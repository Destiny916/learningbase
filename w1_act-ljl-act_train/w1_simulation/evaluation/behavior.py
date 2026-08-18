from __future__ import annotations

import bisect
import json
from pathlib import Path

import numpy as np

from w1_simulation.robot.joints import ACT_STATE_JOINTS

KINEMATIC_BEHAVIOR_LIMITS = {
    "body_mae_rad": 0.15,
    "waist_mae_rad": 0.10,
    "left_arm_mae_rad": 0.18,
    "right_arm_mae_rad": 0.18,
    "gripper_mae": 5.0,
    "waist_amplitude_coverage_min": 0.75,
    "waist_amplitude_coverage_max": 1.25,
}


def reference_states(origin: Path, timestamps: np.ndarray) -> tuple[np.ndarray, float]:
    matches = sorted(origin.glob("pose_record_*.json"))
    if len(matches) != 1:
        raise AssertionError(f"Expected one origin pose record, found {len(matches)}")
    frames = json.loads(matches[0].read_text(encoding="utf-8")).get("frames", [])
    if not frames:
        raise AssertionError("Origin pose record is empty")
    pose_timestamps = [float(frame["timestamp"]) for frame in frames]
    rows: list[list[float]] = []
    deltas_ms: list[float] = []
    for timestamp in timestamps:
        position = bisect.bisect_left(pose_timestamps, float(timestamp))
        candidates = [index for index in (position - 1, position) if 0 <= index < len(frames)]
        index = min(candidates, key=lambda item: abs(pose_timestamps[item] - timestamp))
        rows.append([float(frames[index]["data"][name]) for name in ACT_STATE_JOINTS])
        deltas_ms.append(abs(pose_timestamps[index] - timestamp) * 1000.0)
    return np.asarray(rows, dtype=np.float64), float(np.max(deltas_ms))


def behavior_metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    waist_reference_amplitude = float(np.ptp(reference[:, 0]))
    waist_actual_amplitude = float(np.ptp(actual[:, 0]))
    return {
        "body_mae_rad": float(np.mean(np.abs(actual[:, :17] - reference[:, :17]))),
        "waist_mae_rad": float(np.mean(np.abs(actual[:, 0] - reference[:, 0]))),
        "left_arm_mae_rad": float(np.mean(np.abs(actual[:, 1:8] - reference[:, 1:8]))),
        "right_arm_mae_rad": float(np.mean(np.abs(actual[:, 10:17] - reference[:, 10:17]))),
        "gripper_mae": float(np.mean(np.abs(actual[:, 17:] - reference[:, 17:]))),
        "waist_reference_amplitude_rad": waist_reference_amplitude,
        "waist_actual_amplitude_rad": waist_actual_amplitude,
        "waist_amplitude_coverage": waist_actual_amplitude / max(waist_reference_amplitude, 1e-9),
    }


def enforce_kinematic_behavior(metrics: dict[str, float]) -> None:
    for key in ("body_mae_rad", "waist_mae_rad", "left_arm_mae_rad", "right_arm_mae_rad", "gripper_mae"):
        if metrics[key] > KINEMATIC_BEHAVIOR_LIMITS[key]:
            raise AssertionError(
                f"Kinematic behavior regression: {key}={metrics[key]:.6f}, "
                f"limit={KINEMATIC_BEHAVIOR_LIMITS[key]:.6f}"
            )
    coverage = metrics["waist_amplitude_coverage"]
    lower = KINEMATIC_BEHAVIOR_LIMITS["waist_amplitude_coverage_min"]
    upper = KINEMATIC_BEHAVIOR_LIMITS["waist_amplitude_coverage_max"]
    if not lower <= coverage <= upper:
        raise AssertionError(
            f"Kinematic waist amplitude coverage is {coverage:.6f}, expected [{lower}, {upper}]"
        )
