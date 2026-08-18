from __future__ import annotations

import pytest
from w1_simulation.evaluation.verification import KINEMATIC_BEHAVIOR_LIMITS, _enforce_kinematic_behavior


def _passing_metrics() -> dict[str, float]:
    return {
        "body_mae_rad": 0.10,
        "waist_mae_rad": 0.05,
        "left_arm_mae_rad": 0.09,
        "right_arm_mae_rad": 0.14,
        "gripper_mae": 1.0,
        "waist_amplitude_coverage": 1.0,
    }


def test_kinematic_behavior_gate_accepts_validated_full_run_metrics() -> None:
    _enforce_kinematic_behavior(_passing_metrics())


@pytest.mark.parametrize(
    "key",
    ("body_mae_rad", "waist_mae_rad", "left_arm_mae_rad", "right_arm_mae_rad", "gripper_mae"),
)
def test_kinematic_behavior_gate_rejects_joint_error_regression(key: str) -> None:
    metrics = _passing_metrics()
    metrics[key] = KINEMATIC_BEHAVIOR_LIMITS[key] + 1e-3

    with pytest.raises(AssertionError, match=key):
        _enforce_kinematic_behavior(metrics)


@pytest.mark.parametrize("coverage", (0.74, 1.26))
def test_kinematic_behavior_gate_rejects_waist_amplitude_regression(coverage: float) -> None:
    metrics = _passing_metrics()
    metrics["waist_amplitude_coverage"] = coverage

    with pytest.raises(AssertionError, match="waist amplitude"):
        _enforce_kinematic_behavior(metrics)
