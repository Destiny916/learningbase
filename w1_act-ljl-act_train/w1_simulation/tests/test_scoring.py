from __future__ import annotations

import numpy as np
import pytest
from w1_simulation.evaluation.scoring import compute_run_score


def _quality(steps: int, value: float = 1.0) -> dict[str, np.ndarray]:
    return {
        "quality_pose_score": np.full(steps, value),
        "quality_end_effector_score": np.full(steps, value),
        "quality_motion_direction_score": np.full(steps, value),
        "quality_amplitude_score": np.full(steps, value),
    }


def _timing(*, realtime: bool = True, fps: float = 30.0, p95: float = 1000.0 / 30.0, misses: int = 0):
    return {
        "realtime_requested": realtime,
        "effective_fps": fps,
        "p95_cycle_ms": p95,
        "deadline_misses": misses,
        "telemetry_flush_seconds": 0.0,
    }


def _score(
    actual: np.ndarray,
    reference: np.ndarray,
    *,
    quality: dict[str, np.ndarray] | None = None,
    timing: dict[str, object] | None = None,
    smoothness: bool = True,
    realtime: bool = True,
    control_hz: float = 30.0,
):
    steps = len(actual)
    return compute_run_score(
        quality_arrays=quality or _quality(steps),
        actual_states=actual,
        reference_states=reference,
        state_span=np.ones(19),
        timing=timing or _timing(),
        control_hz=control_hz,
        frames=steps,
        joint_limit_violations=0,
        target_step_errors=np.zeros(steps),
        action_pipeline="bridge",
        enable_smoothness=smoothness,
        enable_realtime=realtime,
    )


def test_perfect_motion_and_timing_score_one_hundred() -> None:
    reference = np.linspace(0.0, 0.1, 10)[:, None] * np.ones((1, 19))

    result = _score(reference.copy(), reference)

    assert result.summary["average_score"] == pytest.approx(100.0)
    assert result.summary["components"] == {
        "motion_reproduction": 100.0,
        "smoothness": 100.0,
        "amplitude": 100.0,
        "realtime": 100.0,
    }
    assert result.summary["comparable"] is True


def test_chunk_boundary_jitter_lowers_smoothness_score() -> None:
    reference = np.linspace(0.0, 0.1, 12)[:, None] * np.ones((1, 19))
    jittered = reference.copy()
    jittered[6] += 0.08

    smooth = _score(reference, reference)
    jitter = _score(jittered, reference)

    assert jitter.summary["components"]["smoothness"] < smooth.summary["components"]["smoothness"]
    assert np.min(jitter.trajectory_arrays["score_smoothness"]) < 100.0


def test_slow_rollout_lowers_realtime_and_average_score() -> None:
    states = np.zeros((100, 19))

    result = _score(
        states,
        states,
        timing=_timing(fps=22.8, p95=39.36, misses=99),
        control_hz=60.0,
    )

    assert result.summary["components"]["realtime"] < 60.0
    assert result.summary["average_score"] < 96.0


def test_disabled_components_are_renormalized_and_marked_noncomparable() -> None:
    states = np.zeros((4, 19))
    quality = {"quality_pose_score": np.full(4, 0.5)}

    result = _score(states, states, quality=quality, smoothness=False, realtime=False)

    assert result.summary["average_score"] == pytest.approx(50.0)
    assert result.summary["normalized_weights"] == {"motion_reproduction": 1.0}
    assert result.summary["comparable"] is False


def test_joint_limit_violation_invalidates_official_average() -> None:
    states = np.zeros((3, 19))

    result = compute_run_score(
        quality_arrays=_quality(3),
        actual_states=states,
        reference_states=states,
        state_span=np.ones(19),
        timing=_timing(),
        control_hz=30.0,
        frames=3,
        joint_limit_violations=1,
        target_step_errors=np.zeros(3),
        action_pipeline="bridge",
    )

    assert result.summary["valid"] is False
    assert result.summary["average_score"] is None
    assert result.summary["safety_status"] == "failed"


def test_nonfinite_input_is_rejected() -> None:
    states = np.zeros((3, 19))
    states[1, 2] = np.nan

    with pytest.raises(ValueError, match="finite"):
        _score(states, np.zeros_like(states))
