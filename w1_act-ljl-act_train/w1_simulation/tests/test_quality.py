from __future__ import annotations

import json

import numpy as np
import pytest
from w1_simulation.evaluation.quality import (
    AsyncMotionQualityEvaluator,
    load_reference_states,
    validate_quality_metrics,
)
from w1_simulation.robot.joints import ACT_STATE_JOINTS


class _FakeEvaluator:
    def __init__(self) -> None:
        self.steps: list[int] = []
        self.references: list[np.ndarray | None] = []

    def evaluate(
        self,
        step: int,
        _qpos: np.ndarray,
        reference_state: np.ndarray | None = None,
    ) -> None:
        self.steps.append(step)
        self.references.append(reference_state)

    def terminal_fragment(self) -> str:
        return f"quality={len(self.steps):.1f}" if self.steps else ""

    def trajectory_arrays(self) -> dict[str, np.ndarray]:
        return {"quality_score": np.asarray(self.steps, dtype=np.float32)}

    def summary(self) -> dict[str, object]:
        return {"steps": list(self.steps)}


def test_quality_metric_selection_is_canonical_and_rejects_unknown_values() -> None:
    assert validate_quality_metrics(["amplitude", "pose", "pose"]) == ("pose", "amplitude")
    with pytest.raises(ValueError, match="Unknown quality metrics"):
        validate_quality_metrics(["smoothness"])


def test_reference_states_are_interpolated_at_control_timestamps(tmp_path) -> None:
    first = dict.fromkeys(ACT_STATE_JOINTS, 0.0)
    second = dict.fromkeys(ACT_STATE_JOINTS, 2.0)
    path = tmp_path / "pose_record_test.json"
    path.write_text(
        json.dumps(
            {
                "frames": [
                    {"timestamp": 10.0, "data": first},
                    {"timestamp": 12.0, "data": second},
                ]
            }
        ),
        encoding="utf-8",
    )

    states, source, max_delta_ms = load_reference_states(tmp_path, np.asarray([10.0, 11.0, 12.0]))

    np.testing.assert_allclose(states[:, 0], [0.0, 1.0, 2.0])
    assert source == path
    assert max_delta_ms == pytest.approx(1000.0)


def test_reference_loader_rejects_non_monotonic_timestamps(tmp_path) -> None:
    state = dict.fromkeys(ACT_STATE_JOINTS, 0.0)
    (tmp_path / "pose_record_test.json").write_text(
        json.dumps(
            {
                "frames": [
                    {"timestamp": 10.0, "data": state},
                    {"timestamp": 10.0, "data": state},
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        load_reference_states(tmp_path, np.asarray([10.0]))


def test_async_quality_evaluator_drains_every_submitted_step() -> None:
    evaluator = _FakeEvaluator()
    async_evaluator = AsyncMotionQualityEvaluator(evaluator)  # type: ignore[arg-type]
    for step in range(5):
        async_evaluator.submit(step, np.zeros(29))

    async_evaluator.close()

    np.testing.assert_array_equal(async_evaluator.trajectory_arrays()["quality_score"], np.arange(5))
    assert async_evaluator.summary() == {"steps": [0, 1, 2, 3, 4]}


def test_async_quality_evaluator_forwards_selected_frame_reference() -> None:
    evaluator = _FakeEvaluator()
    async_evaluator = AsyncMotionQualityEvaluator(evaluator)  # type: ignore[arg-type]
    reference = np.arange(19, dtype=np.float64)

    async_evaluator.submit(0, np.zeros(29), reference)
    reference[:] = -1.0
    async_evaluator.close()

    np.testing.assert_array_equal(evaluator.references[0], np.arange(19, dtype=np.float64))
