from __future__ import annotations

import numpy as np
import pytest
from w1_simulation.control.bridge import LipoControllerConfig
from w1_simulation.control.processing import BridgeActionChunkProcessor


def test_interpolation_preserves_endpoints_and_uses_original_formula() -> None:
    processor = BridgeActionChunkProcessor(sample_factor=2, full_dim=2)
    actions = np.asarray([[0.0, 10.0], [3.0, 40.0]], dtype=np.float32)

    interpolated = processor.process_chunk(actions, None).processed

    np.testing.assert_allclose(
        interpolated,
        [[0.0, 10.0], [1.0, 20.0], [2.0, 30.0], [3.0, 40.0]],
    )


def test_lipo_default_contract_matches_threshold_scheduler() -> None:
    config = LipoControllerConfig()

    assert config.replan_threshold == 0.5
    assert config.trigger_policy_points == 50
    assert config.trigger_control_points == 100
    assert config.lipo_blend_policy_points == 5
    assert config.lipo_blend_control_points == 10
    assert config.inference_budget_policy_points == 9
    assert config.required_policy_points == 16
    assert config.available_policy_points == 34
    assert config.sample_factor == 2
    assert config.body_dimensions == 17


def test_lipo_rejects_threshold_without_budget_blend_and_margin_capacity() -> None:
    with pytest.raises(ValueError, match="does not leave enough policy points"):
        LipoControllerConfig(replan_threshold=0.1)
