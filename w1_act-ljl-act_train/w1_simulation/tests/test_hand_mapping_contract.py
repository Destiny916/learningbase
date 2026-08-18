from __future__ import annotations

import copy

import numpy as np
import pytest
from w1_simulation.robot.mapping import ActHandGestureConfig, ActJointMapper
from w1_simulation.w1_profile import DEFAULT_PROFILE

EXPECTED_ENDPOINTS = {
    "left": {
        "0": [0.0, 100.0, 35.0, 45.0, 47.0, 37.0],
        "100": [0.0, 70.0, 0.0, 0.0, 0.0, 0.0],
    },
    "right": {
        "0": [65.0, 100.0, 70.0, 75.0, 100.0, 100.0],
        "100": [0.0, 70.0, 0.0, 0.0, 0.0, 0.0],
    },
}


def test_default_hand_mapping_json_has_exact_six_dimensional_endpoints() -> None:
    config = ActHandGestureConfig.from_dict(DEFAULT_PROFILE.hands)
    payload = config.as_dict()

    assert payload["mapping"] == "hand_command_thumb1_mcp_thumb2_cmc_to_urdf_range"

    for side in ("left", "right"):
        for scalar in ("0", "100"):
            assert payload[side][scalar] == EXPECTED_ENDPOINTS[side][scalar]
            assert len(payload[side][scalar]) == 6


@pytest.mark.parametrize("side", ("left", "right"))
def test_hand_mapping_midpoint_is_exact_linear_interpolation(side: str) -> None:
    config = ActHandGestureConfig.from_dict(DEFAULT_PROFILE.hands)
    payload = config.as_dict()

    midpoint = ActJointMapper._gesture_percent(
        50.0,
        tuple(payload[side]["0"]),
        tuple(payload[side]["100"]),
    )

    expected = (np.asarray(EXPECTED_ENDPOINTS[side]["0"]) + EXPECTED_ENDPOINTS[side]["100"]) / 2.0
    np.testing.assert_array_equal(midpoint, expected)


@pytest.mark.parametrize("side", ("left", "right"))
@pytest.mark.parametrize("scalar", (0.0, 100.0))
def test_hand_mapping_scalar_endpoints_are_exact(side: str, scalar: float) -> None:
    config = ActHandGestureConfig.from_dict(DEFAULT_PROFILE.hands)
    payload = config.as_dict()

    actual = ActJointMapper._gesture_percent(
        scalar,
        tuple(payload[side]["0"]),
        tuple(payload[side]["100"]),
    )

    np.testing.assert_array_equal(actual, EXPECTED_ENDPOINTS[side][str(int(scalar))])


@pytest.mark.parametrize("length", (0, 5, 7))
def test_hand_mapping_rejects_endpoint_with_invalid_length(length: int) -> None:
    payload = copy.deepcopy(EXPECTED_ENDPOINTS)
    payload["left"]["0"] = [0.0] * length

    with pytest.raises(ValueError, match="exactly six"):
        ActHandGestureConfig.from_dict(payload)


@pytest.mark.parametrize("invalid", (-0.01, 100.01))
def test_hand_mapping_rejects_endpoint_value_outside_percent_range(invalid: float) -> None:
    payload = copy.deepcopy(EXPECTED_ENDPOINTS)
    payload["right"]["100"][3] = invalid

    with pytest.raises(ValueError, match=r"\[0, 100\]"):
        ActHandGestureConfig.from_dict(payload)


def test_hand_mapping_rejects_nan_endpoint_value() -> None:
    payload = copy.deepcopy(EXPECTED_ENDPOINTS)
    payload["left"]["100"][2] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        ActHandGestureConfig.from_dict(payload)
