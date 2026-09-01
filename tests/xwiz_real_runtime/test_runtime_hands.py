import numpy as np
import pytest

from xwiz_real_runtime.runtime import (
    ACT_DEFAULT_HAND_6,
    LEFT_CLOSED,
    LEFT_OPEN,
    RIGHT_CLOSED,
    RIGHT_OPEN,
    RuntimeContractError,
    action_to_commands,
    hand_command_from_openness,
    validate_hands_ready,
)


def test_act_default_hands_accept_percentage_feedback_at_default():
    expected = np.asarray(ACT_DEFAULT_HAND_6, dtype=np.float64)
    assert validate_hands_ready(expected, expected) == 0.0


def test_act_default_hands_reject_non_default_feedback():
    expected = np.asarray(ACT_DEFAULT_HAND_6, dtype=np.float64)
    bad = expected.copy()
    bad[1] = 10.0
    with pytest.raises(RuntimeContractError, match="left hand is not at ACT default pose"):
        validate_hands_ready(bad, expected)


def test_act_default_hands_accept_percentage_feedback():
    expected = np.asarray(ACT_DEFAULT_HAND_6, dtype=np.float64)
    assert validate_hands_ready(expected, expected) == 0.0


def test_openness_below_95_is_clipped_to_closed_endpoint():
    action = np.zeros(19, dtype=np.float64)
    action[17] = 94.999
    action[18] = 95.0

    command = action_to_commands(action)

    np.testing.assert_allclose(command.left_hand, LEFT_CLOSED)
    np.testing.assert_allclose(
        command.right_hand,
        hand_command_from_openness(95.0, RIGHT_CLOSED, RIGHT_OPEN),
    )
