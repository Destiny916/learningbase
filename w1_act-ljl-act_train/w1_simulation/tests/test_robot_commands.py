from __future__ import annotations

import numpy as np
import pytest
from w1_simulation.robot.commands import (
    BODY_POSITION_TOPIC,
    LEFT_HAND_POSITION_TOPIC,
    RIGHT_HAND_POSITION_TOPIC,
    BodyPositionCommand,
    HandPositionCommand,
    W1ControlEndpoints,
    W1PositionCommand,
    position_command_contract,
)
from w1_simulation.robot.joints import BODY_JOINTS, HAND_POSITION_JOINTS


def _body() -> BodyPositionCommand:
    return BodyPositionCommand(BODY_JOINTS, np.zeros(len(BODY_JOINTS)))


def _hand() -> HandPositionCommand:
    return HandPositionCommand(HAND_POSITION_JOINTS, np.zeros(len(HAND_POSITION_JOINTS)))


def test_standard_w1_position_command_matches_robot_control_endpoints() -> None:
    command = W1PositionCommand(_body(), _hand(), _hand())
    contract = position_command_contract()

    assert command.body.name == BODY_JOINTS
    assert command.left_hand.name == HAND_POSITION_JOINTS
    assert command.right_hand.name == HAND_POSITION_JOINTS
    assert contract["body"]["topic"] == BODY_POSITION_TOPIC
    assert contract["left_hand"]["topic"] == LEFT_HAND_POSITION_TOPIC
    assert contract["right_hand"]["topic"] == RIGHT_HAND_POSITION_TOPIC
    assert contract["body"]["fields"] == ["name", "position"]
    assert contract["left_hand"]["fields"] == ["mode", "name", "value"]
    assert contract["right_hand"]["fields"] == ["mode", "name", "value"]


def test_position_commands_reject_unsupported_duplicate_shape_non_finite_and_hand_range() -> None:
    reordered = BodyPositionCommand(tuple(reversed(BODY_JOINTS)), np.zeros(len(BODY_JOINTS)))
    assert reordered.name == tuple(reversed(BODY_JOINTS))
    with pytest.raises(ValueError, match="unsupported"):
        BodyPositionCommand(("ANKLE",), np.zeros(1))
    with pytest.raises(ValueError, match="duplicate"):
        BodyPositionCommand(("WAIST", "WAIST"), np.zeros(2))
    with pytest.raises(ValueError, match="shape"):
        BodyPositionCommand(BODY_JOINTS, np.zeros(len(BODY_JOINTS) - 1))
    invalid_body = np.zeros(len(BODY_JOINTS))
    invalid_body[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        BodyPositionCommand(BODY_JOINTS, invalid_body)
    with pytest.raises(ValueError, match=r"\[0, 100\]"):
        HandPositionCommand(HAND_POSITION_JOINTS, np.asarray([0, 0, 0, 0, 0, 101]))
    with pytest.raises(ValueError, match="Unsupported"):
        HandPositionCommand(HAND_POSITION_JOINTS, np.zeros(6), mode="TORQUE")


def test_position_command_vectors_are_copied_and_read_only() -> None:
    values = np.zeros(len(BODY_JOINTS))
    command = BodyPositionCommand(BODY_JOINTS, values)
    values[0] = 1.0

    assert command.position[0] == 0.0
    with pytest.raises(ValueError, match="read-only"):
        command.position[0] = 1.0


def test_control_endpoints_must_be_distinct_absolute_topics() -> None:
    with pytest.raises(ValueError, match="distinct absolute"):
        W1ControlEndpoints(body="control/joint_position")
    with pytest.raises(ValueError, match="distinct absolute"):
        W1ControlEndpoints(left_hand=BODY_POSITION_TOPIC)
