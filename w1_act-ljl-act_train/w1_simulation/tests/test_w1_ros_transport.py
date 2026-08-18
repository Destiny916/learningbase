from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from w1_simulation.robot.commands import BodyPositionCommand, HandPositionCommand, W1PositionCommand
from w1_simulation.robot.joints import HAND_POSITION_JOINTS
from w1_simulation.runtime.w1_ros_transport import W1RosMessageTypes, W1RosPublishers, W1RosTransport


class FakeBodyMessage:
    def __init__(self) -> None:
        self.header = SimpleNamespace(stamp=None)
        self.name = []
        self.position = []


class FakeHandMessage:
    def __init__(self) -> None:
        self.mode = None
        self.name = []
        self.value = []


class FakePublisher:
    def __init__(self) -> None:
        self.messages = []

    def publish(self, message: object) -> None:
        self.messages.append(message)


def test_transport_publishes_atomic_w1_command_to_three_endpoints() -> None:
    body = FakePublisher()
    left = FakePublisher()
    right = FakePublisher()
    transport = W1RosTransport(
        W1RosPublishers(body=body, left_hand=left, right_hand=right),
        W1RosMessageTypes(body=FakeBodyMessage, hand=FakeHandMessage, hand_position_mode=7),
    )
    selected_body_names = ("RIGHT_J2", "WAIST", "LEFT_J4")
    selected_body_positions = np.asarray([0.2, 0.3, 0.4], dtype=np.float64)
    command = W1PositionCommand(
        body=BodyPositionCommand(selected_body_names, selected_body_positions),
        left_hand=HandPositionCommand(HAND_POSITION_JOINTS, np.arange(6, dtype=np.float64)),
        right_hand=HandPositionCommand(HAND_POSITION_JOINTS, np.arange(6, dtype=np.float64) + 10),
    )

    transport.publish(command, stamp="stamp")

    assert body.messages[0].header.stamp == "stamp"
    assert body.messages[0].name == list(selected_body_names)
    assert body.messages[0].position == list(selected_body_positions)
    assert left.messages[0].mode == 7
    assert left.messages[0].name == list(HAND_POSITION_JOINTS)
    assert left.messages[0].value == list(np.arange(6, dtype=np.float64))
    assert right.messages[0].value == list(np.arange(6, dtype=np.float64) + 10)
