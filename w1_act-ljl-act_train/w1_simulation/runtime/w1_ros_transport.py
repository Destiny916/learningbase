from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from w1_simulation.robot.commands import W1PositionCommand


@dataclass(frozen=True)
class W1RosMessageTypes:
    body: type
    hand: type
    hand_position_mode: Any


@dataclass(frozen=True)
class W1RosPublishers:
    body: Any
    left_hand: Any
    right_hand: Any


class W1RosTransport:
    def __init__(self, publishers: W1RosPublishers, message_types: W1RosMessageTypes) -> None:
        self.publishers = publishers
        self.message_types = message_types

    def body_message(self, command: W1PositionCommand, stamp: Any | None = None) -> Any:
        message = self.message_types.body()
        if stamp is not None and hasattr(message, "header"):
            message.header.stamp = stamp
        message.name = list(command.body.name)
        message.position = [float(value) for value in command.body.position]
        return message

    def hand_message(self, command: W1PositionCommand, side: str) -> Any:
        if side not in {"left", "right"}:
            raise ValueError(f"Unknown W1 hand side: {side}")
        selected = command.left_hand if side == "left" else command.right_hand
        message = self.message_types.hand()
        message.mode = self.message_types.hand_position_mode
        message.name = list(selected.name)
        message.value = [float(value) for value in selected.value]
        return message

    def publish(self, command: W1PositionCommand, stamp: Any | None = None) -> None:
        self.publishers.body.publish(self.body_message(command, stamp))
        self.publishers.left_hand.publish(self.hand_message(command, "left"))
        self.publishers.right_hand.publish(self.hand_message(command, "right"))
