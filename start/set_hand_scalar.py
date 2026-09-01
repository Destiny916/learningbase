"""Set W1 Linker L6 hands from openness scalars only.

The CLI accepts only left/right openness values (0=closed, 100=open). The
six-joint command is derived internally from the verified ACT endpoints.
"""

from __future__ import annotations

import argparse
import time

import rclpy
from end_effector_interfaces.msg import EEJointControl, EEJointControlMode
from rclpy.node import Node

from xwiz_real_runtime.runtime import (
    LEFT_CLOSED,
    LEFT_OPEN,
    RIGHT_CLOSED,
    RIGHT_OPEN,
    hand_command_from_openness,
)


JOINT_NAMES = [
    "T_MCP", "T_CMC_YAW", "IF_MCP_PITCH", "MF_MCP_PITCH",
    "RF_MCP_PITCH", "LF_MCP_PITCH",
]


class ScalarPublisher(Node):
    def __init__(self) -> None:
        super().__init__("w1_hand_scalar_command")
        self.left_pub = self.create_publisher(EEJointControl, "/control/ee/left", 10)
        self.right_pub = self.create_publisher(EEJointControl, "/control/ee/right", 10)

    def publish_scalar(self, side: str, scalar: float) -> tuple[float, ...]:
        if side == "left":
            values = hand_command_from_openness(scalar, LEFT_CLOSED, LEFT_OPEN)
            publisher = self.left_pub
        else:
            values = hand_command_from_openness(scalar, RIGHT_CLOSED, RIGHT_OPEN)
            publisher = self.right_pub
        message = EEJointControl()
        message.mode = EEJointControlMode.POSITION
        message.joint_names = JOINT_NAMES
        message.values = list(values)
        publisher.publish(message)
        return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=float, required=True, help="left openness scalar")
    parser.add_argument("--right", type=float, required=True, help="right openness scalar")
    parser.add_argument("--seconds", type=float, default=2.0)
    args = parser.parse_args()
    rclpy.init()
    node = ScalarPublisher()
    left = node.publish_scalar("left", args.left)
    right = node.publish_scalar("right", args.right)
    end = time.monotonic() + max(0.2, args.seconds)
    while time.monotonic() < end:
        node.publish_scalar("left", args.left)
        node.publish_scalar("right", args.right)
        rclpy.spin_once(node, timeout_sec=0.1)
    print(f"SCALAR_SENT left={args.left:g} right={args.right:g}")
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
