#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from functools import partial
from typing import Optional

import numpy as np
import rclpy
from end_effector_interfaces.msg import EEFeedback
from rclpy.node import Node
from std_msgs.msg import Float64


HAND_GESTURES = {
    # PC1 order: T_MCP, T_CMC_YAW, IF, MF, RF, LF.
    "normal": [0.0, 70.0, 0.0, 0.0, 0.0, 0.0],
    "normal2": [0.0, 100.0, 0.0, 0.0, 0.0, 0.0],
    "cup": [0.0, 100.0, 35.0, 45.0, 47.0, 37.0],
    "pinch": [65.0, 100.0, 70.0, 75.0, 100.0, 100.0],
    "fist": [100.0, 30.0, 100.0, 100.0, 100.0, 100.0],
    "like": [0.0, 0.0, 100.0, 100.0, 100.0, 100.0],
    "heart": [0.0, 100.0, 60.0, 70.0, 60.0, 60.0],
    "bull": [90.0, 80.0, 0.0, 100.0, 100.0, 0.0],
    "gun": [0.0, 0.0, 0.0, 100.0, 100.0, 100.0],
    "six": [0.0, 0.0, 100.0, 100.0, 100.0, 0.0],
    "one": [100.0, 70.0, 0.0, 100.0, 100.0, 100.0],
    "salute": [100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "ok": [60.0, 90.0, 60.0, 0.0, 0.0, 0.0],
}

HAND_JOINT_NAMES = [
    "T_MCP",
    "T_CMC_YAW",
    "IF_MCP_PITCH",
    "MF_MCP_PITCH",
    "RF_MCP_PITCH",
    "LF_MCP_PITCH",
]


class HandScalarNode(Node):
    def __init__(self):
        super().__init__("hand_scalar_node")

        self.declare_parameter("gesture_name_right", "pinch")
        self.declare_parameter("gesture_name_left", "pinch")
        self.declare_parameter("alpha_ema", 0.98)

        self.declare_parameter("src_topic_right", "/feedback_sim/hand/right")
        self.declare_parameter("dst_topic_right", "/hand/right_scalar")
        self.declare_parameter("src_topic_left", "/feedback_sim/hand/left")
        self.declare_parameter("dst_topic_left", "/hand/left_scalar")

        self.gesture_right = self.get_parameter("gesture_name_right").value
        self.gesture_left = self.get_parameter("gesture_name_left").value
        self.alpha_ema = float(self.get_parameter("alpha_ema").value)

        self.src_topic_right = self.get_parameter("src_topic_right").value
        self.dst_topic_right = self.get_parameter("dst_topic_right").value
        self.src_topic_left = self.get_parameter("src_topic_left").value
        self.dst_topic_left = self.get_parameter("dst_topic_left").value

        self.q_normal = np.array(HAND_GESTURES["normal2"], dtype=np.float32)
        self.hand_cfg = {}
        for hand, gesture_name in (("right", self.gesture_right), ("left", self.gesture_left)):
            if gesture_name not in HAND_GESTURES:
                self.get_logger().warn(
                    f'Unknown gesture "{gesture_name}" for {hand}, fallback to "pinch"'
                )
                gesture_name = "pinch"
            q_gesture = np.array(HAND_GESTURES[gesture_name], dtype=np.float32)
            delta = q_gesture - self.q_normal
            den = float(np.dot(delta, delta)) if np.any(delta) else 1.0
            self.hand_cfg[hand] = {"delta": delta, "den": den, "last_alpha": None}

        self.sub_right = self.create_subscription(
            EEFeedback, self.src_topic_right, partial(self.cb, hand="right"), 10
        )
        self.pub_right = self.create_publisher(Float64, self.dst_topic_right, 10)

        self.sub_left = self.create_subscription(
            EEFeedback, self.src_topic_left, partial(self.cb, hand="left"), 10
        )
        self.pub_left = self.create_publisher(Float64, self.dst_topic_left, 10)

        self.get_logger().info(
            f"HandScalar started | gesture_right={self.gesture_right}, gesture_left={self.gesture_left}, "
            f"alpha_ema={self.alpha_ema}\n"
            f"  Right: src={self.src_topic_right} -> dst={self.dst_topic_right}\n"
            f"  Left : src={self.src_topic_left}  -> dst={self.dst_topic_left}"
        )

    def _extract_positions(self, msg: EEFeedback) -> Optional[np.ndarray]:
        states = list(msg.joint_states)
        if not states:
            return None

        by_name = {state.name: float(state.position) for state in states}
        if all(name in by_name for name in HAND_JOINT_NAMES):
            return np.asarray([by_name[name] for name in HAND_JOINT_NAMES], dtype=np.float32)

        if len(states) >= len(HAND_JOINT_NAMES):
            return np.asarray(
                [state.position for state in states[: len(HAND_JOINT_NAMES)]],
                dtype=np.float32,
            )

        return None

    def cb(self, msg: EEFeedback, hand: str):
        q = self._extract_positions(msg)
        if q is None:
            self.get_logger().warn(
                f"Skip {hand} hand scalar: EEFeedback has fewer than {len(HAND_JOINT_NAMES)} joint states"
            )
            return

        cfg = self.hand_cfg[hand]
        num = float(np.dot(q - self.q_normal, cfg["delta"]))
        alpha = 0.0 if cfg["den"] == 0.0 else num / cfg["den"]
        alpha = max(0.0, min(1.0, alpha))

        last = cfg["last_alpha"]
        if last is None:
            smooth = alpha
        else:
            a = self.alpha_ema
            smooth = (1.0 - a) * last + a * alpha

        cfg["last_alpha"] = smooth
        pub = self.pub_right if hand == "right" else self.pub_left

        out = Float64()
        out.data = float(smooth)
        pub.publish(out)


def main():
    rclpy.init()
    node = HandScalarNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
