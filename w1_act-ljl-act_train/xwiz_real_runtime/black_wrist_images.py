"""Publish deterministic black wrist images for the two unused ACT inputs."""

from __future__ import annotations

import argparse

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class BlackWristPublisher(Node):
    def __init__(self, width: int, height: int, frequency: float) -> None:
        super().__init__("xwiz_black_wrist_images")
        self.width = int(width)
        self.height = int(height)
        self.payload = bytes(self.width * self.height * 3)
        self.left = self.create_publisher(Image, "/camera_l/color/image_rect_raw", 2)
        self.right = self.create_publisher(Image, "/camera_r/color/image_rect_raw", 2)
        self.timer = self.create_timer(1.0 / float(frequency), self.publish_pair)

    def make_image(self, frame_id: str) -> Image:
        message = Image()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = frame_id
        message.height = self.height
        message.width = self.width
        message.encoding = "bgr8"
        message.is_bigendian = 0
        message.step = self.width * 3
        message.data = self.payload
        return message

    def publish_pair(self) -> None:
        self.left.publish(self.make_image("black_left_wrist"))
        self.right.publish(self.make_image("black_right_wrist"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--frequency", type=float, default=10.0)
    args = parser.parse_args()
    rclpy.init()
    node = BlackWristPublisher(args.width, args.height, args.frequency)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
