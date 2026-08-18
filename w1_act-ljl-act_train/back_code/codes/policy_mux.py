#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import threading
from typing import List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

# === 消息类型选择 ===
# 默认：JointState；如果你们有 dex_common_msgs，请把下一行取消注释并注释掉 JointState 那行
# from dex_common_msgs.msg import JointPositionControl as JointMsg
from sensor_msgs.msg import JointState as JointMsg  # 默认回退


class PolicyMux(Node):

    def __init__(self):
        super().__init__('policy_mux')

        # 参数
        self.declare_parameter('source_a', '/policy_mux/A')
        self.declare_parameter('source_b', '/policy_mux/B')
        self.declare_parameter('output', '/control/joint_position')
        self.declare_parameter('blend_ms', 300)  # 软切换时长
        self.declare_parameter('loop_hz', 100.0)  # 发布频率

        self.src_a = self.get_parameter(
            'source_a').get_parameter_value().string_value
        self.src_b = self.get_parameter(
            'source_b').get_parameter_value().string_value
        self.out_t = self.get_parameter(
            'output').get_parameter_value().string_value
        self.blend_ms = int(
            self.get_parameter('blend_ms').get_parameter_value().integer_value
            or 300)
        self.loop_hz = float(
            self.get_parameter('loop_hz').get_parameter_value().double_value or
            100.0)

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,  # ← 关键：RELIABLE
            durability=DurabilityPolicy.VOLATILE,
        )

        self.sub_a = self.create_subscription(JointMsg, self.src_a, self.cb_a,
                                              qos)
        self.sub_b = self.create_subscription(JointMsg, self.src_b, self.cb_b,
                                              qos)
        self.pub = self.create_publisher(JointMsg, self.out_t, qos)

        self.last_a: Optional[JointMsg] = None
        self.last_b: Optional[JointMsg] = None
        self.last_out_pos: Optional[List[float]] = None

        self.active = 'A'  # 当前激活的策略
        self.lock = threading.Lock()

        # blending 状态
        self.blending = False
        self.blend_from: Optional[List[float]] = None
        self.blend_start_ns = 0
        self.blend_dur_ns = int(self.blend_ms * 1e6)

        self.timer = self.create_timer(1.0 / max(self.loop_hz, 1.0),
                                       self.on_timer)

        # 键盘切换线程
        threading.Thread(target=self.stdin_loop, daemon=True).start()

        self.get_logger().info(
            f"PolicyMux started. A={self.src_a}, B={self.src_b}, out={self.out_t}, "
            f"blend_ms={self.blend_ms}, loop_hz={self.loop_hz}")
        self.get_logger().info(
            "Press Enter to toggle A↔B, 'a'/'b' to force, or 'q' to quit.")

    def now_ns(self) -> int:
        return self.get_clock().now().nanoseconds

    # ---------- 读取/构造消息的适配 ----------
    def read_fields(self, msg: JointMsg):
        # JointState 与 JointPositionControl 都常见有 name/position
        names = list(getattr(msg, 'name', []))
        pos = list(getattr(msg, 'position', []))
        return names, pos

    def build_msg(self, names: List[str], pos: List[float]) -> JointMsg:
        out = JointMsg()
        if hasattr(out, 'header'):
            out.header.stamp = self.get_clock().now().to_msg()
        if hasattr(out, 'name'):
            out.name = list(names)
        if hasattr(out, 'position'):
            out.position = list(pos)
        return out

    # ---------- 回调 ----------
    def cb_a(self, msg: JointMsg):
        with self.lock:
            self.last_a = msg

    def cb_b(self, msg: JointMsg):
        with self.lock:
            self.last_b = msg

    # ---------- 切换与软过渡 ----------
    def start_blend_from_last_out(self):
        with self.lock:
            if self.last_out_pos is not None:
                self.blend_from = list(self.last_out_pos)
            else:
                src = self.last_a if self.active == 'A' else self.last_b
                if src is not None:
                    _, pos = self.read_fields(src)
                    self.blend_from = list(pos)
                else:
                    self.blend_from = None
            self.blend_start_ns = self.now_ns()
            self.blending = True

    def toggle_to(self, target: str):
        with self.lock:
            if target not in ('A', 'B') or target == self.active:
                return
            candidate = self.last_b if target == 'B' else self.last_a
            if candidate is None:
                self.get_logger().warn(
                    f"Target {target} not warmed yet; keep using {self.active} until first msg arrives."
                )
                # 延迟切换：定时器将持续检查，一旦有帧就能切换
                return
            self.active = target
            self.get_logger().info(
                f"Switching active policy -> {self.active} (soft handoff {self.blend_ms} ms)"
            )
            self.start_blend_from_last_out()

    def toggle(self):
        self.toggle_to('B' if self.active == 'A' else 'A')

    # ---------- 定时发布 ----------
    def on_timer(self):
        with self.lock:
            src_msg = self.last_a if self.active == 'A' else self.last_b
            other_msg = self.last_b if self.active == 'A' else self.last_a

            if src_msg is None:
                src_msg = other_msg
                if src_msg is None:
                    return  # 两个都没有，跳过

            names, target = self.read_fields(src_msg)

            if self.blending and self.blend_from is not None \
               and len(self.blend_from) == len(target) and self.blend_dur_ns > 0:
                t = self.now_ns() - self.blend_start_ns
                if t >= self.blend_dur_ns:
                    self.blending = False
                    out_pos = target
                else:
                    alpha = float(t) / float(self.blend_dur_ns)
                    out_pos = [(1.0 - alpha) * a + alpha * b
                               for a, b in zip(self.blend_from, target)]
            else:
                self.blending = False
                out_pos = target

        out_msg = self.build_msg(names, out_pos)
        self.pub.publish(out_msg)

        with self.lock:
            self.last_out_pos = list(out_pos)

    # ---------- 键盘线程 ----------
    def stdin_loop(self):
        while rclpy.ok():
            line = sys.stdin.readline()
            if not line:
                rclpy.shutdown(context=self.context)
                break
            line = line.strip().lower()
            if line in ("", "toggle"):
                self.toggle()
            elif line == "a":
                self.toggle_to('A')
            elif line == "b":
                self.toggle_to('B')
            elif line == "q":
                self.get_logger().info("Quit requested.")
                rclpy.shutdown(context=self.context)
                break


def main():
    rclpy.init()
    node = PolicyMux()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
