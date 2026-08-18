#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def publish_switch_topic(policy_id: str, topic_name: str, wait_s: float) -> int:
    rclpy.init()
    node = Node("switch_policy_cli")
    pub = node.create_publisher(String, topic_name, 10)

    msg = String()
    msg.data = policy_id

    deadline = time.time() + max(wait_s, 0.1)
    while pub.get_subscription_count() == 0 and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)

    pub.publish(msg)
    node.get_logger().info(f"Published switch request: {policy_id} -> {topic_name}")

    end_time = time.time() + max(wait_s, 0.1)
    while time.time() < end_time:
        rclpy.spin_once(node, timeout_sec=0.05)

    node.destroy_node()
    rclpy.shutdown()
    return 0


def call_switch_service(policy_id: str, service_name: str, service_type: str, timeout_s: float) -> int:
    request = f'{{policy_id: "{policy_id}"}}'
    cmd = [
        'ros2', 'service', 'call', service_name, service_type, request,
    ]
    try:
        completed = subprocess.run(cmd, check=False, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print(f"Timed out calling service {service_name}", file=sys.stderr)
        return 1
    return completed.returncode


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Switch active policy on the rollout client')
    parser.add_argument('policy_id', help='Configured policy id, e.g. social_default')
    parser.add_argument('--mode', choices=('service', 'topic'), default='service',
                        help='service is the default path; topic is fallback only')
    parser.add_argument('--topic', default='/w1_act/switch_policy_fallback', help='Fallback topic used in topic mode')
    parser.add_argument('--service', default='/w1_act/switch_policy', help='Service name used in service mode')
    parser.add_argument('--service-type', default='w1_act_interfaces/srv/SwitchPolicy',
                        help='ROS service type used in service mode')
    parser.add_argument('--wait', type=float, default=0.5, help='Wait time after publish or before service timeout')
    parser.add_argument('--no-fallback', action='store_true', help='Do not fallback to topic if service call fails')
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.mode == 'topic':
        return publish_switch_topic(
            policy_id=args.policy_id,
            topic_name=args.topic,
            wait_s=args.wait,
        )

    rc = call_switch_service(
        policy_id=args.policy_id,
        service_name=args.service,
        service_type=args.service_type,
        timeout_s=max(args.wait, 1.0),
    )
    if rc == 0 or args.no_fallback:
        return rc

    print('Service call failed, fallback to topic publish.', file=sys.stderr)
    return publish_switch_topic(
        policy_id=args.policy_id,
        topic_name=args.topic,
        wait_s=args.wait,
    )


if __name__ == '__main__':
    raise SystemExit(main())
