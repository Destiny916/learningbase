#!/usr/bin/env python3
"""
Sensor Acquisition Test Script
Tests kingfisher camera, hand cameras, hand poses, and whole body joints.
Runs for 5 seconds, measures maximum frame rate, and saves acquired data.

usage: 
1. cd w1_act
2. python -m act_async_infer_distributed_demo.scripts.test.sensor_test --duration 10.0
"""

import threading
import time
import numpy as np
import cv2
import os
import json
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import JointState
from end_effector_interfaces.msg import EEJointFeedback
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

# Import kingfisher module
from act_async_infer_distributed_demo.scripts.client._kingfisher import _kingfisher

# Import utility functions
from act_async_infer_distributed_demo.scripts.utils_distributed import (
    now_sec,
    stamp_to_sec,
    log_info,
    log_warning,
    log_error,
)


class SensorTestNode(Node):
    """Node for testing sensor acquisition."""

    def __init__(self, test_duration=5.0):
        super().__init__("sensor_test_node")

        self.test_duration = test_duration
        self.start_time = None
        self.end_time = None

        # Sensor data buffers
        self.kingfisher_data = []
        self.hand_left_images = []
        self.hand_right_images = []
        self.hand_left_poses = []
        self.hand_right_poses = []
        self.joint_states = []

        # Statistics
        self.kingfisher_count = 0
        self.hand_left_image_count = 0
        self.hand_right_image_count = 0
        self.hand_left_pose_count = 0
        self.hand_right_pose_count = 0
        self.joint_state_count = 0

        # Timestamps for frame rate calculation
        self.kingfisher_timestamps = []
        self.hand_left_image_timestamps = []
        self.hand_right_image_timestamps = []
        self.hand_left_pose_timestamps = []
        self.hand_right_pose_timestamps = []
        self.joint_state_timestamps = []

        # Configuration (similar to original code)
        self.tolerance_s = 0.15
        self.head_target_size = [640, 360]
        self.hand_target_size = [640, 480]

        # Topic names (adjust these according to your system)
        self.cam_hand_left_topic = "/camera/left/image_raw"
        self.cam_hand_right_topic = "/camera/right/image_raw"
        self.joint_topic = "/feedback/joint"
        self.left_hand_qpos_topic = "/feedback/hand/left"
        self.right_hand_qpos_topic = "/feedback/hand/right"

        # Bridge for image conversion
        self.bridge = CvBridge()

        # Kingfisher camera
        self.kingfisher = _kingfisher()

        # Setup QoS profiles
        q_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )

        q_img = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Create subscriptions
        self.create_subscription(
            Image, self.cam_hand_left_topic, self.hand_left_image_callback, q_img
        )
        log_info(f"Subscribed to hand-left image: {self.cam_hand_left_topic}")

        self.create_subscription(
            Image, self.cam_hand_right_topic, self.hand_right_image_callback, q_img
        )
        log_info(f"Subscribed to hand-right image: {self.cam_hand_right_topic}")

        self.create_subscription(
            JointState, self.joint_topic, self.joint_state_callback, q_reliable
        )
        log_info(f"Subscribed to joint state: {self.joint_topic}")

        self.create_subscription(
            EEJointFeedback,
            self.left_hand_qpos_topic,
            self.hand_left_pose_callback,
            q_reliable,
        )
        log_info(f"Subscribed to left hand pose: {self.left_hand_qpos_topic}")

        self.create_subscription(
            EEJointFeedback,
            self.right_hand_qpos_topic,
            self.hand_right_pose_callback,
            q_reliable,
        )
        log_info(f"Subscribed to right hand pose: {self.right_hand_qpos_topic}")

        # Initialize kingfisher
        if self.kingfisher.init_kingfisher():
            log_info("Kingfisher initialized successfully")
        else:
            log_error("Kingfisher initialization failed")

        log_info("Sensor test node initialized. Waiting for sensor data...")

    def hand_left_image_callback(self, msg: Image):
        """Callback for left hand camera images."""
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            t = now_sec()

            self.hand_left_images.append((t, bgr.copy()))
            self.hand_left_image_timestamps.append(t)
            self.hand_left_image_count += 1

        except Exception as e:
            log_warning(f"cv_bridge left-hand fail: {e}")

    def hand_right_image_callback(self, msg: Image):
        """Callback for right hand camera images."""
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            t = now_sec()

            self.hand_right_images.append((t, bgr.copy()))
            self.hand_right_image_timestamps.append(t)
            self.hand_right_image_count += 1

        except Exception as e:
            log_warning(f"cv_bridge right-hand fail: {e}")

    def joint_state_callback(self, msg: JointState):
        """Callback for joint state messages."""
        if not msg.position:
            return

        t = stamp_to_sec(msg.header.stamp)
        joint_qpos = np.asarray(msg.position, dtype=np.float32)

        self.joint_states.append((t, joint_qpos.copy()))
        self.joint_state_timestamps.append(t)
        self.joint_state_count += 1

    def hand_left_pose_callback(self, msg: EEJointFeedback):
        """Callback for left hand pose messages."""
        if len(msg.position) >= 6:
            arr = np.asarray(msg.position[:6], dtype=np.float32)
            t = time.time()

            self.hand_left_poses.append((t, arr.copy()))
            self.hand_left_pose_timestamps.append(t)
            self.hand_left_pose_count += 1
        else:
            log_warning(
                f"Received left hand position with only {len(msg.position)} elements, expected at least 6"
            )

    def hand_right_pose_callback(self, msg: EEJointFeedback):
        """Callback for right hand pose messages."""
        if len(msg.position) >= 6:
            arr = np.asarray(msg.position[:6], dtype=np.float32)
            t = time.time()

            self.hand_right_poses.append((t, arr.copy()))
            self.hand_right_pose_timestamps.append(t)
            self.hand_right_pose_count += 1
        else:
            log_warning(
                f"Received right hand position with only {len(msg.position)} elements, expected at least 6"
            )

    def collect_kingfisher_data(self):
        """Collect data from kingfisher camera periodically."""
        while rclpy.ok() and self.start_time is not None:
            try:
                (
                    kingfisher_left,
                    kingfisher_right,
                    t,
                ) = self.kingfisher.get_kingfisher_images()

                if kingfisher_left is not None and kingfisher_right is not None:
                    self.kingfisher_data.append(
                        (t, kingfisher_left.copy(), kingfisher_right.copy())
                    )
                    self.kingfisher_timestamps.append(t)
                    self.kingfisher_count += 1

            except Exception as e:
                log_warning(f"Kingfisher acquisition error: {e}")

            # Sleep for a short time to avoid overwhelming the system
            time.sleep(0.01)

    def run_test(self):
        """Run the sensor test for specified duration."""
        log_info(f"Starting sensor test for {self.test_duration} seconds...")
        self.start_time = time.time()
        self.end_time = self.start_time + self.test_duration

        # Start kingfisher collection thread
        kingfisher_thread = threading.Thread(
            target=self.collect_kingfisher_data, daemon=True
        )
        kingfisher_thread.start()

        # Main test loop
        while time.time() < self.end_time and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)

            # Print progress every second
            current_time = time.time()
            elapsed = current_time - self.start_time
            if elapsed > 0 and int(elapsed) != int(elapsed - 0.01):
                remaining = self.end_time - current_time
                log_info(
                    f"Test progress: {elapsed:.1f}s elapsed, {remaining:.1f}s remaining"
                )

        self.end_time = time.time()
        log_info("Test completed. Processing results...")

        # Stop kingfisher thread
        kingfisher_thread.join(timeout=0.1)

        # Calculate and display statistics
        self.print_statistics()

        # Save sample data
        self.save_data()

    def calculate_frame_rate(self, timestamps):
        """Calculate frame rate from timestamps."""
        if len(timestamps) < 2:
            return 0.0

        # Sort timestamps
        sorted_ts = sorted(timestamps)

        # Calculate intervals
        intervals = [sorted_ts[i] - sorted_ts[i - 1] for i in range(1, len(sorted_ts))]

        # Filter out unusually long intervals (more than 1 second)
        filtered_intervals = [i for i in intervals if i < 1.0]

        if not filtered_intervals:
            return 0.0

        avg_interval = sum(filtered_intervals) / len(filtered_intervals)
        return 1.0 / avg_interval if avg_interval > 0 else 0.0

    def print_statistics(self):
        """Print test statistics."""
        actual_duration = self.end_time - self.start_time

        log_info("=" * 60)
        log_info("SENSOR TEST RESULTS")
        log_info("=" * 60)
        log_info(f"Test duration: {actual_duration:.2f} seconds")
        log_info("")

        # Kingfisher statistics
        kingfisher_rate = self.calculate_frame_rate(self.kingfisher_timestamps)
        log_info(f"Kingfisher Camera:")
        log_info(f"  Frames acquired: {self.kingfisher_count}")
        log_info(f"  Average frame rate: {kingfisher_rate:.2f} Hz")
        log_info(f"  Data points saved: {len(self.kingfisher_data)}")

        # Hand camera statistics
        hand_left_rate = self.calculate_frame_rate(self.hand_left_image_timestamps)
        hand_right_rate = self.calculate_frame_rate(self.hand_right_image_timestamps)
        log_info(f"Left Hand Camera:")
        log_info(f"  Frames acquired: {self.hand_left_image_count}")
        log_info(f"  Average frame rate: {hand_left_rate:.2f} Hz")
        log_info(f"  Data points saved: {len(self.hand_left_images)}")
        log_info(f"Right Hand Camera:")
        log_info(f"  Frames acquired: {self.hand_right_image_count}")
        log_info(f"  Average frame rate: {hand_right_rate:.2f} Hz")
        log_info(f"  Data points saved: {len(self.hand_right_images)}")

        # Hand pose statistics
        hand_left_pose_rate = self.calculate_frame_rate(self.hand_left_pose_timestamps)
        hand_right_pose_rate = self.calculate_frame_rate(
            self.hand_right_pose_timestamps
        )
        log_info(f"Left Hand Pose:")
        log_info(f"  Samples acquired: {self.hand_left_pose_count}")
        log_info(f"  Average rate: {hand_left_pose_rate:.2f} Hz")
        log_info(f"  Data points saved: {len(self.hand_left_poses)}")
        log_info(f"Right Hand Pose:")
        log_info(f"  Samples acquired: {self.hand_right_pose_count}")
        log_info(f"  Average rate: {hand_right_pose_rate:.2f} Hz")
        log_info(f"  Data points saved: {len(self.hand_right_poses)}")

        # Joint state statistics
        joint_state_rate = self.calculate_frame_rate(self.joint_state_timestamps)
        log_info(f"Joint States:")
        log_info(f"  Samples acquired: {self.joint_state_count}")
        log_info(f"  Average rate: {joint_state_rate:.2f} Hz")
        log_info(f"  Data points saved: {len(self.joint_states)}")

        # Overall statistics
        total_samples = (
            self.kingfisher_count
            + self.hand_left_image_count
            + self.hand_right_image_count
            + self.hand_left_pose_count
            + self.hand_right_pose_count
            + self.joint_state_count
        )
        log_info("")
        log_info(f"Total data samples acquired: {total_samples}")
        log_info(
            f"Overall data rate: {total_samples / actual_duration:.2f} samples/second"
        )

        # Check for any sensors with no data
        sensors_without_data = []
        if self.kingfisher_count == 0:
            sensors_without_data.append("Kingfisher camera")
        if self.hand_left_image_count == 0:
            sensors_without_data.append("Left hand camera")
        if self.hand_right_image_count == 0:
            sensors_without_data.append("Right hand camera")
        if self.hand_left_pose_count == 0:
            sensors_without_data.append("Left hand pose")
        if self.hand_right_pose_count == 0:
            sensors_without_data.append("Right hand pose")
        if self.joint_state_count == 0:
            sensors_without_data.append("Joint states")

        if sensors_without_data:
            log_info("")
            log_warning("Sensors with no data received:")
            for sensor in sensors_without_data:
                log_warning(f"  - {sensor}")
        else:
            log_info("")
            log_info("✓ All sensors received data successfully!")

    def save_data(self):
        """Save sample data to files for verification."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"sensor_test_output_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)

        log_info(f"Saving sample data to: {output_dir}")

        # Save statistics
        stats = {
            "test_duration": self.end_time - self.start_time,
            "kingfisher_count": self.kingfisher_count,
            "hand_left_image_count": self.hand_left_image_count,
            "hand_right_image_count": self.hand_right_image_count,
            "hand_left_pose_count": self.hand_left_pose_count,
            "hand_right_pose_count": self.hand_right_pose_count,
            "joint_state_count": self.joint_state_count,
            "kingfisher_timestamps": self.kingfisher_timestamps,
            "hand_left_image_timestamps": self.hand_left_image_timestamps,
            "hand_right_image_timestamps": self.hand_right_image_timestamps,
            "hand_left_pose_timestamps": self.hand_left_pose_timestamps,
            "hand_right_pose_timestamps": self.hand_right_pose_timestamps,
            "joint_state_timestamps": self.joint_state_timestamps,
        }

        with open(os.path.join(output_dir, "statistics.json"), "w") as f:
            json.dump(stats, f, indent=2, default=str)

        # Save sample images (first frame of each camera)
        if self.kingfisher_data:
            _, left_img, right_img = self.kingfisher_data[0]
            cv2.imwrite(
                os.path.join(output_dir, "kingfisher_left_sample.jpg"), left_img
            )
            cv2.imwrite(
                os.path.join(output_dir, "kingfisher_right_sample.jpg"), right_img
            )

        if self.hand_left_images:
            _, img = self.hand_left_images[0]
            cv2.imwrite(os.path.join(output_dir, "hand_left_sample.jpg"), img)

        if self.hand_right_images:
            _, img = self.hand_right_images[0]
            cv2.imwrite(os.path.join(output_dir, "hand_right_sample.jpg"), img)

        # Save sample pose and joint data
        if self.hand_left_poses:
            _, pose = self.hand_left_poses[0]
            np.save(os.path.join(output_dir, "hand_left_pose_sample.npy"), pose)

        if self.hand_right_poses:
            _, pose = self.hand_right_poses[0]
            np.save(os.path.join(output_dir, "hand_right_pose_sample.npy"), pose)

        if self.joint_states:
            _, joints = self.joint_states[0]
            np.save(os.path.join(output_dir, "joint_state_sample.npy"), joints)

        # Save timestamp distribution plot
        self.plot_timestamp_distribution(output_dir)

        log_info(f"Data saved successfully to {output_dir}")

    def plot_timestamp_distribution(self, output_dir):
        """Create a simple plot of timestamp distribution."""
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 8))

            sensors = [
                ("Kingfisher", self.kingfisher_timestamps, "r"),
                ("Left Hand Camera", self.hand_left_image_timestamps, "g"),
                ("Right Hand Camera", self.hand_right_image_timestamps, "b"),
                ("Left Hand Pose", self.hand_left_pose_timestamps, "c"),
                ("Right Hand Pose", self.hand_right_pose_timestamps, "m"),
                ("Joint States", self.joint_state_timestamps, "y"),
            ]

            for i, (sensor_name, timestamps, color) in enumerate(sensors):
                if timestamps:
                    # Normalize timestamps to start at 0
                    if timestamps:
                        normalized_ts = [ts - min(timestamps) for ts in timestamps]
                        plt.scatter(
                            normalized_ts,
                            [i] * len(normalized_ts),
                            color=color,
                            label=sensor_name,
                            alpha=0.6,
                            s=10,
                        )

            plt.xlabel("Time (seconds from start)")
            plt.ylabel("Sensor")
            plt.yticks(range(len(sensors)), [s[0] for s in sensors])
            plt.title("Sensor Timestamp Distribution")
            plt.legend(loc="upper right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            plot_path = os.path.join(output_dir, "timestamp_distribution.png")
            plt.savefig(plot_path)
            plt.close()

            log_info(f"Timestamp distribution plot saved to: {plot_path}")

        except ImportError:
            log_warning("Matplotlib not available. Skipping timestamp plot.")


def main():
    """Main function to run the sensor test."""
    rclpy.init()

    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Sensor Acquisition Test")
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Test duration in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--topic-left-camera",
        type=str,
        default="/camera/left/image_raw",
        help="Topic for left hand camera",
    )
    parser.add_argument(
        "--topic-right-camera",
        type=str,
        default="/camera/right/image_raw",
        help="Topic for right hand camera",
    )
    parser.add_argument(
        "--topic-joint-state",
        type=str,
        default="/feedback/robot_server_state",
        help="Topic for joint state",
    )
    parser.add_argument(
        "--topic-left-hand-pose",
        type=str,
        default="/brainco_left_hand_qpos",
        help="Topic for left hand pose",
    )
    parser.add_argument(
        "--topic-right-hand-pose",
        type=str,
        default="/brainco_right_hand_qpos",
        help="Topic for right hand pose",
    )

    args = parser.parse_args()

    # Create and configure test node
    test_node = SensorTestNode(test_duration=args.duration)

    # Update topic names if provided
    test_node.cam_hand_left_topic = args.topic_left_camera
    test_node.cam_hand_right_topic = args.topic_right_camera
    test_node.joint_topic = args.topic_joint_state
    test_node.left_hand_qpos_topic = args.topic_left_hand_pose
    test_node.right_hand_qpos_topic = args.topic_right_hand_pose

    try:
        # Run the test
        test_node.run_test()

    except KeyboardInterrupt:
        log_info("Test interrupted by user")
    except Exception as e:
        log_error(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Cleanup
        test_node.destroy_node()
        rclpy.shutdown()

    log_info("Sensor test completed.")


if __name__ == "__main__":
    main()
