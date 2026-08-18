#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 node — Direct stereo capture (SBS), ACT batch builder, and action publisher
with RIGHT-HAND scalar integration.

Changes vs. original:
- The training joint order now includes a HAND SCALAR joint name (e.g., 'RIGHT_HAND').
- Subscribes /hand/right_scalar (std_msgs/Float64) as a 1-D continuous value in [0,1].
- Reorders only BODY joints from /feedback/joint to match training order (excluding hand scalar).
- Aligns (image, body_joints, hand_scalar) in time, builds full state [BODY(10) + HAND(1)].
- Ensures HAND SCALAR is never published to actuators (added to drop set).

Notes:
- Adjust 'ordered_joint_names' and 'hand_scalar_name' to match your training exactly.
- If your policy outputs dim != len(ordered_joint_names), this node will error-log.
"""

from __future__ import annotations
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Tuple, Optional, List

import numpy as np
import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray
from joint_interfaces.msg import JointPositionControl
# from JointPositionController import JointCommandPublisher
# Joint_Position_Controller = JointCommandPublisher()
try:
    from act.modeling_act import ACTPolicy
except ImportError as e:
    raise ImportError(
        "Failed to import ACTPolicy from act.modeling_act. "
        "Please make sure modeling_act.py is in your PYTHONPATH.") from e

# -------------------- Small helpers -------------------- #


def now_sec() -> float:
    return time.time()


def stamp_to_sec(stamp) -> float:
    # Some JointState may have empty header; fall back to wall clock
    if stamp and (stamp.sec or stamp.nanosec):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9
    return now_sec()


def bgr_to_chw_rgb_resized(img_bgr: np.ndarray, size: Tuple[int,
                                                            int]) -> np.ndarray:
    """BGR HxWx3 → RGB CHW float32 in [0,1], resized to (W,H)=size."""
    import cv2
    tw, th = size
    x = cv2.resize(img_bgr, (tw, th), interpolation=cv2.INTER_AREA)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))  # C,H,W
    return x


@dataclass
class TimedFrame:
    t: float
    left_bgr: np.ndarray
    right_bgr: np.ndarray


class W1ACTDirectNode(Node):

    def __init__(self) -> None:
        super().__init__('w1_act_direct_node')

        # ---------- Parameters ---------- #
        self.declare_parameter('policy_path', '/path/to/pretrained_model')
        self.declare_parameter('device', 'cuda')
        # camera
        self.declare_parameter('joint_topic', '/feedback/joint')
        self.declare_parameter('publish_topic',
                               '/w1/policy/desired_joint_positions')
        self.declare_parameter('policy_hz', 15.0)
        self.declare_parameter('tolerance_ms', 60.0)
        self.declare_parameter('target_width', 960)
        self.declare_parameter('target_height', 540)
        # Training keys
        self.declare_parameter('image_left_key',
                               'observation.images.cam_high_left')
        self.declare_parameter('image_right_key',
                               'observation.images.cam_high_right')
        self.declare_parameter('state_key', 'observation.state')

        # === Training joint order: BODY(10) + HAND(1) ===
        # !! IMPORTANT: replace 'RIGHT_HAND' with YOUR real hand-scalar joint name used in training.
        self.declare_parameter(
            'ordered_joint_names',
            [
                'WAIST',
                'NECK1',
                'NECK2',
                'RIGHT_J1',
                'RIGHT_J2',
                'RIGHT_J3',
                'RIGHT_J4',
                'RIGHT_J5',
                'RIGHT_J6',
                'RIGHT_J7',
                'RIGHT_GRIPPER',  # <-- hand scalar joint name (1-D)
            ])
        self.declare_parameter('hand_scalar_name', 'RIGHT_GRIPPER')
        self.declare_parameter('hand_scalar_topic', '/hand/right_scalar')

        # Drop these joints when publishing (by name). We'll also always drop the hand scalar.
        self.declare_parameter('drop_joint_names', ['ANKLE', 'KNEE', 'BUTTOCK'])
        self.hand_gestures = {
            "normal": [0.0, 70.0, 0.0, 0.0, 0.0, 0.0],
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
        # Stats buffers for sync monitoring (can be silent)
        self._dt_hist = deque(maxlen=300)
        self._miss = 0
        self._seen = 0

        # ---------- Read parameters ---------- #
        self.policy_path: str = self.get_parameter(
            'policy_path').get_parameter_value().string_value
        self.device_str: str = self.get_parameter(
            'device').get_parameter_value().string_value
        self.joint_topic: str = self.get_parameter(
            'joint_topic').get_parameter_value().string_value
        self.publish_topic: str = self.get_parameter(
            'publish_topic').get_parameter_value().string_value
        # self.pub_hand = self.create_publisher(JointPositionControl, self.publish_topic, q_reliable)

        self.policy_hz: float = float(
            self.get_parameter('policy_hz').get_parameter_value().double_value)
        self.tolerance_s: float = float(
            self.get_parameter(
                'tolerance_ms').get_parameter_value().double_value) / 1000.0
        self.target_size: Tuple[int, int] = (
            int(
                self.get_parameter(
                    'target_width').get_parameter_value().integer_value),
            int(
                self.get_parameter(
                    'target_height').get_parameter_value().integer_value),
        )
        self.image_left_key: str = self.get_parameter(
            'image_left_key').get_parameter_value().string_value
        self.image_right_key: str = self.get_parameter(
            'image_right_key').get_parameter_value().string_value
        self.state_key: str = self.get_parameter(
            'state_key').get_parameter_value().string_value
        self.training_names: List[str] = list(
            self.get_parameter(
                'ordered_joint_names').get_parameter_value().string_array_value)
        self.hand_name: str = self.get_parameter(
            'hand_scalar_name').get_parameter_value().string_value
        self.hand_scalar_topic: str = self.get_parameter(
            'hand_scalar_topic').get_parameter_value().string_value
        self.drop_joint_names: List[str] = list(
            self.get_parameter(
                'drop_joint_names').get_parameter_value().string_array_value)

        # Sanity: training_names must contain hand_name
        if self.hand_name not in self.training_names:
            raise RuntimeError(
                f"'hand_scalar_name' ({self.hand_name}) not found in 'ordered_joint_names'."
            )

        # Precompute BODY names (training order without the hand scalar)
        self.hand_index: int = self.training_names.index(self.hand_name)
        self.body_names: List[str] = [
            n for n in self.training_names if n != self.hand_name
        ]
        self.body_train_positions: List[int] = [
            i for i, n in enumerate(self.training_names) if n != self.hand_name
        ]

        self.get_logger().info(
            f"[train] body dims={len(self.body_names)}, hand joint='{self.hand_name}' at index {self.hand_index}."
        )

        # ---------- QoS ---------- #
        q_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ---------- Publishers ---------- #
        self.pub_action = self.create_publisher(JointPositionControl,
                                                self.publish_topic, q_reliable)
        self.pub_right_hand = self.create_publisher(
            Float64MultiArray, '/set_brainco_right_hand_qpos', q_reliable)
        # ---------- Subscribers / buffers ---------- #
        # BODY joints (from /feedback/joint) → we will reorder to self.body_names
        self.sub_joint = self.create_subscription(JointState, self.joint_topic,
                                                  self.cb_joint, q_reliable)
        self.body_buf: deque[Tuple[float, np.ndarray]] = deque(
            maxlen=2000)  # vector arranged as self.body_names
        self.incoming_body_index: Optional[
            np.ndarray] = None  # map incoming -> body_names order

        # HAND scalar (Float64) — timestamp using wall clock
        self.sub_hand_scalar = self.create_subscription(Float64,
                                                        self.hand_scalar_topic,
                                                        self.cb_hand_scalar,
                                                        q_reliable)
        self.hand_scalar_buf: deque[Tuple[float, float]] = deque(maxlen=2000)

        # ---------- Policy ---------- #
        if ACTPolicy is None:
            raise RuntimeError('ACTPolicy import failed')
        self.device = torch.device(self.device_str if (
            self.device_str == 'cuda' and torch.cuda.is_available()) else 'cpu')
        self.get_logger().info(
            f'Loading ACTPolicy from {self.policy_path} on {self.device} ...')
        self.policy = ACTPolicy.from_pretrained(self.policy_path)
        self.policy.to(self.device)
        self.policy.eval()
        self.get_logger().info('Policy loaded.')

        # ---------- Direct camera capture thread (SBS) ---------- #
        self.frame_lock = threading.Lock()
        self.latest_frame: Optional[TimedFrame] = None
        self._cap_thread_stop = False
        self._cap_thread = threading.Thread(target=self.capture_loop_sbs,
                                            daemon=True)
        self._cap_thread.start()
        self.get_logger().info('Direct camera capture thread started (SBS).')

        # ---------- Inference timer ---------- #
        self.min_infer_dt = 1.0 / max(1.0, self.policy_hz)
        self.create_timer(self.min_infer_dt, self.timer_infer)

        # ---------- Build publish filter (drop names + always drop hand scalar) ---------- #
        drop_set = set(self.drop_joint_names)
        drop_set.add(self.hand_name)  # never publish hand scalar
        keep_idx = [
            i for i, n in enumerate(self.training_names) if n not in drop_set
        ]
        if not keep_idx:
            raise RuntimeError(
                "All joints were dropped; check 'drop_joint_names' and 'hand_scalar_name'!"
            )
        self.publish_keep_index = np.asarray(keep_idx, dtype=np.int64)
        self.publish_names = [self.training_names[i] for i in keep_idx]
        self.get_logger().info(
            f"[publish] dropping {sorted(list(drop_set))}, keeping {len(self.publish_names)} joints."
        )

        self.get_logger().info(
            'W1 ACT direct node (with hand scalar) initialized.')

    # -------------------- Capture (SBS) -------------------- #
    def capture_loop_sbs(self) -> None:
        import cv2, time
        cap = cv2.VideoCapture('/dev/video99', cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(99)
        if not cap.isOpened():
            self.get_logger().error(
                'Failed to open /dev/video99 with OpenCV VideoCapture')
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)

        try:
            while not self._cap_thread_stop:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.005)
                    continue

                t = now_sec()
                h, w2, _ = frame.shape
                if w2 % 2 != 0:
                    self.get_logger().warn(
                        f'Frame width {w2} is not even; skip frame')
                    continue

                w = w2 // 2
                left_bgr = frame[:, :w]
                right_bgr = frame[:, w:]

                tf = TimedFrame(t=t, left_bgr=left_bgr, right_bgr=right_bgr)
                with self.frame_lock:
                    self.latest_frame = tf
        finally:
            cap.release()

    # -------------------- Subscribers -------------------- #
    def cb_joint(self, msg: JointState) -> None:
        t = stamp_to_sec(msg.header.stamp)
        if not msg.position:
            return
        qpos_in = np.asarray(msg.position, dtype=np.float32)

        # Build reordering map ONCE: incoming msg.name -> body_names (training order without hand scalar)
        if self.incoming_body_index is None and msg.name:
            name_to_idx = {n: i for i, n in enumerate(msg.name)}
            map_idx = []
            missing = []
            for n in self.body_names:
                if n in name_to_idx:
                    map_idx.append(name_to_idx[n])
                else:
                    missing.append(n)
            if missing:
                self.get_logger().warn(
                    f"Incoming JointState missing body joint names: {missing}. Waiting for complete mapping..."
                )
                return
            self.incoming_body_index = np.asarray(map_idx, dtype=np.int64)
            self.get_logger().info(
                'Body joint order mapped to training body order.')

        # Reorder to body_names order
        if self.incoming_body_index is not None:
            try:
                q_body = qpos_in[self.incoming_body_index]
            except Exception as e:
                self.get_logger().warn(f"Failed to reorder body joints: {e}")
                return
        else:
            return

        self.body_buf.append((t, q_body))

    def cb_hand_scalar(self, msg: Float64) -> None:
        t = now_sec()  # Float64 has no header; use receipt time
        self.hand_scalar_buf.append((t, float(msg.data)))

    # -------------------- Inference timer -------------------- #
    @torch.no_grad()
    def timer_infer(self) -> None:
        # Latest frame
        with self.frame_lock:
            tf = self.latest_frame
        if tf is None:
            return

        # Find nearest body joints (q_body) and hand scalar (hand_s)
        nb = self._nearest_body_with_time(tf.t, self.tolerance_s)
        nh = self._nearest_hand_scalar_with_time(tf.t, self.tolerance_s)
        self._seen += 1
        if nb is None or nh is None:
            self._miss += 1
            return
        q_body, tj = nb
        hand_s, th = nh
        dt = tj - tf.t
        self._dt_hist.append(dt)

        # Sanity
        if q_body.shape[0] != len(self.body_names):
            self.get_logger().error(
                f'Body dims mismatch: expected {len(self.body_names)}, got {q_body.shape[0]}'
            )
            return

        # Preprocess images
        left_chw = bgr_to_chw_rgb_resized(tf.left_bgr, self.target_size)
        right_chw = bgr_to_chw_rgb_resized(tf.right_bgr, self.target_size)

        # Build FULL state (BODY in training positions + HAND scalar at hand_index)
        full_dim = len(self.training_names)
        full_state = np.empty(full_dim, dtype=np.float32)
        # place body
        full_state[self.body_train_positions] = q_body
        # place hand scalar
        full_state[self.hand_index] = float(hand_s)

        # Build batch (B=1)
        batch_np = {
            self.image_left_key: left_chw[None, ...],  # (1,3,H,W)
            self.image_right_key: right_chw[None, ...],  # (1,3,H,W)
            self.state_key: full_state[None, ...],  # (1, D)
        }
        batch = {
            k: torch.from_numpy(v).to(self.device) for k, v in batch_np.items()
        }

        # Run policy (ABSOLUTE action mode)
        try:
            action = self.policy.select_action(
                batch)  # Expected shape [D] or [1,D]
        except Exception as e:
            self.get_logger().error(f'policy.select_action failed: {e}')
            return

        if isinstance(action, torch.Tensor):
            act = action.detach().to('cpu').numpy()
            if act.ndim == 2 and act.shape[0] == 1:
                act = act[0]
        else:
            act = np.asarray(action, dtype=np.float32)

        if act.shape[-1] != full_dim:
            self.get_logger().error(
                f'Policy dim mismatch: expected {full_dim}, got {act.shape}')
            return
        hand_s = float(np.clip(act[10], 0.0, 1.0))
        right_hand_values = list(
            np.array(self.hand_gestures['normal']) * (1 - hand_s) +
            np.array(self.hand_gestures['pinch']) * hand_s)
        hand_msg = Float64MultiArray()
        hand_msg.data = right_hand_values
        self.pub_right_hand.publish(hand_msg)

        # -------- PUBLISH: drop specified joints (including hand scalar) --------
        act_pub = act[
            self.
            publish_keep_index]  # filter out dropped joints (includes hand scalar)
        msg = JointPositionControl()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.publish_names)  # kept names only
        msg.position = [float(x) for x in act_pub.tolist()]
        self.pub_action.publish(msg)
        # ----------------------------------------------------------------------

    # -------------------- Utils -------------------- #
    def _nearest_body_with_time(
            self, t: float, tol: float) -> Optional[Tuple[np.ndarray, float]]:
        if not self.body_buf:
            return None
        ts = np.fromiter((jt[0] for jt in self.body_buf), dtype=np.float64)
        idx = int(np.argmin(np.abs(ts - t)))
        dt = float(ts[idx] - t)
        if abs(dt) <= tol:
            return self.body_buf[idx][1], float(ts[idx])
        return None

    def _nearest_hand_scalar_with_time(
            self, t: float, tol: float) -> Optional[Tuple[float, float]]:
        if not self.hand_scalar_buf:
            return None
        ts = np.fromiter((x[0] for x in self.hand_scalar_buf), dtype=np.float64)
        idx = int(np.argmin(np.abs(ts - t)))
        dt = float(ts[idx] - t)
        if abs(dt) <= tol:
            return float(self.hand_scalar_buf[idx][1]), float(ts[idx])
        return None


# -------------------- Main -------------------- #
def main() -> None:
    rclpy.init()
    node = W1ACTDirectNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(node, '_cap_thread_stop'):
            node._cap_thread_stop = True
        if hasattr(node, '_cap_thread') and node._cap_thread.is_alive():
            node._cap_thread.join(timeout=1.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
