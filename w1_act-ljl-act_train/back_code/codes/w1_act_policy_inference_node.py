#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 node (single file) — DIRECT CAMERA CAPTURE (side-by-side), ACT batch builder, and action publisher

• Captures a side‑by‑side stereo frame (e.g., 3840x1080) via GStreamer/OpenCV, splits to left/right,
  resizes to training resolution (default 960x540), and time‑stamps with wall‑clock.
• Subscribes /feedback/joint (sensor_msgs/JointState) at high rate (~100 Hz), keeps a ring buffer.
• At policy_hz (10–20 Hz), aligns the latest frame with the nearest joint sample (±tolerance_ms),
  builds the ACT observation batch with training keys, runs policy.select_action (ABSOLUTE mode),
  and publishes 20‑DoF desired joints to /w1/policy/desired_joint_positions.

Run:
  python3 w1_act_policy_inference_node.py --ros-args \
    -p policy_path:=/ABS/PATH/TO/pretrained_model \
    -p gst_pipeline:='v4l2src device=/dev/video99 ! image/jpeg,width=3840,height=1080,framerate=30/1 ! jpegdec ! videoconvert ! appsink max-buffers=1 drop=True' \
    -p joint_topic:=/feedback/joint \
    -p publish_topic:=/w1/policy/desired_joint_positions \
    -p device:=cuda -p policy_hz:=15.0 -p tolerance_ms:=30.0 \
    -p target_width:=960 -p target_height:=540 \
    -p image_left_key:=observation.images.cam_high_left \
    -p image_right_key:=observation.images.cam_high_right \
    -p state_key:=observation.state

Notes:
- Keys and resolution default to your training set: cam_high_left/right at 960x540 (RGB, CHW, [0,1]).
- Action mode is ABSOLUTE only (matches your request).
- If JointState.name order differs from training, this node will reorder by name once.
"""

from __future__ import annotations
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import os
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

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
        self.declare_parameter(
            'gst_pipeline',
            "v4l2src device=/dev/video99 ! image/jpeg,width=3840,height=1080,framerate=30/1 ! jpegdec ! videoconvert ! appsink max-buffers=1 drop=True"
        )
        self.declare_parameter('joint_topic', '/feedback/joint')
        self.declare_parameter('publish_topic',
                               '/w1/policy/desired_joint_positions')
        self.declare_parameter('policy_hz', 15.0)
        self.declare_parameter('tolerance_ms', 30.0)
        self.declare_parameter('target_width', 960)
        self.declare_parameter('target_height', 540)
        # Training keys
        self.declare_parameter('image_left_key',
                               'observation.images.cam_high_left')
        self.declare_parameter('image_right_key',
                               'observation.images.cam_high_right')
        self.declare_parameter('state_key', 'observation.state')
        # Expected joint order from training (can override)
        self.declare_parameter('ordered_joint_names', [
            'ANKLE',
            'KNEE',
            'BUTTOCK',
            'WAIST',
            'LEFT_J1',
            'LEFT_J2',
            'LEFT_J3',
            'LEFT_J4',
            'LEFT_J5',
            'LEFT_J6',
            'LEFT_J7',
            'NECK1',
            'NECK2',
            'RIGHT_J1',
            'RIGHT_J2',
            'RIGHT_J3',
            'RIGHT_J4',
            'RIGHT_J5',
            'RIGHT_J6',
            'RIGHT_J7',
        ])
        from collections import deque
        self._dt_hist = deque(maxlen=300)  # 最近300次 dt（秒）
        self._miss = 0  # 没对齐的次数
        self._seen = 0  # 尝试对齐的总次数

        self.policy_path: str = self.get_parameter(
            'policy_path').get_parameter_value().string_value
        self.device_str: str = self.get_parameter(
            'device').get_parameter_value().string_value
        self.gst_pipeline: str = self.get_parameter(
            'gst_pipeline').get_parameter_value().string_value
        self.joint_topic: str = self.get_parameter(
            'joint_topic').get_parameter_value().string_value
        self.publish_topic: str = self.get_parameter(
            'publish_topic').get_parameter_value().string_value
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
        self.ordered_joint_names = list(
            self.get_parameter(
                'ordered_joint_names').get_parameter_value().string_array_value)

        # ---------- QoS ---------- #
        q_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ---------- Publishers ---------- #
        self.pub_action = self.create_publisher(Float64MultiArray,
                                                self.publish_topic, q_reliable)

        # ---------- Joint subscriber / buffer ---------- #
        self.sub_joint = self.create_subscription(JointState, self.joint_topic,
                                                  self.cb_joint, q_reliable)
        self.joint_buf: deque[Tuple[float, np.ndarray]] = deque(
            maxlen=2000)  # ~20 s @100 Hz
        self.incoming_joint_index: Optional[
            np.ndarray] = None  # map to training order on first msg

        # ---------- Policy ---------- #
        if ACTPolicy is None:
            self.get_logger().error(
                f"Failed to import ACTPolicy: {_IMPORT_ERR}")
            raise RuntimeError('ACTPolicy import failed')
        self.device = torch.device(self.device_str if (
            self.device_str == 'cuda' and torch.cuda.is_available()) else 'cpu')
        self.get_logger().info(
            f'Loading ACTPolicy from {self.policy_path} on {self.device} ...')
        self.policy = ACTPolicy.from_pretrained(self.policy_path)
        self.policy.to(self.device)
        self.policy.eval()
        self.get_logger().info('Policy loaded.')

        # ---------- Direct capture thread ---------- #
        self.frame_lock = threading.Lock()
        self.latest_frame: Optional[TimedFrame] = None
        self._cap_thread_stop = False
        self._cap_thread = threading.Thread(target=self.capture_loop_sbs,
                                            daemon=True)
        self._cap_thread.start()
        self.get_logger().info('Direct camera capture thread started (SBS).')

        # ---------- Inference timer ---------- #
        self.min_infer_dt = 1.0 / max(1.0, self.policy_hz)
        print(self.min_infer_dt)
        self.last_infer_t = 0.0
        self.create_timer(self.min_infer_dt, self.timer_infer)

        self.get_logger().info('W1 ACT direct node initialized.')

    def capture_loop_sbs(self) -> None:
        import cv2, time

        # 直接打开 /dev/video99（不走 GStreamer），V4L2 更稳
        cap = cv2.VideoCapture('/dev/video99', cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(99)  # 退一步：用设备索引
        if not cap.isOpened():
            self.get_logger().error(
                'Failed to open /dev/video99 with OpenCV VideoCapture')
            return

        # 期望设置（相机可能忽略，但不影响读取）
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

                # 写入最新帧供推理
                tf = TimedFrame(t=t, left_bgr=left_bgr, right_bgr=right_bgr)
                with self.frame_lock:
                    self.latest_frame = tf
        finally:
            cap.release()

    # -------------------- Joint callback -------------------- #
    def cb_joint(self, msg: JointState) -> None:
        t = stamp_to_sec(msg.header.stamp)
        if not msg.position:
            return
        qpos = np.asarray(msg.position, dtype=np.float32)

        # Build reordering map once using msg.name
        if self.incoming_joint_index is None and msg.name:
            name_to_idx = {n: i for i, n in enumerate(msg.name)}
            map_idx = []
            missing = []
            for n in self.ordered_joint_names:
                if n in name_to_idx:
                    map_idx.append(name_to_idx[n])
                else:
                    missing.append(n)
            if missing:
                self.get_logger().warn(
                    f"Incoming JointState missing names: {missing}. Using incoming order as-is."
                )
                self.incoming_joint_index = None
            else:
                self.incoming_joint_index = np.asarray(map_idx, dtype=np.int64)
                self.get_logger().info('Joint order mapped to training order.')

        if self.incoming_joint_index is not None and self.incoming_joint_index.shape[
                0] == qpos.shape[0]:
            qpos = qpos[self.incoming_joint_index]

        self.joint_buf.append((t, qpos))

    # -------------------- Inference timer -------------------- #
    @torch.no_grad()
    def timer_infer(self) -> None:
        # Rate limit
        # if now_sec() - self.last_infer_t < self.min_infer_dt:
        #     return

        # Get latest frame
        with self.frame_lock:
            tf = self.latest_frame
        if tf is None:
            return

        # Find nearest joint sample
        nj = self._nearest_joint_with_time(tf.t, self.tolerance_s)
        self._seen += 1
        if nj is None:
            self._miss += 1
            return
        qpos, tj = nj
        dt = tj - tf.t
        self._dt_hist.append(dt)
        # 每 50 次打印一次对齐统计
        if self._seen % 50 == 0 and len(self._dt_hist) > 0:
            import numpy as np
            a = np.abs(np.fromiter(self._dt_hist, dtype=np.float64))
            p95 = float(np.percentile(a, 95))
            mean = float(a.mean())
            miss_rate = self._miss / max(1, self._seen)
            # self.get_logger().info(f"[sync] |dt| mean={mean*1e3:.1f}ms, p95={p95*1e3:.1f}ms, miss={miss_rate*100:.1f}%")
            # 如果发现总是偏正/偏负，可考虑静态 offset 补偿：用 (tf.t - offset) 去对齐

        if qpos.shape[0] != 20:
            self.get_logger().error(
                f'Expected 20-DoF joints, got {qpos.shape[0]}')
            return

        # Preprocess images to training size
        left_chw = bgr_to_chw_rgb_resized(tf.left_bgr, self.target_size)
        right_chw = bgr_to_chw_rgb_resized(tf.right_bgr, self.target_size)

        # Build batch (B=1)
        batch_np = {
            self.image_left_key: left_chw[None, ...],  # (1,3,H,W)
            self.image_right_key: right_chw[None, ...],  # (1,3,H,W)
            self.state_key: qpos[None, ...],  # (1,20)
        }
        batch = {
            k: torch.from_numpy(v).to(self.device) for k, v in batch_np.items()
        }

        # Run policy (ABSOLUTE action mode)
        try:
            action = self.policy.select_action(batch)  # shape [20] or [1,20]
        except Exception as e:
            self.get_logger().error(f'policy.select_action failed: {e}')
            return

        if isinstance(action, torch.Tensor):
            act = action.detach().to('cpu').numpy()
            if act.ndim == 2 and act.shape[0] == 1:
                act = act[0]
        else:
            act = np.asarray(action, dtype=np.float32)

        if act.shape[-1] != 20:
            self.get_logger().error(f'Expected 20-dim action, got {act.shape}')
            return

        # Publish desired joints as Float64MultiArray
        msg = Float64MultiArray()
        msg.data = [float(x) for x in act.tolist()]
        self.pub_action.publish(msg)

        # self.last_infer_t = now_sec()

    # -------------------- Utils -------------------- #
    def _nearest_joint(self, t: float, tol: float) -> Optional[np.ndarray]:
        if not self.joint_buf:
            return None
        ts = np.fromiter((jt[0] for jt in self.joint_buf), dtype=np.float64)
        idx = int(np.argmin(np.abs(ts - t)))
        if abs(ts[idx] - t) <= tol:
            return self.joint_buf[idx][1]
        return None

    def _nearest_joint_with_time(
            self, t: float, tol: float) -> Optional[Tuple[np.ndarray, float]]:
        if not self.joint_buf:
            return None
        ts = np.fromiter((jt[0] for jt in self.joint_buf), dtype=np.float64)
        idx = int(np.argmin(np.abs(ts - t)))
        dt = float(ts[idx] - t)
        if abs(dt) <= tol:
            return self.joint_buf[idx][1], float(ts[idx])
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
        # Signal capture thread to stop
        if hasattr(node, '_cap_thread_stop'):
            node._cap_thread_stop = True
        if hasattr(node, '_cap_thread') and node._cap_thread.is_alive():
            node._cap_thread.join(timeout=1.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
