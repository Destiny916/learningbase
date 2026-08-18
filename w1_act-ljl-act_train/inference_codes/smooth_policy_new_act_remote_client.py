#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
import threading
from collections import deque
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict

import numpy as np
import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray
from joint_interfaces.msg import JointPositionControl
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from end_effector_interfaces.msg import EEFeedback, EEJointControl, EEJointControlMode
from std_msgs.msg import String      # 新增
import json
from act_async_infer_distributed_demo.scripts.network_utils_act import NetworkClient
# ===== Utils =====

def now_sec() -> float:
    import time
    return time.time()

def stamp_to_sec(stamp) -> float:
    if stamp and (stamp.sec or stamp.nanosec):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9
    return now_sec()

def nearest(buf: deque, t: float, tol_s: float) -> Optional[np.ndarray]:
    """Return latest value in buf nearest to time t within tol."""
    if not buf:
        return None
    dt = float(buf[-1][0] - t)
    if abs(dt) <= tol_s:
        return buf[-1][1]
    return None

def bgr_to_hwc_rgb_resized(img_bgr: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """BGR HxWx3 → RGB HWC uint8, resized to (W,H)."""
    import cv2
    tw, th = size
    x = cv2.resize(img_bgr, (tw, th), interpolation=cv2.INTER_AREA)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    return x

# ===== Policy import =====
try:
    from lerobot.policies.utils import prepare_observation_for_inference
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.utils.constants import ACTION  # 可选，常量

except Exception as e:
    raise ImportError("Import policy pre/postprocessor failed; check your lerobot install") from e


# ===== Data containers =====

@dataclass
class TimedFrame:
    t: float
    left_bgr: np.ndarray
    right_bgr: np.ndarray


# ===== Node =====

class W1ACTFlexibleNode(Node):
    def __init__(self) -> None:
        super().__init__('w1_act_flexible_node')

        # ---------- Parameters ----------
        # Core
        self.declare_parameter('policy_path', '/path/to/pretrained_model')
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('policy_hz', 15.0)
        self.declare_parameter('tolerance_ms', 50.0)
        self.declare_parameter('remote_server_host', '127.0.0.1')
        self.declare_parameter('remote_server_port', 8899)
        self.declare_parameter('remote_connect_timeout_s', 10.0)
        self.declare_parameter('remote_horizon_n', 30)
        self.declare_parameter('remote_replan_trigger', 0)

        # Topics
        self.declare_parameter('joint_topic', '/feedback/robot_server_state')
        self.declare_parameter('publish_topic', '/w1/policy/desired_joint_positions')

        # Images / state keys
        self.declare_parameter('head_target_width', 640)
        self.declare_parameter('head_target_height', 360)
        self.declare_parameter('hand_target_width', 640)
        self.declare_parameter('hand_target_height', 480)
        # self.declare_parameter('image_left_key', 'observation.images.cam_high_left')
        # self.declare_parameter('image_right_key', 'observation.images.cam_high_right')
        # 手部相机图像键
        self.declare_parameter('image_hand_left_key', 'observation.images.cam_hand_left')
        self.declare_parameter('image_hand_right_key', 'observation.images.cam_hand_right')
        self.declare_parameter('state_key', 'observation.state')

        self.declare_parameter('image_keys', [
            "observation.images.cam_high_left",
            "observation.images.cam_high_right",
            "observation.images.cam_hand_left",
            "observation.images.cam_hand_right",
        ])
        # Canonical 20 BODY names
        canonical_body = [
            'ANKLE','KNEE','BUTTOCK','WAIST',
            'LEFT_J1','LEFT_J2','LEFT_J3','LEFT_J4','LEFT_J5','LEFT_J6','LEFT_J7',
            'NECK1','NECK2',
            'RIGHT_J1','RIGHT_J2','RIGHT_J3','RIGHT_J4','RIGHT_J5','RIGHT_J6','RIGHT_J7',
        ]
        self.declare_parameter('ordered_body_names', canonical_body)

        # YOU provide the full list to define model IO order (can include placeholders)
        self.declare_parameter('selected_body_names', canonical_body)

        # Optional drops (publish stage)
        self.declare_parameter('drop_joint_names', ['ANKLE','KNEE','BUTTOCK'])

        # Hands: mode + side(s)
        self.declare_parameter('hand_input_mode', 'none')       # none | scalar | qpos6
        self.declare_parameter('hand_sides', ['left','right'], ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY))
        self.declare_parameter('hand_sides_str', '')            # "left" | "right" | "both"

        # Hand placeholder names (for order definition)
        self.declare_parameter('left_hand_scalar_name', 'LEFT_GRIPPER')
        self.declare_parameter('right_hand_scalar_name', 'RIGHT_GRIPPER')

        # Hand qpos6 names (publishing & optional order expansion)
        self.declare_parameter('left_hand_qpos6_names',  [
            'LEFT_HAND_THUMB1','LEFT_HAND_THUMB2','LEFT_HAND_INDEX',
            'LEFT_HAND_MIDDLE','LEFT_HAND_RING','LEFT_HAND_PINKY'
        ])
        self.declare_parameter('right_hand_qpos6_names', [
            'RIGHT_HAND_THUMB1','RIGHT_HAND_THUMB2','RIGHT_HAND_INDEX',
            'RIGHT_HAND_MIDDLE','RIGHT_HAND_RING','RIGHT_HAND_PINKY'
        ])

        # Input (sensor) topics for hands
        self.declare_parameter('left_hand_scalar_topic',  '/hand/left_scalar')
        self.declare_parameter('right_hand_scalar_topic', '/hand/right_scalar')
        self.declare_parameter('left_hand_qpos6_topic',   '/feedback_sim/hand/left')
        self.declare_parameter('right_hand_qpos6_topic',  '/feedback_sim/hand/right')

        # Output command topics for hands
        self.declare_parameter('set_left_hand_qpos6_topic',  '/control/ee/left')
        self.declare_parameter('set_right_hand_qpos6_topic', '/control/ee/right')

        # 手部相机对应的 topic
        self.declare_parameter('cam_hand_left_topic',  '/camera_l/color/image_rect_raw')
        self.declare_parameter('cam_hand_right_topic', '/camera_r/color/image_rect_raw')

        # v1 手势表（scalar→qpos6 使用）
        self.hand_gestures = {
            "normal":[0.0,   70.0,   0.0,    0.0,    0.0,    0.0],
            "normal2":[0.0,   100.0,   0.0,    0.0,    0.0,    0.0],
            "cup": [0.0,  100.0,  35.0,   45.0,   47.0,  37.0],
            "pinch": [65.0,  100.0,  70.0,   75.0,   100.0,  100.0],
            "fist":  [100.0, 30.0,   100.0,  100.0,  100.0,  100.0],
            "like":  [0.0,   0.0,    100.0,  100.0,  100.0,  100.0],
            "heart": [0.0,   100.0,  60.0,   70.0,   60.0,   60.0],
            "bull":  [90.0,  80.0,   0.0 ,   100.0,  100.0,  0.0],
            "gun":   [0.0,   0.0,    0.0,    100.0,  100.0,  100.0],
            "six":   [0.0,   0.0,    100.0,  100.0,  100.0,  0.0],
            "one":   [100.0, 70.0,   0.0,    100.0,  100.0,  100.0],
            "salute":[100.0, 0.0,    0.0,    0.0,    0.0,    0.0],
            "ok":    [60.0,  90.0,   60.0,   0.0,    0.0,    0.0],
        }

        self.joint_names = [
            "ANKLE","KNEE", "BUTTOCK",  "WAIST", "NECK1", "NECK2",
            "LEFT_J1","LEFT_J2","LEFT_J3","LEFT_J4","LEFT_J5", "LEFT_J6","LEFT_J7",
            "RIGHT_J1","RIGHT_J2", "RIGHT_J3","RIGHT_J4","RIGHT_J5","RIGHT_J6","RIGHT_J7",
        ]
        
        self.declare_parameter("hand_interp_start", "normal2")
        self.declare_parameter("hand_interp_end", "pinch")

        # Gripper postprocess (inference-time, low investment)
        self.declare_parameter('gripper_binarize', True)       # threshold to 0/1
        self.declare_parameter('gripper_thr', 0.5)             # simple threshold
        self.declare_parameter('gripper_hysteresis', True)     # use open/close thresholds
        self.declare_parameter('gripper_thr_close', 0.55)      # 0->1
        self.declare_parameter('gripper_thr_open', 0.45)       # 1->0
        self.declare_parameter('freeze_after_release_s', 0.0)  # >0 to stop re-grasp loop after release


        # ---------- Read params ----------
        gp = self.get_parameter
        self.policy_path: str = gp('policy_path').get_parameter_value().string_value
        self.device_str: str  = gp('device').get_parameter_value().string_value
        self.policy_hz: float = float(gp('policy_hz').get_parameter_value().double_value)
        self.tolerance_s: float = float(gp('tolerance_ms').get_parameter_value().double_value) / 1000.0
        self.remote_server_host: str = gp('remote_server_host').get_parameter_value().string_value
        self.remote_server_port: int = int(gp('remote_server_port').get_parameter_value().integer_value)
        self.remote_connect_timeout_s: float = float(gp('remote_connect_timeout_s').get_parameter_value().double_value)
        self.remote_horizon_n: int = int(gp('remote_horizon_n').get_parameter_value().integer_value)
        self.remote_replan_trigger: int = int(gp('remote_replan_trigger').get_parameter_value().integer_value)

        self.joint_topic: str = gp('joint_topic').get_parameter_value().string_value
        self.publish_topic: str = gp('publish_topic').get_parameter_value().string_value

        self.head_target_size = (
            int(gp('head_target_width').get_parameter_value().integer_value),
            int(gp('head_target_height').get_parameter_value().integer_value),
        )
        self.hand_target_size = (
            int(gp('hand_target_width').get_parameter_value().integer_value),
            int(gp('hand_target_height').get_parameter_value().integer_value),
        )
        self.state_key       = gp('state_key').get_parameter_value().string_value

        self.body_canonical: List[str] = list(gp('ordered_body_names').get_parameter_value().string_array_value)
        selected_raw: List[str] = list(gp('selected_body_names').get_parameter_value().string_array_value)

        self.drop_set = set(list(gp('drop_joint_names').get_parameter_value().string_array_value))

        mode = gp('hand_input_mode').get_parameter_value().string_value.lower().strip()
        assert mode in ('none','scalar','qpos6'), "hand_input_mode must be none|scalar|qpos6"
        self.hand_input_mode = mode

        # sides：hand_sides_str 优先
        sides_arr = [s.lower() for s in list(gp('hand_sides').get_parameter_value().string_array_value)]
        sides_str = gp('hand_sides_str').get_parameter_value().string_value.lower().strip()
        if sides_str in ('left','right','both'):
            self.hand_sides = ['left','right'] if sides_str == 'both' else [sides_str]
        else:
            self.hand_sides = [s for s in ['left','right'] if s in sides_arr]

        self.left_scalar_name  = gp('left_hand_scalar_name').get_parameter_value().string_value or 'LEFT_GRIPPER'
        self.right_scalar_name = gp('right_hand_scalar_name').get_parameter_value().string_value or 'RIGHT_GRIPPER'
        self.left_q6_names  = list(gp('left_hand_qpos6_names').get_parameter_value().string_array_value)
        self.right_q6_names = list(gp('right_hand_qpos6_names').get_parameter_value().string_array_value)

        self.left_scalar_topic  = gp('left_hand_scalar_topic').get_parameter_value().string_value
        self.right_scalar_topic = gp('right_hand_scalar_topic').get_parameter_value().string_value
        self.left_qpos6_topic   = gp('left_hand_qpos6_topic').get_parameter_value().string_value
        self.right_qpos6_topic  = gp('right_hand_qpos6_topic').get_parameter_value().string_value

        self.set_left_qpos6_topic  = gp('set_left_hand_qpos6_topic').get_parameter_value().string_value
        self.set_right_qpos6_topic = gp('set_right_hand_qpos6_topic').get_parameter_value().string_value

        self.hand_interp_start = self.get_parameter("hand_interp_start").get_parameter_value().string_value
        self.hand_interp_end   = self.get_parameter("hand_interp_end").get_parameter_value().string_value

        # Gripper postprocess (inference-time)
        self.gripper_binarize = bool(self.get_parameter('gripper_binarize').get_parameter_value().bool_value)
        self.gripper_thr = float(self.get_parameter('gripper_thr').get_parameter_value().double_value)
        self.gripper_hysteresis = bool(self.get_parameter('gripper_hysteresis').get_parameter_value().bool_value)
        self.gripper_thr_close = float(self.get_parameter('gripper_thr_close').get_parameter_value().double_value)
        self.gripper_thr_open  = float(self.get_parameter('gripper_thr_open').get_parameter_value().double_value)
        self.freeze_after_release_s = float(self.get_parameter('freeze_after_release_s').get_parameter_value().double_value)
        self.freeze_until = 0.0
        self._grip_bin = {'left': 0, 'right': 0}
        self._prev_grip_bin = {'left': 0, 'right': 0}


        ALLOWED_IMAGE_KEYS = {
            "observation.images.cam_high_left",
            "observation.images.cam_high_right",
            "observation.images.cam_hand_left",
            "observation.images.cam_hand_right",
        }
        req_keys = list(self.get_parameter('image_keys').get_parameter_value().string_array_value)
        self.image_keys = [k for k in req_keys if k in ALLOWED_IMAGE_KEYS]
        if not self.image_keys:
            raise ValueError("image_keys 不能为空：必须从四个允许键中至少选择一个。")

        self.cam_hand_left_topic  = self.get_parameter('cam_hand_left_topic').get_parameter_value().string_value
        self.cam_hand_right_topic = self.get_parameter('cam_hand_right_topic').get_parameter_value().string_value

        # ---------- Build IO order ----------
        self.full_order: List[str] = self._build_full_order(selected_raw)
        self.full_dim = len(self.full_order)

        self.body_order: List[str] = [n for n in self.full_order if n in self.body_canonical]

        self.idx_left_scalar: Optional[int] = None
        self.idx_right_scalar: Optional[int] = None
        self.slice_left_q6: Optional[Tuple[int,int]] = None
        self.slice_right_q6: Optional[Tuple[int,int]] = None
        for i, n in enumerate(self.full_order):
            if n == self.left_scalar_name:  self.idx_left_scalar = i
            if n == self.right_scalar_name: self.idx_right_scalar = i
        def find_block(names: List[str]) -> Optional[Tuple[int,int]]:
            L = len(names)
            for i in range(0, len(self.full_order) - L + 1):
                if self.full_order[i:i+L] == names:
                    return (i, i+L)
            return None
        if 'left' in self.hand_sides and self.hand_input_mode == 'qpos6':
            self.slice_left_q6 = find_block(self.left_q6_names)
        if 'right' in self.hand_sides and self.hand_input_mode == 'qpos6':
            self.slice_right_q6 = find_block(self.right_q6_names)

        self.get_logger().info(
            "=== Inference Order (len=%d) ===\n%s" %
            (self.full_dim, "\n".join([f"{i:02d}: {n}" for i,n in enumerate(self.full_order)]))
        )
        self.get_logger().info(
            f"[train IO] BODY={len(self.body_order)} HAND_MODE={self.hand_input_mode} sides={self.hand_sides} → full_dim={self.full_dim}"
        )
        if self.hand_input_mode == 'scalar':
            self.get_logger().info(f"scalar idx: left={self.idx_left_scalar}, right={self.idx_right_scalar}")
        elif self.hand_input_mode == 'qpos6':
            self.get_logger().info(f"q6 slices: left={self.slice_left_q6}, right={self.slice_right_q6}")

        if self.full_dim == 0:
            raise RuntimeError("selected_body_names 展开后为空；请至少选择一个维度")

        # ---------- QoS & pubs/subs ----------
        q_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.pub_action = self.create_publisher(JointPositionControl, self.publish_topic, q_reliable)
        self.pub_set_left_qpos6  = self.create_publisher(EEJointControl, self.set_left_qpos6_topic,  q_reliable)
        self.pub_set_right_qpos6 = self.create_publisher(EEJointControl, self.set_right_qpos6_topic, q_reliable)
        self.hand_joint_name = ["T_CMC_YAW","T_MCP","IF_MCP_PITCH","MF_MCP_PITCH","RF_MCP_PITCH","LF_MCP_PITCH"]

        # Buffers
        self.body_buf: deque = deque(maxlen=2000)
        self.body_index_map: Optional[np.ndarray] = None
        self.hand_left_buf:  deque = deque(maxlen=200)
        self.hand_right_buf: deque = deque(maxlen=200)
        self.bridge = CvBridge()
        self.scalar_left_buf: deque  = deque(maxlen=2000)
        self.scalar_right_buf: deque = deque(maxlen=2000)
        self.qpos6_left_buf: deque   = deque(maxlen=2000)
        self.qpos6_right_buf: deque  = deque(maxlen=2000)

        # Subs
        q_img = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )
        if "observation.images.cam_hand_left" in self.image_keys:
            self.create_subscription(Image, self.cam_hand_left_topic,  self.cb_img_hand_left,  q_img)
            self.get_logger().info(f"Subscribe hand-left image: {self.cam_hand_left_topic}")
        if "observation.images.cam_hand_right" in self.image_keys:
            self.create_subscription(Image, self.cam_hand_right_topic, self.cb_img_hand_right, q_img)
            self.get_logger().info(f"Subscribe hand-right image: {self.cam_hand_right_topic}")

        self.create_subscription(String, self.joint_topic, self.cb_joint, q_reliable)
        self.create_subscription(Float64, self.left_scalar_topic,  self.cb_scalar_left,  q_reliable)
        self.create_subscription(Float64, self.right_scalar_topic, self.cb_scalar_right, q_reliable)
        self.create_subscription(EEFeedback, self.left_qpos6_topic,  self.cb_qpos6_left,  q_reliable)
        self.create_subscription(EEFeedback, self.right_qpos6_topic, self.cb_qpos6_right, q_reliable)

        # Local preprocess/postprocess + remote policy
        self.device = torch.device(self.device_str if (self.device_str == 'cuda' and torch.cuda.is_available()) else 'cpu')
        try:
            self.preprocessor = PolicyProcessorPipeline.from_pretrained(
                self.policy_path,
                config_filename="policy_preprocessor.json",
                local_files_only=True,
            )
            self.postprocessor = PolicyProcessorPipeline.from_pretrained(
                self.policy_path,
                config_filename="policy_postprocessor.json",
                local_files_only=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load policy_preprocessor.json/policy_postprocessor.json from {self.policy_path}: {e}"
            )

        self._configure_processor_devices()
        self.get_logger().info("Loaded policy_preprocessor.json / policy_postprocessor.json.")

        self.network_client = NetworkClient(
            self.remote_server_host, self.remote_server_port
        )
        self.get_logger().info(
            f"Connecting remote policy server at "
            f"{self.remote_server_host}:{self.remote_server_port}"
        )
        if not self.network_client.connect(timeout=self.remote_connect_timeout_s):
            raise RuntimeError(
                f"Could not connect to remote policy server "
                f"{self.remote_server_host}:{self.remote_server_port}"
            )
        reset_resp = self.network_client.send_request("reset_policy")
        if not reset_resp or reset_resp.get("status") != "ok":
            self.get_logger().warn("reset_policy request failed, continue anyway.")
        self.get_logger().info("Remote policy server connected.")

        # ---------- Async planning buffers ----------
        self.plan_lock = threading.Lock()
        self.plan_queue: deque = deque()      # 每个元素: np.ndarray(full_dim)
        self.replan_in_progress: bool = False

        # horizon: 一次重推理拿多少帧 (例如30)
        self.horizon_N: int = max(1, self.remote_horizon_n)
        # <= 这个阈值时触发异步补货 (例如15)
        if self.remote_replan_trigger > 0:
            self.replan_trigger: int = self.remote_replan_trigger
        else:
            # self.replan_trigger = max(1, self.horizon_N // 2)
            self.replan_trigger = self.remote_replan_trigger
        self.get_logger().info(
            f"[Remote ACT] horizon_N={self.horizon_N}, "
            f"replan_trigger={self.replan_trigger}"
        )

        # --- blend相关: chunk衔接淡化 ---
        self.last_cmd_vec = np.zeros(self.full_dim, dtype=np.float32)
        self.have_last_cmd = False
        self.blend_steps = 20  # 过渡长度(帧数)，越小越"看不出来"

        # Blend mask: do NOT blend gripper/hand dims (prevents 'grab then drop' at chunk boundary)
        self.blend_mask = np.ones(self.full_dim, dtype=np.float32)
        if self.hand_input_mode == 'scalar':
            if self.idx_left_scalar is not None:
                self.blend_mask[self.idx_left_scalar] = 0.0
            if self.idx_right_scalar is not None:
                self.blend_mask[self.idx_right_scalar] = 0.0
        elif self.hand_input_mode == 'qpos6':
            if self.slice_left_q6 is not None:
                a,b = self.slice_left_q6
                self.blend_mask[a:b] = 0.0
            if self.slice_right_q6 is not None:
                a,b = self.slice_right_q6
                self.blend_mask[a:b] = 0.0

        # Camera thread (SBS)
        self.frame_lock = threading.Lock()
        self.latest_frame: Optional[TimedFrame] = None
        self._cap_stop = False
        th = threading.Thread(target=self.capture_loop_sbs, daemon=True)
        th.start()
        self._cap_thread = th

        # Timer
        period = 1.0 / max(1.0, self.policy_hz)
        self.create_timer(period, self.timer_infer)

    # ---------- Build order helper ----------
    def _build_full_order(self, selected_raw: List[str]) -> List[str]:
        mode = self.hand_input_mode
        sides = self.hand_sides
        out: List[str] = []
        body_set = set(self.body_canonical)

        for n in selected_raw:
            if n in body_set:
                out.append(n)
                continue

            if mode == 'none':
                continue

            if mode == 'scalar':
                if n == self.left_scalar_name and 'left' in sides:
                    out.append(self.left_scalar_name)
                    continue
                if n == self.right_scalar_name and 'right' in sides:
                    out.append(self.right_scalar_name)
                    continue
                continue

            if mode == 'qpos6':
                if n == self.left_scalar_name and 'left' in sides:
                    out.extend(self.left_q6_names)
                    continue
                if n == self.right_scalar_name and 'right' in sides:
                    out.extend(self.right_q6_names)
                    continue
                if n in self.left_q6_names and 'left' in sides:
                    out.append(n); continue
                if n in self.right_q6_names and 'right' in sides:
                    out.append(n); continue
                continue

        if mode == 'scalar':
            if 'left' in sides and self.left_scalar_name not in out:
                out.append(self.left_scalar_name)
            if 'right' in sides and self.right_scalar_name not in out:
                out.append(self.right_scalar_name)
        elif mode == 'qpos6':
            def ensure_block(block: List[str]):
                if not any(n in out for n in block):
                    out.extend(block)
            if 'left' in sides:
                ensure_block(self.left_q6_names)
            if 'right' in sides:
                ensure_block(self.right_q6_names)

        return out

    # ---------- Camera ----------
    def capture_loop_sbs(self) -> None:
        import cv2, time
        cap = cv2.VideoCapture('/dev/video99', cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(99)
        if not cap.isOpened():
            self.get_logger().error('OpenCV cannot open /dev/video99 (or id 99).')
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)

        try:
            while not self._cap_stop:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.005); continue
                t = now_sec()
                h, w2, _ = frame.shape
                if w2 % 2 != 0:
                    self.get_logger().warn(f'Frame width {w2} not even; skip')
                    continue
                w = w2 // 2
                left_bgr  = frame[:, :w]
                right_bgr = frame[:,  w:]
                with self.frame_lock:
                    self.latest_frame = TimedFrame(t=t, left_bgr=left_bgr, right_bgr=right_bgr)
        finally:
            cap.release()

    # ---------- Subs ----------
    def cb_joint(self, msg: String) -> None:
        # 0. body_order 还是老逻辑
        if len(self.body_order) == 0:
            return

        # 1. 解析 JSON 字符串
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().warn(f"JSON parse error in /feedback/robot_server_state: {e}")
            return
        
        joint_pos = data.get("joint_position", None)
        if joint_pos is None or len(joint_pos) != len(self.joint_names):
            self.get_logger().warn(
                f"joint_position length {0 if joint_pos is None else len(joint_pos)} "
                f"!= expected {len(self.joint_names)}"
            )
            return

        # 2. 时间戳 now_sec()
        t = now_sec()
        q_in = np.asarray(joint_pos, dtype=np.float32)

        # 3. 构建 body_index_map：
        #    joint_position 的顺序固定是 JOINT_NAMES
        if self.body_index_map is None:
            name_to_idx = {n: i for i, n in enumerate(self.joint_names)}
            idxs, missing = [], []
            for n in self.body_order:
                if n in name_to_idx:
                    idxs.append(name_to_idx[n])
                else:
                    missing.append(n)
            if missing:
                self.get_logger().warn(
                    f"Incoming robot_server_state missing BODY joints: {missing}. Waiting..."
                )
                return
            self.body_index_map = np.asarray(idxs, dtype=np.int64)
            self.get_logger().info(
                f"Mapped robot_server_state joint_position to BODY order "
                f"({len(self.body_index_map)} joints)."
            )

        if self.body_index_map is None:
            return

        # 4. 和原来一样：重排 + append 到 buf
        try:
            q_body = q_in[self.body_index_map]
        except Exception as e:
            self.get_logger().warn(f"Reorder BODY failed: {e}")
            return

        self.body_buf.append((t, q_body))

    def cb_scalar_left(self, msg: Float64) -> None:
        self.scalar_left_buf.append((now_sec(), float(msg.data)))

    def cb_scalar_right(self, msg: Float64) -> None:
        self.scalar_right_buf.append((now_sec(), float(msg.data)))

    def _extract_ee_positions(self, msg: EEFeedback, joint_names: List[str]) -> Optional[np.ndarray]:
        states = list(msg.joint_states)
        if not states:
            return None

        by_name = {state.name: float(state.position) for state in states}
        if all(name in by_name for name in joint_names):
            return np.asarray([by_name[name] for name in joint_names], dtype=np.float32)

        if len(states) >= len(joint_names):
            return np.asarray([state.position for state in states[:len(joint_names)]], dtype=np.float32)

        return None

    def cb_qpos6_left(self, msg: EEFeedback) -> None:
        arr = self._extract_ee_positions(msg, self.hand_joint_name)
        if arr is not None:
            self.qpos6_left_buf.append((now_sec(), arr.copy()))

    def cb_qpos6_right(self, msg: EEFeedback) -> None:
        arr = self._extract_ee_positions(msg, self.hand_joint_name)
        if arr is not None:
            self.qpos6_right_buf.append((now_sec(), arr.copy()))

    def cb_img_hand_left(self, msg: Image) -> None:
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge left-hand fail: {e}")
            return
        t = now_sec()
        self.hand_left_buf.append((t, bgr))

    def cb_img_hand_right(self, msg: Image) -> None:
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge right-hand fail: {e}")
            return
        t = now_sec()
        self.hand_right_buf.append((t, bgr))

    # ---------- Scalar → qpos6 ----------
    def map_scalar_to_qpos6(self, s: float) -> np.ndarray:
        s = float(np.clip(s, 0.0, 1.0))
        start, end = self.hand_interp_start, self.hand_interp_end
        if start not in self.hand_gestures or end not in self.hand_gestures:
            raise ValueError(f"Hand gesture must be in {list(self.hand_gestures.keys())}")
        base   = np.asarray(self.hand_gestures[start], dtype=np.float32)
        target = np.asarray(self.hand_gestures[end],   dtype=np.float32)
        return base * (1.0 - s) + target * s

    def _configure_processor_devices(self) -> None:
        """Ensure processor steps run on correct devices.

        - Preprocessor should prepare model inputs on `self.device` (typically cuda).
        - Postprocessor should output actions on CPU for ROS publishing.
        """
        if getattr(self, "preprocessor", None) is not None:
            for step in self.preprocessor.steps:
                if step.__class__.__name__ == "DeviceProcessorStep":
                    step.device = self.device.type
                elif step.__class__.__name__ == "NormalizerProcessorStep" and hasattr(step, "to"):
                    step.to(device=self.device.type)

        if getattr(self, "postprocessor", None) is not None:
            for step in self.postprocessor.steps:
                if step.__class__.__name__ == "DeviceProcessorStep":
                    step.device = "cpu"
                elif step.__class__.__name__ == "UnnormalizerProcessorStep" and hasattr(step, "to"):
                    step.to(device="cpu")


    def _preprocess_observation(self, obs_np: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Prepare observation dict for ACTPolicy.select_action().

        - Uses `prepare_observation_for_inference` to convert numpy images/state to torch tensors.
        - Applies `self.preprocessor` if available (batch/device/normalization).
        """
        obs_t = prepare_observation_for_inference(obs_np, self.device)
        if self.preprocessor is None:
            # Minimal fallback: ensure batch dimension exists.
            out: Dict[str, torch.Tensor] = {}
            for k, v in obs_t.items():
                if isinstance(v, torch.Tensor) and v.ndim in (1, 3):
                    out[k] = v.unsqueeze(0)
                else:
                    out[k] = v
            return out
        return self.preprocessor(obs_t)



    def _postprocess_gripper_scalar(self, s_raw: float, side: str) -> float:
        """Map raw gripper scalar to a stable command.
        - Optional binarization (0/1) with hysteresis
        - Optional freeze-after-release to avoid 'do it twice' loops
        """
        s_raw = float(np.clip(float(s_raw), 0.0, 1.0))
        if not self.gripper_binarize:
            return s_raw

        if not self.gripper_hysteresis:
            s_bin = 1.0 if s_raw >= self.gripper_thr else 0.0
        else:
            st = int(self._grip_bin.get(side, 0))
            if st == 0 and s_raw >= self.gripper_thr_close:
                st = 1
            elif st == 1 and s_raw <= self.gripper_thr_open:
                st = 0
            self._grip_bin[side] = st
            s_bin = float(st)

        prev = int(self._prev_grip_bin.get(side, 0))
        nowb = int(round(s_bin))
        if prev == 1 and nowb == 0 and self.freeze_after_release_s > 0.0:
            self.freeze_until = max(self.freeze_until, now_sec() + self.freeze_after_release_s)
        self._prev_grip_bin[side] = nowb
        return s_bin


    # ---------- Helpers for async inference ----------
    def _action_to_np(self, action) -> Optional[np.ndarray]:
        if isinstance(action, torch.Tensor):
            act = action.detach().cpu().numpy()
            if act.ndim == 2:
                act = act[0]
        else:
            act = np.asarray(action, dtype=np.float32)

        if act.shape[-1] != self.full_dim:
            self.get_logger().error(
                f'Policy dim mismatch: expected {self.full_dim}, got {act.shape}. '
                f'Order: {self.full_order}'
            )
            return None

        return act.astype(np.float32, copy=True)


    def _postprocess_action(self, a: torch.Tensor) -> torch.Tensor:
        """Apply official postprocessor (unnormalize action + move to CPU).

        The exported policy_postprocessor.json typically contains:
        UnnormalizerProcessorStep(ACTION) -> DeviceProcessorStep(cpu)
        We pass {ACTION: a} and retrieve ACTION from the result.
        """
        if getattr(self, "postprocessor", None) is None:
            self.get_logger().warn("Warning: no postprocessor found; returning raw action.")
            return a
        try:
            out = self.postprocessor({ACTION: a})
        except Exception:
            out = self.postprocessor({"action": a})
        if isinstance(out, dict):
            if ACTION in out:
                return out[ACTION]
            if "action" in out:
                return out["action"]
        if isinstance(out, torch.Tensor):
            return out
        self.get_logger().warn("Warning: postprocessor output has unexpected format; returning raw action.")
        return a

    def _publish_action_vector(self, act_np: np.ndarray) -> None:
        # 身体
        pub_names: List[str] = []
        pub_pos:   List[float] = []
        for i, n in enumerate(self.full_order):
            if n in self.body_canonical and n not in self.drop_set:
                pub_names.append(n)
                pub_pos.append(float(act_np[i]))

        if pub_names:
            msg = JointPositionControl()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = pub_names
            msg.position = pub_pos
            self.pub_action.publish(msg)

        # 手
        if self.hand_input_mode == 'scalar':
            if 'left' in self.hand_sides and self.idx_left_scalar is not None:
                s_raw = float(act_np[self.idx_left_scalar])
                s = self._postprocess_gripper_scalar(s_raw, side='left')
                act_np[self.idx_left_scalar] = s
                q6 = self.map_scalar_to_qpos6(s)
                # m = Float64MultiArray(); m.data = [float(x) for x in q6]
                # self.pub_set_left_qpos6.publish(m)
                m = EEJointControl()
                m.mode = EEJointControlMode.POSITION
                m.name = self.hand_joint_name
                m.value = [float(x) for x in q6]   # v 长度=6
                self.pub_set_left_qpos6.publish(m)

            if 'right' in self.hand_sides and self.idx_right_scalar is not None:
                s_raw = float(act_np[self.idx_right_scalar])
                s = self._postprocess_gripper_scalar(s_raw, side='right')
                act_np[self.idx_right_scalar] = s
                q6 = self.map_scalar_to_qpos6(s)
                # m = Float64MultiArray(); m.data = [float(x) for x in q6]
                # self.pub_set_right_qpos6.publish(m)
                m = EEJointControl()
                m.mode = EEJointControlMode.POSITION
                m.name = self.hand_joint_name
                m.value = [float(x) for x in q6]   # v 长度=6
                self.pub_set_right_qpos6.publish(m)

        elif self.hand_input_mode == 'qpos6':
            if 'left' in self.hand_sides and self.slice_left_q6 is not None:
                a, b = self.slice_left_q6
                v = act_np[a:b]
                # m = Float64MultiArray(); m.data = [float(x) for x in v]
                # self.pub_set_left_qpos6.publish(m)
                m = EEJointControl()
                m.mode = EEJointControlMode.POSITION
                m.name = self.hand_joint_name
                m.value = [float(x) for x in v]   # v 长度=6
                self.pub_set_left_qpos6.publish(m)
            if 'right' in self.hand_sides and self.slice_right_q6 is not None:
                a, b = self.slice_right_q6
                v = act_np[a:b]
                # m = Float64MultiArray(); m.data = [float(x) for x in v]
                # self.pub_set_right_qpos6.publish(m)
                m = EEJointControl()
                m.mode = EEJointControlMode.POSITION
                m.name = self.hand_joint_name
                m.value = [float(x) for x in v]   # v 长度=6
                self.pub_set_right_qpos6.publish(m)

        # 记录：我们刚刚真的下发给机器人的这一帧
        self.last_cmd_vec = act_np.astype(np.float32, copy=True)
        self.have_last_cmd = True

    def _make_seed_batch(
    self,
    obs_base: Dict[str, np.ndarray],
    seed_vec_np: np.ndarray
) -> Dict[str, np.ndarray]:
        obs = dict(obs_base)
        obs[self.state_key] = seed_vec_np.astype(np.float32)
        batch_t = self._preprocess_observation(obs)
        batch_np: Dict[str, np.ndarray] = {}
        for key, value in batch_t.items():
            # 只保留 observation.* 开头的字段，其余全部丢弃
            if not key.startswith("observation."):
                continue
            if isinstance(value, torch.Tensor):
                batch_np[key] = value.detach().cpu().numpy()
            else:
                arr = np.asarray(value)
                if arr.dtype != object:
                    batch_np[key] = arr
        return batch_np

    def _ensure_remote_connected(self) -> bool:
        if self.network_client.connected:
            return True
        self.get_logger().warn(
            f"Remote policy disconnected, reconnecting "
            f"{self.remote_server_host}:{self.remote_server_port} ..."
        )
        ok = self.network_client.connect(timeout=self.remote_connect_timeout_s)
        if ok:
            reset_resp = self.network_client.send_request("reset_policy")
            if not reset_resp or reset_resp.get("status") != "ok":
                self.get_logger().warn(
                    "reset_policy after reconnect failed, continue anyway."
                )
        return ok

    def _remote_reset_policy(self) -> bool:
        if not self._ensure_remote_connected():
            return False
        response = self.network_client.send_request("reset_policy")
        return bool(response and response.get("status") == "ok")

    def _remote_select_action(self, batch: Dict[str, np.ndarray]) -> Optional[torch.Tensor]:
        if not self._ensure_remote_connected():
            self.get_logger().error("Reconnect remote policy server failed.")
            return None
        payload = {"batch": {k: np.asarray(v) for k, v in batch.items()}}
        response = self.network_client.send_request("select_action", payload)
        if not response:
            self.get_logger().error("No response from remote policy server.")
            return None
        if response.get("status") != "success":
            self.get_logger().error(f"Remote policy error: {response.get('message')}")
            return None
        action_np = np.asarray(response.get("action"), dtype=np.float32)
        return torch.from_numpy(action_np)

    def _remote_select_action_chunk(
        self, batch: Dict[str, np.ndarray]
    ) -> Optional[np.ndarray]:
        if not self._ensure_remote_connected():
            self.get_logger().error("Reconnect remote policy server failed.")
            return None
        payload = {
            "batch": {k: np.asarray(v) for k, v in batch.items()},
            "n_action_steps": int(self.horizon_N),
        }
        response = self.network_client.send_request("select_action_chunk", payload)
        if not response:
            self.get_logger().error("No response from remote policy server.")
            return None
        if response.get("status") != "success":
            self.get_logger().error(f"Remote policy error: {response.get('message')}")
            return None
        actions = np.asarray(response.get("actions"), dtype=np.float32)
        if actions.ndim == 3:
            actions = actions[0]
        if actions.ndim != 2:
            self.get_logger().error(f"Chunk shape is invalid: {actions.shape}")
            return None
        return actions

    def _spawn_replan(self, batch: Dict[str, np.ndarray], sync: bool = False) -> None:
        if self.replan_in_progress:
            return

        self.replan_in_progress = True

        def worker():
            actions_local: List[np.ndarray] = []
            if not self._remote_reset_policy():
                self.get_logger().error("remote reset_policy failed in worker.")
                self.replan_in_progress = False
                return

            t0 = now_sec()
            try:
                chunk = self._remote_select_action_chunk(batch)
                if chunk is None:
                    self.replan_in_progress = False
                    return
                t1 = now_sec()
                infer_dt = t1 - t0
                if infer_dt > 0.1:
                    self.get_logger().info(
                        f'[async] heavy remote chunk inference {infer_dt:.3f}s (worker)'
                    )

                for i in range(chunk.shape[0]):
                    a = torch.from_numpy(chunk[i])
                    a = self._postprocess_action(a)
                    act_np = self._action_to_np(a)
                    if act_np is None:
                        break
                    actions_local.append(act_np)
            except Exception as e:
                self.get_logger().error(f"remote select_action_chunk failed in worker: {e}")
                self.replan_in_progress = False
                return

            # === 把新chunk平滑拼接到 plan_queue 尾部 ===
            with self.plan_lock:
                # 1) 找到拼接对齐参考：优先用队列最后一帧，否则用上一次下发的命令
                if len(self.plan_queue) > 0:
                    prev_tail = self.plan_queue[-1].astype(np.float32, copy=True)
                    do_blend = True
                elif self.have_last_cmd:
                    prev_tail = self.last_cmd_vec.astype(np.float32, copy=True)
                    do_blend = True
                else:
                    prev_tail = None
                    do_blend = False

                # 2) 对新段的前 blend_steps 帧做 offset 淡化
                if actions_local and do_blend:
                    offset = (prev_tail - actions_local[0]) * self.blend_mask
                    Nblend = min(self.blend_steps, len(actions_local))
                    # alpha从1 -> ~0，线性衰减
                    denom = float(max(Nblend, 1))
                    for i in range(Nblend):
                        alpha = 1.0 - (i / denom)
                        actions_local[i] = (
                            actions_local[i] + offset * alpha
                        ).astype(np.float32, copy=False)

                # 3) 把处理后的帧塞到队列尾
                for x in actions_local:
                    self.plan_queue.append(x)

            self.replan_in_progress = False

        if sync:
            worker()
        else:
            th = threading.Thread(target=worker, daemon=True)
            th.start()

    # ---------- Timer ----------
    @torch.no_grad()
    def timer_infer(self) -> None:
        # 1) 最新相机帧
        with self.frame_lock:
            tf = self.latest_frame
        if tf is None:
            return
        t_anchor = tf.t

        # 2) proprio 状态 (按照 full_order)
        state_vals: List[float] = []

        if len(self.body_order) > 0:
            q_body = nearest(self.body_buf, tf.t, self.tolerance_s)
            # print("collect q body success")
            if q_body is None:
                print("here, no qbody")
                return
            body_map = {n: float(v) for n, v in zip(self.body_order, q_body)}
        else:
            body_map = {}

        left_scalar  = nearest(self.scalar_left_buf, tf.t, self.tolerance_s)
        right_scalar = nearest(self.scalar_right_buf, tf.t, self.tolerance_s)
        left_q6  = nearest(self.qpos6_left_buf, tf.t, self.tolerance_s)
        right_q6 = nearest(self.qpos6_right_buf, tf.t, self.tolerance_s)

        for n in self.full_order:
            if n in self.body_canonical:
                state_vals.append(body_map.get(n, 0.0))
                continue

            if self.hand_input_mode == 'scalar':
                if n == self.left_scalar_name and 'left' in self.hand_sides:
                    state_vals.append(0.0 if left_scalar is None else float(left_scalar))
                    continue
                if n == self.right_scalar_name and 'right' in self.hand_sides:
                    state_vals.append(0.0 if right_scalar is None else float(right_scalar))
                    continue

            elif self.hand_input_mode == 'qpos6':
                if n in self.left_q6_names and 'left' in self.hand_sides:
                    i = self.left_q6_names.index(n)
                    v = 0.0 if left_q6 is None else float(left_q6[i])
                    state_vals.append(v); continue
                if n in self.right_q6_names and 'right' in self.hand_sides:
                    i = self.right_q6_names.index(n)
                    v = 0.0 if right_q6 is None else float(right_q6[i])
                    state_vals.append(v); continue

            state_vals.append(0.0)

        state_vec = np.asarray(state_vals, dtype=np.float32)
        if state_vec.shape[0] != self.full_dim:
            self.get_logger().error(f"State dim {state_vec.shape} != expected {self.full_dim}")
            return

        # 3) 打包 observation (raw numpy)
        obs_np: Dict[str, np.ndarray] = {}
        head_W, head_H = int(self.head_target_size[0]), int(self.head_target_size[1])
        hand_W, hand_H = int(self.hand_target_size[0]), int(self.hand_target_size[1])

        for key in self.image_keys:
            if key == "observation.images.cam_high_left":
                if tf.left_bgr is None:
                    return
                rgb = bgr_to_hwc_rgb_resized(tf.left_bgr, (head_W, head_H))
                obs_np[key] = rgb
            elif key == "observation.images.cam_high_right":
                if tf.right_bgr is None:
                    return
                rgb = bgr_to_hwc_rgb_resized(tf.right_bgr, (head_W, head_H))
                obs_np[key] = rgb
            elif key == "observation.images.cam_hand_left":
                bgr = nearest(self.hand_left_buf, t_anchor, self.tolerance_s)
                if bgr is None:
                    return
                rgb = bgr_to_hwc_rgb_resized(bgr, (hand_W, hand_H))
                obs_np[key] = rgb
            elif key == "observation.images.cam_hand_right":
                bgr = nearest(self.hand_right_buf, t_anchor, self.tolerance_s)
                if bgr is None:
                    return
                rgb = bgr_to_hwc_rgb_resized(bgr, (hand_W, hand_H))
                obs_np[key] = rgb

        obs_np[self.state_key] = state_vec.astype(np.float32)
        # batch: Dict[str, torch.Tensor] = self._preprocess_observation(obs_np)

        # 3.5) 可选：放手后冻结一段时间，避免“做完又做一遍”
        # if self.freeze_until > now_sec():
        #     if self.have_last_cmd:
        #         self._publish_action_vector(self.last_cmd_vec)
        #     return

        # 4) 异步控制逻辑
        # 4a. 没有弹药就同步roll一段
        with self.plan_lock:
            qlen = len(self.plan_queue)
            busy = self.replan_in_progress

        if qlen == 0 and (not busy):
            print("here, qlen is 0 now")
            seed_batch = self._make_seed_batch(obs_np, state_vec)
            self._spawn_replan(seed_batch, sync=True)

            with self.plan_lock:
                qlen = len(self.plan_queue)
                busy = self.replan_in_progress

        # 4b. 从队列里拿一帧下发
        act_np: Optional[np.ndarray] = None
        with self.plan_lock:
            if self.plan_queue:
                act_np = self.plan_queue.popleft()
            qlen_after = len(self.plan_queue)
            busy_after = self.replan_in_progress

        if act_np is not None:
            self._publish_action_vector(act_np)

        # 4c. 弹药快用完了 -> 异步roll下一段（起点=未来目标姿态）
        if (qlen_after <= self.replan_trigger) and (not busy_after):
            with self.plan_lock:
                if self.plan_queue:
                    future_seed = self.plan_queue[-1].astype(np.float32, copy=True)
                else:
                    future_seed = state_vec.astype(np.float32, copy=True)

            seed_batch = self._make_seed_batch(obs_np, future_seed)
            self._spawn_replan(seed_batch, sync=False)

        return


# ===== main =====

def main() -> None:
    rclpy.init()
    node = W1ACTFlexibleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(node, '_cap_stop'):
            node._cap_stop = True
        if hasattr(node, '_cap_thread') and node._cap_thread.is_alive():
            node._cap_thread.join(timeout=1.0)
        if hasattr(node, 'network_client'):
            try:
                node.network_client.close()
            except Exception:
                pass
        node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
