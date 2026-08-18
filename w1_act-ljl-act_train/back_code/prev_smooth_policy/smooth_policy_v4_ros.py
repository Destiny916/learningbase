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

from std_msgs.msg import String
import json

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
    ts = np.fromiter((x[0] for x in buf), dtype=np.float64)
    i = int(np.argmin(np.abs(ts - t)))
    dt = float(ts[i] - t)
    if abs(dt) <= tol_s:
        return buf[i][1]
    return None


def bgr_to_chw_rgb_resized(img_bgr: np.ndarray, size: Tuple[int,
                                                            int]) -> np.ndarray:
    """BGR HxWx3 → RGB CHW float32 [0,1], resized to (W,H)."""
    import cv2
    tw, th = size
    x = cv2.resize(img_bgr, (tw, th), interpolation=cv2.INTER_AREA)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))  # C,H,W
    return x


# ---- 角度最短弧差值 & 映射工具（新增） ----
def shortest_delta_vec(src: np.ndarray, dst: np.ndarray,
                       ang_mask: np.ndarray) -> np.ndarray:
    """
    返回从 src 到 dst 的差值向量：
    - 对角度维：最短弧 Δ = atan2(sin(dst-src), cos(dst-src)) ∈ [-pi, pi]
    - 对非角度维：普通差值 dst - src
    """
    d = dst - src
    if ang_mask.any():
        dang = np.arctan2(np.sin(d), np.cos(d))
        d = np.where(ang_mask, dang, d)
    return d.astype(np.float32, copy=False)


def wrap_to_pi_vec(x: np.ndarray, ang_mask: np.ndarray) -> np.ndarray:
    """仅对角度维做 wrap 到 [-pi, pi]；非角度维不动。"""
    if not ang_mask.any():
        return x
    wrapped = (x + np.pi) % (2 * np.pi) - np.pi
    return np.where(ang_mask, wrapped, x).astype(np.float32, copy=False)


def smoothstep(s: float) -> float:
    """S 曲线（段内插值用）：0→1 平滑，抑制速度跃变"""
    s = float(np.clip(s, 0.0, 1.0))
    return s * s * (3.0 - 2.0 * s)


def half_hamming_weights(K: int) -> np.ndarray:
    """
    生成单调上升的“半个 Hamming 窗”（已归一化到 [0,1]），长度 K。
    用作双边淡化的基形（旧尾用上升， 新头用下降）。
    """
    if K <= 0:
        return np.zeros((0,), dtype=np.float32)
    i = np.arange(K, dtype=np.float32)
    # u 从 ~0.08 上升到 1.0
    u = 0.54 - 0.46 * np.cos(np.pi * (i + 1.0) / float(K))
    u0 = float(u[0])
    denom = max(1e-6, 1.0 - u0)
    r = (u - u0) / denom  # 归一化到 [0,1]
    return r.astype(np.float32)


# ===== Policy import =====
try:
    from act.modeling_act import ACTPolicy
except Exception as e:
    raise ImportError(
        "Import ACTPolicy failed; ensure act/modeling_act.py in PYTHONPATH"
    ) from e

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
        self.declare_parameter('policy_hz', 60.0)
        self.declare_parameter('tolerance_ms', 50.0)

        # >>> 新增：发布频率/插值设置 <<<
        self.declare_parameter('pub_hz', 120.0)  # 发布小定时器频率（粗→细）
        self.declare_parameter('interp_mode', 'linear')  # linear | smoothstep
        self.declare_parameter('blend_gamma', 0.4)  # Hamming 双边淡化最大让步系数 (0~1)

        # Topics
        self.declare_parameter('joint_topic', '/feedback/robot_server_state')
        self.declare_parameter('publish_topic',
                               '/w1/policy/desired_joint_positions')

        # Images / state keys
        self.declare_parameter('head_target_width', 640)
        self.declare_parameter('head_target_height', 360)
        self.declare_parameter('hand_target_width', 640)
        self.declare_parameter('hand_target_height', 360)
        self.declare_parameter('image_hand_left_key',
                               'observation.images.cam_hand_left')
        self.declare_parameter('image_hand_right_key',
                               'observation.images.cam_hand_right')
        self.declare_parameter('state_key', 'observation.state')

        self.declare_parameter('image_keys', [
            "observation.images.cam_high_left",
            "observation.images.cam_high_right",
            "observation.images.cam_hand_left",
            "observation.images.cam_hand_right",
        ])
        # Canonical 20 BODY names
        canonical_body = [
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
        ]
        self.declare_parameter('ordered_body_names', canonical_body)

        # YOU provide the full list to define model IO order (can include placeholders)
        self.declare_parameter('selected_body_names', canonical_body)

        # Optional drops (publish stage)
        #self.declare_parameter('drop_joint_names', ['ANKLE','KNEE','BUTTOCK','WAIST',
        #    'LEFT_J1','LEFT_J2','LEFT_J3','LEFT_J4','LEFT_J5','LEFT_J6','LEFT_J7','RIGHT_J1','RIGHT_J2',
        #    'RIGHT_J3','RIGHT_J4','RIGHT_J5','RIGHT_J6','RIGHT_J7'])
        self.declare_parameter('drop_joint_names',
                               ['ANKLE', 'KNEE', 'BUTTOCK', 'WAIST'])

        # Hands: mode + side(s)
        self.declare_parameter('hand_input_mode',
                               'none')  # none | scalar | qpos6
        self.declare_parameter(
            'hand_sides', ['left', 'right'],
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY))
        self.declare_parameter('hand_sides_str',
                               '')  # "left" | "right" | "both"

        # Hand placeholder names (for order definition)
        self.declare_parameter('left_hand_scalar_name', 'LEFT_GRIPPER')
        self.declare_parameter('right_hand_scalar_name', 'RIGHT_GRIPPER')

        # Hand qpos6 names (publishing & optional order expansion)
        self.declare_parameter('left_hand_qpos6_names', [
            'LEFT_HAND_THUMB1', 'LEFT_HAND_THUMB2', 'LEFT_HAND_INDEX',
            'LEFT_HAND_MIDDLE', 'LEFT_HAND_RING', 'LEFT_HAND_PINKY'
        ])
        self.declare_parameter('right_hand_qpos6_names', [
            'RIGHT_HAND_THUMB1', 'RIGHT_HAND_THUMB2', 'RIGHT_HAND_INDEX',
            'RIGHT_HAND_MIDDLE', 'RIGHT_HAND_RING', 'RIGHT_HAND_PINKY'
        ])

        # Input (sensor) topics for hands
        self.declare_parameter('left_hand_scalar_topic', '/hand/left_scalar')
        self.declare_parameter('right_hand_scalar_topic', '/hand/right_scalar')
        self.declare_parameter('left_hand_qpos6_topic',
                               '/feedback_sim/hand/left')
        self.declare_parameter('right_hand_qpos6_topic',
                               '/feedback_sim/hand/right')

        # Output command topics for hands
        self.declare_parameter('set_left_hand_qpos6_topic',
                               '/control/ee/left')
        self.declare_parameter('set_right_hand_qpos6_topic',
                               '/control/ee/right')

        # 手部相机对应的 topic
        self.declare_parameter('cam_hand_left_topic', '/camera/left/image_raw')
        self.declare_parameter('cam_hand_right_topic',
                               '/camera/right/image_raw')

        self.declare_parameter('cam_head_left_topic', '/camera/left_eye')
        self.declare_parameter('cam_head_right_topic', '/camera/right_eye')
        # v1 手势表（scalar→qpos6 使用）
        self.hand_gestures = {
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
        self.joint_names = [
            "ANKLE",
            "KNEE",
            "BUTTOCK",
            "WAIST",
            "NECK1",
            "NECK2",
            "LEFT_J1",
            "LEFT_J2",
            "LEFT_J3",
            "LEFT_J4",
            "LEFT_J5",
            "LEFT_J6",
            "LEFT_J7",
            "RIGHT_J1",
            "RIGHT_J2",
            "RIGHT_J3",
            "RIGHT_J4",
            "RIGHT_J5",
            "RIGHT_J6",
            "RIGHT_J7",
        ]

        self.declare_parameter("hand_interp_start", "normal2")
        self.declare_parameter("hand_interp_end", "pinch")

        # ---------- Read params ----------
        gp = self.get_parameter
        self.policy_path: str = gp(
            'policy_path').get_parameter_value().string_value
        self.device_str: str = gp('device').get_parameter_value().string_value
        self.policy_hz: float = float(
            gp('policy_hz').get_parameter_value().double_value)
        self.tolerance_s: float = float(
            gp('tolerance_ms').get_parameter_value().double_value) / 1000.0

        self.pub_hz: float = float(
            gp('pub_hz').get_parameter_value().double_value)
        self.interp_mode: str = gp(
            'interp_mode').get_parameter_value().string_value.strip().lower()
        self.blend_gamma: float = float(
            gp('blend_gamma').get_parameter_value().double_value)
        self.blend_gamma = float(np.clip(self.blend_gamma, 0.0, 1.0))

        self.joint_topic: str = gp(
            'joint_topic').get_parameter_value().string_value
        self.publish_topic: str = gp(
            'publish_topic').get_parameter_value().string_value

        self.head_target_size = (
            int(gp('head_target_width').get_parameter_value().integer_value),
            int(gp('head_target_height').get_parameter_value().integer_value),
        )
        self.hand_target_size = (
            int(gp('hand_target_width').get_parameter_value().integer_value),
            int(gp('hand_target_height').get_parameter_value().integer_value),
        )
        self.state_key = gp('state_key').get_parameter_value().string_value

        self.body_canonical: List[str] = list(
            gp('ordered_body_names').get_parameter_value().string_array_value)
        selected_raw: List[str] = list(
            gp('selected_body_names').get_parameter_value().string_array_value)

        self.drop_set = set(
            list(
                gp('drop_joint_names').get_parameter_value().string_array_value)
        )

        mode = gp('hand_input_mode').get_parameter_value().string_value.lower(
        ).strip()
        assert mode in ('none', 'scalar',
                        'qpos6'), "hand_input_mode must be none|scalar|qpos6"
        self.hand_input_mode = mode

        # sides：hand_sides_str 优先
        sides_arr = [
            s.lower() for s in list(
                gp('hand_sides').get_parameter_value().string_array_value)
        ]
        sides_str = gp('hand_sides_str').get_parameter_value(
        ).string_value.lower().strip()
        if sides_str in ('left', 'right', 'both'):
            self.hand_sides = ['left', 'right'
                              ] if sides_str == 'both' else [sides_str]
        else:
            self.hand_sides = [s for s in ['left', 'right'] if s in sides_arr]

        self.left_scalar_name = gp('left_hand_scalar_name').get_parameter_value(
        ).string_value or 'LEFT_GRIPPER'
        self.right_scalar_name = gp(
            'right_hand_scalar_name').get_parameter_value(
            ).string_value or 'RIGHT_GRIPPER'
        self.left_q6_names = list(
            gp('left_hand_qpos6_names').get_parameter_value().string_array_value
        )
        self.right_q6_names = list(
            gp('right_hand_qpos6_names').get_parameter_value().
            string_array_value)

        self.left_scalar_topic = gp(
            'left_hand_scalar_topic').get_parameter_value().string_value
        self.right_scalar_topic = gp(
            'right_hand_scalar_topic').get_parameter_value().string_value
        self.left_qpos6_topic = gp(
            'left_hand_qpos6_topic').get_parameter_value().string_value
        self.right_qpos6_topic = gp(
            'right_hand_qpos6_topic').get_parameter_value().string_value

        self.set_left_qpos6_topic = gp(
            'set_left_hand_qpos6_topic').get_parameter_value().string_value
        self.set_right_qpos6_topic = gp(
            'set_right_hand_qpos6_topic').get_parameter_value().string_value

        self.hand_interp_start = self.get_parameter(
            "hand_interp_start").get_parameter_value().string_value
        self.hand_interp_end = self.get_parameter(
            "hand_interp_end").get_parameter_value().string_value

        ALLOWED_IMAGE_KEYS = {
            "observation.images.cam_high_left",
            "observation.images.cam_high_right",
            "observation.images.cam_hand_left",
            "observation.images.cam_hand_right",
        }
        req_keys = list(
            self.get_parameter(
                'image_keys').get_parameter_value().string_array_value)
        self.image_keys = [k for k in req_keys if k in ALLOWED_IMAGE_KEYS]
        if not self.image_keys:
            raise ValueError("image_keys 不能为空：必须从四个允许键中至少选择一个。")

        self.cam_hand_left_topic = self.get_parameter(
            'cam_hand_left_topic').get_parameter_value().string_value
        self.cam_hand_right_topic = self.get_parameter(
            'cam_hand_right_topic').get_parameter_value().string_value
        self.cam_head_left_topic = self.get_parameter(
            'cam_head_left_topic').get_parameter_value().string_value
        self.cam_head_right_topic = self.get_parameter(
            'cam_head_right_topic').get_parameter_value().string_value

        # ---------- Build IO order ----------
        self.full_order: List[str] = self._build_full_order(selected_raw)
        self.full_dim = len(self.full_order)

        self.body_order: List[str] = [
            n for n in self.full_order if n in self.body_canonical
        ]

        self.idx_left_scalar: Optional[int] = None
        self.idx_right_scalar: Optional[int] = None
        self.slice_left_q6: Optional[Tuple[int, int]] = None
        self.slice_right_q6: Optional[Tuple[int, int]] = None
        for i, n in enumerate(self.full_order):
            if n == self.left_scalar_name:
                self.idx_left_scalar = i
            if n == self.right_scalar_name:
                self.idx_right_scalar = i

        def find_block(names: List[str]) -> Optional[Tuple[int, int]]:
            L = len(names)
            for i in range(0, len(self.full_order) - L + 1):
                if self.full_order[i:i + L] == names:
                    return (i, i + L)
            return None

        if 'left' in self.hand_sides and self.hand_input_mode == 'qpos6':
            self.slice_left_q6 = find_block(self.left_q6_names)
        if 'right' in self.hand_sides and self.hand_input_mode == 'qpos6':
            self.slice_right_q6 = find_block(self.right_q6_names)

        # ---- 角度维 mask（新增） ----
        self.mask_ang = np.zeros(self.full_dim, dtype=bool)
        for i, n in enumerate(self.full_order):
            if n in self.body_canonical:
                self.mask_ang[i] = True
            elif self.hand_input_mode == 'qpos6':
                if self.slice_left_q6 is not None and self.slice_left_q6[
                        0] <= i < self.slice_left_q6[1]:
                    self.mask_ang[i] = True
                if self.slice_right_q6 is not None and self.slice_right_q6[
                        0] <= i < self.slice_right_q6[1]:
                    self.mask_ang[i] = True
        # scalar 手势维（若存在）不视为角度

        self.get_logger().info(
            "=== Inference Order (len=%d) ===\n%s" % (self.full_dim, "\n".join(
                [f"{i:02d}: {n}" for i, n in enumerate(self.full_order)])))
        self.get_logger().info(
            f"[train IO] BODY={len(self.body_order)} HAND_MODE={self.hand_input_mode} sides={self.hand_sides} → full_dim={self.full_dim}"
        )
        if self.hand_input_mode == 'scalar':
            self.get_logger().info(
                f"scalar idx: left={self.idx_left_scalar}, right={self.idx_right_scalar}"
            )
        elif self.hand_input_mode == 'qpos6':
            self.get_logger().info(
                f"q6 slices: left={self.slice_left_q6}, right={self.slice_right_q6}"
            )

        if self.full_dim == 0:
            raise RuntimeError("selected_body_names 展开后为空；请至少选择一个维度")

        # ---------- QoS & pubs/subs ----------
        q_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.pub_action = self.create_publisher(JointPositionControl,
                                                self.publish_topic, q_reliable)
        self.pub_set_left_qpos6 = self.create_publisher(
            Float64MultiArray, self.set_left_qpos6_topic, q_reliable)
        self.pub_set_right_qpos6 = self.create_publisher(
            Float64MultiArray, self.set_right_qpos6_topic, q_reliable)

        # Buffers
        self.body_buf: deque = deque(maxlen=200)
        self.body_index_map: Optional[np.ndarray] = None
        self.hand_left_buf: deque = deque(maxlen=200)
        self.hand_right_buf: deque = deque(maxlen=200)
        self.head_left_buf: deque = deque(maxlen=200)
        self.head_right_buf: deque = deque(maxlen=200)
        self.bridge = CvBridge()
        self.scalar_left_buf: deque = deque(maxlen=2000)
        self.scalar_right_buf: deque = deque(maxlen=2000)
        self.qpos6_left_buf: deque = deque(maxlen=2000)
        self.qpos6_right_buf: deque = deque(maxlen=2000)

        # Subs
        q_img = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )
        # 头部相机
        if "observation.images.cam_high_left" in self.image_keys:
            self.create_subscription(Image, self.cam_head_left_topic,
                                     self.cb_img_head_left, q_img)
            self.get_logger().info(
                f"Subscribe head-left image: {self.cam_head_left_topic}")
        if "observation.images.cam_high_right" in self.image_keys:
            self.create_subscription(Image, self.cam_head_right_topic,
                                     self.cb_img_head_right, q_img)
            self.get_logger().info(
                f"Subscribe head-right image: {self.cam_head_right_topic}")

        if "observation.images.cam_hand_left" in self.image_keys:
            self.create_subscription(Image, self.cam_hand_left_topic,
                                     self.cb_img_hand_left, q_img)
            self.get_logger().info(
                f"Subscribe hand-left image: {self.cam_hand_left_topic}")
        if "observation.images.cam_hand_right" in self.image_keys:
            self.create_subscription(Image, self.cam_hand_right_topic,
                                     self.cb_img_hand_right, q_img)
            self.get_logger().info(
                f"Subscribe hand-right image: {self.cam_hand_right_topic}")

        self.create_subscription(String, self.joint_topic, self.cb_joint,
                                 q_reliable)
        self.create_subscription(Float64, self.left_scalar_topic,
                                 self.cb_scalar_left, q_reliable)
        self.create_subscription(Float64, self.right_scalar_topic,
                                 self.cb_scalar_right, q_reliable)
        self.create_subscription(Float64MultiArray, self.left_qpos6_topic,
                                 self.cb_qpos6_left, q_reliable)
        self.create_subscription(Float64MultiArray, self.right_qpos6_topic,
                                 self.cb_qpos6_right, q_reliable)

        # Policy
        self.device = torch.device(self.device_str if (
            self.device_str == 'cuda' and torch.cuda.is_available()) else 'cpu')
        self.get_logger().info(
            f'Loading ACTPolicy from {self.policy_path} on {self.device} ...')
        self.policy = ACTPolicy.from_pretrained(self.policy_path,
                                                local_files_only=True)
        self.policy.to(self.device).eval()

        from act.modeling_act import ACTTemporalEnsembler
        # without ensembling
        cfg = self.policy.config
        cfg.temporal_ensemble_coeff = None
        cfg.n_action_steps = min(30, cfg.chunk_size)
        self.policy.reset()
        self.get_logger().info('Policy loaded.')
        self.get_logger().info(
            f"[ACT] chunk_size={cfg.chunk_size}, n_action_steps={cfg.n_action_steps}, "
            f"temporal_ensemble_coeff={cfg.temporal_ensemble_coeff}")

        # ---------- Async planning buffers ----------
        self.plan_lock = threading.Lock()
        self.plan_queue: deque = deque()  # 每个元素: np.ndarray(full_dim)
        self.replan_in_progress: bool = False

        # horizon: 一次重推理拿多少帧 (例如30)
        self.horizon_N: int = int(cfg.n_action_steps)
        # <= 这个阈值时触发异步补货 (例如15)
        self.replan_trigger: int = max(24, self.horizon_N // 2)

        # --- blend相关: chunk衔接淡化 ---
        self.last_cmd_vec = np.zeros(self.full_dim, dtype=np.float32)
        self.have_last_cmd = False
        self.blend_steps = 20  # 过渡长度(帧数)
        # 以上 + blend_gamma 控制让步强度

        # Camera thread (SBS)
        # self.frame_lock = threading.Lock()
        # self.latest_frame: Optional[TimedFrame] = None
        # self._cap_stop = False
        # th = threading.Thread(target=self.capture_loop_sbs, daemon=True)
        # th.start()
        # self._cap_thread = th

        # Timers
        period = 1.0 / max(1.0, self.policy_hz)
        self.create_timer(period, self.timer_infer)

        # >>> 新增：发布层插值定时器 <<<
        self.interp_substeps = max(
            1, int(round(self.pub_hz / max(1.0, self.policy_hz))))
        self._seg_has_target = False
        self._seg_start = np.zeros(self.full_dim, dtype=np.float32)
        self._seg_target = np.zeros(self.full_dim, dtype=np.float32)
        self._seg_i = 0
        self._seg_N = self.interp_substeps
        self.create_timer(1.0 / max(1.0, self.pub_hz), self.timer_publish)

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
                    out.append(n)
                    continue
                if n in self.right_q6_names and 'right' in sides:
                    out.append(n)
                    continue
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
            self.get_logger().error(
                'OpenCV cannot open /dev/video99 (or id 99).')
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)

        try:
            while not self._cap_stop:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.005)
                    continue
                t = now_sec()
                h, w2, _ = frame.shape
                if w2 % 2 != 0:
                    self.get_logger().warn(f'Frame width {w2} not even; skip')
                    continue
                w = w2 // 2
                left_bgr = frame[:, :w]
                right_bgr = frame[:, w:]
                with self.frame_lock:
                    self.latest_frame = TimedFrame(t=t,
                                                   left_bgr=left_bgr,
                                                   right_bgr=right_bgr)
        finally:
            cap.release()

    # ---------- Subs ----------
    def cb_joint(self, msg: String) -> None:
        # 0. 还是用 body_order 的映射逻辑
        if len(self.body_order) == 0:
            return

        # 1. 解析 JSON 字符串
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().warn(
                f"JSON parse error in /feedback/robot_server_state: {e}")
            return

        joint_pos = data.get("joint_position", None)
        if joint_pos is None or len(joint_pos) != len(self.joint_names):
            self.get_logger().warn(
                f"joint_position length {0 if joint_pos is None else len(joint_pos)} "
                f"!= expected {len(self.joint_names)}")
            return

        # 2. 时间戳：还是用 now_sec()
        t = now_sec()
        q_in = np.asarray(joint_pos, dtype=np.float32)

        # 3. 构建 body_index_map：用 joint_names 做索引表
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
                f"({len(self.body_index_map)} joints).")

        if self.body_index_map is None:
            return

        # 4. 重排 & 存入 buffer
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

    def cb_qpos6_left(self, msg: Float64MultiArray) -> None:
        arr = np.asarray(msg.data, dtype=np.float32)
        if arr.shape[0] >= 6:
            self.qpos6_left_buf.append((now_sec(), arr[:6].copy()))

    def cb_qpos6_right(self, msg: Float64MultiArray) -> None:
        arr = np.asarray(msg.data, dtype=np.float32)
        if arr.shape[0] >= 6:
            self.qpos6_right_buf.append((now_sec(), arr[:6].copy()))

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

    def cb_img_head_left(self, msg: Image) -> None:
        self._handle_head_image('left', msg)

    def cb_img_head_right(self, msg: Image) -> None:
        self._handle_head_image('right', msg)

    def _handle_head_image(self, side: str, msg: Image) -> None:
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge head-{side} fail: {e}")
            return
        t = now_sec()
        if side == 'left':
            self.head_left_buf.append((t, bgr))
        else:
            self.head_right_buf.append((t, bgr))

    # ---------- Scalar → qpos6 ----------
    def map_scalar_to_qpos6(self, s: float) -> np.ndarray:
        s = float(np.clip(s, 0.0, 1.0))
        start, end = self.hand_interp_start, self.hand_interp_end
        if start not in self.hand_gestures or end not in self.hand_gestures:
            raise ValueError(
                f"Hand gesture must be in {list(self.hand_gestures.keys())}")
        base = np.asarray(self.hand_gestures[start], dtype=np.float32)
        target = np.asarray(self.hand_gestures[end], dtype=np.float32)
        return base * (1.0 - s) + target * s

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
                f'Order: {self.full_order}')
            return None

        return act.astype(np.float32, copy=True)

    def _publish_action_vector(self, act_np: np.ndarray) -> None:
        # 身体
        pub_names: List[str] = []
        pub_pos: List[float] = []
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
                s = float(act_np[self.idx_left_scalar])
                q6 = self.map_scalar_to_qpos6(s)
                m = Float64MultiArray()
                m.data = [float(x) for x in q6]
                self.pub_set_left_qpos6.publish(m)
            if 'right' in self.hand_sides and self.idx_right_scalar is not None:
                s = float(act_np[self.idx_right_scalar])
                q6 = self.map_scalar_to_qpos6(s)
                m = Float64MultiArray()
                m.data = [float(x) for x in q6]
                self.pub_set_right_qpos6.publish(m)

        elif self.hand_input_mode == 'qpos6':
            if 'left' in self.hand_sides and self.slice_left_q6 is not None:
                a, b = self.slice_left_q6
                v = act_np[a:b]
                m = Float64MultiArray()
                m.data = [float(x) for x in v]
                self.pub_set_left_qpos6.publish(m)
            if 'right' in self.hand_sides and self.slice_right_q6 is not None:
                a, b = self.slice_right_q6
                v = act_np[a:b]
                m = Float64MultiArray()
                m.data = [float(x) for x in v]
                self.pub_set_right_qpos6.publish(m)

        # 记录：我们刚刚真的下发给机器人的这一帧
        self.last_cmd_vec = act_np.astype(np.float32, copy=True)
        self.have_last_cmd = True

    def _make_seed_batch(self, batch_base: Dict[str, torch.Tensor],
                         seed_vec_np: np.ndarray) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for k, v in batch_base.items():
            if k == self.state_key:
                out[k] = torch.from_numpy(seed_vec_np[None, ...].astype(
                    np.float32)).to(self.device)
            else:
                out[k] = v
        return out

    def _spawn_replan(self,
                      batch: Dict[str, torch.Tensor],
                      sync: bool = False) -> None:
        if self.replan_in_progress:
            return

        self.replan_in_progress = True

        def worker():
            actions_local: List[np.ndarray] = []
            with torch.no_grad():
                for _ in range(self.horizon_N):
                    t0 = now_sec()
                    try:
                        a = self.policy.select_action(batch)
                    except Exception as e:
                        self.get_logger().error(
                            f'policy.select_action failed in worker: {e}')
                        break
                    t1 = now_sec()
                    infer_dt = t1 - t0
                    if infer_dt > 0.1:
                        self.get_logger().info(
                            f'[async] heavy inference {infer_dt:.3f}s (worker)')

                    act_np = self._action_to_np(a)
                    if act_np is None:
                        break
                    actions_local.append(act_np)

            # === 把新chunk平滑拼接到 plan_queue 尾部 ===
            with self.plan_lock:
                # 1) 对齐参考
                if len(self.plan_queue) > 0:
                    prev_tail = self.plan_queue[-1].astype(np.float32,
                                                           copy=True)
                    do_blend = True
                elif self.have_last_cmd:
                    prev_tail = self.last_cmd_vec.astype(np.float32, copy=True)
                    do_blend = True
                else:
                    prev_tail = None
                    do_blend = False

                # 2) 双边淡化（Hamming，上升给旧尾，下降给新头；所有角度差用最短弧）
                if actions_local and do_blend:
                    K = min(self.blend_steps, len(actions_local),
                            len(self.plan_queue))
                    if K > 0:
                        r = half_hamming_weights(K)  # [0,1] 单调上升
                        c_old = (self.blend_gamma * r).astype(
                            np.float32)  # 旧尾：由轻到重
                        c_new = (self.blend_gamma * r[::-1]).astype(
                            np.float32)  # 新头：由重到轻

                        # 2a) 旧尾朝新头第1帧靠拢（最短弧差）
                        new_head_ref = actions_local[0].astype(np.float32,
                                                               copy=True)
                        for i in range(K):
                            idx_old = len(self.plan_queue) - K + i
                            old_val = self.plan_queue[idx_old].astype(
                                np.float32, copy=True)
                            d_old = shortest_delta_vec(old_val, new_head_ref,
                                                       self.mask_ang)
                            self.plan_queue[idx_old] = (
                                old_val + c_old[i] * d_old).astype(np.float32,
                                                                   copy=False)

                        # 更新“被微调后的旧段最后一帧”
                        prev_tail_adj = self.plan_queue[-1].astype(np.float32,
                                                                   copy=True)

                        # 2b) 新头朝 prev_tail_adj 靠拢（最短弧差）
                        for i in range(K):
                            v = actions_local[i].astype(np.float32, copy=True)
                            d_new = shortest_delta_vec(v, prev_tail_adj,
                                                       self.mask_ang)
                            actions_local[i] = (v + c_new[i] * d_new).astype(
                                np.float32, copy=False)
                    else:
                        # 退化：单边线性窗，但 offset 用最短弧
                        offset = shortest_delta_vec(actions_local[0], prev_tail,
                                                    self.mask_ang)
                        Nblend = min(self.blend_steps, len(actions_local))
                        denom = float(max(Nblend, 1))
                        for i in range(Nblend):
                            alpha = 1.0 - (i / denom)
                            actions_local[i] = (actions_local[i] +
                                                offset * alpha).astype(
                                                    np.float32, copy=False)

                # 3) 入队
                for x in actions_local:
                    self.plan_queue.append(x.astype(np.float32, copy=False))

            self.replan_in_progress = False

        if sync:
            worker()
        else:
            th = threading.Thread(target=worker, daemon=True)
            th.start()

    # ---------- Timer（推理） ----------
    @torch.no_grad()
    def timer_infer(self) -> None:
        # 1) 最新相机帧
        t_candidates: List[float] = []
        if self.head_left_buf:
            t_candidates.append(self.head_left_buf[-1][0])
        if self.head_right_buf:
            t_candidates.append(self.head_right_buf[-1][0])

        if not t_candidates:
            # 没有头相机图像就不推理
            # 你也可以 print("no t candidate") debug
            return

        t_anchor = max(t_candidates)

        # 2) proprio 状态 (按照 full_order)
        state_vals: List[float] = []

        if len(self.body_order) > 0:
            q_body = nearest(self.body_buf, t_anchor, self.tolerance_s)
            # print("collect q body success")
            if q_body is None:
                print("here")
                return
            body_map = {n: float(v) for n, v in zip(self.body_order, q_body)}
        else:
            body_map = {}

        left_scalar = nearest(self.scalar_left_buf, t_anchor, self.tolerance_s)
        right_scalar = nearest(self.scalar_right_buf, t_anchor,
                               self.tolerance_s)
        left_q6 = nearest(self.qpos6_left_buf, t_anchor, self.tolerance_s)
        right_q6 = nearest(self.qpos6_right_buf, t_anchor, self.tolerance_s)

        for n in self.full_order:
            if n in self.body_canonical:
                state_vals.append(body_map.get(n, 0.0))
                continue

            if self.hand_input_mode == 'scalar':
                if n == self.left_scalar_name and 'left' in self.hand_sides:
                    state_vals.append(0.0 if left_scalar is
                                      None else float(left_scalar))
                    continue
                if n == self.right_scalar_name and 'right' in self.hand_sides:
                    state_vals.append(0.0 if right_scalar is
                                      None else float(right_scalar))
                    continue

            elif self.hand_input_mode == 'qpos6':
                if n in self.left_q6_names and 'left' in self.hand_sides:
                    i = self.left_q6_names.index(n)
                    v = 0.0 if left_q6 is None else float(left_q6[i])
                    state_vals.append(v)
                    continue
                if n in self.right_q6_names and 'right' in self.hand_sides:
                    i = self.right_q6_names.index(n)
                    v = 0.0 if right_q6 is None else float(right_q6[i])
                    state_vals.append(v)
                    continue

            state_vals.append(0.0)

        state_vec = np.asarray(state_vals, dtype=np.float32)
        if state_vec.shape[0] != self.full_dim:
            self.get_logger().error(
                f"State dim {state_vec.shape} != expected {self.full_dim}")
            return

        # 3) 打包 batch
        batch_np: Dict[str, np.ndarray] = {}
        head_W, head_H = int(self.head_target_size[0]), int(
            self.head_target_size[1])
        hand_W, hand_H = int(self.hand_target_size[0]), int(
            self.hand_target_size[1])

        for key in self.image_keys:
            if key == "observation.images.cam_high_left":
                bgr = nearest(self.head_left_buf, t_anchor, self.tolerance_s)
                if bgr is None:
                    return
                rgb = bgr_to_chw_rgb_resized(bgr, (head_W, head_H))
                batch_np[key] = rgb[None, ...]
            elif key == "observation.images.cam_high_right":
                bgr = nearest(self.head_right_buf, t_anchor, self.tolerance_s)
                if bgr is None:
                    return
                rgb = bgr_to_chw_rgb_resized(bgr, (head_W, head_H))
                batch_np[key] = rgb[None, ...]
            elif key == "observation.images.cam_hand_left":
                bgr = nearest(self.hand_left_buf, t_anchor, self.tolerance_s)
                if bgr is None:
                    return
                rgb = bgr_to_chw_rgb_resized(bgr, (hand_W, hand_H))
                batch_np[key] = rgb[None, ...]
            elif key == "observation.images.cam_hand_right":
                bgr = nearest(self.hand_right_buf, t_anchor, self.tolerance_s)
                if bgr is None:
                    return
                rgb = bgr_to_chw_rgb_resized(bgr, (hand_W, hand_H))
                batch_np[key] = rgb[None, ...]

        batch_np[self.state_key] = state_vec[None, ...]
        batch: Dict[str, torch.Tensor] = {
            k: torch.from_numpy(v).to(self.device) for k, v in batch_np.items()
        }

        # 4) 异步控制逻辑（不在这里发布；发布挪到 timer_publish）
        with self.plan_lock:
            qlen = len(self.plan_queue)
            busy = self.replan_in_progress

        if qlen == 0 and (not busy):
            seed_batch = self._make_seed_batch(batch, state_vec)
            self._spawn_replan(seed_batch, sync=True)
            with self.plan_lock:
                qlen = len(self.plan_queue)
                busy = self.replan_in_progress

        # 触发异步补货（未来种子=队列尾/第30帧）
        with self.plan_lock:
            qlen_after = len(self.plan_queue)
            busy_after = self.replan_in_progress

        if (qlen_after <= self.replan_trigger) and (not busy_after):
            with self.plan_lock:
                if self.plan_queue:
                    future_seed = self.plan_queue[-1].astype(np.float32,
                                                             copy=True)
                else:
                    future_seed = state_vec.astype(np.float32, copy=True)
            seed_batch = self._make_seed_batch(batch, future_seed)
            self._spawn_replan(seed_batch, sync=False)

        return

    # ---------- Timer（发布层插值：粗→细）【新增】 ----------
    def timer_publish(self) -> None:
        with self.plan_lock:
            # 若当下没有段目标，尝试从队列取一个粗帧
            if not self._seg_has_target:
                if self.plan_queue:
                    self._seg_target = self.plan_queue.popleft().astype(
                        np.float32, copy=True)
                    # 段起点 = 上一次真实下发帧（若无，则用当前目标，避免突变）
                    self._seg_start = self.last_cmd_vec.astype(
                        np.float32, copy=True
                    ) if self.have_last_cmd else self._seg_target.copy()
                    self._seg_i = 0
                    self._seg_N = self.interp_substeps
                    self._seg_has_target = True
                else:
                    # 没有目标就复发上一帧，尽量保持
                    if not self.have_last_cmd:
                        return
                    cmd = self.last_cmd_vec.astype(np.float32, copy=True)
                    self._publish_action_vector(cmd)
                    return

            # 有段目标：做一个子步
            # 差值统一走“最短弧”（角度维），非角度维线性
            off = shortest_delta_vec(self._seg_start, self._seg_target,
                                     self.mask_ang)
            s_lin = float(self._seg_i + 1) / float(max(1, self._seg_N))
            s = smoothstep(s_lin) if self.interp_mode == 'smoothstep' else s_lin
            cmd = (self._seg_start + s * off).astype(np.float32, copy=False)
            # 仅用于显示整洁；实际物理无差
            cmd = wrap_to_pi_vec(cmd, self.mask_ang)

            self._seg_i += 1
            if self._seg_i >= self._seg_N:
                # 最后一个子步：恰好等于粗帧目标；并进入下一段
                cmd = self._seg_target.astype(np.float32, copy=True)
                self._seg_has_target = False
                self._seg_start = cmd.copy()

        # 统一在锁外发布
        self._publish_action_vector(cmd)


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
        node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
