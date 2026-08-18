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
        self.declare_parameter('policy_hz', 15.0)
        self.declare_parameter('tolerance_ms', 50.0)

        # Topics
        self.declare_parameter('joint_topic', '/feedback/joint')
        self.declare_parameter('publish_topic',
                               '/w1/policy/desired_joint_positions')

        # Images / state keys
        self.declare_parameter('head_target_width', 640)
        self.declare_parameter('head_target_height', 360)
        self.declare_parameter('hand_target_width', 640)
        self.declare_parameter('hand_target_height', 480)
        # self.declare_parameter('image_left_key', 'observation.images.cam_high_left')
        # self.declare_parameter('image_right_key', 'observation.images.cam_high_right')
        # 手部相机图像键
        self.declare_parameter('image_hand_left_key',
                               'observation.images.cam_hand_left')
        self.declare_parameter('image_hand_right_key',
                               'observation.images.cam_hand_right')
        self.declare_parameter('state_key', 'observation.state')

        self.declare_parameter(
            'image_keys',
            [
                "observation.images.cam_high_left",
                #"observation.images.cam_high_right",
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
        self.declare_parameter('drop_joint_names', ['ANKLE', 'KNEE', 'BUTTOCK'])

        # Hands: mode + side(s)
        self.declare_parameter('hand_input_mode',
                               'none')  # none | scalar | qpos6
        self.declare_parameter(
            'hand_sides', ['left', 'right'],
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY
                               ))  # e.g. ["right"] or ["left","right"]
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
                               '/brainco_left_hand_qpos')
        self.declare_parameter('right_hand_qpos6_topic',
                               '/brainco_right_hand_qpos')

        # Output command topics for hands
        self.declare_parameter('set_left_hand_qpos6_topic',
                               '/set_brainco_left_hand_qpos')
        self.declare_parameter('set_right_hand_qpos6_topic',
                               '/set_brainco_right_hand_qpos')

        # 手部相机对应的 topic（可在 params.yaml 里改）
        self.declare_parameter('cam_hand_left_topic', '/camera/left/image_raw')
        self.declare_parameter('cam_hand_right_topic',
                               '/camera/right/image_raw')
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
        # 起始手势和目标手势
        self.declare_parameter("hand_interp_start", "normal")
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
        # self.image_left_key  = gp('image_left_key').get_parameter_value().string_value
        # self.image_right_key = gp('image_right_key').get_parameter_value().string_value
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

        # ---------- Build IO order ----------
        # 把 selected_raw 里的占位符按模式/侧别进行保留或展开，得到“模型向量顺序” full_order
        self.full_order: List[str] = self._build_full_order(selected_raw)
        self.full_dim = len(self.full_order)

        # 从 full_order 里提取 body 名单（只保留 canonical body）
        self.body_order: List[str] = [
            n for n in self.full_order if n in self.body_canonical
        ]

        # 记录手部索引（用于切动作向量 & 组装状态）
        self.idx_left_scalar: Optional[int] = None
        self.idx_right_scalar: Optional[int] = None
        self.slice_left_q6: Optional[Tuple[int, int]] = None
        self.slice_right_q6: Optional[Tuple[int, int]] = None
        for i, n in enumerate(self.full_order):
            if n == self.left_scalar_name:
                self.idx_left_scalar = i
            if n == self.right_scalar_name:
                self.idx_right_scalar = i
        # qpos6：扫描全列表中出现的六指名的连续段
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

        # 打印映射（让你一眼看到“第几个是谁”）
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
            raise RuntimeError("你的 selected_body_names 展开后为空；请至少选择一个维度")

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
        self.body_buf: deque = deque(maxlen=2000)
        self.body_index_map: Optional[np.ndarray] = None
        self.hand_left_buf: deque = deque(maxlen=200)
        self.hand_right_buf: deque = deque(maxlen=200)
        self.bridge = CvBridge()
        self.scalar_left_buf: deque = deque(maxlen=2000)
        self.scalar_right_buf: deque = deque(maxlen=2000)
        self.qpos6_left_buf: deque = deque(maxlen=2000)
        self.qpos6_right_buf: deque = deque(maxlen=2000)

        # Subs
        q_img = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  # 图像常用
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )
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

        self.create_subscription(JointState, self.joint_topic, self.cb_joint,
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

        #add temporal ensembling
        from act.modeling_act import ACTTemporalEnsembler  # 路径按你的工程改
        cfg = self.policy.config
        cfg.temporal_ensemble_coeff = 0.005  # 建议从 0.01 起步；>0 更偏“老动作”，<0 更偏“新动作”
        cfg.n_action_steps = 1  # 关键：开启 ensembling 时必须为 1
        self.policy.temporal_ensembler = ACTTemporalEnsembler(
            cfg.temporal_ensemble_coeff, cfg.chunk_size)
        self.policy.reset()  # 让内部队列/状态按新配置重置

        #without ensembling
        # cfg = self.policy.config
        # cfg.temporal_ensemble_coeff = None
        # cfg.n_action_steps = min(3, cfg.chunk_size)
        # self.policy.reset()
        self.get_logger().info('Policy loaded.')
        self.get_logger().info(
            f"[ACT] chunk_size={cfg.chunk_size}, n_action_steps={cfg.n_action_steps}, "
            f"temporal_ensemble_coeff={cfg.temporal_ensemble_coeff}")

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
        """根据 selected_raw、hand_input_mode 和 sides 生成最终的推理顺序。"""
        mode = self.hand_input_mode
        sides = self.hand_sides
        out: List[str] = []

        # 便捷集合
        body_set = set(self.body_canonical)

        # 遍历用户给的顺序，按规则保留/展开
        for n in selected_raw:
            if n in body_set:
                out.append(n)
                continue

            if mode == 'none':
                # 忽略任何手的占位符/名字
                continue

            if mode == 'scalar':
                if n == self.left_scalar_name and 'left' in sides:
                    out.append(self.left_scalar_name)
                    continue
                if n == self.right_scalar_name and 'right' in sides:
                    out.append(self.right_scalar_name)
                    continue
                # 若给了 q6 的具体名字，在 scalar 模式下忽略
                continue

            if mode == 'qpos6':
                if n == self.left_scalar_name and 'left' in sides:
                    out.extend(self.left_q6_names)
                    continue
                if n == self.right_scalar_name and 'right' in sides:
                    out.extend(self.right_q6_names)
                    continue
                # 明确的 q6 名字：保留（仅当属于左右名单）
                if n in self.left_q6_names and 'left' in sides:
                    out.append(n)
                    continue
                if n in self.right_q6_names and 'right' in sides:
                    out.append(n)
                    continue
                # 其它无效名：忽略
                continue

        # 如果用户没在 selected 里放手的占位符/名字，但 sides 里要求了手，就补在末尾
        if mode == 'scalar':
            if 'left' in sides and self.left_scalar_name not in out:
                out.append(self.left_scalar_name)
            if 'right' in sides and self.right_scalar_name not in out:
                out.append(self.right_scalar_name)
        elif mode == 'qpos6':

            def ensure_block(block: List[str]):
                # 若 6 个名字都未出现，则补上整块
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
                if w2 % 2 != 0:  # expect side-by-side
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
    def cb_joint(self, msg: JointState) -> None:
        if not msg.position:
            return
        if len(self.body_order) == 0:
            return

        t = stamp_to_sec(msg.header.stamp)
        q_in = np.asarray(msg.position, dtype=np.float32)

        # 构建一次性 name→idx 的重映射
        if self.body_index_map is None and msg.name:
            name_to_idx = {n: i for i, n in enumerate(msg.name)}
            idxs, missing = [], []
            for n in self.body_order:
                if n in name_to_idx:
                    idxs.append(name_to_idx[n])
                else:
                    missing.append(n)
            if missing:
                self.get_logger().warn(
                    f"Incoming JointState missing selected BODY joints: {missing}. Waiting..."
                )
                return
            self.body_index_map = np.asarray(idxs, dtype=np.int64)
            self.get_logger().info(
                f"Mapped incoming JointState to BODY order ({len(self.body_index_map)} joints)."
            )

        if self.body_index_map is None:
            return

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
        # t = stamp_to_sec(msg.header.stamp)
        t = now_sec()
        self.hand_left_buf.append((t, bgr))

    def cb_img_hand_right(self, msg: Image) -> None:
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge right-hand fail: {e}")
            return
        # t = stamp_to_sec(msg.header.stamp)
        t = now_sec()
        self.hand_right_buf.append((t, bgr))

    # ---------- Scalar → qpos6（可配置插值） ----------
    def map_scalar_to_qpos6(self, s: float) -> np.ndarray:
        s = float(np.clip(s, 0.0, 1.0))
        start, end = self.hand_interp_start, self.hand_interp_end

        if start not in self.hand_gestures or end not in self.hand_gestures:
            raise ValueError(
                f"Hand gesture must be in {list(self.hand_gestures.keys())}")

        base = np.asarray(self.hand_gestures[start], dtype=np.float32)
        target = np.asarray(self.hand_gestures[end], dtype=np.float32)
        return base * (1.0 - s) + target * s

    # ---------- Timer ----------
    @torch.no_grad()
    def timer_infer(self) -> None:
        # 拿图像（用于对齐时戳 & 作为视觉输入）
        # with getattr(self, "frame_lock", threading.Lock()):
        #     tf = getattr(self, "latest_frame", None)
        # if tf is None:
        #     return
        with self.frame_lock:
            tf = self.latest_frame
        if tf is None:
            return
        t_anchor = tf.t
        # 组装 “状态向量” 按 full_order 的顺序逐项填充
        state_vals: List[float] = []

        # 先拿 BODY
        if len(self.body_order) > 0:
            q_body = nearest(self.body_buf, tf.t, self.tolerance_s)
            if q_body is None:
                return
            body_map = {n: float(v) for n, v in zip(self.body_order, q_body)}
        else:
            body_map = {}

        # 手的传感输入
        left_scalar = nearest(self.scalar_left_buf, tf.t, self.tolerance_s)
        right_scalar = nearest(self.scalar_right_buf, tf.t, self.tolerance_s)
        left_q6 = nearest(self.qpos6_left_buf, tf.t, self.tolerance_s)
        right_q6 = nearest(self.qpos6_right_buf, tf.t, self.tolerance_s)

        # 按名字一项项填
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

            # 其它名：默认 0
            state_vals.append(0.0)

        state_vec = np.asarray(state_vals, dtype=np.float32)
        if state_vec.shape[0] != self.full_dim:
            self.get_logger().error(
                f"State dim {state_vec.shape} != expected {self.full_dim}")
            return

        # 图像打包
        batch_np = {}

        head_W, head_H = int(self.head_target_size[0]), int(
            self.head_target_size[1])
        hand_W, hand_H = int(self.hand_target_size[0]), int(
            self.hand_target_size[1])
        for key in self.image_keys:
            if key == "observation.images.cam_high_left":
                # 头部左：来自 tf.left_bgr
                if tf.left_bgr is None:
                    return
                rgb = bgr_to_chw_rgb_resized(tf.left_bgr, (head_W, head_H))
                batch_np[key] = rgb[None, ...]
            elif key == "observation.images.cam_high_right":
                # 头部右：来自 tf.right_bgr
                if tf.right_bgr is None:
                    return
                rgb = bgr_to_chw_rgb_resized(tf.right_bgr, (head_W, head_H))
                batch_np[key] = rgb[None, ...]
            elif key == "observation.images.cam_hand_left":
                # 手部左：从 hand_left_buf 最近邻对齐
                bgr = nearest(self.hand_left_buf, t_anchor, self.tolerance_s)
                if bgr is None:
                    return
                rgb = bgr_to_chw_rgb_resized(bgr, (hand_W, hand_H))
                batch_np[key] = rgb[None, ...]
            elif key == "observation.images.cam_hand_right":
                # 手部右：从 hand_right_buf 最近邻对齐
                bgr = nearest(self.hand_right_buf, t_anchor, self.tolerance_s)
                if bgr is None:
                    return
                rgb = bgr_to_chw_rgb_resized(bgr, (hand_W, hand_H))
                batch_np[key] = rgb[None, ...]

        # 加入状态向量
        batch_np[self.state_key] = state_vec[None, ...]
        batch = {
            k: torch.from_numpy(v).to(self.device) for k, v in batch_np.items()
        }

        # 推理
        try:
            action = self.policy.select_action(batch)
        except Exception as e:
            self.get_logger().error(f'policy.select_action failed: {e}')
            return

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
            return

        # 解析输出（不靠 offset，按名字/切片定长处理）
        # 1) 身体 → JointPositionControl
        pub_names: List[str] = []
        pub_pos: List[float] = []
        for i, n in enumerate(self.full_order):
            if n in self.body_canonical and n not in self.drop_set:
                pub_names.append(n)
                pub_pos.append(float(act[i]))

        if pub_names:
            msg = JointPositionControl()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = pub_names
            msg.position = pub_pos
            self.pub_action.publish(msg)

        # 2) 手 → /set_brainco_* qpos6
        # scalar：把对应 index 的 1 维映射为 6 维；qpos6：用对应切片 6 维
        if self.hand_input_mode == 'scalar':
            if 'left' in self.hand_sides and self.idx_left_scalar is not None:
                s = float(act[self.idx_left_scalar])
                q6 = self.map_scalar_to_qpos6(s)
                m = Float64MultiArray()
                m.data = [float(x) for x in q6]
                self.pub_set_left_qpos6.publish(m)
            if 'right' in self.hand_sides and self.idx_right_scalar is not None:
                s = float(act[self.idx_right_scalar])
                q6 = self.map_scalar_to_qpos6(s)
                m = Float64MultiArray()
                m.data = [float(x) for x in q6]
                self.pub_set_right_qpos6.publish(m)

        elif self.hand_input_mode == 'qpos6':
            if 'left' in self.hand_sides and self.slice_left_q6 is not None:
                a, b = self.slice_left_q6
                v = act[a:b]
                m = Float64MultiArray()
                m.data = [float(x) for x in v]
                self.pub_set_left_qpos6.publish(m)
            if 'right' in self.hand_sides and self.slice_right_q6 is not None:
                a, b = self.slice_right_q6
                v = act[a:b]
                m = Float64MultiArray()
                m.data = [float(x) for x in v]
                self.pub_set_right_qpos6.publish(m)

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
