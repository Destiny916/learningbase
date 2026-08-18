#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import numpy as np
import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from std_msgs.msg import Float64
from joint_interfaces.msg import JointPositionControl
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from end_effector_interfaces.msg import EEFeedback, EEJointControl, EEJointControlMode
from std_msgs.msg import String
import json
from act_async_infer_distributed_demo.scripts.network_utils_act import NetworkClient


# For rollout client
from concurrent.futures import ThreadPoolExecutor
from async_infer.async_infer_typedef import *
from async_infer.rollout_client_base import *
from async_infer.rollout_client_functor import *
from async_infer.policy_client_interface import *
from async_infer.policy_client_async import *
from async_infer.processor_interface import *
from async_infer.timed_sequence_array import *
from async_infer.merge_trajectory import *
from async_infer.w1_rollout.w1_rollout_config import (
    W1PositionCommand,
    W1RolloutConfig,
    W1RolloutGripperProcess,
    W1RolloutRobotDoF,
)

# ===== Utils =====

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "test.json"


def resolve_config_path(config_path: Optional[str]) -> Path:
    if config_path is None:
        return DEFAULT_CONFIG_PATH
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def now_sec_raw() -> float:
    import time
    return time.time()


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


# ===== Data containers =====

@dataclass
class TimedFrame:
    t: float
    left_bgr: np.ndarray
    right_bgr: np.ndarray


# ===== Policy invoke interface =====
class InvokePolicyW1(SynchronizedPolicyClientInterface):

    def __init__(
            self,
            make_request_batch_fn,
            remote_reset_policy_fn,
            remote_select_action_chunk_fn,
            action_chunk_to_vector_fn,
            nominal_dt: float,
    ):
        super().__init__()
        self._make_request_batch_fn = make_request_batch_fn
        self._remote_reset_policy_fn = remote_reset_policy_fn
        self._remote_select_action_chunk_fn = remote_select_action_chunk_fn
        self._action_chunk_to_vector_fn = action_chunk_to_vector_fn
        self._nominal_dt = max(float(nominal_dt), 1e-4)

    def __call__(self, invoke_info: PolicyClientInvokeInfo,
                 observation: RolloutClientObservation,
                 current_cmd_trajectory: TimedSequenceArray) -> PolicyClientResponse:
        if observation is None or (not observation.is_valid):
            return PolicyClientResponse(
                request_meta=invoke_info.meta,
                state_trajectory=None,
                error_str="Invalid observation",
            )
        if not isinstance(observation.observation_map, dict):
            return PolicyClientResponse(
                request_meta=invoke_info.meta,
                state_trajectory=None,
                error_str="Observation map must be a dict for W1 rollout",
            )

        obs_np: Dict[str, np.ndarray] = observation.observation_map
        # end_state = np.asarray(current_cmd_trajectory.raw_data_points[-1, :], dtype=np.float32)
        end_state = np.asarray(observation.state, dtype=np.float32)

        try:
            batch = self._make_request_batch_fn(obs_np, end_state)
        except Exception as exc:
            return PolicyClientResponse(
                request_meta=invoke_info.meta,
                state_trajectory=None,
                error_str=f"Failed to preprocess observation: {exc}",
            )

        if not self._remote_reset_policy_fn():
            return PolicyClientResponse(
                request_meta=invoke_info.meta,
                state_trajectory=None,
                error_str="remote reset_policy failed",
            )

        chunk = self._remote_select_action_chunk_fn(batch)
        if chunk is None:
            return PolicyClientResponse(
                request_meta=invoke_info.meta,
                state_trajectory=None,
                error_str="remote select_action_chunk failed",
            )

        try:
            decoded_chunk = self._action_chunk_to_vector_fn(chunk)
        except Exception as exc:
            return PolicyClientResponse(
                request_meta=invoke_info.meta,
                state_trajectory=None,
                error_str=f"Failed to decode remote action chunk: {exc}",
            )

        if decoded_chunk is None or decoded_chunk.ndim != 2 or decoded_chunk.shape[0] == 0:
            return PolicyClientResponse(
                request_meta=invoke_info.meta,
                state_trajectory=None,
                error_str="Decoded action chunk is empty or invalid",
            )

        n_steps = decoded_chunk.shape[0]
        # time_axis_begin = max(float(current_cmd_trajectory.end()), 0.0)
        # time_axis_begin = 0.0
        time_axis_begin = max(float(invoke_info.meta.observation_sync_time), 0.0)
        time_axis = time_axis_begin + np.arange(n_steps, dtype=np.float64) * self._nominal_dt
        state_trajectory = TimedSequenceArray(
            data=decoded_chunk.astype(np.float64, copy=False),
            time=time_axis,
        )
        return PolicyClientResponse(
            request_meta=invoke_info.meta,
            state_trajectory=state_trajectory,
        )


# ===== Node =====

class W1ACTFlexibleNode(Node):
    def __init__(self, config_path: Optional[str] = None,
                 executor_not_own: Optional[ThreadPoolExecutor] = None) -> None:
        super().__init__('w1_act_flexible_node')

        resolved_config_path = resolve_config_path(config_path)
        if not resolved_config_path.exists():
            raise FileNotFoundError(f"Config JSON not found: {resolved_config_path}")
        self.config = W1RolloutConfig.from_json_file(str(resolved_config_path))
        self.get_logger().info(f"Loaded rollout config: {resolved_config_path}")

        ALLOWED_IMAGE_KEYS = {
            "observation.images.cam_high_left",
            "observation.images.cam_high_right",
            "observation.images.cam_hand_left",
            "observation.images.cam_hand_right",
        }
        req_keys = list(self.config.image_keys)
        self.config.image_keys = [k for k in req_keys if k in ALLOWED_IMAGE_KEYS]
        if not self.config.image_keys:
            raise ValueError("image_keys 不能为空：必须从四个允许键中至少选择一个。")

        # ---------- Build IO order ----------
        self.robot_dof = W1RolloutRobotDoF(config=self.config)
        self.full_order: List[str] = list(self.robot_dof.full_order)
        self.full_dim = int(self.robot_dof.full_dim)
        self.body_order: List[str] = list(self.robot_dof.body_order)
        self.idx_left_scalar: Optional[int] = self.robot_dof.idx_left_scalar
        self.idx_right_scalar: Optional[int] = self.robot_dof.idx_right_scalar
        self.slice_left_q6: Optional[Tuple[int, int]] = self.robot_dof.slice_left_q6
        self.slice_right_q6: Optional[Tuple[int, int]] = self.robot_dof.slice_right_q6
        self.body_index_map: Optional[np.ndarray] = self.robot_dof.body_index_map
        self.gripper_processor = W1RolloutGripperProcess(config=self.config)

        self.get_logger().info(
            "=== Inference Order (len=%d) ===\n%s" %
            (self.full_dim, "\n".join([f"{i:02d}: {n}" for i, n in enumerate(self.full_order)]))
        )
        self.get_logger().info(
            f"[train IO] BODY={len(self.body_order)} HAND_MODE={self.config.hand_input_mode} sides={self.config.hand_sides} → full_dim={self.full_dim}"
        )
        if self.config.hand_input_mode == 'scalar':
            self.get_logger().info(f"scalar idx: left={self.idx_left_scalar}, right={self.idx_right_scalar}")
        elif self.config.hand_input_mode == 'qpos6':
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
        self.pub_action = self.create_publisher(JointPositionControl, self.config.publish_topic, q_reliable)
        self.pub_set_left_qpos6 = self.create_publisher(EEJointControl, self.config.set_left_hand_qpos6_topic,
                                                        q_reliable)
        self.pub_set_right_qpos6 = self.create_publisher(EEJointControl, self.config.set_right_hand_qpos6_topic,
                                                         q_reliable)

        # Buffers
        self.body_buf: deque = deque(maxlen=2000)
        self.hand_left_buf: deque = deque(maxlen=200)
        self.hand_right_buf: deque = deque(maxlen=200)
        self.bridge = CvBridge()
        self.scalar_left_buf: deque = deque(maxlen=2000)
        self.scalar_right_buf: deque = deque(maxlen=2000)
        self.qpos6_left_buf: deque = deque(maxlen=2000)
        self.qpos6_right_buf: deque = deque(maxlen=2000)

        # Subs
        q_img = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )
        if "observation.images.cam_hand_left" in self.config.image_keys:
            self.create_subscription(Image, self.config.cam_hand_left_topic, self.cb_img_hand_left, q_img)
            self.get_logger().info(f"Subscribe hand-left image: {self.config.cam_hand_left_topic}")
        if "observation.images.cam_hand_right" in self.config.image_keys:
            self.create_subscription(Image, self.config.cam_hand_right_topic, self.cb_img_hand_right, q_img)
            self.get_logger().info(f"Subscribe hand-right image: {self.config.cam_hand_right_topic}")

        self.create_subscription(String, self.config.joint_topic, self.cb_joint, q_reliable)
        self.create_subscription(Float64, self.config.left_hand_scalar_topic, self.cb_scalar_left, q_reliable)
        self.create_subscription(Float64, self.config.right_hand_scalar_topic, self.cb_scalar_right, q_reliable)
        self.create_subscription(EEFeedback, self.config.left_hand_qpos6_topic, self.cb_qpos6_left, q_reliable)
        self.create_subscription(EEFeedback, self.config.right_hand_qpos6_topic, self.cb_qpos6_right, q_reliable)

        self.get_logger().info("Using remote server for policy preprocess/postprocess.")

        self.network_client = NetworkClient(self.config.remote_server_host, self.config.remote_server_port)
        self.get_logger().info(
            f"Connecting remote policy server at "
            f"{self.config.remote_server_host}:{self.config.remote_server_port}"
        )
        if not self.network_client.connect(timeout=float(self.config.remote_connect_timeout_s)):
            raise RuntimeError(
                f"Could not connect to remote policy server "
                f"{self.config.remote_server_host}:{self.config.remote_server_port}"
            )
        reset_resp = self.network_client.send_request("reset_policy")
        if not reset_resp or reset_resp.get("status") != "ok":
            self.get_logger().warn("reset_policy request failed, continue anyway.")
        self.get_logger().info("Remote policy server connected.")

        # ---------- Async planning buffers ----------
        self.plan_lock = threading.Lock()
        self.plan_queue: deque = deque()  # 每个元素: np.ndarray(full_dim)
        self.replan_in_progress: bool = False

        # horizon: 一次重推理拿多少帧 (例如30)
        self.horizon_N: int = max(1, int(self.config.remote_horizon_n))
        # <= 这个阈值时触发异步补货 (例如15)
        if self.config.remote_replan_trigger > 0:
            self.replan_trigger: int = int(self.config.remote_replan_trigger)
        else:
            self.replan_trigger = max(1, self.horizon_N // 2)
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
        if self.config.hand_input_mode == 'scalar':
            if self.idx_left_scalar is not None:
                self.blend_mask[self.idx_left_scalar] = 0.0
            if self.idx_right_scalar is not None:
                self.blend_mask[self.idx_right_scalar] = 0.0
        elif self.config.hand_input_mode == 'qpos6':
            if self.slice_left_q6 is not None:
                a, b = self.slice_left_q6
                self.blend_mask[a:b] = 0.0
            if self.slice_right_q6 is not None:
                a, b = self.slice_right_q6
                self.blend_mask[a:b] = 0.0

        # Camera thread (SBS)
        self.frame_lock = threading.Lock()
        self.latest_frame: Optional[TimedFrame] = None
        self._cap_stop = False
        th = threading.Thread(target=self.capture_loop_sbs, daemon=True)
        th.start()
        self._cap_thread = th

        # Buffer for rollout client, only accessed on main (timer) thread
        self._client_obs: Optional[RolloutClientObservation] = None
        obs_operation: RolloutClientObservationOperation = lambda t, is_invoke: self._client_take_obs(t, is_invoke)

        # For cmd publish
        self._client_cmd: Optional[NpArray1d] = None
        publish_cmd_fn = lambda t_now, cmd_state, seq_idx: self._client_publish_cmd(t_now, cmd_state, seq_idx)
        logging_fn = lambda content: self.get_logger().info(content)
        output_operation = RolloutClientOutputOperation(publish_cmd_fn=publish_cmd_fn, logger=logging_fn)

        # For policy
        assert executor_not_own is not None
        sync_policy_client = InvokePolicyW1(
            make_request_batch_fn=self._make_seed_batch,
            remote_reset_policy_fn=self._remote_reset_policy,
            remote_select_action_chunk_fn=self._remote_select_action_chunk,
            action_chunk_to_vector_fn=self._decode_remote_action_chunk,
            nominal_dt=1.0 / max(1.0, float(self.config.policy_hz)),
        )
        policy_client = PolicyClientByFunctorOneActive(func=sync_policy_client, executor_now_owned=executor_not_own)
        invoke_after_ratio = 1.0 - (float(self.replan_trigger) / float(max(self.horizon_N, 1)))
        invoke_after_ratio = min(max(invoke_after_ratio, 0.01), 1.0)
        time_before_trajectory_end = float(self.replan_trigger) / max(1.0, float(self.config.policy_hz))
        self.get_logger().info(
            f"[Rollout invoke] invoke_after_ratio={invoke_after_ratio:.3f}, "
            f"time_before_end={time_before_trajectory_end:.3f}s"
        )
        should_invoke_fn = CheckShouldStartNewPolicyInvokeInterface(
            invoke_after_trajectory_ratio=invoke_after_ratio,
            time_before_trajectory_end=time_before_trajectory_end,
        )
        invoke_policy_operation = RolloutClientInvokePolicyOperation(policy_client=policy_client,
                                                                     should_start_new_policy_invoke=should_invoke_fn)

        # Make client
        discrete_state_indices: List[int] = []
        if self.config.hand_input_mode == 'scalar':
            if self.idx_left_scalar is not None:
                discrete_state_indices.append(self.idx_left_scalar)
            if self.idx_right_scalar is not None:
                discrete_state_indices.append(self.idx_right_scalar)
        state_dim_config = AsyncInferStateDimensionConfig(
            state_dim=self.full_dim,
            discrete_tool_state_indices=discrete_state_indices,
        )
        rollout_cmd_config = RolloutClientCommandOption(state_dim_config=state_dim_config,
                                                        merge_option=MergeTrajectoryOption(
                                                            merge_type=MergeTrajectoryType.MergeByNearest,
                                                            merge_blend_ratio=0.1))
        self._rollout_client = RolloutClientFunctor(command_option=rollout_cmd_config,
                                                    observation_operation=obs_operation,
                                                    invoke_operation=invoke_policy_operation,
                                                    output_operation=output_operation)

        # Timer
        self._now_sec_offset = now_sec_raw()
        period = 1.0 / max(1.0, float(self.config.policy_hz))
        self.create_timer(period, self.timer_infer2)

    def now_sec(self) -> float:
        return now_sec_raw() - self._now_sec_offset

    # For rollout client
    def _client_take_obs(self, t_now: float, is_invoke_active: bool):
        return self._client_obs

    def _client_publish_cmd(self, t_now: float, cmd_state: NpArray1d, seq_index: int):
        self._client_cmd = np.copy(cmd_state)

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
                    time.sleep(0.005);
                    continue
                t = self.now_sec()
                h, w2, _ = frame.shape
                if w2 % 2 != 0:
                    self.get_logger().warn(f'Frame width {w2} not even; skip')
                    continue
                w = w2 // 2
                left_bgr = frame[:, :w]
                right_bgr = frame[:, w:]
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
        if joint_pos is None or len(joint_pos) != len(self.config.joint_names):
            self.get_logger().warn(
                f"joint_position length {0 if joint_pos is None else len(joint_pos)} "
                f"!= expected {len(self.config.joint_names)}"
            )
            return

        # 2. 时间戳 now_sec()
        t = self.now_sec()
        q_in = np.asarray(joint_pos, dtype=np.float32)

        # 3. body_index_map 在初始化阶段已由 W1RolloutRobotDoF 构建
        if self.body_index_map is None:
            self.get_logger().warn("body_index_map is empty; skip joint callback.")
            return

        # 4. 和原来一样：重排 + append 到 buf
        try:
            q_body = q_in[self.body_index_map]
        except Exception as e:
            self.get_logger().warn(f"Reorder BODY failed: {e}")
            return
        self.body_buf.append((t, q_body))

    def cb_scalar_left(self, msg: Float64) -> None:
        self.scalar_left_buf.append((self.now_sec(), float(msg.data)))

    def cb_scalar_right(self, msg: Float64) -> None:
        self.scalar_right_buf.append((self.now_sec(), float(msg.data)))

    def _extract_ee_positions(self, msg: EEFeedback) -> Optional[np.ndarray]:
        states = list(msg.joint_states)
        if not states:
            return None

        joint_names = list(self.config.publish_hand_joint_name)
        by_name = {state.name: float(state.position) for state in states}
        if all(name in by_name for name in joint_names):
            return np.asarray([by_name[name] for name in joint_names], dtype=np.float32)

        if len(states) >= len(joint_names):
            return np.asarray([state.position for state in states[:len(joint_names)]], dtype=np.float32)

        return None

    def cb_qpos6_left(self, msg: EEFeedback) -> None:
        arr = self._extract_ee_positions(msg)
        if arr is not None:
            self.qpos6_left_buf.append((self.now_sec(), arr.copy()))

    def cb_qpos6_right(self, msg: EEFeedback) -> None:
        arr = self._extract_ee_positions(msg)
        if arr is not None:
            self.qpos6_right_buf.append((self.now_sec(), arr.copy()))

    def cb_img_hand_left(self, msg: Image) -> None:
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge left-hand fail: {e}")
            return
        t = self.now_sec()
        self.hand_left_buf.append((t, bgr))

    def cb_img_hand_right(self, msg: Image) -> None:
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge right-hand fail: {e}")
            return
        t = self.now_sec()
        self.hand_right_buf.append((t, bgr))

    def _make_action(self, act_np: np.ndarray) -> W1PositionCommand:
        return self.robot_dof.make_action(
            act_np=act_np,
            processor=self.gripper_processor,
            t_now_second=self.now_sec(),
        )

    def _publish_action(self, position_cmd: W1PositionCommand) -> None:
        if position_cmd.robot_cmd is not None:
            msg = JointPositionControl()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = list(position_cmd.robot_cmd.joint_names)
            msg.position = list(position_cmd.robot_cmd.joint_values)
            self.pub_action.publish(msg)

        if position_cmd.left_hand_cmd is not None:
            msg = EEJointControl()
            msg.mode = EEJointControlMode.POSITION
            msg.name = list(position_cmd.left_hand_cmd.joint_names)
            msg.value = list(position_cmd.left_hand_cmd.joint_values)
            self.pub_set_left_qpos6.publish(msg)

        if position_cmd.right_hand_cmd is not None:
            msg = EEJointControl()
            msg.mode = EEJointControlMode.POSITION
            msg.name = list(position_cmd.right_hand_cmd.joint_names)
            msg.value = list(position_cmd.right_hand_cmd.joint_values)
            self.pub_set_right_qpos6.publish(msg)

    def _publish_action_vector(self, act_np: np.ndarray) -> None:
        position_cmd = self._make_action(act_np)
        self._publish_action(position_cmd)

        # 记录：我们刚刚真的下发给机器人的这一帧
        self.last_cmd_vec = act_np.astype(np.float32, copy=True)
        self.have_last_cmd = True

    def _make_seed_batch(
            self,
            obs_base: Dict[str, np.ndarray],
            seed_vec_np: np.ndarray
    ) -> Dict[str, np.ndarray]:
        obs = dict(obs_base)
        obs[self.config.state_key] = seed_vec_np.astype(np.float32)
        batch_np: Dict[str, np.ndarray] = {}
        for key, value in obs.items():
            arr = np.asarray(value)
            if arr.dtype == np.object_:
                continue
            batch_np[key] = arr
        return batch_np

    def _ensure_remote_connected(self) -> bool:
        if self.network_client.connected:
            return True
        self.get_logger().warn(
            f"Remote policy disconnected, reconnecting "
            f"{self.config.remote_server_host}:{self.config.remote_server_port} ..."
        )
        ok = self.network_client.connect(timeout=float(self.config.remote_connect_timeout_s))
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

    def _decode_remote_action_chunk(self, chunk: np.ndarray) -> Optional[np.ndarray]:
        actions_local: List[np.ndarray] = []
        for i in range(chunk.shape[0]):
            act_np = self.robot_dof.action_to_np(chunk[i])
            if act_np is None:
                return None
            actions_local.append(act_np)
        if len(actions_local) == 0:
            return None
        return np.stack(actions_local, axis=0).astype(np.float32, copy=False)

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

            t0 = self.now_sec()
            try:
                chunk = self._remote_select_action_chunk(batch)
                if chunk is None:
                    self.replan_in_progress = False
                    return
                t1 = self.now_sec()
                infer_dt = t1 - t0
                if infer_dt > 0.1:
                    self.get_logger().info(
                        f'[async] heavy remote chunk inference {infer_dt:.3f}s (worker)'
                    )

                for i in range(chunk.shape[0]):
                    act_np = self.robot_dof.action_to_np(chunk[i])
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
            q_body = nearest(self.body_buf, tf.t, float(self.config.tolerance_ms) / 1000.0)
            # print("collect q body success")
            if q_body is None:
                print("here, no qbody")
                return
            body_map = {n: float(v) for n, v in zip(self.body_order, q_body)}
        else:
            body_map = {}

        tolerance_s = float(self.config.tolerance_ms) / 1000.0
        left_scalar = nearest(self.scalar_left_buf, tf.t, tolerance_s)
        right_scalar = nearest(self.scalar_right_buf, tf.t, tolerance_s)
        left_q6 = nearest(self.qpos6_left_buf, tf.t, tolerance_s)
        right_q6 = nearest(self.qpos6_right_buf, tf.t, tolerance_s)

        for n in self.full_order:
            if n in self.config.body_canonical:
                state_vals.append(body_map.get(n, 0.0))
                continue

            if self.config.hand_input_mode == 'scalar':
                if n == self.config.left_scalar_name and 'left' in self.config.hand_sides:
                    state_vals.append(0.0 if left_scalar is None else float(left_scalar))
                    continue
                if n == self.config.right_scalar_name and 'right' in self.config.hand_sides:
                    state_vals.append(0.0 if right_scalar is None else float(right_scalar))
                    continue

            elif self.config.hand_input_mode == 'qpos6':
                if n in self.config.left_q6_names and 'left' in self.config.hand_sides:
                    i = self.config.left_q6_names.index(n)
                    v = 0.0 if left_q6 is None else float(left_q6[i])
                    state_vals.append(v);
                    continue
                if n in self.config.right_q6_names and 'right' in self.config.hand_sides:
                    i = self.config.right_q6_names.index(n)
                    v = 0.0 if right_q6 is None else float(right_q6[i])
                    state_vals.append(v);
                    continue

            state_vals.append(0.0)

        state_vec = np.asarray(state_vals, dtype=np.float32)
        if state_vec.shape[0] != self.full_dim:
            self.get_logger().error(f"State dim {state_vec.shape} != expected {self.full_dim}")
            return

        # 3) 打包 observation (raw numpy)
        obs_np: Dict[str, np.ndarray] = {}
        head_W, head_H = int(self.config.head_target_width), int(self.config.head_target_height)
        hand_W, hand_H = int(self.config.hand_target_width), int(self.config.hand_target_height)

        for key in self.config.image_keys:
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
                bgr = nearest(self.hand_left_buf, t_anchor, tolerance_s)
                if bgr is None:
                    return
                rgb = bgr_to_hwc_rgb_resized(bgr, (hand_W, hand_H))
                obs_np[key] = rgb
            elif key == "observation.images.cam_hand_right":
                bgr = nearest(self.hand_right_buf, t_anchor, tolerance_s)
                if bgr is None:
                    return
                rgb = bgr_to_hwc_rgb_resized(bgr, (hand_W, hand_H))
                obs_np[key] = rgb

        obs_np[self.config.state_key] = state_vec.astype(np.float32)
        # batch: Dict[str, torch.Tensor] = self._preprocess_observation(obs_np)

        # 4) 异步控制逻辑
        # 4a. 没有弹药就同步roll一段
        with self.plan_lock:
            qlen = len(self.plan_queue)
            busy = self.replan_in_progress

        if qlen == 0 and (not busy):
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

    def timer_infer2(self) -> None:
        # 1) 最新相机帧
        with self.frame_lock:
            tf = self.latest_frame
        if tf is None:
            return
        t_anchor = tf.t
        t_now = self.now_sec()

        # 2) proprio 状态 (按照 full_order)
        state_vals: List[float] = []

        if len(self.body_order) > 0:
            q_body = nearest(self.body_buf, tf.t, float(self.config.tolerance_ms) / 1000.0)
            # print("collect q body success")
            if q_body is None:
                print("here, no qbody")
                return
            body_map = {n: float(v) for n, v in zip(self.body_order, q_body)}
        else:
            body_map = {}

        tolerance_s = float(self.config.tolerance_ms) / 1000.0
        left_scalar = nearest(self.scalar_left_buf, tf.t, tolerance_s)
        right_scalar = nearest(self.scalar_right_buf, tf.t, tolerance_s)
        left_q6 = nearest(self.qpos6_left_buf, tf.t, tolerance_s)
        right_q6 = nearest(self.qpos6_right_buf, tf.t, tolerance_s)

        for n in self.full_order:
            if n in self.config.body_canonical:
                state_vals.append(body_map.get(n, 0.0))
                continue

            if self.config.hand_input_mode == 'scalar':
                if n == self.config.left_scalar_name and 'left' in self.config.hand_sides:
                    state_vals.append(0.0 if left_scalar is None else float(left_scalar))
                    continue
                if n == self.config.right_scalar_name and 'right' in self.config.hand_sides:
                    state_vals.append(0.0 if right_scalar is None else float(right_scalar))
                    continue

            elif self.config.hand_input_mode == 'qpos6':
                if n in self.config.left_q6_names and 'left' in self.config.hand_sides:
                    i = self.config.left_q6_names.index(n)
                    v = 0.0 if left_q6 is None else float(left_q6[i])
                    state_vals.append(v);
                    continue
                if n in self.config.right_q6_names and 'right' in self.config.hand_sides:
                    i = self.config.right_q6_names.index(n)
                    v = 0.0 if right_q6 is None else float(right_q6[i])
                    state_vals.append(v);
                    continue

            state_vals.append(0.0)

        state_vec = np.asarray(state_vals, dtype=np.float32)
        if state_vec.shape[0] != self.full_dim:
            self.get_logger().error(f"State dim {state_vec.shape} != expected {self.full_dim}")
            return

        # 3) 打包 observation (raw numpy)
        obs_np: Dict[str, np.ndarray] = {}
        head_W, head_H = int(self.config.head_target_width), int(self.config.head_target_height)
        hand_W, hand_H = int(self.config.hand_target_width), int(self.config.hand_target_height)

        for key in self.config.image_keys:
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
                bgr = nearest(self.hand_left_buf, t_anchor, tolerance_s)
                if bgr is None:
                    return
                rgb = bgr_to_hwc_rgb_resized(bgr, (hand_W, hand_H))
                obs_np[key] = rgb
            elif key == "observation.images.cam_hand_right":
                bgr = nearest(self.hand_right_buf, t_anchor, tolerance_s)
                if bgr is None:
                    return
                rgb = bgr_to_hwc_rgb_resized(bgr, (hand_W, hand_H))
                obs_np[key] = rgb

        obs_np[self.config.state_key] = state_vec.astype(np.float32)

        # Copy obs
        self._client_obs = RolloutClientObservation(sync_time=t_anchor, observation_map=obs_np,
                                                    process_obs_status_str=None,
                                                    explicit_state=ensure_immutable_numpy(state_vec))

        # Invoke client
        self._rollout_client.loop_once(t_now=t_now)

        # Get action
        if self._client_cmd is None:
            return
        act_np = np.copy(self._client_cmd)

        if act_np is not None:
            self._publish_action_vector(act_np)
        return


# ===== main =====
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to rollout config json",
    )
    args = parser.parse_args()

    rclpy.init()
    executor = ThreadPoolExecutor(max_workers=1)
    node = W1ACTFlexibleNode(config_path=args.config, executor_not_own=executor)
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
    executor.shutdown()


if __name__ == '__main__':
    main()
