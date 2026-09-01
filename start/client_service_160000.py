"""PC1 real-robot client for the isolated ACT-DINOv3 160000 server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from queue import Empty, Queue
import sys
import time
import types

import cv2
import numpy as np


W1_ACT_ROOT = Path(__file__).resolve().parents[1]
if str(W1_ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(W1_ACT_ROOT))

_lipo_name = "act_async_infer_distributed_demo.scripts.action_lipo"
if _lipo_name not in sys.modules:
    disabled_lipo = types.ModuleType(_lipo_name)
    disabled_lipo.ActionLiPo = type("DisabledActionLiPo", (), {})
    sys.modules[_lipo_name] = disabled_lipo

import rclpy
from end_effector_interfaces.msg import EEFeedback, EEJointControl, EEJointControlMode
import end_effector_interfaces.msg as ee_messages

if not hasattr(ee_messages, "EEJointFeedback"):
    ee_messages.EEJointFeedback = EEFeedback

from act_async_infer_distributed_demo.scripts.client.robot_client_with_kingfisher import (
    OptimizedRobotClient,
    RobotState,
)
from act_async_infer_distributed_demo.scripts.inference_config import (
    ClientConfig,
    RequestType,
    ResponseKey,
)
from act_async_infer_distributed_demo.scripts.w1_mapping import CommonKey
from act_async_infer_distributed_demo.scripts.utils_distributed import TimedAction, log_info

from .client_runtime_160000 import (
    AdjacentFeedbackBuffer,
    Chunk16Gate,
    Client160000Error,
    assemble_absolute_state,
    hand_command_from_openness,
    hand_openness_from_feedback,
    latest_fresh_image,
    preprocess_images,
)
from .runtime import (
    BODY_ORDER,
    LEFT_CLOSED,
    LEFT_OPEN,
    RIGHT_CLOSED,
    RIGHT_OPEN,
    action_to_commands,
    validate_feedback_freshness,
    validate_observation_buffers,
    validate_robot_health,
    validate_robot_ready,
)


HAND_NAMES = ("T_MCP", "T_CMC_YAW", "IF_MCP_PITCH", "MF_MCP_PITCH", "RF_MCP_PITCH", "LF_MCP_PITCH")
_vendor_setup = OptimizedRobotClient._cmd_setup_config
_vendor_joint_callback = OptimizedRobotClient.joint_state_callback


def _hand_values(message):
    by_name = {str(item.name): float(item.position) for item in message.joint_states}
    missing = [name for name in HAND_NAMES if name not in by_name]
    if missing:
        raise Client160000Error(f"hand feedback missing joints: {', '.join(missing)}")
    values = np.asarray([by_name[name] for name in HAND_NAMES], dtype=np.float32)
    if not np.isfinite(values).all():
        raise Client160000Error("hand feedback must be finite")
    return values


def _hand_callback(side):
    def callback(self, message):
        try:
            values = _hand_values(message)
        except Client160000Error as exc:
            self.get_logger().error(str(exc))
            return
        target = self.hand_qpos6_left_buf if side == "left" else self.hand_qpos6_right_buf
        target.append((time.monotonic(), values))
        if side == "left":
            self._left_hand_received_at_160000 = time.monotonic()
        else:
            self._right_hand_received_at_160000 = time.monotonic()
    return callback


def _joint_callback(self, message):
    try:
        payload = json.loads(message.data)
        positions = np.asarray(payload["joint_position"], dtype=np.float32)
        if not self.hand_qpos6_left_buf or not self.hand_qpos6_right_buf:
            raise Client160000Error("both real hand feedback streams are required")
        now = time.monotonic()
        validate_feedback_freshness(
            getattr(self, "_left_hand_received_at_160000", 0.0),
            now=now,
            timeout_seconds=1.0,
        )
        validate_feedback_freshness(
            getattr(self, "_right_hand_received_at_160000", 0.0),
            now=now,
            timeout_seconds=1.0,
        )
        left = hand_openness_from_feedback(self.hand_qpos6_left_buf[-1][1], LEFT_CLOSED, LEFT_OPEN)
        right = hand_openness_from_feedback(self.hand_qpos6_right_buf[-1][1], RIGHT_CLOSED, RIGHT_OPEN)
        state = assemble_absolute_state(positions, left, right)
        timestamp = float(payload["timestamp"])
    except (KeyError, TypeError, ValueError, Client160000Error) as exc:
        self.get_logger().error(f"160000 feedback snapshot rejected: {exc}")
    else:
        if not hasattr(self, "_feedback_160000"):
            self._feedback_160000 = AdjacentFeedbackBuffer()
        try:
            self._feedback_160000.append(
                state,
                timestamp,
                received_at=time.monotonic(),
            )
        except Client160000Error as exc:
            self.get_logger().error(f"160000 feedback snapshot rejected: {exc}")
    _vendor_joint_callback(self, message)
    try:
        self._latest_robot_state = payload
        self._latest_robot_state_received_at = time.monotonic()
    except UnboundLocalError:
        self._latest_robot_state = None


def _raw_camera_callback(buffer_name, expected_hw):
    def callback(self, message):
        try:
            bgr = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            if bgr.shape[:2] != expected_hw:
                raise Client160000Error(
                    f"{buffer_name} expected {expected_hw}, got {bgr.shape[:2]}"
                )
            rgb = bgr[..., ::-1].copy()
        except Exception as exc:
            self.get_logger().error(f"160000 camera rejected: {exc}")
            return
        getattr(self, buffer_name).append((time.monotonic(), rgb))
    return callback


def _get_observation(self):
    history = getattr(self, "_feedback_160000", None)
    if history is None or history.newest_sequence < 1:
        raise Client160000Error("two adjacent real feedback frames are required")
    now = time.monotonic()
    images = preprocess_images(
        latest_fresh_image(self.head_right_buf, "head right", now=now),
        latest_fresh_image(self.hand_left_buf, "physical left wrist", now=now),
        latest_fresh_image(self.hand_right_buf, "physical right wrist", now=now),
        resize=lambda image, size: cv2.resize(image, size, interpolation=cv2.INTER_AREA),
    )
    return {CommonKey.IMAGES.value: images, CommonKey.STATES.value: {}}, time.time()


def _send_observation(self, observation):
    if not self.start_infer:
        return True
    try:
        pair = self._feedback_160000.take_new_pair(now=time.monotonic(), max_age_seconds=1.0)
        raw = observation.get_observation()
        payload = {
            **pair,
            "timestamp": observation.get_timestamp(),
            "timestep": observation.get_timestep(),
            "must_go": observation.must_go,
            "start_infer": self.start_infer,
            "image_target_size": [224, 224],
        }
        for key, image in raw[CommonKey.IMAGES.value].items():
            payload[key] = np.asarray(image, dtype=np.uint8).tobytes()
        with self.socket_lock:
            response = self.network_client.send_request(RequestType.OBSERVATION, payload)
        if not response or response.get(ResponseKey.STATUS) != "received":
            raise Client160000Error(f"server rejected observation: {response}")
        self._gate_160000.mark_requested(pair["current_feedback"]["sequence"])
        self.start_infer = False
        return True
    except Exception as exc:
        self._set_state(RobotState.ERROR, str(exc))
        return False


def _ready(self):
    if not self.allow_infer_check:
        return None
    history = getattr(self, "_feedback_160000", None)
    if history is None:
        return None
    with self.action_queue_lock:
        queue_size = self.action_queue.qsize()
    if not self._gate_160000.can_request(
        queue_size=queue_size,
        newest_feedback_sequence=history.newest_sequence,
    ):
        return None
    self.must_go.set()
    self.start_infer = True
    self.allow_infer_check = False


def _get_actions(self):
    with self.socket_lock:
        response = self.network_client.send_request(RequestType.GET_ACTIONS)
    if not response or response.get(ResponseKey.STATUS) != ResponseKey.SUCCESS:
        return None
    if response.get("protocol_version") != 2 or response.get("action_representation") != "absolute":
        raise Client160000Error("server did not return protocol-v2 absolute actions")
    if response.get("action_shape") != [16, 19]:
        raise Client160000Error("server action_shape must be [16, 19]")
    grouped = response["actions"]["qpos"]
    order = ("waistqpos", "left_armqpos", "headqpos", "right_armqpos", "left_eefgripper", "right_eefgripper")
    rows = []
    for step in range(16):
        rows.append(np.concatenate([np.asarray(grouped[key][step], dtype=np.float32) for key in order]))
    actions = np.asarray(rows, dtype=np.float32)
    if actions.shape != (16, 19) or not np.isfinite(actions).all():
        raise Client160000Error("received actions must be finite 16x19")
    self.current_joint_order = list(order)
    return self._time_action_chunk(float(response["timestamp"]), actions, int(response["timestep"]))


def _direct_queue(self, incoming_actions, aggregate_fn=None):
    if len(incoming_actions) != 16:
        raise Client160000Error("one chunk must contain exactly 16 actions")
    queue = Queue()
    for action in incoming_actions:
        queue.put(TimedAction(action.timestamp, action.timestep, np.asarray(action.action).copy()))
    with self.action_queue_lock:
        if not self.action_queue.empty():
            raise Client160000Error("prior action queue must be empty before next chunk")
        self.action_queue = queue
    self.action_chunk_size = 16
    self.first_get_actions = False


def _ee_message(self, values):
    message = EEJointControl()
    message.header.stamp = self._now()
    message.mode = EEJointControlMode.POSITION
    message.joint_names = list(HAND_NAMES)
    message.values = [float(value) for value in values]
    return message


def _exec_action(self, timed_action):
    state = getattr(self, "_latest_robot_state", None)
    if not state:
        raise Client160000Error("robot feedback disappeared")
    validate_feedback_freshness(
        getattr(self, "_latest_robot_state_received_at", 0.0),
        now=time.monotonic(), timeout_seconds=1.0,
    )
    validate_robot_health(state, allowed_status=("Idle", "Running"))
    command = action_to_commands(timed_action.get_action())
    self.publish_joint_positions(BODY_ORDER, command.body_positions, clip=False)
    self._pub_hand_left.publish(_ee_message(self, command.left_hand))
    self._pub_hand_right.publish(_ee_message(self, command.right_hand))
    self.current_step += 1
    progress = self._gate_160000.mark_published()
    if progress.chunk_complete:
        self._gate_160000.mark_chunk_completed(self._feedback_160000.newest_sequence)
        self._last_chunk_completed_at = time.monotonic()
        self.allow_infer_check = True
        while not self.observation_queue.empty():
            try:
                self.observation_queue.get_nowait()
            except Empty:
                break
    return timed_action


def _setup(self, client_config, server_config, _home_position):
    required = {
        "mode": 2, "action_horizon": 16,
        "sample_factor": 1, "chunk_size_threshold": 0,
    }
    for key, expected in required.items():
        if client_config.get(key) != expected:
            return {ResponseKey.SUCCESS: False, ResponseKey.MESSAGE: f"{key} must be {expected}"}
    for key in ("control_frequency", "collect_frequency"):
        if client_config.get(key) not in (10, 30):
            return {ResponseKey.SUCCESS: False, ResponseKey.MESSAGE: f"{key} must be 10 or 30"}
    if not getattr(self, "_latest_robot_state", None):
        return {ResponseKey.SUCCESS: False, ResponseKey.MESSAGE: "real feedback unavailable"}
    # Keep the health/feedback checks, but allow an explicitly authorized
    # deployment to bypass only the ACT default-pose gate.
    if not bool(client_config.get("skip_default_pose_check", False)):
        validate_robot_ready(self._latest_robot_state, tolerance_rad=0.05)
    validate_observation_buffers(
        {
            "head_left": self.head_left_buf, "head_right": self.head_right_buf,
            "wrist_left": self.hand_left_buf, "wrist_right": self.hand_right_buf,
            "joint_state": self.joint_state_buf,
            "left_hand": self.hand_qpos6_left_buf, "right_hand": self.hand_qpos6_right_buf,
        }, use_wrist_images=True,
    )
    self._feedback_160000 = AdjacentFeedbackBuffer()
    self._gate_160000 = Chunk16Gate()
    server_config = dict(server_config)
    server_config.update(data_type="real", protocol_version=2, action_horizon=16)
    return _vendor_setup(self, client_config, server_config, "")


def install_hooks():
    OptimizedRobotClient.joint_state_callback = _joint_callback
    OptimizedRobotClient.hand_qpos6_left_callback = _hand_callback("left")
    OptimizedRobotClient.hand_qpos6_right_callback = _hand_callback("right")
    OptimizedRobotClient.cb_img_head_right = _raw_camera_callback("head_right_buf", (540, 960))
    OptimizedRobotClient.cb_img_hand_left = _raw_camera_callback("hand_left_buf", (360, 640))
    OptimizedRobotClient.cb_img_hand_right = _raw_camera_callback("hand_right_buf", (480, 640))
    OptimizedRobotClient._cmd_setup_config = _setup
    OptimizedRobotClient.get_real_obs = _get_observation
    OptimizedRobotClient.send_observation = _send_observation
    OptimizedRobotClient._ready_to_send_observation = _ready
    OptimizedRobotClient.get_actions = _get_actions
    OptimizedRobotClient._aggregate_action_queues = _direct_queue
    OptimizedRobotClient.exec_action = _exec_action
    OptimizedRobotClient.ros_spin = lambda self: None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    install_hooks()
    config = ClientConfig.from_json_file(args.config)
    config.service = True
    rclpy.init()
    client = OptimizedRobotClient(config)
    client._start_manager_listener()
    try:
        while rclpy.ok():
            rclpy.spin_once(client, timeout_sec=0.1)
    finally:
        client.stop()


if __name__ == "__main__":
    main()
