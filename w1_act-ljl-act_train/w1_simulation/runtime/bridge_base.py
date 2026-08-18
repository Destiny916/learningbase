from __future__ import annotations

import json
import os
import threading
import time
import xml.etree.ElementTree as ET
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from multiprocessing import shared_memory
from multiprocessing.connection import Client
from typing import Any

import numpy as np

from w1_simulation.robot.act_adapter import ActHandGestureConfig, W1ActAdapter
from w1_simulation.robot.joints import BODY_FEEDBACK_JOINTS, BODY_JOINTS
from w1_simulation.runtime.w1_ros_transport import W1RosMessageTypes, W1RosPublishers, W1RosTransport
from w1_simulation.w1_profile import DEFAULT_PROFILE

ROS_IMPORT_ERROR: ModuleNotFoundError | None = None
try:
    import rclpy
    from cv_bridge import CvBridge
    from end_effector_interfaces.msg import EEJointControl, EEJointControlMode
    from joint_interfaces.msg import JointPositionControl
    from rcl_interfaces.msg import ParameterDescriptor, ParameterType
    from rclpy.executors import ExternalShutdownException as _ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image
    from std_msgs.msg import String

    ExternalShutdownException = _ExternalShutdownException
except ModuleNotFoundError as exc:
    ROS_IMPORT_ERROR = exc
    rclpy = None
    Node = object


DEFAULT_POLICY_HZ = float(DEFAULT_PROFILE.runtime["policy_hz"])
CHUNK_SIZE = int(DEFAULT_PROFILE.act["n_action_steps"])
GRIPPER_MIN = 0.0
GRIPPER_MAX = 100.0
MAX_OBSERVATION_AGE_S = 0.25
IPC_HANDSHAKE_TIMEOUT_S = 10.0
IPC_INFERENCE_TIMEOUT_S = 1.0


def now_sec() -> float:
    return time.time()


def message_timestamp(message: Any) -> float:
    stamp = getattr(getattr(message, "header", None), "stamp", None)
    if stamp is None:
        return now_sec()
    seconds = float(getattr(stamp, "sec", 0)) + float(getattr(stamp, "nanosec", 0)) * 1e-9
    return seconds if seconds > 0.0 else now_sec()


def normalize_timestamp(value: Any, fallback: float) -> float:
    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        return fallback
    if not np.isfinite(timestamp) or timestamp <= 0.0:
        return fallback
    if timestamp > 1e17:
        return timestamp / 1e9
    if timestamp > 1e14:
        return timestamp / 1e6
    if timestamp > 1e11:
        return timestamp / 1e3
    return timestamp


@dataclass(frozen=True)
class BufferedSample:
    sample_time: float
    received_at: float
    value: Any


def load_revolute_joint_limits(path: str) -> dict[str, tuple[float, float]]:
    resolved = os.path.abspath(os.path.expanduser(path))
    if not resolved or not os.path.isfile(resolved):
        raise FileNotFoundError(f"URDF does not exist: {resolved}")
    try:
        root = ET.parse(resolved).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"URDF is not valid XML: {resolved}") from exc
    limits: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        if joint.get("type") != "revolute":
            continue
        name = str(joint.get("name") or "").strip()
        limit = joint.find("limit")
        if not name or limit is None:
            continue
        try:
            lower = float(limit.get("lower"))
            upper = float(limit.get("upper"))
        except (TypeError, ValueError):
            continue
        if lower <= upper:
            limits[name] = (lower, upper)
    if not limits:
        raise ValueError(f"URDF contains no usable revolute joint limits: {resolved}")
    return limits


def prepare_resized(image_bgr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    target_width, target_height = size
    if image_bgr.shape[:2] == (target_height, target_width):
        return np.asarray(image_bgr, dtype=np.uint8)
    import cv2

    return cv2.resize(image_bgr, (target_width, target_height), interpolation=cv2.INTER_LINEAR)


def interpolate_actions(actions: np.ndarray, sample_factor: int) -> np.ndarray:
    values = np.asarray(actions, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError(f"actions must be a finite 2D array, got {values.shape}")
    if isinstance(sample_factor, bool) or not isinstance(sample_factor, int) or sample_factor < 1:
        raise ValueError("sample_factor must be a positive integer")
    if sample_factor == 1 or len(values) <= 1:
        return values.copy()
    old_positions = np.linspace(0.0, 1.0, len(values))
    new_positions = np.linspace(0.0, 1.0, len(values) * sample_factor)
    result = np.empty((len(new_positions), values.shape[1]), dtype=np.float32)
    for dimension in range(values.shape[1]):
        result[:, dimension] = np.interp(new_positions, old_positions, values[:, dimension])
    return result


class PolicyIPCClient:
    def __init__(
        self,
        *,
        server_port: int,
        full_dim: int,
        horizon: int,
        image_keys: list[str],
        state_key: str,
        head_shape: tuple[int, int, int],
        hand_shape: tuple[int, int, int],
    ) -> None:
        self.server_port = int(server_port)
        self.full_dim = int(full_dim)
        self.horizon = int(horizon)
        self.image_keys = list(image_keys)
        self.state_key = str(state_key)
        self.head_shape = tuple(head_shape)
        self.hand_shape = tuple(hand_shape)
        self.image_shapes = {
            key: self.hand_shape if "hand" in key else self.head_shape for key in self.image_keys
        }
        self.slot_size = max(int(np.prod(shape)) for shape in self.image_shapes.values())
        self.num_slots = len(self.image_keys)
        self.obs_size = self.num_slots * self.slot_size + self.full_dim * 4
        self.acts_size = self.horizon * self.full_dim * 4
        token = f"{os.getpid()}_{id(self):x}"
        self.obs_name = f"policy_obs_{token}"
        self.acts_name = f"policy_acts_{token}"
        self.shm_obs = shared_memory.SharedMemory(create=True, name=self.obs_name, size=self.obs_size)
        self.shm_acts = shared_memory.SharedMemory(create=True, name=self.acts_name, size=self.acts_size)
        self._connection = None
        self._connected_event = threading.Event()
        self._lock = threading.RLock()

    @property
    def connected(self) -> bool:
        return self._connected_event.is_set()

    def _disconnect_locked(self) -> None:
        if self._connection is not None:
            with suppress(Exception):
                self._connection.close()
        self._connection = None
        self._connected_event.clear()

    @staticmethod
    def _receive(connection: Any, timeout_s: float, operation: str) -> Any:
        if not connection.poll(timeout_s):
            raise TimeoutError(f"model server timed out during {operation}")
        return connection.recv()

    def disconnect(self) -> None:
        with self._lock:
            self._disconnect_locked()

    def _write_observation(self, observation: dict[str, np.ndarray]) -> None:
        for slot, key in enumerate(self.image_keys):
            image = np.asarray(observation[key], dtype=np.uint8)
            expected_shape = self.image_shapes[key]
            if image.shape != expected_shape:
                raise ValueError(f"{key} has shape {image.shape}, expected {expected_shape}")
            offset = slot * self.slot_size
            view = np.ndarray(image.shape, dtype=np.uint8, buffer=self.shm_obs.buf, offset=offset)
            np.copyto(view, image[:, :, ::-1])
        state = np.asarray(observation[self.state_key], dtype=np.float32)
        if state.shape != (self.full_dim,) or not np.isfinite(state).all():
            raise ValueError(f"state must have shape ({self.full_dim},) and be finite")
        state_offset = self.num_slots * self.slot_size
        view = np.ndarray((self.full_dim,), dtype=np.float32, buffer=self.shm_obs.buf, offset=state_offset)
        np.copyto(view, state)

    def _read_actions(self, count: int) -> np.ndarray:
        if count <= 0 or count > self.horizon:
            raise ValueError(f"invalid action count from server: {count}")
        actions = np.ndarray(
            (self.horizon, self.full_dim),
            dtype=np.float32,
            buffer=self.shm_acts.buf,
        )
        result = actions[:count].copy()
        if not np.isfinite(result).all():
            raise ValueError("model server returned non-finite actions")
        return result

    def _dummy_observation(self) -> dict[str, np.ndarray]:
        observation = {key: np.zeros(shape, dtype=np.uint8) for key, shape in self.image_shapes.items()}
        observation[self.state_key] = np.zeros(self.full_dim, dtype=np.float32)
        return observation

    def connect(self, active_model_id: str) -> None:
        with self._lock:
            self._disconnect_locked()
            connection = Client(("127.0.0.1", self.server_port), authkey=b"w1_simulation_secret")
            try:
                connection.send(
                    {
                        "cmd": "SHM_INIT",
                        "obs_name": self.obs_name,
                        "acts_name": self.acts_name,
                        "obs_size": self.obs_size,
                        "acts_size": self.acts_size,
                        "num_slots": self.num_slots,
                        "slot_size": self.slot_size,
                        "state_dim": self.full_dim,
                        "horizon_N": self.horizon,
                        "image_keys": self.image_keys,
                        "state_key": self.state_key,
                        "head_shape": list(self.head_shape),
                        "hand_shape": list(self.hand_shape),
                    }
                )
                if self._receive(connection, IPC_HANDSHAKE_TIMEOUT_S, "SHM_INIT") != "OK":
                    raise RuntimeError("SHM_INIT failed")
                self._write_observation(self._dummy_observation())
                connection.send({"cmd": "INFER_CHUNK", "steps": 1})
                warmup = self._receive(connection, IPC_HANDSHAKE_TIMEOUT_S, "warmup")
                if not isinstance(warmup, dict) or warmup.get("status") != "OK":
                    raise RuntimeError(f"model warmup failed: {warmup!r}")
                connection.send({"cmd": "SWITCH_MODEL", "target": str(active_model_id)})
                if self._receive(connection, IPC_HANDSHAKE_TIMEOUT_S, "model switch") != "OK":
                    raise RuntimeError(f"model switch failed: {active_model_id}")
            except Exception:
                connection.close()
                raise
            self._connection = connection
            self._connected_event.set()

    def infer_chunk(self, observation: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
        with self._lock:
            if self._connection is None:
                raise ConnectionError("model server is not connected")
            started = time.perf_counter()
            try:
                self._connection.send({"cmd": "RESET"})
                if self._receive(self._connection, IPC_INFERENCE_TIMEOUT_S, "policy reset") != "OK":
                    raise RuntimeError("policy reset failed")
                self._write_observation(observation)
                self._connection.send({"cmd": "INFER_CHUNK", "steps": self.horizon})
                response = self._receive(
                    self._connection,
                    IPC_INFERENCE_TIMEOUT_S,
                    "chunk inference",
                )
                if not isinstance(response, dict) or response.get("status") != "OK":
                    raise RuntimeError(f"inference failed: {response!r}")
                action_count = int(response["n_steps"])
                if action_count != self.horizon:
                    raise RuntimeError(
                        f"model server returned {action_count} actions, expected {self.horizon}"
                    )
                actions = self._read_actions(action_count)
            except Exception:
                self._disconnect_locked()
                raise
            return actions, (time.perf_counter() - started) * 1000.0

    def close(self) -> None:
        self.disconnect()
        for memory in (self.shm_obs, self.shm_acts):
            try:
                memory.close()
                memory.unlink()
            except FileNotFoundError:
                pass


@dataclass(frozen=True)
class ObservationSnapshot:
    anchor_time: float
    images: dict[str, np.ndarray]
    state: np.ndarray
    state_source: str


class W1PolicyBridgeBase(Node):
    def __init__(self) -> None:
        if ROS_IMPORT_ERROR is not None:
            raise RuntimeError(
                "ROS2 Python dependencies are required to run policy_bridge.py"
            ) from ROS_IMPORT_ERROR
        super().__init__("policy_bridge")

        string_array = ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY)
        self.declare_parameter("policy_hz", DEFAULT_POLICY_HZ)
        self.declare_parameter("sample_factor", int(DEFAULT_PROFILE.runtime["sample_factor"]))
        self.declare_parameter("shadow_mode", False)
        self.declare_parameter("tolerance_ms", 50.0)
        self.declare_parameter("server_port", int(DEFAULT_PROFILE.runtime["server_port"]))
        self.declare_parameter("active_model_id", "0")
        self.declare_parameter("state_key", "observation.state")
        self.declare_parameter("joint_topic", str(DEFAULT_PROFILE.runtime["joint_feedback_topic"]))
        runtime_cameras = DEFAULT_PROFILE.runtime["camera_topics"]
        self.declare_parameter(
            "cam_high_left_topic", str(runtime_cameras["observation.images.cam_high_left"])
        )
        self.declare_parameter("cam_high_right_topic", "/camera/right_eye_resize")
        self.declare_parameter(
            "cam_hand_left_topic", str(runtime_cameras["observation.images.cam_hand_left"])
        )
        self.declare_parameter(
            "cam_hand_right_topic", str(runtime_cameras["observation.images.cam_hand_right"])
        )
        self.declare_parameter("head_target_width", 640)
        self.declare_parameter("head_target_height", 360)
        self.declare_parameter("hand_target_width", 640)
        self.declare_parameter("hand_target_height", 360)
        self.declare_parameter(
            "image_keys",
            list(DEFAULT_PROFILE.act["image_keys"]),
            string_array,
        )
        self.declare_parameter("feedback_joint_names", list(BODY_FEEDBACK_JOINTS), string_array)
        self.declare_parameter("hand_input_mode", "scalar")
        self.declare_parameter("hand_sides", ["left", "right"], string_array)
        self.declare_parameter("left_hand_scalar_name", "LEFT_GRIPPER")
        self.declare_parameter("right_hand_scalar_name", "RIGHT_GRIPPER")
        self.declare_parameter("urdf_path", str(DEFAULT_PROFILE.urdf))

        def get_value(name: str) -> Any:
            return self.get_parameter(name).value

        self.policy_hz = float(get_value("policy_hz"))
        sample_factor = get_value("sample_factor")
        if isinstance(sample_factor, bool) or not isinstance(sample_factor, int):
            raise ValueError("sample_factor must be a positive integer")
        self.sample_factor = sample_factor
        if self.policy_hz <= 0.0:
            raise ValueError("policy_hz must be positive")
        if self.sample_factor < 1:
            raise ValueError("sample_factor must be a positive integer")
        self.control_hz = self.policy_hz * self.sample_factor
        self.shadow_mode = bool(get_value("shadow_mode"))
        self.tolerance_s = float(get_value("tolerance_ms")) / 1000.0
        self.active_model_id = str(get_value("active_model_id"))
        self.state_key = str(get_value("state_key"))
        self.image_keys = list(get_value("image_keys"))
        self.body_order = list(BODY_JOINTS)
        self.feedback_joint_names = list(get_value("feedback_joint_names"))
        self.hand_sides = [str(side).lower() for side in get_value("hand_sides")]
        if str(get_value("hand_input_mode")).lower() != "scalar":
            raise ValueError("policy_bridge.py supports scalar gripper state only")
        self.left_scalar_name = str(get_value("left_hand_scalar_name"))
        self.right_scalar_name = str(get_value("right_hand_scalar_name"))
        self.full_order = self.body_order.copy()
        if "left" in self.hand_sides:
            self.full_order.append(self.left_scalar_name)
        if "right" in self.hand_sides:
            self.full_order.append(self.right_scalar_name)
        self.full_dim = len(self.full_order)
        if self.full_dim != 19:
            raise ValueError(
                f"ACT checkpoint contract requires 19 state/action dimensions, got {self.full_dim}"
            )
        self.left_gripper_index = (
            self.full_order.index(self.left_scalar_name) if self.left_scalar_name in self.full_order else None
        )
        self.right_gripper_index = (
            self.full_order.index(self.right_scalar_name)
            if self.right_scalar_name in self.full_order
            else None
        )
        self.gripper_command_state = {
            "left": GRIPPER_MIN,
            "right": GRIPPER_MIN,
        }
        self.last_command_state: np.ndarray | None = None
        self.feedback_state_source = "feedback_bootstrap"

        self.head_target_size = (
            int(get_value("head_target_width")),
            int(get_value("head_target_height")),
        )
        self.hand_target_size = (
            int(get_value("hand_target_width")),
            int(get_value("hand_target_height")),
        )
        self.image_topics = {
            "observation.images.cam_high_left": str(get_value("cam_high_left_topic")),
            "observation.images.cam_high_right": str(get_value("cam_high_right_topic")),
            "observation.images.cam_hand_left": str(get_value("cam_hand_left_topic")),
            "observation.images.cam_hand_right": str(get_value("cam_hand_right_topic")),
        }
        unknown_keys = set(self.image_keys) - set(self.image_topics)
        if unknown_keys:
            raise ValueError(f"unsupported image keys: {sorted(unknown_keys)}")

        urdf_path = str(get_value("urdf_path"))
        try:
            self.joint_limit_map = load_revolute_joint_limits(urdf_path)
        except (FileNotFoundError, ValueError) as exc:
            if not self.shadow_mode:
                raise
            self.get_logger().warning(f"Shadow mode has no URDF limit protection: {exc}")
            self.joint_limit_map = {}
        missing_limits = [name for name in self.body_order if name not in self.joint_limit_map]
        if missing_limits:
            message = f"Missing URDF limits for controlled joints: {missing_limits}"
            if not self.shadow_mode:
                raise ValueError(message)
            self.get_logger().warning(message)
        self.action_lower, self.action_upper, self.action_clip_mask = self._joint_limit_arrays()
        self.act_adapter = W1ActAdapter(
            dict(zip(self.body_order, self.action_lower[: len(self.body_order)], strict=True)),
            dict(zip(self.body_order, self.action_upper[: len(self.body_order)], strict=True)),
            ActHandGestureConfig.from_dict(DEFAULT_PROFILE.hands),
            DEFAULT_PROFILE.body_command_names,
        )

        reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.pub_action = self.create_publisher(
            JointPositionControl, DEFAULT_PROFILE.endpoints.body, reliable
        )
        self.pub_left_hand = self.create_publisher(
            EEJointControl, DEFAULT_PROFILE.endpoints.left_hand, reliable
        )
        self.pub_right_hand = self.create_publisher(
            EEJointControl, DEFAULT_PROFILE.endpoints.right_hand, reliable
        )
        self.command_transport = W1RosTransport(
            W1RosPublishers(
                body=self.pub_action,
                left_hand=self.pub_left_hand,
                right_hand=self.pub_right_hand,
            ),
            W1RosMessageTypes(
                body=JointPositionControl,
                hand=EEJointControl,
                hand_position_mode=EEJointControlMode.POSITION,
            ),
        )

        self.bridge = CvBridge()
        self.buffer_lock = threading.RLock()
        self.image_buffers = {key: deque(maxlen=1) for key in self.image_keys}
        for key in self.image_keys:
            self.create_subscription(
                Image,
                self.image_topics[key],
                lambda message, image_key=key: self._image_callback(image_key, message),
                image_qos,
            )
        self.body_buffer: deque = deque(maxlen=1)
        self.create_subscription(String, str(get_value("joint_topic")), self._body_callback, reliable)

        head_shape = (self.head_target_size[1], self.head_target_size[0], 3)
        hand_shape = (self.hand_target_size[1], self.hand_target_size[0], 3)
        self.ipc = PolicyIPCClient(
            server_port=int(get_value("server_port")),
            full_dim=self.full_dim,
            horizon=CHUNK_SIZE,
            image_keys=self.image_keys,
            state_key=self.state_key,
            head_shape=head_shape,
            hand_shape=hand_shape,
        )
        self.stop_event = threading.Event()
        self.subscriber_lock = threading.Lock()
        self.subscriber_present = False
        self.subscriber_session = 0
        self.subscriber_reset_pending = False
        self.block_count = 0
        self.waiting_for_inputs_logged = False
        self.waiting_for_subscriber_logged = False

        self.subscriber_timer = self.create_timer(0.05, self._monitor_subscriber)
        threading.Thread(target=self._connection_loop, daemon=True).start()
        threading.Thread(target=self._execution_loop, daemon=True).start()
        self.get_logger().info(
            f"Blocking Policy Bridge ready: dim={self.full_dim} policy={self.policy_hz:.1f}Hz "
            f"control={self.control_hz:.1f}Hz "
            f"sample_factor={self.sample_factor} "
            f"chunk={CHUNK_SIZE * self.sample_factor}control_points "
            f"shadow={self.shadow_mode}"
        )

    def _joint_limit_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lower = np.full(self.full_dim, -np.inf, dtype=np.float32)
        upper = np.full(self.full_dim, np.inf, dtype=np.float32)
        mask = np.zeros(self.full_dim, dtype=bool)
        for index, name in enumerate(self.full_order):
            if name in self.joint_limit_map:
                lower[index], upper[index] = self.joint_limit_map[name]
                mask[index] = True
        return lower, upper, mask

    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        result = np.asarray(action, dtype=np.float32).copy()
        result[self.action_clip_mask] = np.clip(
            result[self.action_clip_mask],
            self.action_lower[self.action_clip_mask],
            self.action_upper[self.action_clip_mask],
        )
        return result

    def _connection_loop(self) -> None:
        while not self.stop_event.is_set() and rclpy.ok():
            if self.ipc.connected:
                self.stop_event.wait(0.5)
                continue
            try:
                self.ipc.connect(self.active_model_id)
                self.get_logger().info(f"Model server connected: model={self.active_model_id}")
            except Exception as exc:
                self.get_logger().warning(f"Waiting for model server: {exc}")
                self.stop_event.wait(1.0)

    def _image_callback(self, key: str, message: Any) -> None:
        received_at = time.monotonic()
        try:
            image = self.bridge.imgmsg_to_cv2(message, "bgr8")
            with self.buffer_lock:
                self.image_buffers[key].append(BufferedSample(message_timestamp(message), received_at, image))
        except Exception as exc:
            self.get_logger().warning(f"Image decode failed for {key}: {exc}")

    def _body_callback(self, message: Any) -> None:
        wall_received_at = now_sec()
        monotonic_received_at = time.monotonic()
        try:
            payload = json.loads(message.data)
            positions = payload.get("joint_position")
            if positions is None:
                return
            if isinstance(positions, dict):
                values = {str(name): float(value) for name, value in positions.items()}
            else:
                names = payload.get("joint_names") or payload.get("name") or self.feedback_joint_names
                if len(names) != len(positions):
                    raise ValueError("joint feedback names and positions have different lengths")
                values = {str(name): float(value) for name, value in zip(names, positions, strict=True)}
            if not all(np.isfinite(value) for value in values.values()):
                raise ValueError("joint feedback contains non-finite values")
            timestamp = normalize_timestamp(payload.get("timestamp"), wall_received_at)
            with self.buffer_lock:
                self.body_buffer.append(BufferedSample(timestamp, monotonic_received_at, values))
        except Exception as exc:
            self.get_logger().warning(f"Joint feedback decode failed: {exc}")

    def _latest_snapshot(self) -> ObservationSnapshot | None:
        with self.buffer_lock:
            if (
                not self.image_keys
                or not self.body_buffer
                or any(not self.image_buffers[key] for key in self.image_keys)
            ):
                return None
            samples = [self.image_buffers[key][-1] for key in self.image_keys]
            body_sample = self.body_buffer[-1]
            received_at = [sample.received_at for sample in samples]
            received_at.append(body_sample.received_at)
            current_time = time.monotonic()
            if any(not 0.0 <= current_time - timestamp <= MAX_OBSERVATION_AGE_S for timestamp in received_at):
                return None
            if max(received_at) - min(received_at) > self.tolerance_s:
                return None
            images = {
                key: np.asarray(sample.value).copy()
                for key, sample in zip(self.image_keys, samples, strict=True)
            }
            body = body_sample.value
            if body is None or any(name not in body for name in self.body_order):
                return None
            if self.last_command_state is None:
                state_values = [float(body[name]) for name in self.body_order]
                if "left" in self.hand_sides:
                    state_values.append(self.gripper_command_state["left"])
                if "right" in self.hand_sides:
                    state_values.append(self.gripper_command_state["right"])
                state = np.asarray(state_values, dtype=np.float32)
                state_source = self.feedback_state_source
            else:
                state = self.last_command_state.copy()
                state_source = "last_command"
            if state.shape != (self.full_dim,) or not np.isfinite(state).all():
                return None
            anchor_key = (
                "observation.images.cam_high_left"
                if "observation.images.cam_high_left" in self.image_keys
                else self.image_keys[0]
            )
            anchor_index = self.image_keys.index(anchor_key)
            return ObservationSnapshot(
                float(samples[anchor_index].sample_time),
                images,
                state,
                state_source,
            )

    def _build_observation(self, snapshot: ObservationSnapshot) -> dict[str, np.ndarray]:
        observation: dict[str, np.ndarray] = {}
        for key, image in snapshot.images.items():
            size = self.hand_target_size if "hand" in key else self.head_target_size
            observation[key] = prepare_resized(image, size).copy()
        observation[self.state_key] = snapshot.state.copy()
        return observation

    def _publish_action(self, action: np.ndarray) -> None:
        values = self._clip_action(action)
        gripper_updates: dict[str, float] = {}
        if self.left_gripper_index is not None:
            left_scalar = float(np.clip(values[self.left_gripper_index], GRIPPER_MIN, GRIPPER_MAX))
            values[self.left_gripper_index] = left_scalar
            gripper_updates["left"] = left_scalar
        if self.right_gripper_index is not None:
            right_scalar = float(np.clip(values[self.right_gripper_index], GRIPPER_MIN, GRIPPER_MAX))
            values[self.right_gripper_index] = right_scalar
            gripper_updates["right"] = right_scalar
        command = self.act_adapter.action_to_command(values)
        if not self.shadow_mode:
            self.command_transport.publish(command, self.get_clock().now().to_msg())
        self.gripper_command_state.update(gripper_updates)
        self.last_command_state = values.copy()

    def _has_subscriber(self) -> bool:
        return self.shadow_mode or self.pub_action.get_subscription_count() > 0

    def _subscriber_state(self) -> tuple[bool, int]:
        present = self._has_subscriber()
        lost = False
        with self.subscriber_lock:
            if present != self.subscriber_present:
                self.subscriber_present = present
                if not present:
                    self.subscriber_session += 1
                    self.subscriber_reset_pending = True
                    lost = True
            session = self.subscriber_session
        if lost:
            self.get_logger().warning(f"Action subscriber lost: session={session}; current block invalidated")
        return present, session

    def _monitor_subscriber(self) -> None:
        self._subscriber_state()

    def _consume_subscriber_reset(self) -> bool:
        with self.subscriber_lock:
            if not self.subscriber_reset_pending:
                return False
            self.subscriber_reset_pending = False
            session = self.subscriber_session
        self.last_command_state = None
        self.feedback_state_source = "feedback_resubscribe"
        self.get_logger().info(
            f"Command state cleared for subscriber session={session}; "
            "next inference will use real body feedback"
        )
        return True

    def _execution_loop(self) -> None:
        raise NotImplementedError("A W1 policy bridge must provide its scheduling loop")

    def close(self) -> None:
        self.stop_event.set()
        self.ipc.close()
