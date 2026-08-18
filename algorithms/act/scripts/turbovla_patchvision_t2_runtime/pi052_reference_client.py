#!/usr/bin/env python3
"""Isolated bimanual ACT deployment client for the 0806swap checkpoint.

The script intentionally reuses LeRobot's asynchronous gRPC client and only
replaces the hardware factory.  It leaves the existing single-arm runtime
untouched.  The model observes 20D state and emits a 14D joint action.
"""

from __future__ import annotations

import argparse
import signal
import threading
import time
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable

import cv2
import numpy as np

LEFT_ACTION_NAMES = tuple(f"left_joint_{index}" for index in range(6)) + ("left_gripper",)
RIGHT_ACTION_NAMES = tuple(f"right_joint_{index}" for index in range(6)) + ("right_gripper",)
ACTION_NAMES = LEFT_ACTION_NAMES + RIGHT_ACTION_NAMES
LEFT_STATE_NAMES = tuple(f"left_joint_{index}" for index in range(6)) + (
    "left_endpoint_x", "left_endpoint_y", "left_endpoint_z", "left_gripper"
)
RIGHT_STATE_NAMES = tuple(f"right_joint_{index}" for index in range(6)) + (
    "right_endpoint_x", "right_endpoint_y", "right_endpoint_z", "right_gripper"
)
STATE_NAMES = LEFT_STATE_NAMES + RIGHT_STATE_NAMES
DEG_MILLI_PER_RAD = 1000.0 * 180.0 / np.pi
METERS_PER_PIPER_ENDPOSE_UNIT = 1e-6
RELATIVE_JOINT_PREVIOUS_STATE_KEY = "__relative_joint_previous_state"
RELATIVE_JOINT_PREVIOUS_SEQUENCE_KEY = "__relative_joint_previous_sequence"
RELATIVE_JOINT_CURRENT_SEQUENCE_KEY = "__relative_joint_current_sequence"
RELATIVE_JOINT_PREVIOUS_TIMESTAMP_KEY = "__relative_joint_previous_timestamp"
RELATIVE_JOINT_CURRENT_TIMESTAMP_KEY = "__relative_joint_current_timestamp"
DEFAULT_LEFT_PIKA_PORT = (
    "/dev/serial/by-path/pci-0000:c4:00.3-usb-0:3.4:1.0-port0"
)
DEFAULT_RIGHT_PIKA_PORT = (
    "/dev/serial/by-path/pci-0000:c6:00.4-usb-0:1.4:1.0-port0"
)


@dataclass(frozen=True)
class ActualStateSample:
    sequence: int
    timestamp: float
    values: np.ndarray


class ActualStateSampler:
    """Continuously retain the two newest actual 20D hardware feedback samples."""

    def __init__(self, reader: Callable[[], np.ndarray], fps: float = 30.0) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.reader = reader
        self.period_s = 1.0 / float(fps)
        self._samples: deque[ActualStateSample] = deque(maxlen=2)
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._sequence = 0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        with self._condition:
            self._samples.clear()
            self._error = None
            self._sequence = 0
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="dual-act-actual-state",
            daemon=True,
        )
        self._thread.start()

    def _loop(self) -> None:
        deadline = time.perf_counter()
        while not self._stop.is_set():
            try:
                values = np.asarray(self.reader(), dtype=np.float64)
                if values.shape != (20,):
                    raise ValueError(f"Actual state must contain exactly 20 values, got {values.shape}")
                if not np.isfinite(values).all():
                    raise ValueError("Actual state must contain only finite values")
                sample = ActualStateSample(
                    sequence=self._sequence,
                    timestamp=time.time(),
                    values=values.copy(),
                )
            except BaseException as error:
                with self._condition:
                    self._error = error
                    self._condition.notify_all()
                return

            with self._condition:
                self._samples.append(sample)
                self._sequence += 1
                self._condition.notify_all()

            deadline += self.period_s
            now = time.perf_counter()
            if deadline < now - self.period_s:
                deadline = now
            self._stop.wait(max(0.0, deadline - now))

    def snapshot(self, timeout_s: float = 1.0) -> tuple[ActualStateSample, ActualStateSample]:
        deadline = time.perf_counter() + timeout_s
        with self._condition:
            while len(self._samples) < 2:
                if self._error is not None:
                    raise self._error
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for two actual state samples")
                self._condition.wait(remaining)

            previous, current = self._samples
            return (
                ActualStateSample(previous.sequence, previous.timestamp, previous.values.copy()),
                ActualStateSample(current.sequence, current.timestamp, current.values.copy()),
            )

    def stop(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None


def next_chunk_observation_timestep(latest_action: int) -> int:
    """Return the first identifier of the next non-overlapping action chunk."""
    return max(int(latest_action) + 1, 0)


def build_full_chunk_client_class(base_class: type) -> type:
    """Build a RobotClient variant that requests one complete chunk at a time."""

    class FullChunkRobotClient(base_class):
        def _single_flight_observation_requests(self) -> bool:
            return self.config.observation_request_policy == "single_flight"

        def _ready_to_send_observation(self) -> bool:
            if self._single_flight_observation_requests():
                if self._observation_worker_busy.is_set():
                    return False
                request_queue = self._observation_request_queue
                if request_queue is not None and not request_queue.empty():
                    return False
            return super()._ready_to_send_observation()

        def _make_observation_request(self, task: str, verbose: bool):
            request = super()._make_observation_request(task, verbose)
            return replace(
                request,
                latest_action=next_chunk_observation_timestep(request.latest_action),
            )

    FullChunkRobotClient.__name__ = "FullChunkRobotClient"
    return FullChunkRobotClient


class D405RGB:
    """Latest-frame RGB reader for one D405, using the device serial number."""

    def __init__(self, serial: str) -> None:
        self.serial = serial
        self._pipeline: Any | None = None

    def connect(self) -> None:
        import pyrealsense2 as rs

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        pipeline.start(config)
        self._pipeline = pipeline
        for _ in range(10):
            self.read()

    def read(self) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError(f"D405 {self.serial} is not connected")
        frame = self._pipeline.wait_for_frames(1000).get_color_frame()
        if not frame:
            raise RuntimeError(f"D405 {self.serial} returned no color frame")
        image = np.asanyarray(frame.get_data())
        if image.shape != (480, 640, 3) or image.dtype != np.uint8:
            raise RuntimeError(f"D405 {self.serial} returned {image.shape} {image.dtype}")
        return image

    def disconnect(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None


def process_top_stereo_frame(frame: np.ndarray) -> np.ndarray:
    """Extract the model's right eye and convert OpenCV BGR to RGB."""
    if frame.shape != (1080, 3840, 3) or frame.dtype != np.uint8:
        raise ValueError(f"Top stereo frame must be 1080x3840 uint8 BGR, got {frame.shape} {frame.dtype}")
    right_bgr = cv2.resize(
        frame[:, 1920:3840],
        (720, 405),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)


def process_raw_decoded_bgr_frame(frame: np.ndarray) -> np.ndarray:
    """Apply the established model preprocessing to one raw decoded BGR frame."""
    return process_top_stereo_frame(frame)


class TopStereoRightRaw:
    """Decode the NVCAM raw YUY2 stream and cache the model's right RGB eye."""

    libav_options = {
        "input_format": "yuyv422",
        "video_size": "3840x1080",
        "framerate": "60",
    }

    def __init__(self, device: str, fps: float = 30.0) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.device = device
        self.period_s = 1.0 / float(fps)
        self._container: Any | None = None
        self._latest_right: np.ndarray | None = None
        self._latest_sequence = -1
        self._latest_timestamp = 0.0
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def connect(self) -> None:
        import av

        self._stop.clear()
        self._error = None
        self._container = av.open(
            self.device,
            format="v4l2",
            mode="r",
            options=self.libav_options,
        )
        self._thread = threading.Thread(target=self._loop, name="dual-act-top-raw", daemon=True)
        self._thread.start()
        self.read_right(timeout_s=3.0)

    def _loop(self) -> None:
        assert self._container is not None
        deadline = time.perf_counter()
        try:
            stream = self._container.streams.video[0]
            for decoded_frame in self._container.decode(stream):
                if self._stop.is_set():
                    return
                now = time.perf_counter()
                if now < deadline:
                    continue
                deadline = max(deadline + self.period_s, now)
                bgr_frame = decoded_frame.to_ndarray(format="bgr24")
                right_rgb = process_raw_decoded_bgr_frame(bgr_frame)
                self._publish_right_frame(right_rgb, timestamp=now)
        except BaseException as error:
            if not self._stop.is_set():
                self._error = error

    def _publish_right_frame(self, image: np.ndarray, *, timestamp: float | None = None) -> None:
        timestamp = time.monotonic() if timestamp is None else float(timestamp)
        with self._condition:
            self._latest_right = image.copy()
            self._latest_sequence += 1
            self._latest_timestamp = timestamp
            self._condition.notify_all()

    def read_right_after(
        self,
        sequence: int,
        timeout_s: float = 1.0,
    ) -> tuple[int, float, np.ndarray]:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while self._latest_sequence <= sequence:
                if self._error is not None:
                    raise RuntimeError("Raw top camera failed") from self._error
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("No new raw top-right RGB frame available")
                self._condition.wait(remaining)
            assert self._latest_right is not None
            return self._latest_sequence, self._latest_timestamp, self._latest_right.copy()

    def read_right(self, timeout_s: float = 0.2) -> np.ndarray:
        return self.read_right_after(-1, timeout_s=timeout_s)[2]

    def disconnect(self) -> None:
        self._stop.set()
        if self._container is not None:
            self._container.close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        self._container = None


class TopStereoRightRGB:
    """Read the combined MJPG stereo feed and cache the model's right RGB eye.

    ROS Humble on the robot host is built for Python 3.10 while the LeRobot
    client requires Python 3.12.  The inference process therefore reads the
    camera directly.  A separate Python 3.10 ROS bridge may publish both eyes
    for inspection, but does not sit on the inference path.
    """

    def __init__(self, device: str, fps: float = 30.0) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.device = device
        self.fps = fps
        self.period_s = 1.0 / float(fps)
        self._capture: cv2.VideoCapture | None = None
        self._latest_right: np.ndarray | None = None
        self._latest_sequence = -1
        self._latest_timestamp = 0.0
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def connect(self) -> None:
        self._stop.clear()
        capture = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open top stereo camera {self.device}")
        self._capture = capture
        self._thread = threading.Thread(target=self._loop, name="dual-act-top-camera", daemon=True)
        self._thread.start()
        self.read_right(timeout_s=3.0)

    def _loop(self) -> None:
        assert self._capture is not None
        deadline = time.perf_counter()
        while not self._stop.is_set():
            ok, frame = self._capture.read()
            if ok and frame is not None and frame.shape == (1080, 3840, 3):
                right_rgb = process_top_stereo_frame(frame)
                self._publish_right_frame(right_rgb)

            deadline += self.period_s
            now = time.perf_counter()
            if deadline < now - self.period_s:
                deadline = now
            self._stop.wait(max(0.0, deadline - now))

    def _publish_right_frame(self, image: np.ndarray, *, timestamp: float | None = None) -> None:
        timestamp = time.monotonic() if timestamp is None else float(timestamp)
        with self._condition:
            self._latest_right = image.copy()
            self._latest_sequence += 1
            self._latest_timestamp = timestamp
            self._condition.notify_all()

    def read_right_after(
        self,
        sequence: int,
        timeout_s: float = 1.0,
    ) -> tuple[int, float, np.ndarray]:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while self._latest_sequence <= sequence:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("No new top-right RGB frame available")
                self._condition.wait(remaining)
            assert self._latest_right is not None
            return self._latest_sequence, self._latest_timestamp, self._latest_right.copy()

    def read_right(self, timeout_s: float = 0.2) -> np.ndarray:
        return self.read_right_after(-1, timeout_s=timeout_s)[2]

    def disconnect(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._capture is not None:
            self._capture.release()
        self._thread = None
        self._capture = None


class PikaPiperGripper:
    def __init__(self, side: str, port: str, *, enable_on_connect: bool, max_width_m: float) -> None:
        self.side = side
        self.port = port
        self.enable_on_connect = enable_on_connect
        self.max_width_m = max_width_m
        self.device: Any | None = None

    def connect(self) -> None:
        from pika.gripper import Gripper

        self.device = Gripper(self.port)
        if not self.device.connect():
            self.device = None
            raise RuntimeError(f"Cannot connect {self.side} PikaPiper on {self.port}")
        if self.enable_on_connect and not self.device.enable():
            self.disconnect()
            raise RuntimeError(f"Cannot enable {self.side} PikaPiper on {self.port}")

    def read_width(self) -> float:
        if self.device is None:
            raise RuntimeError(f"{self.side} PikaPiper is not connected")
        return max(float(self.device.get_gripper_distance()) / 1000.0, 0.0)

    def execute_width(self, width_m: float) -> float:
        if self.device is None:
            raise RuntimeError(f"{self.side} PikaPiper is not connected")
        width_m = min(max(float(width_m), 0.0), self.max_width_m)
        if not self.device.set_gripper_distance(width_m * 1000.0):
            raise RuntimeError(f"{self.side} PikaPiper rejected width {width_m:.5f} m")
        return width_m

    def disconnect(self) -> None:
        if self.device is not None:
            self.device.disconnect()
            self.device = None


class DualActHardware:
    """Robot client hardware for 20D endpoint state and 14D joint action."""

    observation_features = OrderedDict(
        [(name, float) for name in STATE_NAMES]
        + [
            ("top", (405, 720, 3)),
            ("gripper_right", (480, 640, 3)),
            ("gripper_left", (480, 640, 3)),
        ]
    )
    action_features = OrderedDict((name, float) for name in ACTION_NAMES)

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.left_arm: Any | None = None
        self.right_arm: Any | None = None
        self.left_gripper = PikaPiperGripper(
            "left", args.left_pika_port, enable_on_connect=args.enable_grippers, max_width_m=args.gripper_max_m
        )
        self.right_gripper = PikaPiperGripper(
            "right", args.right_pika_port, enable_on_connect=args.enable_grippers, max_width_m=args.gripper_max_m
        )
        self.left_camera = D405RGB(args.left_d405_serial)
        self.right_camera = D405RGB(args.right_d405_serial)
        camera_class = TopStereoRightRaw if args.top_codec == "raw" else TopStereoRightRGB
        self.top_camera = camera_class(args.top_device, fps=args.fps)
        self.state_sampler = ActualStateSampler(self._read_actual_state, fps=args.fps)
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        from piper_sdk import C_PiperInterface_V2

        self.left_arm = C_PiperInterface_V2(self.args.left_can)
        self.right_arm = C_PiperInterface_V2(self.args.right_can)
        self.left_arm.ConnectPort()
        self.right_arm.ConnectPort()
        if self.args.enable_arms:
            for side, arm in (("left", self.left_arm), ("right", self.right_arm)):
                deadline = time.perf_counter() + 10.0
                while not arm.EnablePiper():
                    if time.perf_counter() > deadline:
                        raise TimeoutError(f"Timed out enabling {side} Piper")
                    time.sleep(0.01)
        self.left_gripper.connect()
        self.right_gripper.connect()
        self.left_camera.connect()
        self.right_camera.connect()
        self.top_camera.connect()
        self.state_sampler.start()
        self.state_sampler.snapshot(timeout_s=3.0)
        self._connected = True

    @staticmethod
    def _read_arm_state(arm: Any, gripper: PikaPiperGripper) -> list[float]:
        joints = arm.GetArmJointMsgs().joint_state
        values = [float(getattr(joints, f"joint_{index}")) / DEG_MILLI_PER_RAD for index in range(1, 7)]
        pose = arm.GetArmEndPoseMsgs().end_pose
        values.extend(
            float(getattr(pose, axis)) * METERS_PER_PIPER_ENDPOSE_UNIT
            for axis in ("X_axis", "Y_axis", "Z_axis")
        )
        values.append(gripper.read_width())
        return values

    def _read_actual_state(self) -> np.ndarray:
        if self.left_arm is None or self.right_arm is None:
            raise RuntimeError("Dual ACT arm feedback is not connected")
        state = self._read_arm_state(self.left_arm, self.left_gripper)
        state.extend(self._read_arm_state(self.right_arm, self.right_gripper))
        return np.asarray(state, dtype=np.float64)

    def get_state_observation(self) -> dict[str, Any]:
        if not self._connected or self.left_arm is None or self.right_arm is None:
            raise RuntimeError("Dual ACT hardware is not connected")
        previous, current = self.state_sampler.snapshot()
        observation: dict[str, Any] = {
            name: float(current.values[index]) for index, name in enumerate(STATE_NAMES)
        }
        observation[RELATIVE_JOINT_PREVIOUS_STATE_KEY] = previous.values.astype(
            np.float32,
            copy=True,
        )
        observation[RELATIVE_JOINT_PREVIOUS_SEQUENCE_KEY] = previous.sequence
        observation[RELATIVE_JOINT_CURRENT_SEQUENCE_KEY] = current.sequence
        observation[RELATIVE_JOINT_PREVIOUS_TIMESTAMP_KEY] = previous.timestamp
        observation[RELATIVE_JOINT_CURRENT_TIMESTAMP_KEY] = current.timestamp
        return observation

    def get_observation(self) -> dict[str, Any]:
        observation = self.get_state_observation()
        observation["top"] = self.top_camera.read_right()
        observation["gripper_right"] = self.right_camera.read()
        observation["gripper_left"] = self.left_camera.read()
        return observation

    @staticmethod
    def _send_arm(arm: Any, values: list[float]) -> None:
        joints = [int(round(value * DEG_MILLI_PER_RAD)) for value in values]
        arm.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        arm.JointCtrl(*joints)

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        if not self.args.enable_arms or not self.args.enable_grippers:
            raise RuntimeError("Motion is disabled. Re-run with --enable-arms --enable-grippers only after dry-run.")
        expected = set(ACTION_NAMES)
        if set(action) != expected:
            missing = sorted(expected - set(action))
            extra = sorted(set(action) - expected)
            raise ValueError(f"Expected exact 14D action names; missing={missing}, extra={extra}")
        assert self.left_arm is not None and self.right_arm is not None
        self._send_arm(self.left_arm, [action[name] for name in LEFT_ACTION_NAMES[:6]])
        self._send_arm(self.right_arm, [action[name] for name in RIGHT_ACTION_NAMES[:6]])
        left_width = self.left_gripper.execute_width(action["left_gripper"])
        right_width = self.right_gripper.execute_width(action["right_gripper"])
        performed = dict(action)
        performed["left_gripper"] = left_width
        performed["right_gripper"] = right_width
        return performed

    def disconnect(self) -> None:
        self.state_sampler.stop()
        self.top_camera.disconnect()
        self.left_camera.disconnect()
        self.right_camera.disconnect()
        self.left_gripper.disconnect()
        self.right_gripper.disconnect()
        if self.args.enable_arms:
            for arm in (self.left_arm, self.right_arm):
                if arm is not None:
                    arm.DisablePiper()
        self.left_arm = None
        self.right_arm = None
        self._connected = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-address", default="127.0.0.1:18187")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--policy-type", default="act")
    parser.add_argument("--actions-per-chunk", type=int, default=16)
    parser.add_argument("--task", default="grasp bread")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--left-can", default="left_piper")
    parser.add_argument("--right-can", default="right_piper")
    parser.add_argument("--left-d405-serial", default="412622273326")
    parser.add_argument("--right-d405-serial", default="260622271788")
    parser.add_argument("--top-device", default="/dev/video26")
    parser.add_argument("--top-codec", choices=("raw", "opencv"), default="raw")
    parser.add_argument("--left-pika-port", default=DEFAULT_LEFT_PIKA_PORT)
    parser.add_argument("--right-pika-port", default=DEFAULT_RIGHT_PIKA_PORT)
    parser.add_argument("--gripper-max-m", type=float, default=0.09)
    parser.add_argument("--enable-arms", action="store_true")
    parser.add_argument("--enable-grippers", action="store_true")
    parser.add_argument("--execute-robot-actions", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.actions_per_chunk <= 0:
        raise SystemExit("--actions-per-chunk must be positive")
    if args.execute_robot_actions and not (args.enable_arms and args.enable_grippers):
        raise SystemExit("--execute-robot-actions requires --enable-arms and --enable-grippers")
    hardware = DualActHardware(args)
    try:
        if args.self_test:
            hardware.connect()
            observation = hardware.get_observation()
            print("self_test_state", [round(float(observation[name]), 6) for name in STATE_NAMES])
            print("self_test_images", {key: list(observation[key].shape) for key in ("top", "gripper_right", "gripper_left")})
            return

        from lerobot.async_inference import robot_client as robot_client_module
        from lerobot.async_inference.configs import RobotClientConfig
        from lerobot.robots.piper_follower import PiperFollowerConfig

        robot_client_module.make_robot_from_config = lambda _config: hardware
        config = RobotClientConfig(
            policy_type=args.policy_type,
            pretrained_name_or_path=args.checkpoint,
            robot=PiperFollowerConfig(can_name=args.right_can, enable_on_connect=False, cameras={}),
            actions_per_chunk=args.actions_per_chunk,
            task=args.task,
            server_address=args.server_address,
            policy_device="cuda",
            client_device="cpu",
            fps=args.fps,
            chunk_size_threshold=0.0,
            aggregate_fn_name="latest_only",
            execute_robot_actions=args.execute_robot_actions,
            async_observation=True,
            observation_request_policy="single_flight",
            use_pika_gripper=False,
        )
        client_class = build_full_chunk_client_class(robot_client_module.RobotClient)
        client = client_class(config)
        if not client.start():
            client.stop()
            raise SystemExit("Cannot connect to policy server")
        receiver = threading.Thread(target=client.receive_actions, daemon=True)
        receiver.start()

        def stop(_signum: int, _frame: Any) -> None:
            client.shutdown_event.set()

        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        try:
            client.control_loop(task=args.task)
        finally:
            client.stop()
            receiver.join(timeout=2.0)
    finally:
        hardware.disconnect()


if __name__ == "__main__":
    main()
