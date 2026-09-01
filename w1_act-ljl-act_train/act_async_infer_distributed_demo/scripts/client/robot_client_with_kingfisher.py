import threading
import time
import numpy as np
import cv2
import os
import socket
import json

from collections import deque, OrderedDict
from queue import Queue, Empty
from copy import deepcopy
from typing import Callable, Optional
import rclpy
import traceback

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from end_effector_interfaces.msg import EEJointFeedback, EEJointControl, EEJointControlMode
from std_msgs.msg import String
from joint_interfaces.msg import JointPositionControl
from array import array
from act_async_infer_distributed_demo.scripts.action_lipo import ActionLiPo
from act_async_infer_distributed_demo.scripts.utils_distributed import (
    TimedObservation,
    TimedAction,
    log_info,
    log_error,
    log_warning,
    log_debug,
    setup_logger,
    now_sec,
    nearest_without_tol,
    draw_actionchunks,
    _save_frames_to_file,
    interpolate_2d,
)
from act_async_infer_distributed_demo.scripts.network_utils import (
    NetworkClient,
)
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from act_async_infer_distributed_demo.scripts.w1_mapping import (
    CommonKey,
    ImageKey,
    InferenceConfigKey,
    JointNamesKey,
    w1qpos_names_map,
    origin_joint_names,
    JointLimit,
    TrajectoryKeys,
)
from act_async_infer_distributed_demo.scripts.inference_config import (
    ClientConfig,
    RequestType,
    ResponseKey,
    ManagerKey,
    
)
from enum import Enum



DEFAULT_HANDBUF_CLEAN_LEN = 100
DEAFULT_SIM_TORSO_SPEED_RATIO = 1.0
DEAFULT_SIM_HEAD_SPEED_RATIO = 1.0
DEFAULT_SIM_MOVE_SPEED = 1.0
DEFAULT_GET_ACTION_FREQUENCY = 15.0
GRIPPER_PUBLISH_TIMES = 10
ACTIVATE_TIME = 50.0
WAIT_FOR_JOINT_DATA = 5.0

class RobotState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    ERROR = "error"
    
    
class OptimizedRobotClient(Node):
    _logger_initialized = False

    def __init__(self, client_config: ClientConfig):
        super().__init__("optimized_robot_client")
        self.cfg = client_config

        self.tracer = None
        self.state = RobotState.IDLE
        self.error_message: Optional[str] = None
        self.state_lock = threading.Lock()
        self.post_opt = None
        self.inference_delay = None
        self.start_inference_time = None

        # QoS
        q_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=30,
            durability=DurabilityPolicy.VOLATILE,
        )

        q_img = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=30,
            durability=DurabilityPolicy.VOLATILE,
        )


        self.create_subscription(
            Image, self.cfg.cam_head_left_topic, self.cb_img_head_left, q_img
        )
        self.get_logger().info(f"Subscribe head-left image: {self.cfg.cam_head_left_topic}")

        self.create_subscription(
            Image, self.cfg.cam_head_right_topic, self.cb_img_head_right, q_img
        )
        self.get_logger().info(f"Subscribe head-right image: {self.cfg.cam_head_right_topic}")

        self.create_subscription(
            Image, self.cfg.cam_hand_left_topic, self.cb_img_hand_left, q_img
        )
        self.get_logger().info(f"Subscribe hand-left image: {self.cfg.cam_hand_left_topic}")

        self.create_subscription(
            Image, self.cfg.cam_hand_right_topic, self.cb_img_hand_right, q_img
        )
        self.get_logger().info(
            f"Subscribe hand-right image: {self.cfg.cam_hand_right_topic}"
        )

        self.create_subscription(
            String, self.cfg.joint_topic, self.joint_state_callback, q_reliable
        )
        self.create_subscription(
            EEJointFeedback,
            self.cfg.left_hand_qpos6_topic,
            self.hand_qpos6_left_callback,
            q_reliable,
        )
        self.create_subscription(
            EEJointFeedback,
            self.cfg.right_hand_qpos6_topic,
            self.hand_qpos6_right_callback,
            q_reliable,
        )
        self.create_subscription(
            EEJointFeedback,
            self.cfg.left_gripper_qpos_topic,
            self.gripper_qpos_left_callback,
            q_reliable,
        )
        self.create_subscription(
            EEJointFeedback,
            self.cfg.right_gripper_qpos_topic,
            self.gripper_qpos_right_callback,
            q_reliable,
        )
        
        # publisher
        self._pub_joint = None
        self._pub_hand_left = None
        self._pub_hand_right = None
        self._pub_gripper_left = None
        self._pub_gripper_right = None
        self._hand_joint_names = TrajectoryKeys.HAND_JOINT_NAMES
        self._gripper_left_names = TrajectoryKeys.PIPER_LEFT_JOINT
        self._gripper_right_names = TrajectoryKeys.PIPER_RIGHT_JOINT
        self._joint_limits = JointLimit

        

        # 网络客户端
        self.network_client = NetworkClient(self.cfg.server_host, self.cfg.server_port)

        # 动作队列系统
        self.aggregate_fn = None
        self.allow_infer_check = True
        self.connection_points = []

        self.action_queue = Queue()
        self.action_queue_lock = threading.Lock()
        self.action_queue_size = []
        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.latest_action_timestamp = -1
        self.action_chunk_size = 0
        self.get_action_times = deque(maxlen=50)
        self.actions = OrderedDict()
        self.actionchunks = []

        self.get_action_frequency = DEFAULT_GET_ACTION_FREQUENCY
        self.environment_dt = 0.0
        self.time_delay = 0
        self.blending_horizon = 0

        self.current_step = 0
        self.must_go = threading.Event()
        self.must_go.set()
        self.shutdown_event = threading.Event()
        self._shutdown_event = threading.Event()
        self._inference_running = False
        self.start_infer = False
        self.start_barrier = threading.Barrier(5)
        self.observation_queue = Queue(maxsize=1)
        self.observation_sending_thread = None


        self.joint_state_buf = []
        self.joint_state_lock = threading.Lock()
        self.body_index_map: Optional[np.ndarray] = None

        self.gripper_qpos_left_buf = deque(maxlen=2000)
        self.gripper_qpos_right_buf = deque(maxlen=2000)
        self.head_left_buf: deque = deque(maxlen=200)
        self.head_right_buf: deque = deque(maxlen=200)

        self.hand_qpos6_left_buf = deque(maxlen=2000)
        self.hand_qpos6_right_buf = deque(maxlen=2000)
        self.hand_left_buf: deque = deque(maxlen=200)
        self.hand_right_buf: deque = deque(maxlen=200)
        self.bridge = CvBridge()
        self.head_lock = threading.Lock()
        self.hand_qpos6_right_lock = threading.Lock()
        self.hand_qpos6_left_lock = threading.Lock()
        self.hand_left_lock = threading.Lock()
        self.hand_right_lock = threading.Lock()
        self.body_buf_lock = threading.Lock()

        self.action_times = deque(maxlen=100)
        self.observation_times = deque(maxlen=100)
        self.observation_check_times = deque(maxlen=100)

        self.observation_check_frequency = 10
        self.first_get_actions = True
        self.inference_delay_zone = []
        self.blend_zone = []

        self.socket_lock = threading.RLock()

        self._mgr_conn = None
        self._mgr_sock = None
        self.joint_names_from_topic = None

    def joint_state_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            return

        joint_position = data.get("joint_position")
        if not joint_position:
            return

        joint_qpos = np.asarray(joint_position, dtype=np.float32)

        if self.joint_names_from_topic is None:
            self.joint_names_from_topic = origin_joint_names

        with self.joint_state_lock:
            self.joint_state_buf = joint_qpos.tolist()
            self.joint_state_time = time.time()


    def cb_img_head_left(self, msg: Image) -> None:
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            current_time = time.time()
        except Exception as e:
            self.get_logger().warn(f"cv_bridge left-head fail: {e}")
            return
        bgr = cv2.resize(
            bgr, (self.cfg.head_target_size[0], self.cfg.head_target_size[1]),
            interpolation=cv2.INTER_AREA
        )
        t = now_sec()
        self.head_left_buf.append((t, bgr))
        if len(self.head_left_buf) % DEFAULT_HANDBUF_CLEAN_LEN == 0:
            self.clean_old_data(self.head_left_buf, current_time, max_age_seconds=5.0)

    def cb_img_head_right(self, msg: Image) -> None:
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            current_time = time.time()
        except Exception as e:
            self.get_logger().warn(f"cv_bridge right-head fail: {e}")
            return
        bgr = cv2.resize(
            bgr, (self.cfg.head_target_size[0], self.cfg.head_target_size[1]),
            interpolation=cv2.INTER_AREA
        )
        t = now_sec()
        self.head_right_buf.append((t, bgr))
        if len(self.head_right_buf) % DEFAULT_HANDBUF_CLEAN_LEN == 0:
            self.clean_old_data(self.head_right_buf, current_time, max_age_seconds=5.0)

    def cb_img_hand_left(self, msg: Image) -> None:
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            current_time = time.time()
        except Exception as e:
            self.get_logger().warn(f"cv_bridge left-hand fail: {e}")
            return
        size = getattr(self.cfg, "hand_left_target_size", self.cfg.hand_target_size)
        bgr = cv2.resize(
            bgr, (size[0], size[1]),
            interpolation=cv2.INTER_AREA
        )
        t = now_sec()
        self.hand_left_buf.append((t, bgr))
        if len(self.hand_left_buf) % DEFAULT_HANDBUF_CLEAN_LEN == 0:
            self.clean_old_data(self.hand_left_buf, current_time, max_age_seconds=5.0)

    def cb_img_hand_right(self, msg: Image) -> None:
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            current_time = time.time()
        except Exception as e:
            self.get_logger().warn(f"cv_bridge right-hand fail: {e}")
            return
        size = getattr(self.cfg, "hand_right_target_size", self.cfg.hand_target_size)
        bgr = cv2.resize(
            bgr, (size[0], size[1]),
            interpolation=cv2.INTER_AREA
        )
        t = now_sec()
        self.hand_right_buf.append((t, bgr))
        if len(self.hand_right_buf) % DEFAULT_HANDBUF_CLEAN_LEN == 0:
            self.clean_old_data(self.hand_right_buf, current_time, max_age_seconds=5.0)

    def hand_qpos6_left_callback(self, msg: EEJointFeedback):
        if len(msg.position) >= 6:
            current_time = time.time()
            arr = np.asarray(msg.position[:6], dtype=np.float32)
            self.hand_qpos6_left_buf.append((current_time, arr.copy()))
            if len(self.hand_qpos6_left_buf) % DEFAULT_HANDBUF_CLEAN_LEN == 0:
                self.clean_old_data(
                    self.hand_qpos6_left_buf, current_time, max_age_seconds=5.0
                )
        else:
            log_warning(
                f"Received left hand position with only {len(msg.position)} elements, expected at least 6"
            )

    def hand_qpos6_right_callback(self, msg: EEJointFeedback):
        if len(msg.position) >= 6:
            current_time = time.time()
            arr = np.asarray(msg.position[:6], dtype=np.float32)
            self.hand_qpos6_right_buf.append((current_time, arr.copy()))
            if len(self.hand_qpos6_right_buf) % DEFAULT_HANDBUF_CLEAN_LEN == 0:
                self.clean_old_data(
                    self.hand_qpos6_right_buf, current_time, max_age_seconds=5.0
                )
        else:
            log_warning(
                f"Received right hand position with only {len(msg.position)} elements, expected at least 6"
            )

    def gripper_qpos_left_callback(self, msg: EEJointFeedback):
        if len(msg.position) >= 1:
            current_time = time.time()
            arr = [msg.position[0]]
            self.gripper_qpos_left_buf.append((current_time, arr.copy()))
            if len(self.gripper_qpos_left_buf) % DEFAULT_HANDBUF_CLEAN_LEN == 0:
                self.clean_old_data(
                    self.gripper_qpos_left_buf, current_time, max_age_seconds=5.0
                )
        else:
            log_warning(
                f"Received left gripper position with only {len(msg.position)} elements, expected at least 1"
            )

    def gripper_qpos_right_callback(self, msg: EEJointFeedback):
        if len(msg.position) >= 1:
            current_time = time.time()
            arr = [msg.position[0]]
            self.gripper_qpos_right_buf.append((current_time, arr.copy()))
            if len(self.gripper_qpos_right_buf) % DEFAULT_HANDBUF_CLEAN_LEN == 0:
                self.clean_old_data(
                    self.gripper_qpos_right_buf, current_time, max_age_seconds=5.0
                )
        else:
            log_warning(
                f"Received right gripper position with only {len(msg.position)} elements, expected at least 1"
            )

    def clean_old_data(self, buffer, current_time, max_age_seconds=5.0):
        if not buffer:
            return
        cutoff_time = current_time - max_age_seconds
        while buffer and buffer[0][0] < cutoff_time:
            buffer.popleft()


    @property
    def running(self):
        return not self.shutdown_event.is_set()

    def init_client(self):
        log_info("Initializing Robot Client...")
        if not OptimizedRobotClient._logger_initialized:
            setup_logger()
            OptimizedRobotClient._logger_initialized = True

        # 配置相关参数推导
        self.environment_dt = 1 / self.cfg.control_frequency
        self.time_delay = int(self.cfg.time_infer * self.cfg.control_frequency)
        steps_delay = self.cfg.time_infer * self.cfg.control_frequency
        self.blending_horizon = int(
            np.ceil(self.cfg.action_horizon * self.cfg.chunk_size_threshold - steps_delay)
        )
        self.cfg.max_steps = self.cfg.max_steps * self.cfg.sample_factor
        self.get_action_frequency = DEFAULT_GET_ACTION_FREQUENCY

        log_info(f"dt = {self.environment_dt}")
        log_info(f"time_delay = {self.time_delay}")
        log_info(f"blending_horizon = {self.blending_horizon}")
        log_info(f"time_infer = {self.cfg.time_infer}")
        if self.cfg.chunk_size_threshold > 0:
            if self.blending_horizon < self.time_delay + 3:
                log_warning("blending_horizon must bigger than time_delay + 3, "
                            "please adjust time_infer or chunk_size_threshold or control_frequency!")

            predict_past_action_len = (
                self.cfg.action_horizon * self.cfg.chunk_size_threshold
            ) - (self.cfg.control_frequency * self.cfg.time_infer)
            if predict_past_action_len < self.time_delay + 3 and self.cfg.use_td:
                self.cfg.chunk_size_threshold = 0
                log_warning("could not use lipo, adjust chunk_size_threshold or control_frequency!")
                log_warning("Now use Sync mode!")

        # 队列清空
        while not self.action_queue.empty():
            try:
                self.action_queue.get_nowait()
            except Empty:
                break
        while not self.observation_queue.empty():
            try:
                self.observation_queue.get_nowait()
            except Empty:
                break

        # 计数器重置
        self.current_step = 0
        self.latest_action = -1
        self.latest_action_timestamp = -1
        self.action_chunk_size = 0
        self.action_dim = None

        # 集合/状态重置
        self.actions = OrderedDict()
        self.actionchunks = []
        self.action_queue_size = []
        self.inference_delay_zone = []
        self.blend_zone = []
        self.connection_points = []

        # 事件重置
        self.must_go.set()
        self.start_infer = False
        self.first_get_actions = True
        self.allow_infer_check = True

        # 推理对象重置
        self.post_opt = None
        self.inference_delay = None
        self.start_inference_time = None


        # 传感器 buffer 清空
        self.head_left_buf.clear()
        self.head_right_buf.clear()
        self.hand_left_buf.clear()
        self.hand_right_buf.clear()
        self.hand_qpos6_left_buf.clear()
        self.hand_qpos6_right_buf.clear()
        self.gripper_qpos_left_buf.clear()
        self.gripper_qpos_right_buf.clear()
        self.joint_state_buf.clear()

        log_info("Client initialized.")


    def wait_for_policy_server(self):
        if self.network_client.connected:
            log_info("Policy server already connected (reusing config connection)")
            return True
        start_time = time.perf_counter()
        log_info(
            f"Connecting to policy server at {self.cfg.server_host}:{self.cfg.server_port}..."
        )
        if self.network_client.connect():
            log_info("Policy server connected")
            end_time = time.perf_counter()
            log_info(f"Connection established in {end_time - start_time}s")
            return True
        else:
            self._set_state(RobotState.ERROR,f"连接服务器失败，请检查端口是否正确，网络是否正常！")
            return False


    def actions_available(self):
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def _ready_to_send_observation(self):
        if not self.allow_infer_check:
            return

        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()

            if queue_size == 0:
                self.must_go.set()
                self.start_infer = True
                self.allow_infer_check = False
                if self.current_step % 50 == 0:
                    log_warning("动作队列为空，触发推理")
            elif (
                self.action_chunk_size > 0
                and queue_size / self.action_chunk_size <= self.cfg.chunk_size_threshold
            ):
                self.start_infer = True
                self.allow_infer_check = False
                log_warning(f"队列长度 {queue_size} 低于阈值，触发推理")

    def _time_action_chunk(
        self, t_0: float, action_chunk: np.ndarray, i_0: int
    ) -> list[TimedAction]:
        """返回带动作时间戳的动作块"""
        return [
            TimedAction(
                timestamp=t_0 + i * self.environment_dt, timestep=i_0 + i, action=action
            )
            for i, action in enumerate(action_chunk)
        ]

    def get_actions(self):
        """从服务器获取动作"""
        try:
            time_get_actions = time.perf_counter()
            with self.socket_lock:
                response = self.network_client.send_request(RequestType.GET_ACTIONS)
            if response and response.get(ResponseKey.STATUS) == ResponseKey.SUCCESS:
                actions = response["actions"]["qpos"]
                timestamp = response[CommonKey.TIMESTAMP.value]
                timestep = response[CommonKey.TIMESTEP.value]
                available_keys = list(actions.keys())

                expected_order = [
                    InferenceConfigKey.WAISTQPOS.value,
                    InferenceConfigKey.LEFT_ARMQPOS.value,
                    InferenceConfigKey.HEADQPOS.value,
                    InferenceConfigKey.RIGHT_ARMQPOS.value,
                    InferenceConfigKey.ANKLEQPOS.value,
                    InferenceConfigKey.KNEEQPOS.value,
                    InferenceConfigKey.BUTTOCKQPOS.value,
                    (
                        InferenceConfigKey.LEFT_EEFHAND.value
                        if self.cfg.end_effector_type == InferenceConfigKey.HAND.value
                        else InferenceConfigKey.LEFT_EEFGRIPPER.value
                    ),
                    (
                        InferenceConfigKey.RIGHT_EEFHAND.value
                        if self.cfg.end_effector_type == InferenceConfigKey.HAND.value
                        else InferenceConfigKey.RIGHT_EEFGRIPPER.value
                    ),
                ]

                ordered_keys = [key for key in expected_order if key in available_keys]

                _actions = []
                last_concatenated = None
                for step in range(self.cfg.action_horizon):
                    step_actions = []
                    for key in ordered_keys:
                        action_data = actions.get(key)
                        if action_data is not None and len(action_data) > step:
                            step_actions.append(action_data[step])
                    if step_actions:
                        concatenated = np.concatenate(
                            [arr for arr in step_actions if len(arr) > 0]
                        )
                        _actions.append(concatenated)
                        last_concatenated = concatenated
                if self.action_dim is None and last_concatenated is not None:
                    self.action_dim = len(last_concatenated)

                action_chunk = self._time_action_chunk(timestamp, _actions, timestep)
                log_info(f"Actions received for timestep {timestep}")
                self.get_action_times.append(
                    (time.perf_counter() - time_get_actions) * 1000
                )
                avg_time = np.mean(self.get_action_times)
                log_warning(f"Get action cost {avg_time:.4f}ms")

                self.current_joint_order = ordered_keys
                return action_chunk
            else:
                return None

        except Exception as e:
            self._set_state(RobotState.ERROR,f"获取动作失败，请检查服务器或客户端日志查看问题: {e}")
            return None

    def receive_actions(self):
        self.start_barrier.wait()
        log_info("Action receiving thread starting")
        get_actions_interval = 1.0 / self.get_action_frequency
        next_recv_time = time.perf_counter()

        while self.running and self.current_step < self.cfg.max_steps:
            try:
                loop_start = time.perf_counter()

                timed_actions = self.get_actions()
                if timed_actions is not None:
                    self.action_chunk_size = max(
                        self.action_chunk_size,
                        int(len(timed_actions) * self.cfg.sample_factor),
                    )

                    self._aggregate_action_queues(
                        deepcopy(timed_actions), self.aggregate_fn
                    )
                    self.allow_infer_check = True
                    log_info(f"动作聚合完成，重新允许推理检查")

                elapsed = time.perf_counter() - loop_start
                sleep_time = max(0, get_actions_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

                next_recv_time += get_actions_interval
                current_time = time.perf_counter()
                if current_time < next_recv_time:
                    time.sleep(next_recv_time - current_time)
                else:
                    next_recv_time = current_time

            except Exception as e:
                self._set_state(RobotState.ERROR, f"接收动作异常: {e}")
                time.sleep(0.01)

    def _aggregate_action_queues(
        self,
        incoming_actions: list[TimedAction],
        aggregate_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    ):
        """动作聚合"""
        if aggregate_fn is None:
            def aggregate_fn(x1, x2):
                return x2

        aggregate_start_time = time.perf_counter()
        future_action_queue = Queue()
        timed_actions = []
        if self.start_inference_time is not None:
            self.inference_delay = time.time() - self.start_inference_time
        else:
            self.inference_delay = 0.0
        aggregate_delay = time.time()

        # caculate run_blend
        if self.inference_delay - self.cfg.time_infer >= 0:
            run_blend = int(
                self.blending_horizon
                - (
                    (self.inference_delay - self.cfg.time_infer)
                    / self.environment_dt
                )
            )
        else:
            run_blend = self.blending_horizon

        log_warning(f"run blend = {run_blend}")
        self.inference_delay_zone.append(self.inference_delay)
        self.blend_zone.append(run_blend * self.environment_dt)

        for new_action in incoming_actions:
            timed_actions.append(
                TimedAction(
                    timestamp=new_action.get_timestamp(),
                    timestep=new_action.get_timestep(),
                    action=new_action.get_action(),
                )
            )

            if self.first_get_actions:
                timed_actions[-1].timestamp = (
                    timed_actions[-1].timestamp + self.inference_delay
                )
                new_action.timestamp = new_action.timestamp + self.inference_delay
            elif (
                new_action.get_timestamp()
                <= incoming_actions[0].timestamp + self.inference_delay
                and self.cfg.chunk_size_threshold > 0
            ):  # 新来的时间步比现在的旧就跳过
                log_info(f"skip time stamp {new_action.get_timestamp()}")
                continue

            future_action_queue.put(new_action)
        if self.cfg.save_actionchunks:
            self.actionchunks.append(deepcopy(timed_actions))

        if not self.first_get_actions and self.cfg.chunk_size_threshold > 0:
            start_merge = time.perf_counter()
            future_actions = np.array(
                [action.get_action() for action in future_action_queue.queue]
            )

            with self.action_queue_lock:
                internal_queue = deepcopy(self.action_queue.queue)
                current_action_queue = {
                    action.get_timestep(): action.get_action()
                    for action in internal_queue
                }

                past_actions = np.array(list(current_action_queue.values()))

                if len(past_actions) > 0:
                    _past_actions = interpolate_2d(past_actions, 1 / self.cfg.sample_factor)
                else:
                    log_warning("异步平滑: 历史动作队列为空，跳过 LiPo 合并")
                    _past_actions = np.array([])

            if len(_past_actions) > 0:
                log_debug("Start lipo merge")
                past_action_len = _past_actions.shape[0]
                log_debug(f"past_actions: {_past_actions.shape}")
                if _past_actions.shape[0] < run_blend + self.post_opt.JM:
                    padding_len = run_blend + self.post_opt.JM - past_action_len
                    padding_factor = float(past_action_len + padding_len + 1) / float(
                        past_action_len
                    )
                    _past_actions = interpolate_2d(_past_actions, padding_factor)

                    log_debug(
                        f"past_actions interpolate_len: {padding_len}, After interpolate: {_past_actions.shape}"
                    )

                log_debug(
                    f"past_actions: {_past_actions.shape} | future_actions: {future_actions.shape}"
                )
                merged_actions, _ = self.post_opt.solve_padding(
                    future_actions,
                    _past_actions,
                    len_past_actions=run_blend,
                    padding_mode=0,
                    use_td=self.cfg.use_td,
                )
                for i, action in enumerate(merged_actions):
                    future_action_queue.queue[i].action = action

                log_debug(
                    f"Use lipo merge cost time: {(time.perf_counter() - start_merge)*1000 :.2f}ms"
                )

        future_actions = np.array(
            [action.get_action() for action in future_action_queue.queue]
        )

        future_actions = interpolate_2d(future_actions, self.cfg.sample_factor)
        origin_timestamp_list = []
        if len(incoming_actions) > 0:
            _latest_action_timestamp = incoming_actions[0].timestamp + self.inference_delay
        else:
            _latest_action_timestamp = time.time()
        for i in range(future_actions.shape[0]):
            if self.first_get_actions:
                origin_timestamp_list.append(
                    incoming_actions[0].timestamp
                    + i * self.environment_dt * (1 / self.cfg.sample_factor)
                )
            else:
                origin_timestamp_list.append(
                    _latest_action_timestamp
                    + i * self.environment_dt * (1 / self.cfg.sample_factor)
                )
        future_action_queue = Queue()
        for i, action in enumerate(future_actions):
            future_action_queue.put(
                TimedAction(
                    timestamp=origin_timestamp_list[i],
                    timestep=self.latest_action + i + 1,
                    action=action,
                )
            )

        log_debug(
            f"Aggregate_action_queue cost time: {(time.perf_counter() - aggregate_start_time)*1000 :.2f}ms"
        )
        if self.first_get_actions == True:
            self.first_get_actions = False

        if not future_action_queue.empty():
            first_action = list(future_action_queue.queue)[0]
            self.connection_points.append(
                {
                    CommonKey.TIMESTAMP.value: first_action.get_timestamp(),
                    CommonKey.TIMESTEP.value: first_action.get_timestep(),
                    CommonKey.ACTION.value: first_action.get_action(),
                    "chunk_index": len(self.actionchunks),
                }
            )
            log_info(
                f"Marked connect point: timestep={first_action.get_timestep()}, chunk_index={len(self.actionchunks)-1}"
            )

        with self.action_queue_lock:
            self.action_queue = future_action_queue


    def action_execution_loop(self):
        self.start_barrier.wait()
        log_info("Action execution thread starting")

        action_interval = 1.0 / (self.cfg.control_frequency * self.cfg.sample_factor)
        next_action_time = time.perf_counter()

        action_count = 0
        action_times = deque(maxlen=50)

        while self.running and self.current_step < self.cfg.max_steps:
            loop_start = time.perf_counter()

            if self.actions_available():
                action_start = time.perf_counter()
                _performed_action = self.control_loop_action()
                action_time = time.perf_counter() - action_start
                self.action_times.append(action_time)
                action_times.append(action_time)
                if self.cfg.save_actionchunks:
                    self.actions[_performed_action.timestep] = deepcopy(
                        _performed_action
                    )
                action_count += 1

            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0, action_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                if action_count % 100 == 0:
                    log_warning(f"动作执行线程超时: {-sleep_time*1000:.2f}ms")

            next_action_time += action_interval
            current_time = time.perf_counter()
            if current_time < next_action_time:
                time.sleep(next_action_time - current_time)
            else:
                next_action_time = current_time


        log_debug("Action execution thread stopped")

    def observation_collection_loop(self):
        self.start_barrier.wait()
        log_info("Observation collection thread starting")

        collect_obs_interval = 1.0 / self.cfg.collect_frequency
        next_obs_time = time.perf_counter()
        collection_count = 0
        collection_times = deque(maxlen=20)

        while self.running and self.current_step < self.cfg.max_steps:
            loop_start = time.perf_counter()

            collection_start = time.perf_counter()
            self.collect_and_queue_observation()
            collection_time = time.perf_counter() - collection_start
            collection_times.append(collection_time)

            collection_count += 1

            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0, collect_obs_interval - elapsed)


            if sleep_time > 0:
                time.sleep(sleep_time)

            next_obs_time += collect_obs_interval
            current_time = time.perf_counter()
            if current_time < next_obs_time:
                # collect_frequency控制get_real_obs频率
                time.sleep(next_obs_time - current_time)
            else:
                next_obs_time = current_time

            if collection_count % 20 == 0 and collection_times:
                avg_time = np.mean(collection_times) * 1000
                log_info(f"观测收集平均耗时: {avg_time:.2f}ms")

        log_debug("Observation collection thread stopped")

    def observation_check_loop(self):
        self.start_barrier.wait()
        log_info("Observation check thread starting")

        check_interval = 1.0 / self.observation_check_frequency
        next_check_time = time.perf_counter()

        check_count = 0
        check_times = deque(maxlen=30)

        while self.running and self.current_step < self.cfg.max_steps:
            loop_start = time.perf_counter()

            check_start = time.perf_counter()
            self._ready_to_send_observation()
            check_time = time.perf_counter() - check_start
            check_times.append(check_time)

            check_count += 1

            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0, check_interval - elapsed)

            if (
                self.post_opt is None
                and self.action_dim is not None
                and self.cfg.chunk_size_threshold > 0
            ):
                log_info(f"init ActionLipo")
                start_time = time.perf_counter()
                self.post_opt = ActionLiPo(
                    chunk_size=self.cfg.action_horizon,
                    blending_horizon=self.blending_horizon,
                    action_dim=self.action_dim,
                    len_time_delay=self.time_delay,
                    dt=self.environment_dt,
                )
                log_info(
                    f"init ActionLipo cost {((time.perf_counter() - start_time)*1000):.4f}ms"
                )

            # observation_check_frequency控制_ready_to_send_observation频率
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                if check_count % 100 == 0:
                    log_warning(f"观测检查线程超时: {-sleep_time*1000:.2f}ms")

            next_check_time += check_interval
            current_time = time.perf_counter()
            if current_time < next_check_time:
                # observation_check_frequency控制_ready_to_send_observation频率
                time.sleep(next_check_time - current_time)
            else:
                next_check_time = current_time
            if log_error.call_count > 0:
                log_warning(f"检测到 {log_error.call_count} 次错误，强制终止推理")
                self.current_step = self.cfg.max_steps


            if check_count % 30 == 0 and check_times:
                avg_time = np.mean(check_times) * 1000
                log_info(f"观测检查平均耗时: {avg_time:.2f}ms")
        log_debug("Observation check thread stopped")

    def collect_and_queue_observation(self):
        try:
            raw_obs, obs_timestamp = self.get_real_obs()

            if raw_obs is None:
                return

            with self.latest_action_lock:
                latest_action = self.latest_action

            observation = TimedObservation(
                timestamp=obs_timestamp,
                observation=raw_obs,
                timestep=max(latest_action, 0),
            )

            with self.action_queue_lock:
                observation.must_go = (
                    self.must_go.is_set() and self.action_queue.empty()
                )

            try:
                self.observation_queue.put_nowait(observation)
            except:
                try:
                    self.observation_queue.get_nowait()
                    self.observation_queue.put_nowait(observation)
                except:
                    pass

        except Exception as e:
            self._set_state(RobotState.ERROR, f"收集观测失败: {e}")

    def observation_sender_loop(self):
        self.start_barrier.wait()
        log_info("Observation sender thread starting")

        while self.running and self.current_step < self.cfg.max_steps:
            try:
                try:
                    observation = self.observation_queue.get_nowait()
                except Empty:
                    time.sleep(0.001)
                    continue

                success = self.send_observation(observation)

                if success:
                    if observation.must_go:
                        self.must_go.clear()

            except Exception as e:
                self._set_state(RobotState.ERROR, f"观测发送错误，请查看服务器或客户端日志以获取详细信息: {e}")
                time.sleep(0.01)

        log_debug("Observation sender thread stopped")


    def control_loop_action(self) -> TimedAction:
        with self.action_queue_lock:
            timed_action = self.action_queue.get_nowait()
            self.action_queue_size.append(self.action_queue.qsize())
        _performed_action = self.exec_action(timed_action)

        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()
            self.latest_action_timestamp = timed_action.get_timestamp()

        return _performed_action

    def exec_action(self, timed_action: TimedAction):
        """执行单个动作帧"""
        exec_start = time.perf_counter()

        action = timed_action.get_action()

        if hasattr(self, "current_joint_order") and self.current_joint_order:
            body_actions = {}
            hand_actions = {}
            joint_dims = {
                InferenceConfigKey.WAISTQPOS.value: len(
                    w1qpos_names_map[InferenceConfigKey.WAISTQPOS.value]
                ),
                InferenceConfigKey.LEFT_ARMQPOS.value: len(
                    w1qpos_names_map[InferenceConfigKey.LEFT_ARMQPOS.value]
                ),
                InferenceConfigKey.HEADQPOS.value: len(
                    w1qpos_names_map[InferenceConfigKey.HEADQPOS.value]
                ),
                InferenceConfigKey.RIGHT_ARMQPOS.value: len(
                    w1qpos_names_map[InferenceConfigKey.RIGHT_ARMQPOS.value]
                ),
                InferenceConfigKey.ANKLEQPOS.value: len(
                    w1qpos_names_map[InferenceConfigKey.ANKLEQPOS.value]
                ),
                InferenceConfigKey.KNEEQPOS.value: len(
                    w1qpos_names_map[InferenceConfigKey.KNEEQPOS.value]
                ),
                InferenceConfigKey.BUTTOCKQPOS.value: len(
                    w1qpos_names_map[InferenceConfigKey.BUTTOCKQPOS.value]
                )
            }
            if self.cfg.end_effector_type == InferenceConfigKey.HAND.value:
                joint_dims.update({
                    InferenceConfigKey.LEFT_EEFHAND.value: len(
                        w1qpos_names_map[InferenceConfigKey.LEFT_EEFHAND.value]
                    ),
                    InferenceConfigKey.RIGHT_EEFHAND.value: len(
                        w1qpos_names_map[InferenceConfigKey.RIGHT_EEFHAND.value]
                    )
                })
            elif self.cfg.end_effector_type == InferenceConfigKey.GRIPPER.value:
                joint_dims.update({
                    InferenceConfigKey.LEFT_EEFGRIPPER.value: len(
                        w1qpos_names_map[InferenceConfigKey.LEFT_EEFGRIPPER.value]
                    ),
                    InferenceConfigKey.RIGHT_EEFGRIPPER.value: len(
                        w1qpos_names_map[InferenceConfigKey.RIGHT_EEFGRIPPER.value]
                    )
                })
            start_idx = 0
            data_dict = {}

            for key in self.current_joint_order:
                dim = joint_dims.get(key, 0)
                if dim > 0 and start_idx + dim <= len(action):
                    joint_names = w1qpos_names_map.get(key, [])
                    for i in range(dim):
                        if i < len(joint_names):
                            data_dict[joint_names[i]] = float(action[start_idx + i])
                    if (
                        InferenceConfigKey.LEFT_EEFHAND.value in key
                        or InferenceConfigKey.RIGHT_EEFHAND.value in key
                    ):
                        hand_actions[key] = action[start_idx : start_idx + dim]
                    elif (
                        InferenceConfigKey.LEFT_EEFGRIPPER.value in key
                        or InferenceConfigKey.RIGHT_EEFGRIPPER.value in key
                    ):
                        hand_actions[key] = action[start_idx : start_idx + dim]
                    else:
                        body_actions[key] = action[start_idx : start_idx + dim]
                    start_idx += dim

            if body_actions:
                pub_names = []
                pub_pos = []
                with self.action_queue_lock:
                    for key, values in body_actions.items():
                        joint_names = w1qpos_names_map.get(key)
                        for i, value in enumerate(values):
                            if i < len(joint_names):
                                pub_names.append(joint_names[i])
                                pub_pos.append(float(value))

                self.publish_joint_positions(pub_names, pub_pos, clip=True)

            if self.cfg.end_effector_type == InferenceConfigKey.HAND.value:
                left_key = InferenceConfigKey.LEFT_EEFHAND.value
                right_key = InferenceConfigKey.RIGHT_EEFHAND.value
                left_hand = hand_actions.get(left_key)
                right_hand = hand_actions.get(right_key)

                if left_hand is not None:
                    self.publish_hand_positions("left", left_hand, clip=True)
                if right_hand is not None:
                    self.publish_hand_positions("right", right_hand, clip=True)

            else:
                left_key = InferenceConfigKey.LEFT_EEFGRIPPER.value
                right_key = InferenceConfigKey.RIGHT_EEFGRIPPER.value
                left_grip = hand_actions.get(left_key)
                right_grip = hand_actions.get(right_key)

                if left_grip is not None:
                    self.publish_gripper_position(
                        "left", left_grip[0] if len(left_grip) > 0 else 0.0, clip=True
                    )
                if right_grip is not None:
                    self.publish_gripper_position(
                        "right", right_grip[0] if len(right_grip) > 0 else 0.0, clip=True
                    )

            if self.cfg.save_exec_action:
                frame_dict = {
                    "frame_id": timed_action.timestep,
                    "timestamp": timed_action.timestamp,
                    "data": data_dict,
                }

                if not hasattr(self, "recorded_frames"):
                    self.recorded_frames = []
                if not hasattr(self, "record_filename"):
                    self.record_filename = "client_output/recorded_actions.json"

                self.recorded_frames.append(frame_dict)
                _save_frames_to_file(self.recorded_frames, self.record_filename)

        log_info(
            f"Executed Action Step: {timed_action.timestep}, "
            f"Remaining Action Queue Size: {self.action_queue_size[-1] if self.action_queue_size else 0}"
        )
        self.current_step += 1

        return timed_action


    def get_real_obs(self):
        obs = {
            CommonKey.IMAGES.value: {},
            CommonKey.STATES.value: {},
            CommonKey.DISPS.value: [None],
        }

        t = now_sec()

        head_left = nearest_without_tol(self.head_left_buf, self.head_lock)
        head_right = nearest_without_tol(self.head_right_buf, self.head_lock)
        if head_left is None or head_right is None:
            self._set_state(RobotState.ERROR, "读取头部相机数据失败，请检查头部相机话题是否有数据！")

        obs[CommonKey.IMAGES.value][ImageKey.CAM_HIGH.value] = head_left
        obs[CommonKey.IMAGES.value][ImageKey.CAM_HIGH_R.value] = head_right

        with self.joint_state_lock:
            qpos = np.array(self.joint_state_buf, dtype=np.float32)

            n_qpos = len(qpos)

            def _safe_slice(arr, key_enum, n):
                start, end = key_enum.value[0], key_enum.value[1]
                end = min(end, n - 1)
                if start >= n:
                    return []
                return [arr[idx] for idx in range(start, end + 1)]

            ankle_qpos = _safe_slice(qpos, JointNamesKey.W1_ANJLE_num, n_qpos)
            knee_qpos = _safe_slice(qpos, JointNamesKey.W1_KNEE_num, n_qpos)
            buttock_qpos = _safe_slice(qpos, JointNamesKey.W1_BUTTOCK_num, n_qpos)
            waist_qpos = _safe_slice(qpos, JointNamesKey.W1_WAIST_num, n_qpos)
            head_qpos = _safe_slice(qpos, JointNamesKey.W1_HEAD_num, n_qpos)
            left_arm_qpos = _safe_slice(qpos, JointNamesKey.W1_LEFT_num, n_qpos)
            right_arm_qpos = _safe_slice(qpos, JointNamesKey.W1_RIGHT_num, n_qpos)

        obs[CommonKey.STATES.value][InferenceConfigKey.WAISTQPOS.value] = np.array(
            waist_qpos
        )
        obs[CommonKey.STATES.value][InferenceConfigKey.ANKLEQPOS.value] = np.array(
            ankle_qpos
        )
        obs[CommonKey.STATES.value][InferenceConfigKey.HEADQPOS.value] = np.array(
            head_qpos
        )
        obs[CommonKey.STATES.value][InferenceConfigKey.BUTTOCKQPOS.value] = np.array(
            buttock_qpos
        )
        obs[CommonKey.STATES.value][InferenceConfigKey.KNEEQPOS.value] = np.array(
            knee_qpos
        )
        obs[CommonKey.STATES.value][InferenceConfigKey.LEFT_ARMQPOS.value] = np.array(
            left_arm_qpos
        )
        obs[CommonKey.STATES.value][InferenceConfigKey.RIGHT_ARMQPOS.value] = np.array(
            right_arm_qpos
        )

        # 获取 gripper/hand 数据
        with self.hand_qpos6_right_lock:
            if self.cfg.end_effector_type == InferenceConfigKey.GRIPPER.value:
                left_gripper = nearest_without_tol(
                    self.gripper_qpos_left_buf, self.hand_left_lock
                )
                right_gripper = nearest_without_tol(
                    self.gripper_qpos_right_buf, self.hand_right_lock
                )
                if left_gripper is None or right_gripper is None:
                    self._set_state(
                        RobotState.ERROR,
                        "读取二指夹数据失败，请检查二指夹是否正常运行或机器人是否使用灵巧手！"
                    )
                obs[CommonKey.STATES.value][
                    InferenceConfigKey.LEFT_EEFGRIPPER.value
                ] = np.array(left_gripper if left_gripper is not None else [])
                obs[CommonKey.STATES.value][
                    InferenceConfigKey.RIGHT_EEFGRIPPER.value
                ] = np.array(right_gripper if right_gripper is not None else [])

            elif self.cfg.end_effector_type == InferenceConfigKey.HAND.value:
                left_q6 = nearest_without_tol(
                    self.hand_qpos6_left_buf, self.hand_left_lock
                )
                right_q6 = nearest_without_tol(
                    self.hand_qpos6_right_buf, self.hand_right_lock
                )
                if left_q6 is None or right_q6 is None:
                    self._set_state(
                        RobotState.ERROR,
                        "读取灵巧手数据失败，请检查灵巧手是否正常运行或机器人是否使用二指夹！"
                    )
                    return None, None
                obs[CommonKey.STATES.value][
                    InferenceConfigKey.LEFT_EEFHAND.value
                ] = np.array(left_q6 if left_q6 is not None else [])
                obs[CommonKey.STATES.value][
                    InferenceConfigKey.RIGHT_EEFHAND.value
                ] = np.array(right_q6 if right_q6 is not None else [])

        # 手部相机
        if self.cfg.use_hand_camera:
            hand_left = nearest_without_tol(self.hand_left_buf, self.hand_left_lock)
            hand_right = nearest_without_tol(self.hand_right_buf, self.hand_left_lock)
            if hand_left is None or hand_right is None:
                self._set_state(
                    RobotState.ERROR,
                    f"读取腕部相机数据失败，请检查腕部相机话题是否有数据！"
                )
            obs[CommonKey.IMAGES.value][ImageKey.CAM_HAND_LEFT.value] = hand_left
            obs[CommonKey.IMAGES.value][ImageKey.CAM_HAND_RIGHT.value] = hand_right
        return obs, t

    def send_observation(self, obs: TimedObservation):
        try:
            timestamp = obs.get_timestamp()
            timestep = obs.get_timestep()
            must_go = obs.must_go

            raw_obs = obs.get_observation()
            images = raw_obs[CommonKey.IMAGES.value]
            states = raw_obs[CommonKey.STATES.value]

            observation = {
                CommonKey.TIMESTAMP.value: timestamp,
                CommonKey.TIMESTEP.value: timestep,
                CommonKey.MUST_GO.value: must_go,
                CommonKey.STATES.value: states,
                "start_infer": self.start_infer,
                "head_target_size": self.cfg.head_target_size,
                "hand_target_size": self.cfg.hand_target_size,
                "hand_left_target_size": getattr(self.cfg, "hand_left_target_size", self.cfg.hand_target_size),
                "hand_right_target_size": getattr(self.cfg, "hand_right_target_size", self.cfg.hand_target_size),
                "time_infer": self.cfg.time_infer,
                CommonKey.END_EFFECTOR_LIMIT.value: self.cfg.end_effector_position_limit,
                CommonKey.INSTRUCTION.value: [self.cfg.prompt]
            }

            if self.start_infer:
                log_info(f"发送推理请求，时间步: {timestep}, 时间戳：{timestamp}")
                self.start_inference_time = timestamp
            self.start_infer = False

            keys = list(images.keys())
            for image_key in keys:
                img_data = images[image_key]
                if img_data is not None:
                    observation[image_key] = img_data.astype(np.uint8).tobytes()


            with self.socket_lock:
                response = self.network_client.send_request(RequestType.OBSERVATION, observation)

            if response and response.get(ResponseKey.STATUS) == "received":
                if observation.get("start_infer"):
                    log_info("推理请求已发送，等待服务器响应")
                return True
            else:
                err_msg = response.get(ResponseKey.MESSAGE, "server returned error") if response else "no response"
                self._set_state(RobotState.ERROR, f"[server] {err_msg}")
                return False


        except Exception as e:
            traceback.print_exc()
            self._set_state(RobotState.ERROR, f"[client] 发送观测失败: {e}")
            return False


    def start(self):
        if self.wait_for_policy_server():
            self.init_client()
            log_error.reset_counter()

            spin_thread = threading.Thread(target=self.ros_spin, daemon=True)
            spin_thread.start()

            log_info("Starting action receiver thread...")
            action_receiver_thread = threading.Thread(
                target=self.receive_actions, daemon=True
            )
            action_receiver_thread.start()

            log_info("Starting observation sender thread...")
            self.observation_sending_thread = threading.Thread(
                target=self.observation_sender_loop, daemon=True
            )
            self.observation_sending_thread.start()

            log_info("Starting action execution thread...")
            action_execution_thread = threading.Thread(
                target=self.action_execution_loop, daemon=True
            )
            action_execution_thread.start()

            log_info("Starting observation collection thread...")
            observation_collection_thread = threading.Thread(
                target=self.observation_collection_loop, daemon=True
            )
            time.sleep(0.25)
            observation_collection_thread.start()

            log_info("Starting observation check thread...")
            observation_check_thread = threading.Thread(
                target=self.observation_check_loop, daemon=True
            )
            observation_check_thread.start()

            try:
                while self.running and self.current_step < self.cfg.max_steps:
                    if self._shutdown_event.is_set():
                        break
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass
            finally:
                self._stop_inference()
                if self.state != RobotState.ERROR:
                    self._set_state(RobotState.IDLE)
                if not self.cfg.service:
                    self.stop()
        else:
            time.sleep(1.0)

    def ros_spin(self):
        try:
            while self.running:
                rclpy.spin_once(self, timeout_sec=0.01)
        except Exception as e:
            self._set_state(RobotState.ERROR, f"ROS通信异常: {e}")
            self.running = False



    def _set_state(self, new_state: RobotState, error_msg: str = ""):
        with self.state_lock:
            old, self.state = self.state, new_state
            self.error_message = error_msg if new_state == RobotState.ERROR else None
            log_info(f"Client state: {old.value} → {new_state.value}"
                    + (f" ({error_msg})" if error_msg else ""))
            if new_state == RobotState.ERROR and error_msg:
                log_error(error_msg)
            self._notify_manager_state()



    def get_status(self) -> dict:
        with self.state_lock:
            # 查询server状态
            server_state = "unknown"
            server_error = None
            if self.network_client and self.network_client.connected:
                try:
                    s = self.network_client.send_request(RequestType.STATUS)
                    if s:
                        server_state = s.get(ResponseKey.STATE, "unknown")
                        server_error = s.get(ResponseKey.ERROR)
                except Exception:
                    server_state = "unreachable"
                    server_error = "server unreachable"


            if server_error and not self.error_message:
                # 只有server报错
                tagged_error = f"[server] {server_error}"
            elif server_error and self.error_message:
                # server和client都报错,根因是 server
                tagged_error = f"[client] {self.error_message} | root: [server] {server_error}"
            elif self.error_message:
                tagged_error = f"[client] {self.error_message}"
            else:
                tagged_error = None

            return {
                ResponseKey.STATE: self.state.value,
                ResponseKey.ERROR: tagged_error,
                "server": f"{self.cfg.server_host}:{self.cfg.server_port}",
                "server_state": server_state,
            }

    def _start_manager_listener(self):
        from act_async_infer_distributed_demo.scripts.network_utils import SimpleJsonTcpServer
        self._mgr_server = SimpleJsonTcpServer("0.0.0.0", self.cfg.manager_port, timeout=1.0)
        self._mgr_thread = self._mgr_server.start_in_thread(self._dispatch_mgr)

    def _dispatch_mgr(self, msg: dict) -> dict:
        self._suppress_state_notify = True
        try:
            return self._process_mgr_cmd(msg.get("command", ""), msg.get("payload", {}))
        finally:
            self._suppress_state_notify = False



    def _process_mgr_cmd(self, command: str, payload: dict) -> dict:
        if command == RequestType.SETUP_CONFIG:
            self._set_state(RobotState.IDLE)
            return self._cmd_setup_config(payload.get("client_config", {}),
                                          payload.get("server_config", {}),
                                          payload.get("home_position", ""))
        elif command == RequestType.STOP:
            return self._cmd_stop()
        elif command == RequestType.STATUS:
            return {ResponseKey.SUCCESS: True, ResponseKey.STATUS: self.get_status()}
        return {ResponseKey.SUCCESS: False, ResponseKey.MESSAGE: f"Unknown: {command}"}


    def _notify_manager_state(self):
        if hasattr(self, '_mgr_conn') and self._mgr_conn:
            try:
                msg = json.dumps({ResponseKey.TYPE: ManagerKey.STATE_UPDATE, ResponseKey.STATE: self.state.value,
                                  ResponseKey.ERROR: self.error_message}).encode("utf-8")
                self._mgr_conn.sendall(len(msg).to_bytes(4, "big"))
                self._mgr_conn.sendall(msg)
            except Exception:
                pass


    def _cmd_setup_config(self, client_cfg: dict, server_cfg: dict,
                        home_position: str) -> dict:
        # 1. 合并配置
        self.cfg.apply_update(client_cfg)
        # 重新同步到 self 属性
        for k, v in client_cfg.items():
            if hasattr(self, k) and k in ClientConfig.RUNTIME_NAMES:
                setattr(self, k, v)
        log_debug(f"Updating client configuration: {client_cfg}")


        # 2 初始化 publisher
        self._init_publishers()

        # 3. go home
        if home_position:
            self._exec_home(home_position)
            time.sleep(0.5)
        # 4. 连接 server 并转发配置
        if not self.network_client.connected:
            self.network_client.host = self.cfg.server_host
            self.network_client.port = self.cfg.server_port
            if not self.network_client.connect():
                self._set_state(RobotState.ERROR, "Cannot connect to server")
                return {ResponseKey.SUCCESS: False, ResponseKey.MESSAGE: "Server connect failed"}
        log_debug(f"_cmd_setup_config:{server_cfg}")
        resp = self.network_client.send_request(RequestType.SETUP_CONFIG, {ResponseKey.CONFIG: server_cfg})
        if not resp or not resp.get(ResponseKey.SUCCESS):
            msg = resp.get(ResponseKey.ERROR, "Server config failed") if resp else "No response"
            self._set_state(RobotState.ERROR, msg)
            return {ResponseKey.SUCCESS: False, ResponseKey.MESSAGE: msg}

        # 5. 等待 server RUNNING
        deadline = time.time() + ACTIVATE_TIME
        while time.time() < deadline:
            s = self.network_client.send_request(RequestType.STATUS)
            if s and s.get(ResponseKey.STATE) == "running":
                break
            time.sleep(0.5)
        else:
            self._set_state(RobotState.ERROR, "Server startup timeout")
            return {ResponseKey.SUCCESS: False, ResponseKey.MESSAGE: "Server startup timeout"}

        # 6. 启动推理
        self._start_inference_thread()
        self._set_state(RobotState.RUNNING)
        return {ResponseKey.SUCCESS: True, ResponseKey.MESSAGE: "Running"}

    def _cmd_stop(self) -> dict:
        if self.network_client.connected:
            try:
                self.network_client.send_request(RequestType.STOP)
            except Exception:
                pass
        self._stop_inference()
        self._set_state(RobotState.IDLE)
        return {ResponseKey.SUCCESS: True, ResponseKey.MESSAGE: "Stopped"}


    def _exec_home(self, home_position: str):
        from act_async_infer_distributed_demo.scripts.traj_player import JointTrajectoryLoader
        home_position = JointTrajectoryLoader.normalize_json_path(home_position)
        if not home_position.startswith("/"):
            records_dir = os.environ.get(
                "W1_RECORDS_DIR",
                "/home/dexforce/w1/dexe_mobile_application/script/records",
            )
            home_position = os.path.join(records_dir, home_position)

        targets = JointTrajectoryLoader.load_targets(home_position)
        self.joint_state_buf = []
        left_hand  = [targets[n] for n in TrajectoryKeys.FINGER_KEYS[:6]]
        right_hand = [targets[n] for n in TrajectoryKeys.FINGER_KEYS[6:]]
        lg = targets[TrajectoryKeys.GRIPPER_KEYS[0]] * 100.0
        rg = targets[TrajectoryKeys.GRIPPER_KEYS[1]] * 100.0

        for _ in range(GRIPPER_PUBLISH_TIMES):
            self.publish_hand_positions("left", left_hand)
            self.publish_hand_positions("right", right_hand)
            self.publish_gripper_position("left", lg)
            self.publish_gripper_position("right", rg)
            time.sleep(0.01)

        # 手动spin等待joint state数据
        deadline = time.time() + WAIT_FOR_JOINT_DATA
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            with self.joint_state_lock:
                if len(self.joint_state_buf) > 0 and self.joint_names_from_topic is not None:
                    break
        else:
            log_error("Timeout waiting for joint state data, cannot execute home")
            return

        with self.joint_state_lock:
            current = np.array(self.joint_state_buf, dtype=np.float64)
            names = list(self.joint_names_from_topic) if self.joint_names_from_topic else []
        idxs = [i for i, n in enumerate(names) if n in targets]
        if len(idxs) == 0:
            log_error("home 轨迹中无匹配关节名")
            return
        cur = current[idxs]
        tgt = np.array([targets[names[i]] for i in idxs], dtype=np.float64)
        traj = JointTrajectoryLoader.build_trajectory(cur, tgt, n_frames=200)

        for step in range(200):
            self.publish_joint_positions(
                names=[names[i] for i in idxs],
                positions=traj[step].tolist(),
            )
            time.sleep(0.01)
        log_info("Home done")



    def _init_publishers(self):
        if self.cfg.mode == 1:
            j, hl, hr, gl, gr = (
                "/mj_sim/control/joint_position",
                "/mj_sim/control/ee/left", "/mj_sim/control/ee/right",
                "/mj_sim/control/gripper/left", "/mj_sim/control/gripper/right",
            )
        else:
            j  = self.cfg.joint_control_topic
            hl = self.cfg.set_left_hand_qpos6_topic
            hr = self.cfg.set_right_hand_qpos6_topic
            gl = self.cfg.set_left_gripper_qpos_topic
            gr = self.cfg.set_right_gripper_qpos_topic


        self._pub_joint = self.create_publisher(JointPositionControl, j, 10)
        self._pub_hand_left  = self.create_publisher(EEJointControl, hl, 10)
        self._pub_hand_right = self.create_publisher(EEJointControl, hr, 10)
        self._pub_gripper_left  = self.create_publisher(EEJointControl, gl, 10)
        self._pub_gripper_right = self.create_publisher(EEJointControl, gr, 10)
        self.end_effector_positon_limit = self.cfg.end_effector_position_limit

    def _now(self):
        return self.get_clock().now().to_msg()

    def publish_joint_positions(self, names, positions, clip=True):
        if clip and self._joint_limits:
            positions = [float(np.clip(p, *self._joint_limits[n]))
                         if n in self._joint_limits else float(p)
                         for n, p in zip(names, positions)]
        from joint_interfaces.msg import JointPositionControl
        msg = JointPositionControl()
        msg.header.stamp = self._now()
        msg.name = names
        msg.position = [float(p) for p in positions]
        if self._pub_joint:
            self._pub_joint.publish(msg)

    def publish_hand_positions(self, side, positions, clip=True):
        if clip and self.end_effector_positon_limit:
            positions = [float(np.clip(p, *self.end_effector_positon_limit)) for p in positions]
        else:
            positions = [float(p) for p in positions]
        positions = (positions + [0.0] * 6)[:6]
        msg = EEJointControl()
        msg.header.stamp = self._now()
        msg.mode = EEJointControlMode.POSITION
        msg.name = self._hand_joint_names
        msg.value = positions
        (self._pub_hand_left if side == "left" else self._pub_hand_right).publish(msg)

    def publish_gripper_position(self, side, position, clip=True):
        if clip and self.end_effector_positon_limit:
            position = float(np.clip(position, *self.end_effector_positon_limit))
        names = self._gripper_left_names if side == "left" else self._gripper_right_names

        msg = EEJointControl()
        msg.header.stamp = self._now()
        msg.mode = EEJointControlMode.POSITION
        msg.name = names
        msg.value = array("d", [position])
        (self._pub_gripper_left if side == "left" else self._pub_gripper_right).publish(msg)


    def _start_inference_thread(self):
        self._shutdown_event.clear()
        self.shutdown_event.clear() 
        self._inference_running = True

        def run():
            try:
                self.start()
            except Exception as e:
                log_error(f"Inference error: {e}")
                traceback.print_exc()
                self.network_client.close()
                self._set_state(RobotState.ERROR, str(e))

        self._inference_thread = threading.Thread(target=run, daemon=True)
        self._inference_thread.start()



    def _stop_inference(self):
        self._shutdown_event.set()
        self.shutdown_event.set() 
        self._inference_running = False




    def stop(self):
        if isinstance(self.actions, OrderedDict):
            self.actions = list(self.actions.values())
        if not os.path.exists(self.cfg.output_dir):
            os.makedirs(self.cfg.output_dir)
        if self.cfg.save_actionchunks:
            draw_actionchunks(
                self.actions,
                self.actionchunks,
                os.path.join(self.cfg.output_dir, "actionchunks.png"),
                self.inference_delay_zone,
                self.blend_zone,
                self.connection_points,
            )
        log_debug("Stopping robot client...")
        self.shutdown_event.set()
        self.network_client.close()
        self.destroy_node()
        time.sleep(1.0) # sleep 1s for error feedback
        log_error.reset_counter()
        rclpy.shutdown()
