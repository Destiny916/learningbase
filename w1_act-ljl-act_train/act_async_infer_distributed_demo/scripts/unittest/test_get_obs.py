import threading
import numpy as np
import cv2
import time
from typing import Dict, List, Tuple, Optional, Deque, Any
from collections import deque
from dataclasses import dataclass
import logging

from act_async_infer_distributed_demo.scripts.w1_mapping import (
    CommonKey,
    ImageKey,
    InferenceConfigKey,
    JointNamesKey,
)

from act_async_infer_distributed_demo.scripts.utils_distributed import nearest

"""
    usage: 
        cd w1_act
        python -m act_async_infer_distributed_demo.scripts.unittest.test_get_obs
"""


class MockKingfisher:
    def __init__(self):
        self.left_image = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
        self.right_image = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)

    def get_kingfisher_images(self):
        timestamp = time.time()
        return self.left_image, self.right_image, timestamp


@dataclass
class ObservationConfig:
    """与 ClientConfig 对齐的观测参数"""
    tolerance_s: float = 0.2
    head_target_size: Tuple[int, int] = (640, 360)
    hand_target_size: Tuple[int, int] = (640, 480)
    use_hand_camera: bool = True
    end_effector_type: str = InferenceConfigKey.HAND.value
    end_effector_position_limit: Tuple[float, float] = (0.0, 100.0)

    def __post_init__(self):
        if self.tolerance_s <= 0.0:
            self.tolerance_s = 0.2
        if self.end_effector_type not in (
            InferenceConfigKey.HAND.value,
            InferenceConfigKey.GRIPPER.value,
        ):
            self.end_effector_type = InferenceConfigKey.HAND.value


class DataBuffer:
    def __init__(self, maxlen: int = 2000):
        self.buffer: Deque[Tuple[float, Any]] = deque(maxlen=maxlen)
        self.lock = threading.Lock()

    def append(self, timestamp: float, data: Any):
        with self.lock:
            self.buffer.append((timestamp, data))

    def get_nearest(self, target_time: float, tolerance: float) -> Optional[Any]:
        with self.lock:
            if not self.buffer:
                return None
            return nearest(self.buffer, target_time, tolerance)

    def clear(self):
        with self.lock:
            self.buffer.clear()

    def size(self) -> int:
        with self.lock:
            return len(self.buffer)


class ObservationCollector:
    """
    观测收集器 —— 与 robot_client_with_kingfisher.py 的 get_real_obs() 对齐
    关键变化：
    - end_effector_type 决定取灵巧手(LEFT_EEFHAND)还是二指夹(LEFT_EEFGRIPPER)数据
    - 传感器缺失不抛异常，返回 None/空数组，由外部 log_error() 处理
    """

    def __init__(self, config: ObservationConfig):
        self.config = config

        # 头部相机
        self.head_left_buffer = DataBuffer(maxlen=200)
        self.head_right_buffer = DataBuffer(maxlen=200)

        # 手部相机
        self.hand_left_buffer = DataBuffer(maxlen=200)
        self.hand_right_buffer = DataBuffer(maxlen=200)

        # 灵巧手 qpos 缓冲区
        self.hand_qpos_left_buffer = DataBuffer(maxlen=2000)
        self.hand_qpos_right_buffer = DataBuffer(maxlen=2000)

        # 二指夹 qpos 缓冲区
        self.gripper_qpos_left_buffer = DataBuffer(maxlen=2000)
        self.gripper_qpos_right_buffer = DataBuffer(maxlen=2000)

        self.joint_state = []
        self.joint_state_time = 0.0
        self.kingfisher_provider = None

    def set_kingfisher_provider(self, provider):
        self.kingfisher_provider = provider

    # ── 数据更新接口 ──────────────────────

    def update_head_image_left(self, timestamp: float, image: np.ndarray):
        self.head_left_buffer.append(timestamp, image)

    def update_head_image_right(self, timestamp: float, image: np.ndarray):
        self.head_right_buffer.append(timestamp, image)

    def update_hand_image_left(self, timestamp: float, image: np.ndarray):
        self.hand_left_buffer.append(timestamp, image)

    def update_hand_image_right(self, timestamp: float, image: np.ndarray):
        self.hand_right_buffer.append(timestamp, image)

    def update_hand_qpos_left(self, timestamp: float, qpos: np.ndarray):
        if len(qpos) >= 6:
            self.hand_qpos_left_buffer.append(timestamp, qpos[:6])

    def update_hand_qpos_right(self, timestamp: float, qpos: np.ndarray):
        if len(qpos) >= 6:
            self.hand_qpos_right_buffer.append(timestamp, qpos[:6])

    # 二指夹更新接口
    def update_gripper_qpos_left(self, timestamp: float, qpos: np.ndarray):
        self.gripper_qpos_left_buffer.append(timestamp, qpos)

    def update_gripper_qpos_right(self, timestamp: float, qpos: np.ndarray):
        self.gripper_qpos_right_buffer.append(timestamp, qpos)

    def update_joint_state(self, timestamp: float, joint_positions: List[float]):
        self.joint_state = joint_positions.copy()
        self.joint_state_time = timestamp

    # ── 观测获取（对齐 get_real_obs） ──────

    def get_observation(self) -> Tuple[Dict, float]:
        obs = {
            CommonKey.IMAGES.value: {},
            CommonKey.STATES.value: {},
            CommonKey.DISPS.value: [None],
        }

        if self.kingfisher_provider:
            kingfisher_left, kingfisher_right, timestamp = self.kingfisher_provider.get_kingfisher_images()
        else:
            kingfisher_left = kingfisher_right = None
            timestamp = time.time()

        head_w, head_h = self.config.head_target_size
        hand_w, hand_h = self.config.hand_target_size

        # 头部相机（kingfisher）
        if kingfisher_left is not None:
            obs[CommonKey.IMAGES.value][ImageKey.CAM_HIGH.value] = cv2.resize(
                kingfisher_left, (head_w, head_h), interpolation=cv2.INTER_AREA
            )
        if kingfisher_right is not None:
            obs[CommonKey.IMAGES.value][ImageKey.CAM_HIGH_R.value] = cv2.resize(
                kingfisher_right, (head_w, head_h), interpolation=cv2.INTER_AREA
            )

        # 关节状态切片
        qpos = np.array(self.joint_state, dtype=np.float32)
        n_qpos = len(qpos)

        def _safe_slice(range_def: Tuple[int, int]) -> List[float]:
            start, end = range_def
            end = min(end, n_qpos - 1)
            if start >= n_qpos:
                return []
            return [qpos[idx] for idx in range(start, end + 1)]

        obs[CommonKey.STATES.value][InferenceConfigKey.WAISTQPOS.value] = np.array(
            _safe_slice(JointNamesKey.W1_WAIST_num.value)
        )
        obs[CommonKey.STATES.value][InferenceConfigKey.ANKLEQPOS.value] = np.array(
            _safe_slice(JointNamesKey.W1_ANJLE_num.value)
        )
        obs[CommonKey.STATES.value][InferenceConfigKey.HEADQPOS.value] = np.array(
            _safe_slice(JointNamesKey.W1_HEAD_num.value)
        )
        obs[CommonKey.STATES.value][InferenceConfigKey.BUTTOCKQPOS.value] = np.array(
            _safe_slice(JointNamesKey.W1_BUTTOCK_num.value)
        )
        obs[CommonKey.STATES.value][InferenceConfigKey.KNEEQPOS.value] = np.array(
            _safe_slice(JointNamesKey.W1_KNEE_num.value)
        )
        obs[CommonKey.STATES.value][InferenceConfigKey.LEFT_ARMQPOS.value] = np.array(
            _safe_slice(JointNamesKey.W1_LEFT_num.value)
        )
        obs[CommonKey.STATES.value][InferenceConfigKey.RIGHT_ARMQPOS.value] = np.array(
            _safe_slice(JointNamesKey.W1_RIGHT_num.value)
        )

        if self.config.tolerance_s <= 0.0:
            self.config.tolerance_s = 0.2

        # 按 end_effector_type 分支取手/二指夹数据
        if self.config.end_effector_type == InferenceConfigKey.HAND.value:
            left_q6 = self.hand_qpos_left_buffer.get_nearest(timestamp, self.config.tolerance_s)
            right_q6 = self.hand_qpos_right_buffer.get_nearest(timestamp, self.config.tolerance_s)
            obs[CommonKey.STATES.value][InferenceConfigKey.LEFT_EEFHAND.value] = (
                np.array(left_q6) if left_q6 is not None else np.array([])
            )
            obs[CommonKey.STATES.value][InferenceConfigKey.RIGHT_EEFHAND.value] = (
                np.array(right_q6) if right_q6 is not None else np.array([])
            )
        elif self.config.end_effector_type == InferenceConfigKey.GRIPPER.value:
            left_grip = self.gripper_qpos_left_buffer.get_nearest(timestamp, self.config.tolerance_s)
            right_grip = self.gripper_qpos_right_buffer.get_nearest(timestamp, self.config.tolerance_s)
            obs[CommonKey.STATES.value][InferenceConfigKey.LEFT_EEFGRIPPER.value] = (
                np.array(left_grip) if left_grip is not None else np.array([])
            )
            obs[CommonKey.STATES.value][InferenceConfigKey.RIGHT_EEFGRIPPER.value] = (
                np.array(right_grip) if right_grip is not None else np.array([])
            )

        # 手部相机
        if self.config.use_hand_camera:
            hand_left = self.hand_left_buffer.get_nearest(timestamp, self.config.tolerance_s)
            if hand_left is not None:
                obs[CommonKey.IMAGES.value][ImageKey.CAM_HAND_LEFT.value] = cv2.resize(
                    hand_left, (hand_w, hand_h), interpolation=cv2.INTER_AREA
                )
            hand_right = self.hand_right_buffer.get_nearest(timestamp, self.config.tolerance_s)
            if hand_right is not None:
                obs[CommonKey.IMAGES.value][ImageKey.CAM_HAND_RIGHT.value] = cv2.resize(
                    hand_right, (hand_w, hand_h), interpolation=cv2.INTER_AREA
                )

        return obs, timestamp


class MockDataGenerator:
    @staticmethod
    def generate_joint_state(num_joints: int = 30) -> List[float]:
        return list(np.random.randn(num_joints).astype(np.float32))

    @staticmethod
    def generate_hand_pose() -> np.ndarray:
        return np.random.randn(6).astype(np.float32)

    @staticmethod
    def generate_gripper_pos() -> np.ndarray:
        return np.array([np.random.uniform(0, 100)], dtype=np.float32)

    @staticmethod
    def generate_hand_image() -> np.ndarray:
        return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)


def example_usage(_tolerance_s, _use_hand_camera, _end_effector_type="hand"):
    config = ObservationConfig(
        tolerance_s=_tolerance_s,
        head_target_size=(640, 360),
        hand_target_size=(640, 480),
        use_hand_camera=_use_hand_camera,
        end_effector_type=_end_effector_type,
    )

    collector = ObservationCollector(config)
    kingfisher_provider = MockKingfisher()
    collector.set_kingfisher_provider(kingfisher_provider)

    current_time = time.time()

    collector.update_hand_image_left(current_time, MockDataGenerator.generate_hand_image())
    collector.update_hand_image_right(current_time, MockDataGenerator.generate_hand_image())

    if _end_effector_type == InferenceConfigKey.HAND.value:
        collector.update_hand_qpos_left(current_time, MockDataGenerator.generate_hand_pose())
        collector.update_hand_qpos_right(current_time, MockDataGenerator.generate_hand_pose())
    else:
        collector.update_gripper_qpos_left(current_time, MockDataGenerator.generate_gripper_pos())
        collector.update_gripper_qpos_right(current_time, MockDataGenerator.generate_gripper_pos())

    collector.update_joint_state(current_time, MockDataGenerator.generate_joint_state(30))

    obs, timestamp = collector.get_observation()

    logging.info(f"obs timestamp: {timestamp}")
    logging.info(f"image keys: {list(obs[CommonKey.IMAGES.value].keys())}")
    logging.info(f"states keys: {list(obs[CommonKey.STATES.value].keys())}")

    return obs


import unittest
from unittest.mock import Mock, patch


class TestObservationCollector(unittest.TestCase):
    def setUp(self):
        self.config = ObservationConfig(
            tolerance_s=0.2,
            head_target_size=(640, 360),
            hand_target_size=(640, 480),
            use_hand_camera=True,
            end_effector_type=InferenceConfigKey.HAND.value,
        )
        self.collector = ObservationCollector(self.config)
        self.mock_kingfisher = MockKingfisher()
        self.collector.set_kingfisher_provider(self.mock_kingfisher)
        self.test_time = time.time()

    # ── 原有测试（适配新 obs 键格式 .value） ──────

    def test_normal_operation(self):
        """测试正常操作场景"""
        self.collector.update_hand_image_left(self.test_time, np.ones((480, 640, 3), dtype=np.uint8) * 128)
        self.collector.update_hand_image_right(self.test_time, np.ones((480, 640, 3), dtype=np.uint8) * 255)
        self.collector.update_hand_qpos_left(self.test_time, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32))
        self.collector.update_hand_qpos_right(self.test_time, np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32))
        self.collector.update_joint_state(self.test_time, list(range(30)))

        obs, timestamp = self.collector.get_observation()

        self.assertIsInstance(obs, dict)
        self.assertIn(CommonKey.IMAGES.value, obs)
        self.assertIn(CommonKey.STATES.value, obs)

        images = obs[CommonKey.IMAGES.value]
        self.assertIn(ImageKey.CAM_HIGH.value, images)
        self.assertIn(ImageKey.CAM_HIGH_R.value, images)
        self.assertIn(ImageKey.CAM_HAND_LEFT.value, images)
        self.assertIn(ImageKey.CAM_HAND_RIGHT.value, images)
        self.assertEqual(images[ImageKey.CAM_HIGH.value].shape, (360, 640, 3))
        self.assertEqual(images[ImageKey.CAM_HAND_LEFT.value].shape, (480, 640, 3))

        states = obs[CommonKey.STATES.value]
        self.assertIn(InferenceConfigKey.LEFT_EEFHAND.value, states)
        self.assertIn(InferenceConfigKey.RIGHT_EEFHAND.value, states)
        self.assertEqual(len(states[InferenceConfigKey.LEFT_EEFHAND.value]), 6)
        self.assertEqual(len(states[InferenceConfigKey.RIGHT_EEFHAND.value]), 6)

    def test_missing_hand_camera(self):
        """测试禁用手部相机配置"""
        self.collector.config.use_hand_camera = False
        self.collector.update_hand_image_left(self.test_time, np.ones((480, 640, 3), dtype=np.uint8))
        obs, _ = self.collector.get_observation()
        images = obs[CommonKey.IMAGES.value]
        self.assertNotIn(ImageKey.CAM_HAND_LEFT.value, images)
        self.assertNotIn(ImageKey.CAM_HAND_RIGHT.value, images)
        self.assertIn(ImageKey.CAM_HIGH.value, images)
        self.assertIn(ImageKey.CAM_HIGH_R.value, images)

    def test_empty_buffers(self):
        """测试数据缓冲区为空"""
        self.collector.hand_left_buffer.clear()
        self.collector.hand_qpos_left_buffer.clear()
        self.collector.joint_state = []
        obs, _ = self.collector.get_observation()
        self.assertIsInstance(obs, dict)
        states = obs[CommonKey.STATES.value]
        left_hand = states.get(InferenceConfigKey.LEFT_EEFHAND.value)
        self.assertTrue(left_hand is None or isinstance(left_hand, np.ndarray))

    def test_time_mismatch(self):
        """测试时间不匹配场景"""
        old_time = self.test_time - 1.0
        self.collector.update_hand_qpos_left(old_time, np.array([1.0] * 6))
        new_time = self.test_time + 0.1
        self.collector.update_hand_qpos_left(new_time, np.array([2.0] * 6))
        obs, timestamp = self.collector.get_observation()
        states = obs[CommonKey.STATES.value]
        left_hand = states.get(InferenceConfigKey.LEFT_EEFHAND.value)
        self.assertIsNotNone(left_hand)
        self.assertTrue(
            np.allclose(left_hand, np.array([2.0] * 6))
            or np.allclose(left_hand, np.array([1.0] * 6))
        )

    def test_joint_index_out_of_range(self):
        """测试关节索引越界"""
        self.collector.update_joint_state(self.test_time, [0.1, 0.2, 0.3])
        try:
            obs, _ = self.collector.get_observation()
            states = obs[CommonKey.STATES.value]
            for key in [InferenceConfigKey.WAISTQPOS.value, InferenceConfigKey.ANKLEQPOS.value]:
                if key in states:
                    self.assertIsInstance(states[key], np.ndarray)
        except IndexError:
            self.fail("代码应处理关节索引越界")

    def test_kingfisher_none(self):
        """测试Kingfisher返回None图像"""
        mock_none = Mock()
        mock_none.get_kingfisher_images.return_value = (None, None, self.test_time)
        self.collector.set_kingfisher_provider(mock_none)
        obs, _ = self.collector.get_observation()
        images = obs[CommonKey.IMAGES.value]
        self.assertNotIn(ImageKey.CAM_HIGH.value, images)
        self.assertNotIn(ImageKey.CAM_HIGH_R.value, images)

    def test_kingfisher_provider_none(self):
        """测试未设置Kingfisher"""
        self.collector.set_kingfisher_provider(None)
        obs, timestamp = self.collector.get_observation()
        self.assertIsInstance(obs, dict)

    def test_hand_pose_insufficient_data(self):
        """测试手部姿势数据不足"""
        self.collector.update_hand_qpos_left(self.test_time, np.array([1.0, 2.0, 3.0]))
        obs, _ = self.collector.get_observation()
        states = obs[CommonKey.STATES.value]
        left_hand = states.get(InferenceConfigKey.LEFT_EEFHAND.value)
        # 数据不足6维 → 不会写入缓冲区 → 取不到 → 空数组
        self.assertTrue(left_hand is not None)  # 现在是空数组而非 None

    def test_image_resize_invalid(self):
        """测试无效尺寸图像重缩放"""
        invalid_image = np.ones((100, 100, 3), dtype=np.uint8)
        self.collector.update_hand_image_left(self.test_time, invalid_image)
        obs, _ = self.collector.get_observation()
        images = obs[CommonKey.IMAGES.value]
        if ImageKey.CAM_HAND_LEFT.value in images:
            self.assertEqual(images[ImageKey.CAM_HAND_LEFT.value].shape, (480, 640, 3))

    def test_multiple_observations(self):
        """测试多次观测获取"""
        self.collector.update_joint_state(self.test_time, [1.0] * 30)
        obs1, ts1 = self.collector.get_observation()
        time.sleep(0.1)
        self.collector.update_joint_state(time.time(), [2.0] * 30)
        obs2, ts2 = self.collector.get_observation()
        self.assertNotEqual(ts1, ts2)

    def test_buffer_overflow(self):
        """测试数据缓冲区溢出"""
        num_entries = 3000
        for i in range(num_entries):
            ts = self.test_time + i * 0.001
            self.collector.update_hand_qpos_left(ts, np.array([float(i)] * 6))
        buffer_size = self.collector.hand_qpos_left_buffer.size()
        self.assertLessEqual(buffer_size, 2000)
        obs, _ = self.collector.get_observation()
        self.assertIsInstance(obs, dict)

    def test_zero_tolerance(self):
        """测试零容差配置"""
        self.collector.config.tolerance_s = 0.0
        exact_time = self.test_time
        self.collector.update_hand_qpos_left(exact_time, np.array([1.0] * 6))
        obs, timestamp = self.collector.get_observation()
        states = obs[CommonKey.STATES.value]
        left_hand = states.get(InferenceConfigKey.LEFT_EEFHAND.value)
        self.assertIsNotNone(left_hand)

    def test_large_tolerance(self):
        """测试大容差配置"""
        self.collector.config.tolerance_s = 10.0
        old_time = self.test_time - 5.0
        self.collector.update_hand_qpos_left(old_time, np.array([5.0] * 6))
        obs, _ = self.collector.get_observation()
        states = obs[CommonKey.STATES.value]
        left_hand = states.get(InferenceConfigKey.LEFT_EEFHAND.value)
        self.assertIsNotNone(left_hand)
        self.assertTrue(np.allclose(left_hand, np.array([5.0] * 6)))

    def test_hand_image_none(self):
        """测试手部图像正常情况"""
        self.collector.update_hand_image_left(self.test_time, np.ones((480, 640, 3), dtype=np.uint8))
        obs, _ = self.collector.get_observation()
        images = obs[CommonKey.IMAGES.value]
        self.assertIn(ImageKey.CAM_HAND_LEFT.value, images)

    def test_joint_state_empty(self):
        """测试空关节状态"""
        self.collector.update_joint_state(self.test_time, [])
        obs, _ = self.collector.get_observation()
        states = obs[CommonKey.STATES.value]
        for key in [InferenceConfigKey.WAISTQPOS.value, InferenceConfigKey.ANKLEQPOS.value]:
            if key in states:
                self.assertEqual(len(states[key]), 0)

    def test_thread_safety(self):
        """测试多线程并发更新"""
        import concurrent.futures
        def update_data(i):
            ts = self.test_time + i * 0.001
            self.collector.update_hand_qpos_left(ts, np.array([float(i)] * 6))
            self.collector.update_joint_state(ts, [float(i)] * 30)
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(update_data, i) for i in range(100)]
            concurrent.futures.wait(futures)
        obs, _ = self.collector.get_observation()
        self.assertIsInstance(obs, dict)

    def test_config_changes(self):
        """测试配置变更"""
        self.collector.config.head_target_size = (320, 240)
        self.collector.config.hand_target_size = (320, 240)
        self.collector.update_hand_image_left(self.test_time, np.ones((480, 640, 3), dtype=np.uint8))
        obs, _ = self.collector.get_observation()
        images = obs[CommonKey.IMAGES.value]
        if ImageKey.CAM_HIGH.value in images:
            self.assertEqual(images[ImageKey.CAM_HIGH.value].shape, (240, 320, 3))
        if ImageKey.CAM_HAND_LEFT.value in images:
            self.assertEqual(images[ImageKey.CAM_HAND_LEFT.value].shape, (240, 320, 3))

    def test_all_data_none(self):
        """测试所有数据源都为空"""
        self.collector.set_kingfisher_provider(None)
        self.collector.hand_left_buffer.clear()
        self.collector.hand_qpos_left_buffer.clear()
        self.collector.joint_state = []
        obs, timestamp = self.collector.get_observation()
        self.assertIsInstance(obs, dict)
        images = obs[CommonKey.IMAGES.value]
        self.assertEqual(len(images), 0)

    def test_hand_images_different_sizes(self):
        """测试不同尺寸的手部图像输入"""
        small_image = np.ones((240, 320, 3), dtype=np.uint8)
        large_image = np.ones((960, 1280, 3), dtype=np.uint8)
        self.collector.update_hand_image_left(self.test_time, small_image)
        self.collector.update_hand_image_right(self.test_time, large_image)
        obs, _ = self.collector.get_observation()
        images = obs[CommonKey.IMAGES.value]
        if ImageKey.CAM_HAND_LEFT.value in images:
            self.assertEqual(images[ImageKey.CAM_HAND_LEFT.value].shape, (480, 640, 3))
        if ImageKey.CAM_HAND_RIGHT.value in images:
            self.assertEqual(images[ImageKey.CAM_HAND_RIGHT.value].shape, (480, 640, 3))

    # 二指夹模式测试

    def test_gripper_mode_observation(self):
        """测试二指夹模式下的观测收集"""
        config = ObservationConfig(
            tolerance_s=0.2,
            head_target_size=(640, 360),
            hand_target_size=(640, 480),
            use_hand_camera=True,
            end_effector_type=InferenceConfigKey.GRIPPER.value,
        )
        collector = ObservationCollector(config)
        collector.set_kingfisher_provider(self.mock_kingfisher)
        collector.update_hand_image_left(self.test_time, np.ones((480, 640, 3), dtype=np.uint8) * 128)
        collector.update_gripper_qpos_left(self.test_time, np.array([50.0], dtype=np.float32))
        collector.update_gripper_qpos_right(self.test_time, np.array([75.0], dtype=np.float32))
        collector.update_joint_state(self.test_time, list(range(30)))

        obs, _ = collector.get_observation()
        states = obs[CommonKey.STATES.value]

        # 二指夹数据存在
        self.assertIn(InferenceConfigKey.LEFT_EEFGRIPPER.value, states)
        self.assertIn(InferenceConfigKey.RIGHT_EEFGRIPPER.value, states)
        self.assertEqual(states[InferenceConfigKey.LEFT_EEFGRIPPER.value][0], 50.0)
        self.assertEqual(states[InferenceConfigKey.RIGHT_EEFGRIPPER.value][0], 75.0)

        # 灵巧手数据不应存在
        self.assertNotIn(InferenceConfigKey.LEFT_EEFHAND.value, states)
        self.assertNotIn(InferenceConfigKey.RIGHT_EEFHAND.value, states)

    def test_gripper_empty_buffer(self):
        """测试二指夹缓冲区为空"""
        self.collector.config.end_effector_type = InferenceConfigKey.GRIPPER.value
        self.collector.joint_state = list(range(30))

        obs, _ = self.collector.get_observation()
        states = obs[CommonKey.STATES.value]

        left_grip = states.get(InferenceConfigKey.LEFT_EEFGRIPPER.value)
        right_grip = states.get(InferenceConfigKey.RIGHT_EEFGRIPPER.value)
        # 空缓冲区 → 空数组
        self.assertEqual(len(left_grip), 0)
        self.assertEqual(len(right_grip), 0)

    def test_switch_end_effector_type(self):
        """测试运行时切换末端执行器类型"""
        # 先用手模式
        self.collector.update_hand_qpos_left(self.test_time, np.array([1.0] * 6))
        obs, _ = self.collector.get_observation()
        self.assertIn(InferenceConfigKey.LEFT_EEFHAND.value, obs[CommonKey.STATES.value])

        # 切换到二指夹
        self.collector.config.end_effector_type = InferenceConfigKey.GRIPPER.value
        self.collector.update_gripper_qpos_left(self.test_time, np.array([50.0]))
        obs, _ = self.collector.get_observation()
        self.assertIn(InferenceConfigKey.LEFT_EEFGRIPPER.value, obs[CommonKey.STATES.value])
        self.assertNotIn(InferenceConfigKey.LEFT_EEFHAND.value, obs[CommonKey.STATES.value])


if __name__ == "__main__":
    logging.info("=== 观测收集器示例（手模式） ===")
    obs_hand = example_usage(0.2, True, InferenceConfigKey.HAND.value)

    logging.info("\n=== 观测收集器示例（二指夹模式） ===")
    obs_gripper = example_usage(0.2, True, InferenceConfigKey.GRIPPER.value)

    logging.info("\n=== 单元测试 ===")
    unittest.main(argv=[""], verbosity=2)
