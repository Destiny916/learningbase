import unittest
from unittest.mock import Mock, patch
import numpy as np
import time
import sys
import os
import logging
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from act_async_infer_distributed_demo.scripts.w1_mapping import (
    InferenceConfigKey,
    w1qpos_names_map,
    hand_joint_names,
)

from act_async_infer_distributed_demo.scripts.utils_distributed import (
    TimedAction,
)
"""
    usage: 
        cd w1_act
        python -m act_async_infer_distributed_demo.scripts.unittest.test_diff_control_parts
"""


class TestClientActionExecutionWithControlMasking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        json_file_path = (
            "act_async_infer_distributed_demo/scripts/unittest/step_output.json"
        )
        try:
            cls.full_body_data = cls.load_full_body_json(json_file_path)
            logging.info(f"已加载完整全身数据，包含关节键: {list(cls.full_body_data.keys())}")

            cls.sample_action = cls._extract_first_timestep_action(cls.full_body_data)
            logging.info(f"样本动作维度: {cls.sample_action.shape}")

        except Exception as e:
            logging.warning(f"加载JSON文件失败: {e}")
            cls.full_body_data = cls._create_mock_full_body_data()
            cls.sample_action = cls._extract_first_timestep_action(cls.full_body_data)

        # 关节维度 —— 从 w1qpos_names_map 动态计算
        cls.joint_dims = {}
        for key_enum in [
            InferenceConfigKey.WAISTQPOS,
            InferenceConfigKey.LEFT_ARMQPOS,
            InferenceConfigKey.HEADQPOS,
            InferenceConfigKey.RIGHT_ARMQPOS,
            InferenceConfigKey.ANKLEQPOS,
            InferenceConfigKey.KNEEQPOS,
            InferenceConfigKey.BUTTOCKQPOS,
            InferenceConfigKey.LEFT_EEFHAND,
            InferenceConfigKey.RIGHT_EEFHAND,
            InferenceConfigKey.LEFT_EEFGRIPPER,
            InferenceConfigKey.RIGHT_EEFGRIPPER,
        ]:
            names = w1qpos_names_map.get(key_enum.value, [])
            cls.joint_dims[key_enum.value] = len(names)
        logging.info(f"关节维度: {cls.joint_dims}")

        # 定义不同控制部位的配置
        cls.control_configs = {
            "full_body_hand": {
                "description": "控制全身+灵巧手",
                "end_effector_type": InferenceConfigKey.HAND.value,
                "joint_order": [
                    InferenceConfigKey.WAISTQPOS.value,
                    InferenceConfigKey.LEFT_ARMQPOS.value,
                    InferenceConfigKey.HEADQPOS.value,
                    InferenceConfigKey.RIGHT_ARMQPOS.value,
                    InferenceConfigKey.ANKLEQPOS.value,
                    InferenceConfigKey.KNEEQPOS.value,
                    InferenceConfigKey.BUTTOCKQPOS.value,
                    InferenceConfigKey.LEFT_EEFHAND.value,
                    InferenceConfigKey.RIGHT_EEFHAND.value,
                ],
                "expected_body_joints": True,
                "expected_hand_joints": True,
                "expected_total_dim": sum(
                    cls.joint_dims[k] for k in [
                        InferenceConfigKey.WAISTQPOS.value,
                        InferenceConfigKey.LEFT_ARMQPOS.value,
                        InferenceConfigKey.HEADQPOS.value,
                        InferenceConfigKey.RIGHT_ARMQPOS.value,
                        InferenceConfigKey.ANKLEQPOS.value,
                        InferenceConfigKey.KNEEQPOS.value,
                        InferenceConfigKey.BUTTOCKQPOS.value,
                        InferenceConfigKey.LEFT_EEFHAND.value,
                        InferenceConfigKey.RIGHT_EEFHAND.value,
                    ]
                ),
            },
            "full_body_gripper": {
                "description": "控制全身+二指夹",
                "end_effector_type": InferenceConfigKey.GRIPPER.value,
                "joint_order": [
                    InferenceConfigKey.WAISTQPOS.value,
                    InferenceConfigKey.LEFT_ARMQPOS.value,
                    InferenceConfigKey.HEADQPOS.value,
                    InferenceConfigKey.RIGHT_ARMQPOS.value,
                    InferenceConfigKey.ANKLEQPOS.value,
                    InferenceConfigKey.KNEEQPOS.value,
                    InferenceConfigKey.BUTTOCKQPOS.value,
                    InferenceConfigKey.LEFT_EEFGRIPPER.value,
                    InferenceConfigKey.RIGHT_EEFGRIPPER.value,
                ],
                "expected_body_joints": True,
                "expected_hand_joints": False,
                "expected_gripper": True,
                "expected_total_dim": sum(
                    cls.joint_dims[k] for k in [
                        InferenceConfigKey.WAISTQPOS.value,
                        InferenceConfigKey.LEFT_ARMQPOS.value,
                        InferenceConfigKey.HEADQPOS.value,
                        InferenceConfigKey.RIGHT_ARMQPOS.value,
                        InferenceConfigKey.ANKLEQPOS.value,
                        InferenceConfigKey.KNEEQPOS.value,
                        InferenceConfigKey.BUTTOCKQPOS.value,
                        InferenceConfigKey.LEFT_EEFGRIPPER.value,
                        InferenceConfigKey.RIGHT_EEFGRIPPER.value,
                    ]
                ),
            },
            "hand_only": {
                "description": "仅控制灵巧手",
                "end_effector_type": InferenceConfigKey.HAND.value,
                "joint_order": [
                    InferenceConfigKey.LEFT_EEFHAND.value,
                    InferenceConfigKey.RIGHT_EEFHAND.value,
                ],
                "expected_body_joints": False,
                "expected_hand_joints": True,
                "expected_total_dim": (
                    cls.joint_dims[InferenceConfigKey.LEFT_EEFHAND.value]
                    + cls.joint_dims[InferenceConfigKey.RIGHT_EEFHAND.value]
                ),
            },
            "gripper_only": {
                "description": "仅控制二指夹",
                "end_effector_type": InferenceConfigKey.GRIPPER.value,
                "joint_order": [
                    InferenceConfigKey.LEFT_EEFGRIPPER.value,
                    InferenceConfigKey.RIGHT_EEFGRIPPER.value,
                ],
                "expected_body_joints": False,
                "expected_hand_joints": False,
                "expected_gripper": True,
                "expected_total_dim": (
                    cls.joint_dims[InferenceConfigKey.LEFT_EEFGRIPPER.value]
                    + cls.joint_dims[InferenceConfigKey.RIGHT_EEFGRIPPER.value]
                ),
            },
            "body_only": {
                "description": "仅控制身体",
                "end_effector_type": InferenceConfigKey.HAND.value,
                "joint_order": [
                    InferenceConfigKey.WAISTQPOS.value,
                    InferenceConfigKey.LEFT_ARMQPOS.value,
                    InferenceConfigKey.HEADQPOS.value,
                    InferenceConfigKey.RIGHT_ARMQPOS.value,
                    InferenceConfigKey.ANKLEQPOS.value,
                    InferenceConfigKey.KNEEQPOS.value,
                    InferenceConfigKey.BUTTOCKQPOS.value,
                ],
                "expected_body_joints": True,
                "expected_hand_joints": False,
                "expected_total_dim": sum(
                    cls.joint_dims[k] for k in [
                        InferenceConfigKey.WAISTQPOS.value,
                        InferenceConfigKey.LEFT_ARMQPOS.value,
                        InferenceConfigKey.HEADQPOS.value,
                        InferenceConfigKey.RIGHT_ARMQPOS.value,
                        InferenceConfigKey.ANKLEQPOS.value,
                        InferenceConfigKey.KNEEQPOS.value,
                        InferenceConfigKey.BUTTOCKQPOS.value,
                    ]
                ),
            },
            "left_hand_only": {
                "description": "仅控制左手",
                "end_effector_type": InferenceConfigKey.HAND.value,
                "joint_order": [InferenceConfigKey.LEFT_EEFHAND.value],
                "expected_body_joints": False,
                "expected_hand_joints": True,
                "expected_total_dim": cls.joint_dims[InferenceConfigKey.LEFT_EEFHAND.value],
            },
            "right_hand_only": {
                "description": "仅控制右手",
                "end_effector_type": InferenceConfigKey.HAND.value,
                "joint_order": [InferenceConfigKey.RIGHT_EEFHAND.value],
                "expected_body_joints": False,
                "expected_hand_joints": True,
                "expected_total_dim": cls.joint_dims[InferenceConfigKey.RIGHT_EEFHAND.value],
            },
            "upper_body_only": {
                "description": "仅控制上半身",
                "end_effector_type": InferenceConfigKey.HAND.value,
                "joint_order": [
                    InferenceConfigKey.WAISTQPOS.value,
                    InferenceConfigKey.LEFT_ARMQPOS.value,
                    InferenceConfigKey.HEADQPOS.value,
                    InferenceConfigKey.RIGHT_ARMQPOS.value,
                ],
                "expected_body_joints": True,
                "expected_hand_joints": False,
                "expected_total_dim": sum(
                    cls.joint_dims[k] for k in [
                        InferenceConfigKey.WAISTQPOS.value,
                        InferenceConfigKey.LEFT_ARMQPOS.value,
                        InferenceConfigKey.HEADQPOS.value,
                        InferenceConfigKey.RIGHT_ARMQPOS.value,
                    ]
                ),
            },
        }

    @staticmethod
    def load_full_body_json(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "qpos" not in data:
                raise ValueError("JSON文件必须包含'qpos'键")

            def convert_to_numpy(obj):
                if isinstance(obj, dict):
                    return {k: convert_to_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    if obj and isinstance(obj[0], list):
                        return np.array(obj, dtype=np.float32)
                    else:
                        return np.array(obj, dtype=np.float32)
                else:
                    return obj

            qpos_data = convert_to_numpy(data["qpos"])
            for key, value in qpos_data.items():
                if hasattr(value, "shape"):
                    logging.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            return qpos_data
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON文件 {filename} 未找到")
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON解析错误: {e}")

    @classmethod
    def _extract_first_timestep_action(cls, full_body_data):
        expected_order = [
            InferenceConfigKey.WAISTQPOS.value,
            InferenceConfigKey.LEFT_ARMQPOS.value,
            InferenceConfigKey.HEADQPOS.value,
            InferenceConfigKey.RIGHT_ARMQPOS.value,
            InferenceConfigKey.ANKLEQPOS.value,
            InferenceConfigKey.KNEEQPOS.value,
            InferenceConfigKey.BUTTOCKQPOS.value,
            InferenceConfigKey.LEFT_EEFHAND.value,
            InferenceConfigKey.RIGHT_EEFHAND.value,
        ]
        available_keys = list(full_body_data.keys())
        ordered_keys = [key for key in expected_order if key in available_keys]

        action_parts = []
        for key in ordered_keys:
            data = full_body_data[key]
            if hasattr(data, "shape") and len(data.shape) >= 2:
                action_parts.append(data[0])
            elif isinstance(data, np.ndarray) and len(data.shape) >= 2:
                action_parts.append(data[0])
            else:
                action_parts.append(data)
        if action_parts:
            return np.concatenate([arr for arr in action_parts])
        else:
            total_dim = sum(cls.joint_dims.get(k, 0) for k in expected_order)
            return np.random.randn(total_dim).astype(np.float32)

    @staticmethod
    def _create_mock_full_body_data():
        return {
            InferenceConfigKey.WAISTQPOS.value: np.random.randn(30, 1).astype(np.float32) * 0.1,
            InferenceConfigKey.LEFT_ARMQPOS.value: np.random.randn(30, 7).astype(np.float32) * 0.5,
            InferenceConfigKey.HEADQPOS.value: np.random.randn(30, 2).astype(np.float32) * 0.2,
            InferenceConfigKey.RIGHT_ARMQPOS.value: np.random.randn(30, 7).astype(np.float32) * 0.5,
            InferenceConfigKey.ANKLEQPOS.value: np.random.randn(30, 1).astype(np.float32) * 0.1,
            InferenceConfigKey.KNEEQPOS.value: np.random.randn(30, 1).astype(np.float32) * 0.1,
            InferenceConfigKey.BUTTOCKQPOS.value: np.random.randn(30, 1).astype(np.float32) * 0.1,
            InferenceConfigKey.LEFT_EEFHAND.value: (np.random.rand(30, 6).astype(np.float32) - 0.5) * 2.0,
            InferenceConfigKey.RIGHT_EEFHAND.value: (np.random.rand(30, 6).astype(np.float32) - 0.5) * 2.0,
        }

    def _create_mock_client(self, control_config_name="full_body_hand"):
        client = Mock()
        config = self.control_configs[control_config_name]

        client.cfg = Mock()
        client.cfg.end_effector_type = config.get("end_effector_type", InferenceConfigKey.HAND.value)
        client.cfg.save_exec_action = False
        client.end_effector_type = client.cfg.end_effector_type
        client.end_effector_positon_limit = (0.0, 100.0)
        client.current_joint_order = config["joint_order"]
        client.joint_dims = self.joint_dims
        client.w1qpos_names_map = w1qpos_names_map
        client.hand_joint_names = hand_joint_names

        return client, config

    # ─── 测试方法 ───────────────────────────────────

    def test_different_control_configs_with_full_body_data(self):
        """使用完整身体数据测试不同的控制配置"""

        for config_name, config in self.control_configs.items():
            with self.subTest(config_name=config_name):
                logging.info(f"测试配置: {config_name}")
                logging.info(f"描述: {config['description']}")

                client, config_data = self._create_mock_client(config_name)

                action_array = self._create_action_for_joint_order(
                    self.sample_action, config_data["joint_order"]
                )

                logging.info(f"动作维度: {action_array.shape}")
                logging.info(f"期望维度: {config_data['expected_total_dim']}")

                self.assertEqual(
                    len(action_array),
                    config_data["expected_total_dim"],
                    f"动作维度应该为{config_data['expected_total_dim']}，实际为{len(action_array)},"
                    f"config={config}",
                )

                timed_action = TimedAction(
                    timestamp=time.time(), timestep=1, action=action_array
                )

                with patch.object(client, "publish_joint_positions") as mock_publish_body:
                    hand_calls = []
                    gripper_calls = []

                    def mock_publish_hand(side_name=None, hand_position=None, clip=True):
                        hand_calls.append({"side": side_name, "position": hand_position})

                    def mock_publish_gripper(side_name=None, grip_position=None, clip=True):
                        pos = grip_position
                        if clip and hasattr(client, 'end_effector_positon_limit') and client.end_effector_positon_limit:
                            pos = float(np.clip(pos, *client.end_effector_positon_limit))
                        gripper_calls.append({"side": side_name, "position": pos})

                    client.publish_hand_positions = Mock(side_effect=mock_publish_hand)
                    client.publish_gripper_position = Mock(side_effect=mock_publish_gripper)

                    self._simulate_exec_action_with_new_interface(client, timed_action)

                    if config_data.get("expected_body_joints", False):
                        mock_publish_body.assert_called()
                    else:
                        mock_publish_body.assert_not_called()

                    if config_data.get("expected_hand_joints", False):
                        self.assertGreater(len(hand_calls), 0)
                    if config_data.get("expected_gripper", False):
                        self.assertGreater(len(gripper_calls), 0)

                logging.info(f"测试完成: {config_name}\n")

    def _create_action_for_joint_order(self, full_body_action, joint_order):
        """
        根据 joint_order 从样本动作中切片。
        缺失的 key（如二指夹不在 step_output.json 中）用随机数据补齐。
        """
        full_order = [
            InferenceConfigKey.WAISTQPOS.value,
            InferenceConfigKey.LEFT_ARMQPOS.value,
            InferenceConfigKey.HEADQPOS.value,
            InferenceConfigKey.RIGHT_ARMQPOS.value,
            InferenceConfigKey.ANKLEQPOS.value,
            InferenceConfigKey.KNEEQPOS.value,
            InferenceConfigKey.BUTTOCKQPOS.value,
            InferenceConfigKey.LEFT_EEFHAND.value,
            InferenceConfigKey.RIGHT_EEFHAND.value,
            InferenceConfigKey.LEFT_EEFGRIPPER.value,
            InferenceConfigKey.RIGHT_EEFGRIPPER.value,
        ]

        start_idx = 0
        joint_indices = {}
        for key in full_order:
            dim = self.joint_dims.get(key, 0)
            if dim > 0:
                joint_indices[key] = (start_idx, start_idx + dim)
                start_idx += dim

        action_parts = []
        for key in joint_order:
            dim = self.joint_dims.get(key, 0)
            if dim == 0:
                continue
            if key in joint_indices:
                start, end = joint_indices[key]
                if start < len(full_body_action) and end <= len(full_body_action):
                    action_parts.append(full_body_action[start:end])
                else:
                    # ✅ 数据不足 → 随机补齐
                    logging.info(f"  {key}: 样本数据不足 (需要 [{start}:{end}], 共 {len(full_body_action)} 维), 随机补齐")
                    action_parts.append(
                        np.random.randn(dim).astype(np.float32) * 0.5
                    )
            else:
                # key 不在 full_order 中 → 纯随机
                action_parts.append(np.random.randn(dim).astype(np.float32) * 0.5)

        if action_parts:
            return np.concatenate(action_parts)
        else:
            total_dim = sum(self.joint_dims.get(key, 0) for key in joint_order)
            return np.random.randn(total_dim).astype(np.float32)

    def _simulate_exec_action_with_new_interface(self, client, timed_action):
        """模拟 exec_action 方法逻辑 —— 与 robot_client_with_kingfisher.py 对齐"""
        action = timed_action.get_action()

        if hasattr(client, "current_joint_order") and client.current_joint_order:
            body_actions = {}
            hand_actions = {}
            gripper_actions = {}

            joint_dims = {}
            for key in client.current_joint_order:
                joint_names = w1qpos_names_map.get(key, [])
                joint_dims[key] = len(joint_names)

            start_idx = 0
            for key in client.current_joint_order:
                dim = joint_dims.get(key, 0)
                if dim > 0 and start_idx + dim <= len(action):
                    joint_data = action[start_idx : start_idx + dim]

                    if (InferenceConfigKey.LEFT_EEFHAND.value in key
                            or InferenceConfigKey.RIGHT_EEFHAND.value in key):
                        hand_actions[key] = joint_data
                    elif (InferenceConfigKey.LEFT_EEFGRIPPER.value in key
                          or InferenceConfigKey.RIGHT_EEFGRIPPER.value in key):
                        gripper_actions[key] = joint_data
                    else:
                        body_actions[key] = joint_data

                    start_idx += dim

            if body_actions:
                pub_names = []
                pub_pos = []
                for key, values in body_actions.items():
                    joint_names = w1qpos_names_map.get(key, [])
                    for i, value in enumerate(values):
                        if i < len(joint_names):
                            pub_names.append(joint_names[i])
                            pub_pos.append(float(value))
                client.publish_joint_positions(pub_names, pub_pos, clip=True)
                logging.info(f"发布身体关节: {len(pub_names)}个")

            if client.end_effector_type == InferenceConfigKey.HAND.value:
                left_hand = hand_actions.get(InferenceConfigKey.LEFT_EEFHAND.value)
                right_hand = hand_actions.get(InferenceConfigKey.RIGHT_EEFHAND.value)
                if left_hand is not None:
                    client.publish_hand_positions("left", left_hand, clip=True)
                if right_hand is not None:
                    client.publish_hand_positions("right", right_hand, clip=True)
            else:
                left_grip = gripper_actions.get(InferenceConfigKey.LEFT_EEFGRIPPER.value)
                right_grip = gripper_actions.get(InferenceConfigKey.RIGHT_EEFGRIPPER.value)
                if left_grip is not None:
                    client.publish_gripper_position(
                        "left", left_grip[0] if len(left_grip) > 0 else 0.0, clip=True
                    )
                if right_grip is not None:
                    client.publish_gripper_position(
                        "right", right_grip[0] if len(right_grip) > 0 else 0.0, clip=True
                    )

    # ─── 手部测试 ───────────────────────────────────

    def test_hand_control_with_clamping_new_interface(self):
        """测试手部控制接口的数值钳制逻辑"""
        logging.info("测试手部控制接口数值钳制")
        test_hand_data = np.array([-10.0, 150.0, 50.0, 50.0, 50.0, 50.0], dtype=np.float32)

        published_calls = []

        def mock_publish_hand(side_name=None, hand_position=None, clip=True):
            if hand_position is not None and side_name is not None:
                values = []
                for pos in hand_position:
                    val = float(pos)
                    if clip:
                        if val < 0.0:
                            val = 0.0
                        elif val > 100.0:
                            val = 100.0
                    values.append(val)
                if len(values) < 6:
                    values = values + [0.0] * (6 - len(values))
                elif len(values) > 6:
                    values = values[:6]
                published_calls.append({"side": side_name, "values": values})

        client, _ = self._create_mock_client("left_hand_only")
        client.publish_hand_positions = Mock(side_effect=mock_publish_hand)

        client.publish_hand_positions("left", test_hand_data, clip=True)

        self.assertEqual(len(published_calls), 1)
        values = published_calls[0]["values"]
        for val in values:
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 100.0)
        self.assertEqual(values[0], 0.0)
        self.assertEqual(values[1], 100.0)
        logging.info("手部控制接口数值钳制测试完成\n")

    def test_single_hand_control_configs(self):
        """测试单只手控制配置"""
        single_hand_configs = [
            ("left_hand_only", InferenceConfigKey.LEFT_EEFHAND.value),
            ("right_hand_only", InferenceConfigKey.RIGHT_EEFHAND.value),
        ]
        for config_name, expected_hand_key in single_hand_configs:
            with self.subTest(config_name=config_name):
                logging.info(f"测试单只手控制: {config_name}")
                client, config = self._create_mock_client(config_name)
                action_array = self._create_action_for_joint_order(
                    self.sample_action, config["joint_order"]
                )
                timed_action = TimedAction(
                    timestamp=time.time(), timestep=1, action=action_array
                )
                with patch.object(client, "publish_joint_positions") as mock_publish_body:
                    hand_calls = []

                    def mock_publish_hand(side_name=None, hand_position=None, clip=True):
                        hand_calls.append({"side": side_name, "position": hand_position})

                    client.publish_hand_positions = Mock(side_effect=mock_publish_hand)
                    self._simulate_exec_action_with_new_interface(client, timed_action)

                    mock_publish_body.assert_not_called()
                    self.assertEqual(len(hand_calls), 1)
                    self.assertIn(hand_calls[0]["side"], ["left", "right"])
                    logging.info(f"✓ {config_name} 测试通过\n")

    # ─── 二指夹测试 ───────────────────────────────────

    def test_gripper_control_config(self):
        """测试二指夹控制配置"""
        logging.info("测试二指夹控制配置")
        client, config = self._create_mock_client("gripper_only")
        self.assertEqual(client.end_effector_type, InferenceConfigKey.GRIPPER.value)

        action_array = self._create_action_for_joint_order(
            self.sample_action, config["joint_order"]
        )
        timed_action = TimedAction(
            timestamp=time.time(), timestep=1, action=action_array
        )

        with patch.object(client, "publish_joint_positions") as mock_publish_body:
            gripper_calls = []

            def mock_publish_gripper(side_name=None, grip_position=None, clip=True):
                gripper_calls.append({"side": side_name, "position": grip_position})

            client.publish_gripper_position = Mock(side_effect=mock_publish_gripper)
            self._simulate_exec_action_with_new_interface(client, timed_action)

            mock_publish_body.assert_not_called()
            self.assertEqual(len(gripper_calls), 2)
            sides = sorted([c["side"] for c in gripper_calls])
            self.assertEqual(sides, ["left", "right"])
            logging.info(f"✓ 二指夹控制测试通过: sides={sides}\n")

    def test_gripper_clamping(self):
        """测试二指夹数值钳制"""
        logging.info("测试二指夹数值钳制")

        published_calls = []

        def mock_publish_gripper(side_name=None, grip_position=None, clip=True):
            pos = grip_position
            if clip:
                pos = float(np.clip(pos, 0.0, 100.0))
            published_calls.append({"side": side_name, "position": pos})

        client, _ = self._create_mock_client("gripper_only")
        client.end_effector_positon_limit = (0.0, 100.0)
        client.publish_gripper_position = Mock(side_effect=mock_publish_gripper)

        client.publish_gripper_position("left", 150.0, clip=True)
        client.publish_gripper_position("right", -10.0, clip=True)

        self.assertEqual(len(published_calls), 2)
        self.assertEqual(published_calls[0]["position"], 100.0)
        self.assertEqual(published_calls[1]["position"], 0.0)
        logging.info("二指夹数值钳制测试完成\n")

    # ─── 通用测试 ───────────────────────────────────

    def test_action_parsing_with_new_interface(self):
        """测试动作解析逻辑"""
        logging.info("测试动作解析逻辑")
        client, config = self._create_mock_client("full_body_hand")
        action_array = self._create_action_for_joint_order(
            self.sample_action, config["joint_order"]
        )
        parsed_joints = self._parse_action_array(action_array, client.current_joint_order)
        self.assertEqual(len(parsed_joints), len(client.current_joint_order))

        total_dim = 0
        for key in client.current_joint_order:
            self.assertIn(key, parsed_joints)
            joint_data = parsed_joints[key]
            expected_dim = self.joint_dims.get(key, 0)
            self.assertEqual(len(joint_data), expected_dim)
            total_dim += expected_dim
            logging.info(f"  {key}: 维度={len(joint_data)}")

        self.assertEqual(total_dim, len(action_array))
        logging.info("动作解析逻辑测试完成\n")

    def _parse_action_array(self, action_array, joint_order):
        parsed = {}
        start_idx = 0
        for key in joint_order:
            dim = self.joint_dims.get(key, 0)
            if dim > 0 and start_idx + dim <= len(action_array):
                parsed[key] = action_array[start_idx : start_idx + dim]
                start_idx += dim
        return parsed

    def test_hand_position_message_structure(self):
        """测试手部控制消息结构"""
        logging.info("测试手部控制消息结构")
        test_hand_data = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=np.float32)

        published_messages = []

        def mock_publish_hand(side_name=None, hand_position=None, clip=True):
            if hand_position is not None and side_name is not None:
                msg_values = []
                for pos in hand_position:
                    val = float(pos)
                    if clip:
                        if val < 0.0:
                            val = 0.0
                        elif val > 100.0:
                            val = 100.0
                    msg_values.append(val)
                if len(msg_values) < 6:
                    msg_values = msg_values + [0.0] * (6 - len(msg_values))
                elif len(msg_values) > 6:
                    msg_values = msg_values[:6]
                published_messages.append({
                    "side": side_name, "values": msg_values, "num_values": len(msg_values),
                })

        client, _ = self._create_mock_client("left_hand_only")
        client.publish_hand_positions = Mock(side_effect=mock_publish_hand)
        client.publish_hand_positions("left", test_hand_data, clip=True)

        self.assertEqual(len(published_messages), 1)
        msg = published_messages[0]
        self.assertEqual(msg["side"], "left")
        self.assertEqual(msg["num_values"], 6)
        for i in range(6):
            self.assertEqual(msg["values"][i], test_hand_data[i])
        logging.info("手部控制消息结构测试完成\n")

    def test_control_configuration_combinations_new_interface(self):
        """测试控制配置组合"""
        test_cases = [
            {
                "name": "双臂+灵巧手控制",
                "end_effector_type": InferenceConfigKey.HAND.value,
                "joint_order": [
                    InferenceConfigKey.LEFT_ARMQPOS.value,
                    InferenceConfigKey.RIGHT_ARMQPOS.value,
                    InferenceConfigKey.LEFT_EEFHAND.value,
                    InferenceConfigKey.RIGHT_EEFHAND.value,
                ],
                "expected_body_dim": 7 + 7,
                "expected_hand_calls": 2,
                "expected_gripper_calls": 0,
            },
            {
                "name": "双臂+二指夹控制",
                "end_effector_type": InferenceConfigKey.GRIPPER.value,
                "joint_order": [
                    InferenceConfigKey.LEFT_ARMQPOS.value,
                    InferenceConfigKey.RIGHT_ARMQPOS.value,
                    InferenceConfigKey.LEFT_EEFGRIPPER.value,
                    InferenceConfigKey.RIGHT_EEFGRIPPER.value,
                ],
                "expected_body_dim": 7 + 7,
                "expected_hand_calls": 0,
                "expected_gripper_calls": 2,
            },
            {
                "name": "腰部+头部控制",
                "end_effector_type": InferenceConfigKey.HAND.value,
                "joint_order": [
                    InferenceConfigKey.WAISTQPOS.value,
                    InferenceConfigKey.HEADQPOS.value,
                ],
                "expected_body_dim": 1 + 2,
                "expected_hand_calls": 0,
                "expected_gripper_calls": 0,
            },
            {
                "name": "下半身控制",
                "end_effector_type": InferenceConfigKey.HAND.value,
                "joint_order": [
                    InferenceConfigKey.ANKLEQPOS.value,
                    InferenceConfigKey.KNEEQPOS.value,
                    InferenceConfigKey.BUTTOCKQPOS.value,
                ],
                "expected_body_dim": 1 + 1 + 1,
                "expected_hand_calls": 0,
                "expected_gripper_calls": 0,
            },
        ]

        for test_case in test_cases:
            with self.subTest(test_case["name"]):
                logging.info(f"测试组合: {test_case['name']}")

                client = Mock()
                client.cfg = Mock()
                client.cfg.end_effector_type = test_case["end_effector_type"]
                client.end_effector_type = test_case["end_effector_type"]
                client.end_effector_positon_limit = (0.0, 100.0)
                client.current_joint_order = test_case["joint_order"]
                client.joint_dims = self.joint_dims
                client.w1qpos_names_map = w1qpos_names_map
                client.hand_joint_names = hand_joint_names

                total_dim = test_case["expected_body_dim"]
                if InferenceConfigKey.LEFT_EEFHAND.value in test_case["joint_order"]:
                    total_dim += self.joint_dims[InferenceConfigKey.LEFT_EEFHAND.value]
                if InferenceConfigKey.RIGHT_EEFHAND.value in test_case["joint_order"]:
                    total_dim += self.joint_dims[InferenceConfigKey.RIGHT_EEFHAND.value]
                if InferenceConfigKey.LEFT_EEFGRIPPER.value in test_case["joint_order"]:
                    total_dim += self.joint_dims[InferenceConfigKey.LEFT_EEFGRIPPER.value]
                if InferenceConfigKey.RIGHT_EEFGRIPPER.value in test_case["joint_order"]:
                    total_dim += self.joint_dims[InferenceConfigKey.RIGHT_EEFGRIPPER.value]

                action_array = np.random.randn(total_dim).astype(np.float32)
                timed_action = TimedAction(
                    timestamp=time.time(), timestep=1, action=action_array
                )

                with patch.object(client, "publish_joint_positions") as mock_publish_body:
                    hand_calls = []
                    gripper_calls = []

                    client.publish_hand_positions = Mock(
                        side_effect=lambda side_name=None, hand_position=None, clip=True:
                            hand_calls.append({"side": side_name, "position": hand_position})
                    )
                    client.publish_gripper_position = Mock(
                        side_effect=lambda side_name=None, grip_position=None, clip=True:
                            gripper_calls.append({"side": side_name, "position": grip_position})
                    )

                    self._simulate_exec_action_with_new_interface(client, timed_action)
                    self.assertEqual(len(action_array), total_dim)
                    self.assertEqual(len(hand_calls), test_case["expected_hand_calls"])
                    self.assertEqual(len(gripper_calls), test_case["expected_gripper_calls"])
                    logging.info(f"✓ 发布验证通过\n")


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2)
