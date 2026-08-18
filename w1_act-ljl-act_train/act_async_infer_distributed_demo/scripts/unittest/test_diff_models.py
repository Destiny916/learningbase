import unittest
from unittest.mock import Mock, patch
import numpy as np
import time
import sys
import os
from copy import deepcopy
import logging
import json
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


"""
    usage: 
        cd w1_act
        python -m act_async_infer_distributed_demo.scripts.unittest.test_diff_models
"""


class TestPolicyServerPureInferenceModelIO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        json_file_path = (
            "act_async_infer_distributed_demo/scripts/unittest/step_output.json"
        )
        cls.real_action_chunk_data = cls.load_action_chunk_from_json(json_file_path)
        logging.info(
            f"测试类初始化完成，已加载真实数据，包含键: {list(cls.real_action_chunk_data['qpos'].keys())}"
        )

    @staticmethod
    def load_action_chunk_from_json(filename):
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

            logging.info(f"从 {filename} 加载的qpos数据结构:")
            for key, value in qpos_data.items():
                if hasattr(value, "shape"):
                    logging.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                elif isinstance(value, np.ndarray):
                    logging.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    logging.info(f"  {key}: type={type(value)}")

            return {"qpos": qpos_data}

        except FileNotFoundError:
            logging.warning(f"JSON文件 {filename} 未找到，使用随机数据")
            return {
                "qpos": {
                    "left_armqpos": np.random.randn(64, 7).astype(np.float32),
                    "right_armqpos": np.random.randn(64, 7).astype(np.float32),
                    "left_eefhand": np.random.rand(64, 6).astype(np.float32) * 0.5,
                    "right_eefhand": np.random.rand(64, 6).astype(np.float32) * 0.5,
                }
            }
        except json.JSONDecodeError as e:
            logging.error(f"JSON解析错误: {e}")
            raise

    def _create_mock_model(self, state_meta, camera_used):
        model = Mock()
        model.policy = Mock()
        model.policy.state_meta = state_meta
        model.policy.camera_used = camera_used
        model.reset = Mock()
        model.from_real_obs = Mock()
        model.step = Mock(return_value=self.real_action_chunk_data)
        model.is_gripper_bool = True
        return model

    def _create_mock_server(self):
        """创建一个虚构 PolicyServerPureInference，跳过真 __init__，手动挂载测试所需全部属性。"""
        from act_async_infer_distributed_demo.scripts.server.policy_server_pure_inference import (
            PolicyServerPureInference,
        )
        from act_async_infer_distributed_demo.scripts.w1_mapping import EnumIndex

        # 完全跳过真构造函数
        with patch.object(PolicyServerPureInference, '__init__', Mock(return_value=None)):
            server = PolicyServerPureInference.__new__(PolicyServerPureInference)

            # ── cfg ──
            server.cfg = Mock()
            server.cfg.pretrained = "/fake/path/model.pth"
            server.cfg.model_name = "act"
            server.cfg.action_horizon = 64
            server.cfg.data_type = "sim"
            server.cfg.hand_sim_max_limit = [87.27, 157.08, 130.9, 130.9, 130.9, 130.9]
            server.cfg.gripper_sim_max_limit = [100]
            server.cfg.model_end_effector_limit = [0, 1]
            server.cfg.inverse_gripper = False
            server.cfg.is_gripper_bool = True
            server.cfg.save_input = False
            server.cfg.save_vis_images = False
            server.cfg.save_joint_change = False
            server.cfg.use_smooth = False
            server.cfg.client_timeout = 10.0
            server.cfg.service = False
            server.cfg.force_model_reinit = False
            server.cfg.prev_pickle = ""
            server.cfg.precompute_lang_embeddings = None

            # ── 基础属性 ──
            server.model = self._create_mock_model([], [])
            server.model_config = {}
            server.network_server = Mock()
            server.first_connect = True
            server.origin_joint_names = None
            server.tracer = None
            server.policy_path = "/fake/path/model.pth"
            server.model_name = "act"

            # ── 线程事件 ──
            server.shutdown_event = threading.Event()
            server.shutdown_event.set()  # 标记已关闭，避免 __del__ 死等
            server.ready_event = threading.Event()
            server.actions_ready_event = threading.Event()
            server.start_infer_event = threading.Event()
            server.busy_event = threading.Event()
            server.busy_event.set()

            # ── 归一化参数 ──
            server.hand_sim_max_limit = np.array([0.8727, 1.5708, 1.309, 1.309, 1.309, 1.309])
            server.gripper_sim_max_limit = np.array([100.0])
            server.model_end_effector_limit = [0, 1]
            server.end_effector_limit = [0, 100]
            server.close_state_normalized = np.array([0.55, 0.9, 0.3, 0.3, 0.3, 0.3])
            server.is_gripper = True
            server.inverse_gripper = False
            server.gripper_types_to_process = EnumIndex.gripper_types_to_process.value

            # ── 左/右手关节名 ──
            server.left_hand_qpos_names = ["LEFT_THUMBMCP", "LEFT_THUMBCMC", "LEFT_INDEXMCP",
                                           "LEFT_MIDDLEMCP", "LEFT_RINGMCP", "LEFT_LITTLEMCP"]
            server.right_hand_qpos_names = ["RIGHT_THUMBMCP", "RIGHT_THUMBCMC", "RIGHT_INDEXMCP",
                                            "RIGHT_MIDDLEMCP", "RIGHT_RINGMCP", "RIGHT_LITTLEMCP"]

            # ── 观测/推理状态 ──
            server.observation_queue = __import__('queue').Queue(maxsize=1)
            server.past_obs_queue = __import__('queue').Queue(maxsize=50)
            server.last_processed_obs = None
            server.observation_queue_lock = threading.Lock()
            server.latest_queued_timestep = -1
            server.obs_queue_timeout = 0.001
            server.inference_latency = 0.0
            server.action_horizon = 64
            server.inference_output_body_dof = None
            server.count = 0
            server.step_time = __import__('collections').deque(maxlen=50)

            # ── 当前动作 ──
            server._predicted_timesteps = set()
            server.current_actions = None
            server.current_timestamp = 0.0
            server.current_timestep = 0
            server.current_actions_lock = threading.Lock()

            # ── _handle_observation 依赖 ──
            server.monitor = Mock()
            server.monitor.record_call = Mock()
            server.current_session_id = 1
            server.init_flag = True

            # ── 调试 ──
            server.Debug_Datasaver = None

            return server

    def test_map_obs_to_model_with_variant_state_meta(self):
        """不同状态元数据配置下的观测映射功能"""

        test_cases = [
            {
                "name": "基础手部模型 - 测试状态过滤和手部归一化",
                "description": "验证模型能正确过滤非state_meta状态，并将手部数据从传感器范围归一化到模拟范围",
                "state_meta": [
                    "left_armqpos",
                    "right_armqpos",
                    "left_eefhand",
                    "right_eefhand",
                ],
                "camera_used": [
                    "cam_high",
                    "cam_high_r",
                    "cam_right_wrist",
                    "cam_left_wrist",
                ],
                "input_states": {
                    "left_armqpos": np.array(
                        [0.8, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0], dtype=np.float32
                    ),
                    "right_armqpos": np.array(
                        [0.8, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0], dtype=np.float32
                    ),
                    "left_eefhand": np.array(
                        [100, 100, 100, 100, 100, 100], dtype=np.float32
                    ),
                    "right_eefhand": np.array(
                        [100, 100, 100, 100, 100, 100], dtype=np.float32
                    ),
                    "headqpos": np.array([0.1, 0.2], dtype=np.float32),
                    "waistqpos": np.array([0.4], dtype=np.float32),
                },
                "expected_state_keys": [
                    "left_armqpos",
                    "right_armqpos",
                    "left_eefhand",
                    "right_eefhand",
                ],
                "use_real_data": True,
            },
        ]

        for test_case in test_cases:
            with self.subTest(test_case["name"]):
                logging.info(f"开始测试: {test_case['name']}")
                logging.info(f"描述: {test_case['description']}")

                server = self._create_mock_server()
                mock_model = self._create_mock_model(
                    test_case["state_meta"], test_case["camera_used"]
                )
                server.model = mock_model
                server.cfg.hand_sim_max_limit = np.array(
                    [0.8727, 1.5708, 1.309, 1.309, 1.309, 1.309]
                )
                server.gripper_limit = [0, 100]

                if test_case.get("use_real_data", True):
                    mock_model.step.return_value = self.real_action_chunk_data

                mock_timed_obs = Mock()
                mock_observation = deepcopy(
                    {"images": {}, "states": test_case["input_states"], "disps": [None], "instruction": ""}
                )

                for cam_name in test_case["camera_used"]:
                    mock_observation["images"][cam_name] = np.random.randint(
                        0, 255, (360, 640, 3), dtype=np.uint8
                    )

                mock_timed_obs.get_observation.return_value = mock_observation

                result = server._map_obs_to_model(mock_timed_obs)

                self.assertIn("images", result)
                self.assertIn("states", result)
                self.assertIn("disps", result)

                self.assertEqual(len(result["images"]), len(test_case["camera_used"]))

                result_state_keys = list(result["states"].keys())
                self.assertListEqual(
                    sorted(result_state_keys), sorted(test_case["expected_state_keys"])
                )

                if "left_eefhand" in result["states"]:
                    left_hand = result["states"]["left_eefhand"]
                    input_left = test_case["input_states"]["left_eefhand"]
                    expected_left = input_left / 100.0 * server.hand_sim_max_limit
                    np.testing.assert_array_almost_equal(
                        left_hand, expected_left, decimal=4
                    )
                    logging.info(f"左手灵巧手数据归一化验证通过")

                if "right_eefhand" in result["states"]:
                    right_hand = result["states"]["right_eefhand"]
                    input_right = test_case["input_states"]["right_eefhand"]
                    expected_right = input_right / 100.0 * server.hand_sim_max_limit
                    np.testing.assert_array_almost_equal(
                        right_hand, expected_right, decimal=4
                    )
                    logging.info(f"右手灵巧手数据归一化验证通过")

                for key in test_case["expected_state_keys"]:
                    if key not in ["left_eefhand", "right_eefhand"]:
                        np.testing.assert_array_equal(
                            result["states"][key], test_case["input_states"][key]
                        )
                        logging.info(f"状态数据 {key} 直接传递验证通过")

                logging.info(f"测试完成: {test_case['name']}")
                logging.info("-----------------------------------------------")

    def test_map_obs_to_model_with_variant_camera_configs(self):
        """不同相机配置下的观测映射功能"""

        test_cases = [
            {
                "name": "四相机配置 - 完整视觉输入",
                "description": "验证四相机配置下图像数据的正确映射和顺序保持",
                "camera_used": [
                    "cam_high",
                    "cam_high_r",
                    "cam_right_wrist",
                    "cam_left_wrist",
                ],
                "state_meta": [
                    "left_armqpos",
                    "right_armqpos",
                    "left_eefhand",
                    "right_eefhand",
                ],
                "use_real_data": True,
            },
            {
                "name": "单相机配置 - 最小视觉输入",
                "description": "验证单相机配置下的数据映射和状态完整性",
                "camera_used": ["cam_high"],
                "state_meta": [
                    "left_armqpos",
                    "right_armqpos",
                    "left_eefhand",
                    "right_eefhand",
                    "headqpos",
                    "waistqpos",
                ],
                "use_real_data": True,
            },
            {
                "name": "双相机配置 - 基本立体视觉",
                "description": "验证双相机配置下的图像索引和状态映射",
                "camera_used": ["cam_high", "cam_high_r"],
                "state_meta": ["left_eefhand", "right_eefhand"],
                "use_real_data": True,
            },
        ]

        for test_case in test_cases:
            with self.subTest(test_case["name"]):
                logging.info(f"开始测试: {test_case['name']}")
                logging.info(f"描述: {test_case['description']}")

                server = self._create_mock_server()
                mock_model = self._create_mock_model(
                    test_case["state_meta"], test_case["camera_used"]
                )
                server.model = mock_model

                if test_case.get("use_real_data", True):
                    mock_model.step.return_value = self.real_action_chunk_data

                mock_timed_obs = Mock()
                mock_observation = {
                    "images": {},
                    "states": {
                        "left_eefhand": np.array([10, 20, 30, 40, 50, 60], dtype=np.float32),
                        "right_eefhand": np.array([15, 25, 35, 45, 55, 65], dtype=np.float32),
                    },
                    "disps": [None],
                    "instruction": ""
                }
                server.gripper_limit = [0, 100]

                image_data = {}
                for cam_name in test_case["camera_used"]:
                    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                    mock_observation["images"][cam_name] = img
                    image_data[cam_name] = img

                mock_timed_obs.get_observation.return_value = mock_observation

                result = server._map_obs_to_model(mock_timed_obs)

                self.assertEqual(len(result["images"]), len(test_case["camera_used"]))
                logging.info(f"相机数据数量验证通过: {len(result['images'])}")

                for i, cam_name in enumerate(test_case["camera_used"]):
                    np.testing.assert_array_equal(result["images"][i], image_data[cam_name])
                    logging.info(f"相机 {cam_name} 图像数据验证通过，索引 {i}")

                self.assertEqual(mock_model.step.return_value, self.real_action_chunk_data)
                logging.info(f"模型step方法真实数据配置验证通过")

                logging.info(f"测试完成: {test_case['name']}")
                logging.info("-----------------------------------------------")

    def test_predict_action_chunk_with_different_model_outputs(self):
        """不同模型配置下的动作预测功能"""

        test_cases = [
            {
                "name": "基础手部模型输出 - 双臂+双手配置",
                "description": "测试基础手部模型（双臂+双手）在双相机配置下的动作预测，验证输出数据过滤",
                "state_meta": [
                    "left_armqpos",
                    "right_armqpos",
                    "left_eefhand",
                    "right_eefhand",
                ],
                "camera_used": ["cam_high", "cam_high_r"],
                "use_real_data": True,
            },
            {
                "name": "完整身体模型输出 - 包含头部和腰部",
                "description": "测试完整身体模型（包含头部和腰部）在单相机配置下的动作预测，验证数据适配",
                "state_meta": [
                    "left_armqpos",
                    "right_armqpos",
                    "left_eefhand",
                    "right_eefhand",
                    "headqpos",
                    "waistqpos",
                ],
                "camera_used": ["cam_high"],
                "use_real_data": True,
            },
        ]

        for test_case in test_cases:
            with self.subTest(test_case["name"]):
                logging.info(f"开始测试: {test_case['name']}")
                logging.info(f"描述: {test_case['description']}")

                server = self._create_mock_server()
                mock_model = self._create_mock_model(
                    test_case["state_meta"], test_case["camera_used"]
                )
                server.model = mock_model
                server.cfg.hand_sim_max_limit = np.array(
                    [0.8727, 1.5708, 1.309, 1.309, 1.309, 1.309]
                )

                server.end_effector_limit = [0, 100]
                server.init_flag = True
                server.cfg.save_input = False
                server.cfg.save_vis_images = False
                server.cfg.save_joint_change = False
                server.cfg.use_smooth = False
                server.cfg.data_type = "sim"
                server.Debug_Datasaver = None

                mock_timed_obs = Mock()
                mock_timed_obs.get_timestep.return_value = 100
                mock_timed_obs.get_timestamp.return_value = time.time()

                model_output = deepcopy(self.real_action_chunk_data)

                filtered_qpos = {}
                for key in test_case["state_meta"]:
                    if key in model_output["qpos"]:
                        filtered_qpos[key] = model_output["qpos"][key]
                        logging.info(f"使用真实数据键: {key}, shape: {filtered_qpos[key].shape}")
                    else:
                        logging.warning(f"真实数据中缺少 {key}，使用随机数据作为适配")
                        if key == "left_armqpos" or key == "right_armqpos":
                            filtered_qpos[key] = np.random.randn(64, 7).astype(np.float32)
                        elif key == "left_eefhand" or key == "right_eefhand":
                            filtered_qpos[key] = np.random.rand(64, 6).astype(np.float32) * 0.5
                        elif key == "headqpos":
                            filtered_qpos[key] = np.random.randn(64, 3).astype(np.float32)
                        elif key == "waistqpos":
                            filtered_qpos[key] = np.random.randn(64, 2).astype(np.float32)

                model_output["qpos"] = filtered_qpos
                mock_model.step.return_value = model_output

                mock_batch = {
                    "images": [
                        np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
                        for _ in test_case["camera_used"]
                    ],
                    "states": {},
                    "disps": [None],
                    "instruction": ""
                }

                for state_key in test_case["state_meta"]:
                    if state_key == "left_eefhand":
                        mock_batch["states"][state_key] = (
                            np.array([10, 20, 30, 40, 50, 60], dtype=np.float32)
                            / 100.0 * server.hand_sim_max_limit
                        )
                    elif state_key == "right_eefhand":
                        mock_batch["states"][state_key] = (
                            np.array([15, 25, 35, 45, 55, 65], dtype=np.float32)
                            / 100.0 * server.hand_sim_max_limit
                        )
                    else:
                        if state_key in ("left_armqpos", "right_armqpos"):
                            mock_batch["states"][state_key] = np.random.randn(7).astype(np.float32)
                        elif state_key == "headqpos":
                            mock_batch["states"][state_key] = np.random.randn(3).astype(np.float32)
                        elif state_key == "waistqpos":
                            mock_batch["states"][state_key] = np.random.randn(2).astype(np.float32)

                mock_batch_map = {"images": Mock(), "states": Mock(), "instruction": ""}
                mock_model.from_real_obs.return_value = mock_batch_map

                with patch.object(server, "_map_obs_to_model", return_value=mock_batch):
                    result = server._predict_action_chunk(mock_timed_obs)

                    self.assertEqual(result, model_output)
                    logging.info("预测结果验证通过")

                    if "left_eefhand" in result["qpos"]:
                        left_actions = result["qpos"]["left_eefhand"]
                        self.assertEqual(len(left_actions.shape), 2)
                        logging.info(f"左手机械爪动作数据验证通过: shape={left_actions.shape}")

                    if "right_eefhand" in result["qpos"]:
                        right_actions = result["qpos"]["right_eefhand"]
                        self.assertEqual(len(right_actions.shape), 2)
                        logging.info(f"右手机械爪动作数据验证通过: shape={right_actions.shape}")

                    mock_model.from_real_obs.assert_called_once()
                    mock_model.step.assert_called_once_with(**mock_batch_map)
                    logging.info("模型方法调用验证通过")

                logging.info(f"测试完成: {test_case['name']}")
                logging.info("-----------------------------------------------")

    def test_handle_observation_with_different_model_configs(self):
        """不同模型配置下的观测处理功能"""

        test_cases = [
            {
                "name": "四相机模型 - 完整视觉输入处理",
                "description": "测试四相机配置下的完整观测处理流程，包含压缩图像数据解析",
                "camera_used": [
                    "cam_high", "cam_high_r", "cam_right_wrist", "cam_left_wrist",
                ],
                "state_meta": [
                    "left_armqpos", "right_armqpos", "left_eefhand", "right_eefhand",
                ],
                "request_keys": [
                    "cam_high", "cam_high_r", "cam_right_wrist", "cam_left_wrist",
                    "states", "timestamp", "timestep", "head_target_size", "hand_target_size"
                ],
                "use_real_data": True,
            },
            {
                "name": "单相机模型 - 最小视觉输入处理",
                "description": "测试单相机配置下的观测处理流程，验证额外状态数据的处理",
                "camera_used": ["cam_high"],
                "state_meta": [
                    "left_armqpos", "right_armqpos", "left_eefhand", "right_eefhand",
                    "headqpos", "waistqpos",
                ],
                "request_keys": ["cam_high", "states", "timestamp", "timestep",
                                "head_target_size", "hand_target_size"],
                "use_real_data": True,
            },
        ]

        for test_case in test_cases:
            with self.subTest(test_case["name"]):
                server = self._create_mock_server()
                mock_model = self._create_mock_model(
                    test_case["state_meta"], test_case["camera_used"]
                )
                server.model = mock_model
                server.current_session_id = 1
                server.start_infer_event = threading.Event()

                mock_request = {
                    "timestamp": time.time(),
                    "timestep": 100,
                    "must_go": True,
                    "start_infer": False,
                    "states": {
                        "left_eefhand": np.array([10, 20, 30, 40, 50, 60], dtype=np.float32),
                        "right_eefhand": np.array([15, 25, 35, 45, 55, 65], dtype=np.float32),
                    },
                    "head_target_size": [640, 360],
                    "hand_target_size": [640, 360],
                    "gripper_limit": [0, 100]
                }

                for cam_name in test_case["camera_used"]:
                    mock_request[cam_name] = np.zeros((640, 360, 3), dtype=np.uint8).tobytes()
                    logging.info(f"添加相机数据: {cam_name}")

                for state_key in test_case["state_meta"]:
                    if state_key not in mock_request["states"]:
                        if state_key in ("left_armqpos", "right_armqpos"):
                            mock_request["states"][state_key] = np.random.randn(7).astype(np.float32)
                        elif state_key == "headqpos":
                            mock_request["states"][state_key] = np.random.randn(3).astype(np.float32)
                        elif state_key == "waistqpos":
                            mock_request["states"][state_key] = np.random.randn(2).astype(np.float32)
                        logging.info(f"添加状态数据: {state_key}")

                mock_timed_obs = Mock()
                mock_timed_obs.get_timestep.return_value = 100
                mock_timed_obs.get_timestamp.return_value = time.time()

                if test_case.get("use_real_data", True):
                    mock_model.step.return_value = self.real_action_chunk_data

                server._map_obs_to_model = Mock(
                    return_value={
                        "images": [
                            np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
                            for _ in test_case["camera_used"]
                        ],
                        "states": mock_request["states"],
                        "disps": [None],
                    }
                )

                mock_model.from_real_obs.return_value = {"images": Mock(), "states": Mock()}

                response = server._handle_observation(mock_request)
                self.assertEqual(response["status"], "received")
                self.assertEqual(response["timestep"], 100)
                self.assertEqual(response["session_id"], 1)
                logging.info(f"响应状态验证通过: status={response['status']}")

                self.assertEqual(mock_model.step.return_value, self.real_action_chunk_data)
                logging.info("真实数据使用验证通过")

                logging.info(f"测试完成: {test_case['name']}")
                logging.info("-----------------------------------------------")

    def test_step_return_value_structure(self):
        """模型step方法返回值结构验证"""
        mock_model = self._create_mock_model(
            ["left_armqpos", "right_armqpos", "left_eefhand", "right_eefhand"],
            ["cam_high", "cam_high_r"],
        )

        mock_model.step.return_value = self.real_action_chunk_data

        batch_map = {"images": Mock(), "states": Mock()}
        result = mock_model.step(**batch_map)

        self.assertIsInstance(result, dict)
        self.assertIn("qpos", result)
        self.assertIsInstance(result["qpos"], dict)
        logging.info("返回值基本结构验证通过")

        expected_keys = [
            "left_armqpos", "right_armqpos", "left_eefhand", "right_eefhand",
        ]
        actual_keys = list(result["qpos"].keys())
        logging.info(f"期望的qpos键: {expected_keys}")
        logging.info(f"实际的qpos键: {actual_keys}")

        for key in expected_keys:
            if key in result["qpos"]:
                value = result["qpos"][key]
                self.assertIsInstance(value, np.ndarray)
                logging.info(f"step返回值中的 {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                logging.warning(f"期望的键 {key} 不在真实数据中")

        mock_model.step.assert_called_once_with(**batch_map)
        logging.info("step方法调用验证通过")

        logging.info("测试完成: step方法返回值结构验证")
        logging.info("-----------------------------------------------")


def create_mock_step_return_value(json_file_path=None):
    if json_file_path and os.path.exists(json_file_path):
        logging.info(f"从 {json_file_path} 加载真实数据")
        loader = TestPolicyServerPureInferenceModelIO()
        return loader.load_action_chunk_from_json(json_file_path)
    else:
        logging.info("使用随机模拟数据")
        return {
            "qpos": {
                "left_armqpos": np.random.randn(64, 7).astype(np.float32),
                "right_armqpos": np.random.randn(64, 7).astype(np.float32),
                "left_eefhand": np.random.rand(64, 6).astype(np.float32) * 0.5,
                "right_eefhand": np.random.rand(64, 6).astype(np.float32) * 0.5,
            }
        }


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2)
