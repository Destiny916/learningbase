import unittest
import os
import tempfile
import json
import numpy as np

from models.async_infer.w1_rollout.w1_rollout_config import W1RolloutConfig, W1RolloutGripperProcess, W1RolloutRobotDoF, \
    PositionCommand, W1PositionCommand


class TestW1RolloutConfig(unittest.TestCase):

    def setUp(self):
        self.test_json_path = None

    def tearDown(self):
        if self.test_json_path and os.path.exists(self.test_json_path):
            os.remove(self.test_json_path)

    def _create_temp_file(self):
        fd, self.test_json_path = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        return self.test_json_path

    def test_default_values(self):
        config = W1RolloutConfig()
        self.assertEqual(config.policy_path, '/path/to/pretrained_model')
        self.assertEqual(config.device, 'cuda')
        self.assertEqual(config.policy_hz, 15.0)
        self.assertEqual(config.tolerance_ms, 50.0)
        self.assertEqual(config.remote_server_host, '127.0.0.1')
        self.assertEqual(config.remote_server_port, 8899)
        self.assertEqual(config.joint_topic, '/feedback/robot_server_state')
        self.assertEqual(config.hand_input_mode, 'none')
        self.assertEqual(config.gripper_binarize, True)
        self.assertEqual(config.gripper_thr, 0.5)

    def test_custom_values(self):
        config = W1RolloutConfig(
            policy_path='/custom/model',
            device='cpu',
            policy_hz=30.0,
            tolerance_ms=100.0,
            remote_server_host='192.168.1.1',
            remote_server_port=9999,
            hand_input_mode='qpos6',
            gripper_thr=0.7,
        )
        self.assertEqual(config.policy_path, '/custom/model')
        self.assertEqual(config.device, 'cpu')
        self.assertEqual(config.policy_hz, 30.0)
        self.assertEqual(config.tolerance_ms, 100.0)
        self.assertEqual(config.remote_server_host, '192.168.1.1')
        self.assertEqual(config.remote_server_port, 9999)
        self.assertEqual(config.hand_input_mode, 'qpos6')
        self.assertEqual(config.gripper_thr, 0.7)

    def test_serialize_to_json_file(self):
        config = W1RolloutConfig(policy_hz=100, device='cpu')
        test_path = self._create_temp_file()
        config.to_json_file(test_path)

        with open(test_path, 'r') as f:
            data = json.load(f)

        self.assertEqual(data['policy_hz'], 100)
        self.assertEqual(data['device'], 'cpu')
        self.assertEqual(data['model_type'], 'w1_rollout_config')

    def test_deserialize_from_json_file(self):
        config = W1RolloutConfig(policy_hz=50, remote_server_port=7777)
        test_path = self._create_temp_file()
        config.to_json_file(test_path)

        loaded_config = W1RolloutConfig.from_json_file(test_path)

        self.assertEqual(loaded_config.policy_hz, 50)
        self.assertEqual(loaded_config.remote_server_port, 7777)
        self.assertEqual(loaded_config.device, 'cuda')

    def test_roundtrip_serialization(self):
        original = W1RolloutConfig(
            policy_path='/test/path',
            device='cpu',
            policy_hz=25.0,
            tolerance_ms=75.0,
            remote_server_host='10.0.0.1',
            remote_server_port=5555,
            joint_topic='/test/joint',
            image_keys=['key1', 'key2'],
            ordered_body_names=['BODY1', 'BODY2'],
            hand_sides=['left'],
            gripper_binarize=False,
            gripper_thr=0.3,
        )
        test_path = self._create_temp_file()
        original.to_json_file(test_path)

        restored = W1RolloutConfig.from_json_file(test_path)

        self.assertEqual(restored.policy_path, '/test/path')
        self.assertEqual(restored.device, 'cpu')
        self.assertEqual(restored.policy_hz, 25.0)
        self.assertEqual(restored.tolerance_ms, 75.0)
        self.assertEqual(restored.remote_server_host, '10.0.0.1')
        self.assertEqual(restored.remote_server_port, 5555)
        self.assertEqual(restored.joint_topic, '/test/joint')
        self.assertEqual(restored.image_keys, ['key1', 'key2'])
        self.assertEqual(restored.ordered_body_names, ['BODY1', 'BODY2'])
        self.assertEqual(restored.hand_sides, ['left'])
        self.assertEqual(restored.gripper_binarize, False)
        self.assertEqual(restored.gripper_thr, 0.3)

    def test_list_attributes_serialization(self):
        config = W1RolloutConfig()
        test_path = self._create_temp_file()
        config.to_json_file(test_path)

        loaded = W1RolloutConfig.from_json_file(test_path)

        expected_image_keys = [
            "observation.images.cam_high_left",
            "observation.images.cam_high_right",
            "observation.images.cam_hand_left",
            "observation.images.cam_hand_right",
        ]
        self.assertEqual(loaded.image_keys, expected_image_keys)

        expected_body_names = [
            'ANKLE', 'KNEE', 'BUTTOCK', 'WAIST',
            'LEFT_J1', 'LEFT_J2', 'LEFT_J3', 'LEFT_J4', 'LEFT_J5', 'LEFT_J6', 'LEFT_J7',
            'NECK1', 'NECK2',
            'RIGHT_J1', 'RIGHT_J2', 'RIGHT_J3', 'RIGHT_J4', 'RIGHT_J5', 'RIGHT_J6', 'RIGHT_J7',
        ]
        self.assertEqual(loaded.ordered_body_names, expected_body_names)

        self.assertEqual(loaded.hand_sides, ['left', 'right'])

        expected_left_qpos6 = [
            'LEFT_HAND_THUMB1', 'LEFT_HAND_THUMB2', 'LEFT_HAND_INDEX',
            'LEFT_HAND_MIDDLE', 'LEFT_HAND_RING', 'LEFT_HAND_PINKY'
        ]
        self.assertEqual(loaded.left_hand_qpos6_names, expected_left_qpos6)

    def test_to_dict(self):
        config = W1RolloutConfig(policy_hz=60)
        config_dict = config.to_dict()

        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['policy_hz'], 60)
        self.assertEqual(config_dict['model_type'], 'w1_rollout_config')

    def test_from_dict(self):
        config_dict = {
            'model_type': 'w1_rollout_config',
            'policy_hz': 45,
            'device': 'cpu',
            'remote_server_port': 1234,
        }
        config = W1RolloutConfig.from_dict(config_dict)

        self.assertEqual(config.policy_hz, 45)
        self.assertEqual(config.device, 'cpu')
        self.assertEqual(config.remote_server_port, 1234)

    def test_to_json_string(self):
        config = W1RolloutConfig(policy_hz=120)
        json_str = config.to_json_string()

        self.assertIn('"policy_hz": 120', json_str)
        self.assertIn('"model_type": "w1_rollout_config"', json_str)

    def test_model_type(self):
        config = W1RolloutConfig()
        self.assertEqual(config.model_type, 'w1_rollout_config')

    def test_all_config_attributes_exist(self):
        config = W1RolloutConfig()
        attrs = [
            'policy_path', 'device', 'policy_hz', 'tolerance_ms',
            'remote_server_host', 'remote_server_port', 'remote_connect_timeout_s',
            'remote_horizon_n', 'remote_replan_trigger',
            'joint_topic', 'publish_topic',
            'head_target_width', 'head_target_height', 'hand_target_width', 'hand_target_height',
            'image_hand_left_key', 'image_hand_right_key', 'state_key', 'image_keys',
            'ordered_body_names', 'selected_body_names', 'drop_joint_names', 'joint_names',
            'hand_input_mode', 'hand_sides', 'hand_sides_str',
            'left_hand_scalar_name', 'right_hand_scalar_name',
            'left_hand_qpos6_names', 'right_hand_qpos6_names',
            'left_hand_scalar_topic', 'right_hand_scalar_topic',
            'left_hand_qpos6_topic', 'right_hand_qpos6_topic',
            'set_left_hand_qpos6_topic', 'set_right_hand_qpos6_topic',
            'cam_hand_left_topic', 'cam_hand_right_topic',
            'hand_interp_start', 'hand_interp_end',
            'gripper_binarize', 'gripper_thr', 'gripper_hysteresis',
            'gripper_thr_close', 'gripper_thr_open', 'freeze_after_release_s',
            'hand_gestures',
        ]
        for attr in attrs:
            self.assertTrue(hasattr(config, attr), f"Missing attribute: {attr}")

    def test_hand_gestures_default_values(self):
        config = W1RolloutConfig()
        self.assertIsInstance(config.hand_gestures, dict)
        self.assertIn('normal', config.hand_gestures)
        self.assertIn('pinch', config.hand_gestures)
        self.assertIn('fist', config.hand_gestures)
        self.assertEqual(config.hand_gestures['pinch'], [65.0, 100.0, 70.0, 75.0, 100.0, 100.0])
        self.assertEqual(config.hand_gestures['fist'], [100.0, 30.0, 100.0, 100.0, 100.0, 100.0])

    def test_hand_gestures_custom_values(self):
        custom_gestures = {
            "test1": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "test2": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
        config = W1RolloutConfig(hand_gestures=custom_gestures)
        self.assertEqual(config.hand_gestures['test1'], [10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        self.assertEqual(config.hand_gestures['test2'], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_hand_gestures_serialization(self):
        config = W1RolloutConfig()
        test_path = self._create_temp_file()
        config.to_json_file(test_path)

        loaded = W1RolloutConfig.from_json_file(test_path)

        self.assertIsInstance(loaded.hand_gestures, dict)
        self.assertEqual(loaded.hand_gestures['pinch'], [65.0, 100.0, 70.0, 75.0, 100.0, 100.0])
        self.assertEqual(loaded.hand_gestures['fist'], [100.0, 30.0, 100.0, 100.0, 100.0, 100.0])
        self.assertEqual(loaded.hand_gestures['normal'], [0.0, 70.0, 0.0, 0.0, 0.0, 0.0])

    def test_hand_gestures_roundtrip(self):
        custom_gestures = {
            "my_gesture": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "another": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
        original = W1RolloutConfig(hand_gestures=custom_gestures)
        test_path = self._create_temp_file()
        original.to_json_file(test_path)

        restored = W1RolloutConfig.from_json_file(test_path)

        self.assertEqual(restored.hand_gestures['my_gesture'], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.assertEqual(restored.hand_gestures['another'], [10.0, 20.0, 30.0, 40.0, 50.0, 60.0])

    def test_w1_rollout_gripper_process_no_binarization(self):
        config = W1RolloutConfig(gripper_binarize=False)
        process = W1RolloutGripperProcess(config)

        # Test that values are clipped and returned as-is
        self.assertAlmostEqual(process.postprocess_gripper_scalar(0.5, 'left', 0.0), 0.5)
        self.assertAlmostEqual(process.postprocess_gripper_scalar(1.5, 'left', 0.0), 1.0)
        self.assertAlmostEqual(process.postprocess_gripper_scalar(-0.5, 'left', 0.0), 0.0)

    def test_w1_rollout_gripper_process_binarization_no_hysteresis(self):
        config = W1RolloutConfig(gripper_binarize=True, gripper_hysteresis=False, gripper_thr=0.5)
        process = W1RolloutGripperProcess(config)

        # Test binarization without hysteresis
        self.assertEqual(process.postprocess_gripper_scalar(0.4, 'left', 0.0), 0.0)
        self.assertEqual(process.postprocess_gripper_scalar(0.6, 'left', 0.0), 1.0)
        self.assertEqual(process.postprocess_gripper_scalar(0.5, 'left', 0.0), 1.0)

    def test_w1_rollout_gripper_process_binarization_with_hysteresis(self):
        config = W1RolloutConfig(
            gripper_binarize=True,
            gripper_hysteresis=True,
            gripper_thr_close=0.6,
            gripper_thr_open=0.4
        )
        process = W1RolloutGripperProcess(config)

        # Test hysteresis behavior
        # Start from 0, should stay 0 until threshold
        self.assertEqual(process.postprocess_gripper_scalar(0.5, 'left', 0.0), 0.0)
        # Cross upper threshold, should switch to 1
        self.assertEqual(process.postprocess_gripper_scalar(0.7, 'left', 0.0), 1.0)
        # Should stay 1 even when below upper threshold but above lower
        self.assertEqual(process.postprocess_gripper_scalar(0.5, 'left', 0.0), 1.0)
        # Cross lower threshold, should switch back to 0
        self.assertEqual(process.postprocess_gripper_scalar(0.3, 'left', 0.0), 0.0)

    def test_w1_rollout_gripper_process_freeze_after_release(self):
        config = W1RolloutConfig(
            gripper_binarize=True,
            freeze_after_release_s=1.0
        )
        process = W1RolloutGripperProcess(config)

        # Set to 1
        process.postprocess_gripper_scalar(1.0, 'left', 0.0)
        # Release (go to 0) at time 1.0
        process.postprocess_gripper_scalar(0.0, 'left', 1.0)
        # Should still be 0 even if we try to go back to 1 within freeze period
        process.postprocess_gripper_scalar(1.0, 'left', 1.5)
        # After freeze period, should respond again
        result = process.postprocess_gripper_scalar(1.0, 'left', 2.1)
        self.assertEqual(result, 1.0)

    def test_w1_rollout_robot_dof_initialization(self):
        # Test initialization with different hand input modes
        config = W1RolloutConfig(hand_input_mode='none')
        dof_none = W1RolloutRobotDoF(config)
        self.assertIsInstance(dof_none, W1RolloutRobotDoF)

        config = W1RolloutConfig(hand_input_mode='scalar')
        dof_scalar = W1RolloutRobotDoF(config)
        self.assertIsInstance(dof_scalar, W1RolloutRobotDoF)

        config = W1RolloutConfig(hand_input_mode='qpos6')
        dof_qpos6 = W1RolloutRobotDoF(config)
        self.assertIsInstance(dof_qpos6, W1RolloutRobotDoF)

    def test_w1_rollout_robot_dof_map_scalar_to_qpos6(self):
        config = W1RolloutConfig()
        dof = W1RolloutRobotDoF(config)

        # Test mapping scalar to qpos6
        qpos6_0 = dof.map_scalar_to_qpos6(0.0)
        self.assertEqual(len(qpos6_0), 6)

        qpos6_1 = dof.map_scalar_to_qpos6(1.0)
        self.assertEqual(len(qpos6_1), 6)

        qpos6_05 = dof.map_scalar_to_qpos6(0.5)
        self.assertEqual(len(qpos6_05), 6)

    def test_w1_rollout_robot_dof_action_to_np(self):
        config = W1RolloutConfig()
        dof = W1RolloutRobotDoF(config)

        # Test with list
        action_list = [0.0] * dof.full_dim
        result = dof.action_to_np(action_list)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (dof.full_dim,))

        # Test with numpy array
        action_np = np.zeros(dof.full_dim)
        result = dof.action_to_np(action_np)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (dof.full_dim,))

    def test_w1_rollout_robot_dof_make_action_scalar(self):
        config = W1RolloutConfig(hand_input_mode='scalar')
        dof = W1RolloutRobotDoF(config)
        processor = W1RolloutGripperProcess(config)

        # Create action array with gripper values
        action = np.zeros(dof.full_dim)
        if dof.idx_left_scalar is not None:
            action[dof.idx_left_scalar] = 1.0
        if dof.idx_right_scalar is not None:
            action[dof.idx_right_scalar] = 0.0

        # Test make_action
        cmd = dof.make_action(action, processor, 0.0)
        self.assertIsInstance(cmd, W1PositionCommand)

    def test_w1_rollout_robot_dof_make_action_qpos6(self):
        config = W1RolloutConfig(hand_input_mode='qpos6')
        dof = W1RolloutRobotDoF(config)
        processor = W1RolloutGripperProcess(config)

        # Create action array
        action = np.zeros(dof.full_dim)

        # Test make_action
        cmd = dof.make_action(action, processor, 0.0)
        self.assertIsInstance(cmd, W1PositionCommand)


if __name__ == '__main__':
    unittest.main()
