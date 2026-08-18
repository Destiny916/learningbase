import unittest
import numpy as np
import json_numpy
from async_infer.json_encode_obs_action import *
from async_infer.async_infer_typedef import ObservationMap, AsyncInferObservationKeys


class TestJsonEncodeObs(unittest.TestCase):

    def setUp(self):
        """Set up test data"""
        self.state_dim = 10
        self.obs_keys = AsyncInferObservationKeys(
            state_key='state',
            rgb_images=['img0', 'img1'],
            depth_images=['depth0', 'depth1'],
            other_keys=['step', 'episode', 'test_array', 'to_remove']
        )
        self.obs_map = ObservationMap(
            observation_time=1.0,
            state=np.random.randn(self.state_dim, ).astype(np.float32),
            tensor_dict={
                'img0': np.random.randn(224, 224, 3),
                'img1': np.random.randn(224, 224, 3),
                'depth0': np.random.randn(224, 224),
                'depth1': np.random.randn(224, 224)
            },
            misc_dict={
                'step': 10,
                'episode': 1,
                'test_array': np.array([1, 2, 3, 4, 5]),
                'to_remove': 'should not be in output'
            }
        )

    def test_default_encoding(self):
        """Test default encoding behavior"""
        encoded_dict = encode_observation_into_json_dict(self.obs_map, self.obs_keys)

        # Basic assertions
        self.assertIsNotNone(encoded_dict)
        self.assertIn('img0', encoded_dict)
        self.assertIn('depth0', encoded_dict)
        self.assertIn('step', encoded_dict)
        self.assertIn('episode', encoded_dict)
        self.assertIn('test_array', encoded_dict)
        self.assertIn('to_remove', encoded_dict)

        # Check types
        self.assertIsInstance(encoded_dict['step'], int)
        self.assertIsInstance(encoded_dict['episode'], int)
        self.assertIsInstance(encoded_dict['test_array'], str)  # Default is ToNumpyJsonStr

    def test_encoding_with_options(self):
        """Test encoding with various options"""
        option = EncodeOption()
        option.removed_entry_keys.add('to_remove')
        option.option_dict['step'] = EntryEncodeOption(new_key='timestep')
        option.option_dict['test_array'] = EntryEncodeOption(np_array_encode_option=NpArrayEncodeOption.ToList)
        option.option_dict['episode'] = EntryEncodeOption(remove_me_in_encoded=True)

        encoded_dict = encode_observation_into_json_dict(self.obs_map, self.obs_keys, option)

        # Basic assertions
        self.assertIsNotNone(encoded_dict)
        self.assertIn('img0', encoded_dict)
        self.assertIn('depth0', encoded_dict)
        self.assertIn('timestep', encoded_dict)  # Renamed from 'step'
        self.assertIn('test_array', encoded_dict)

        # Check that removed keys are not present
        self.assertNotIn('step', encoded_dict)  # Original key should be gone
        self.assertNotIn('episode', encoded_dict)  # Should be removed
        self.assertNotIn('to_remove', encoded_dict)  # Should be removed

        # Check types
        self.assertIsInstance(encoded_dict['timestep'], int)
        self.assertIsInstance(encoded_dict['test_array'], list)  # Should be list due to ToList option

    def test_np_array_encode_options(self):
        """Test different numpy array encoding options"""
        # Test DoNothing option
        option = EncodeOption(global_np_array_encode_option=NpArrayEncodeOption.DoNothing)
        encoded_dict = encode_observation_into_json_dict(self.obs_map, self.obs_keys, option)
        self.assertIsInstance(encoded_dict['test_array'], np.ndarray)

        # Test ToList option
        option = EncodeOption(global_np_array_encode_option=NpArrayEncodeOption.ToList)
        encoded_dict = encode_observation_into_json_dict(self.obs_map, self.obs_keys, option)
        self.assertIsInstance(encoded_dict['test_array'], list)

        # Test ToNumpyJsonStr option
        option = EncodeOption(global_np_array_encode_option=NpArrayEncodeOption.ToNumpyJsonStr)
        encoded_dict = encode_observation_into_json_dict(self.obs_map, self.obs_keys, option)
        self.assertIsInstance(encoded_dict['test_array'], str)

    def test_invalid_observation(self):
        """Test encoding with invalid observation"""
        # Test None observation
        result = encode_observation_into_json_dict(None, self.obs_keys)
        self.assertIsNone(result)

        # Test observation with negative request time
        invalid_obs = ObservationMap(
            observation_time=-1.0,
            state=np.random.randn(self.state_dim, ).astype(np.float32),
            tensor_dict={},
            misc_dict={}
        )
        result = encode_observation_into_json_dict(invalid_obs, self.obs_keys)
        self.assertIsNone(result)


class TestDecodeAction(unittest.TestCase):

    def setUp(self):
        pass

    def test_json_numpy_encoded_trajectory(self):
        """Test decoding json_numpy encoded trajectory"""
        from async_infer.policy_client_interface import PolicyClientRequestMeta
        import numpy as np
        
        # Create request meta
        request_meta = PolicyClientRequestMeta(reqeust_time=1.0, reqeust_seq_index=1)
        
        # Create test data
        trajectory = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32)
        time_data = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        
        # Create response dict with json_numpy encoded data
        response_dict = {
            'action.trajectory': json_numpy.dumps(trajectory),
            'action.trajectory_time': json_numpy.dumps(time_data)
        }
        
        # Decode response
        response = decode_policy_response(request_meta, response_dict)
        
        # Assertions
        self.assertTrue(response.is_valid)
        self.assertEqual(response.state_trajectory.raw_data_points.shape, trajectory.shape)
        self.assertEqual(response.state_trajectory.raw_data_times.shape, time_data.shape)

    def test_list_encoded_trajectory(self):
        """Test decoding list encoded trajectory"""
        from async_infer.policy_client_interface import PolicyClientRequestMeta
        import numpy as np
        
        # Create request meta
        request_meta = PolicyClientRequestMeta(reqeust_time=1.0, reqeust_seq_index=1)
        
        # Create test data
        trajectory = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32)
        time_data = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        
        # Create response dict with list encoded data
        response_dict = {
            'action.trajectory': trajectory.tolist(),
            'action.trajectory_time': time_data.tolist()
        }
        
        # Decode response
        response = decode_policy_response(request_meta, response_dict)
        
        # Assertions
        self.assertTrue(response.is_valid)
        self.assertEqual(response.state_trajectory.raw_data_points.shape, trajectory.shape)
        self.assertEqual(response.state_trajectory.raw_data_times.shape, time_data.shape)

    def test_numpy_array_trajectory(self):
        """Test decoding numpy array trajectory"""
        from async_infer.policy_client_interface import PolicyClientRequestMeta
        import numpy as np
        
        # Create request meta
        request_meta = PolicyClientRequestMeta(reqeust_time=1.0, reqeust_seq_index=1)
        
        # Create test data
        trajectory = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32)
        time_data = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        
        # Create response dict with numpy arrays
        response_dict = {
            'action.trajectory': trajectory,
            'action.trajectory_time': time_data
        }
        
        # Decode response
        response = decode_policy_response(request_meta, response_dict)
        
        # Assertions
        self.assertTrue(response.is_valid)
        self.assertEqual(response.state_trajectory.raw_data_points.shape, trajectory.shape)
        self.assertEqual(response.state_trajectory.raw_data_times.shape, time_data.shape)

    def test_without_time_data(self):
        """Test decoding without time data (should generate evenly spaced time)"""
        from async_infer.policy_client_interface import PolicyClientRequestMeta
        import numpy as np
        
        # Create request meta
        request_meta = PolicyClientRequestMeta(reqeust_time=1.0, reqeust_seq_index=1)
        
        # Create test data
        trajectory = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32)
        
        # Create response dict without time data
        response_dict = {
            'action.trajectory': trajectory.tolist()
        }
        
        # Decode response
        response = decode_policy_response(request_meta, response_dict)
        
        # Assertions
        self.assertTrue(response.is_valid)
        self.assertEqual(response.state_trajectory.raw_data_points.shape, trajectory.shape)
        self.assertEqual(response.state_trajectory.raw_data_times.shape, (3,))

    def test_with_error_string(self):
        """Test decoding with error string"""
        from async_infer.policy_client_interface import PolicyClientRequestMeta
        import numpy as np
        
        # Create request meta
        request_meta = PolicyClientRequestMeta(reqeust_time=1.0, reqeust_seq_index=1)
        
        # Create test data
        trajectory = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32)
        
        # Create response dict with error string
        response_dict = {
            'action.trajectory': trajectory.tolist(),
            'action.error': 'Test error'
        }
        
        # Decode response
        response = decode_policy_response(request_meta, response_dict)
        
        # Assertions
        self.assertTrue(response.is_valid)
        self.assertEqual(response.error_str, 'Test error')

    def test_missing_trajectory_key(self):
        """Test error case - missing trajectory key"""
        from async_infer.policy_client_interface import PolicyClientRequestMeta
        import numpy as np
        
        # Create request meta
        request_meta = PolicyClientRequestMeta(reqeust_time=1.0, reqeust_seq_index=1)
        
        # Create response dict without trajectory key
        response_dict = {
            'action_time': np.array([0.0, 0.5, 1.0]).tolist()
        }
        
        # Decode response
        response = decode_policy_response(request_meta, response_dict)
        
        # Assertions
        self.assertFalse(response.is_valid)
        self.assertIn('Missing trajectory key', response.error_str)

    def test_invalid_trajectory_shape(self):
        """Test error case - invalid trajectory shape"""
        from async_infer.policy_client_interface import PolicyClientRequestMeta
        import numpy as np
        
        # Create request meta
        request_meta = PolicyClientRequestMeta(reqeust_time=1.0, reqeust_seq_index=1)
        
        # Create response dict with 1D trajectory (should be 2D)
        response_dict = {
            'action': np.array([0.1, 0.2, 0.3]).tolist()
        }
        
        # Decode response
        response = decode_policy_response(request_meta, response_dict)
        
        # Assertions
        self.assertFalse(response.is_valid)
        self.assertIn('Missing trajectory key', response.error_str)

    def test_error_in_response_and_failed_decode_trajectory(self):
        """Test error case - error in response + failed to decode trajectory"""
        from async_infer.policy_client_interface import PolicyClientRequestMeta
        
        # Create request meta
        request_meta = PolicyClientRequestMeta(reqeust_time=1.0, reqeust_seq_index=1)
        
        # Create response dict with invalid trajectory data
        response_dict = {
            'action.trajectory': 'invalid_json',
            'action.error': 'Server error'
        }
        
        # Decode response
        response = decode_policy_response(request_meta, response_dict)
        
        # Assertions
        self.assertFalse(response.is_valid)
        self.assertIn('Response error: Server error', response.error_str)
        self.assertIn('Decode error: Failed to decode trajectory data', response.error_str)

    def test_error_in_response_and_invalid_trajectory_shape(self):
        """Test error case - error in response + invalid trajectory shape"""
        from async_infer.policy_client_interface import PolicyClientRequestMeta
        import numpy as np
        
        # Create request meta
        request_meta = PolicyClientRequestMeta(reqeust_time=1.0, reqeust_seq_index=1)
        
        # Create response dict with 1D trajectory and error
        response_dict = {
            'action.trajectory': np.array([0.1, 0.2, 0.3]).tolist(),  # 1D instead of 2D
            'action.error': 'Server error'
        }
        
        # Decode response
        response = decode_policy_response(request_meta, response_dict)
        
        # Assertions
        self.assertFalse(response.is_valid)
        self.assertIn('Response error: Server error', response.error_str)
        self.assertIn('Decode error: Invalid trajectory shape', response.error_str)

    def test_error_in_response_and_failed_decode_time_data(self):
        """Test error case - error in response + failed to decode time data"""
        from async_infer.policy_client_interface import PolicyClientRequestMeta
        import numpy as np
        
        # Create request meta
        request_meta = PolicyClientRequestMeta(reqeust_time=1.0, reqeust_seq_index=1)
        
        # Create test data
        trajectory = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32)
        
        # Create response dict with invalid time data
        response_dict = {
            'action.trajectory': trajectory.tolist(),
            'action.trajectory_time': 'invalid_json',
            'action.error': 'Server error'
        }
        
        # Decode response
        response = decode_policy_response(request_meta, response_dict)
        
        # Assertions
        self.assertFalse(response.is_valid)
        self.assertIn('Response error: Server error', response.error_str)
        self.assertIn('Decode error: Failed to decode time data', response.error_str)

    def test_error_in_response_and_invalid_time_shape(self):
        """Test error case - error in response + invalid time shape"""
        from async_infer.policy_client_interface import PolicyClientRequestMeta
        import numpy as np
        
        # Create request meta
        request_meta = PolicyClientRequestMeta(reqeust_time=1.0, reqeust_seq_index=1)
        
        # Create test data
        trajectory = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32)
        
        # Create response dict with wrong size time data
        response_dict = {
            'action.trajectory': trajectory.tolist(),
            'action.trajectory_time': np.array([0.0, 0.5]).tolist(),  # Wrong size
            'action.error': 'Server error'
        }
        
        # Decode response
        response = decode_policy_response(request_meta, response_dict)
        
        # Assertions
        self.assertFalse(response.is_valid)
        self.assertIn('Response error: Server error', response.error_str)
        self.assertIn('Decode error: Invalid time shape', response.error_str)


if __name__ == '__main__':
    unittest.main()
