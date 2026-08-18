import unittest
import numpy as np
from async_infer.async_infer_typedef import AsyncInferStateDimensionConfig
from async_infer.timed_sequence_array import *
from async_infer.merge_trajectory import *


class TestTimedSequenceArray(unittest.TestCase):

    def test_array_basic(self):
        data = np.zeros(shape=(3, 5), dtype=np.float64)
        data[0, :] = 0.0
        data[1, :] = 1.0
        data[2, :] = 2.0
        time = np.array([10, 11, 12], dtype=np.float64)
        seq = TimedSequenceArray(data, time)
        assert len(seq) == 3

        v = seq.get_at_time(9.0, True)
        assert v is not None and abs(v[0]) < 1e-6
        v = seq.get_at_time(10.0, True)
        assert v is not None and abs(v[0]) < 1e-6
        v = seq.get_at_time(11.0, True)
        assert v is not None and abs(v[0] - 1) < 1e-6
        v = seq.get_at_time(11.1, True)
        assert v is not None and abs(v[0] - 1.1) < 1e-6
        v = seq.get_at_time(10.1, True)
        assert v is not None and abs(v[0] - 0.1) < 1e-6
        v = seq.get_at_time(10.7, True)
        assert v is not None and abs(v[0] - 0.7) < 1e-6

        out = seq.get_at_times(t_in=np.array([0.1, 0.2, 9.0, 10.1, 11.0, 11.5, 11.99, 15, 20]))
        assert (out[:3, ...] < 1e-7).all()
        assert (np.abs(out[3, ...] - 0.1) < 1e-7).all()
        assert (np.abs(out[4, ...] - 1.0) < 1e-7).all()
        assert (np.abs(out[5, ...] - 1.5) < 1e-7).all()
        assert (np.abs(out[6, ...] - 1.99) < 1e-7).all()
        assert (np.abs(out[7:, ...] - 2.0) < 1e-7).all()
        # print(out[:, 0])

    def test_array_blend_basic(self):
        state_dim: int = 5
        data = np.zeros(shape=(3, state_dim), dtype=np.float64)
        data[0, :] = 0.0
        data[1, :] = 1.0
        data[2, :] = 2.0
        time = np.array([10, 11, 12], dtype=np.float64)
        seq = TimedSequenceArray(data, time)
        assert len(seq) == 3
        state_config = AsyncInferStateDimensionConfig(state_dim=state_dim, discrete_tool_state_indices=[],
                                                      state_distance_weight=None)

        # Run blend
        out = seq.get_at_times(t_in=np.array([0.1, 0.2, 9.0, 10.1, 11.0, 11.5, 11.99, 15, 20]))
        state_begin_np = out[0, ...] + 1.0
        state_begin = MergeTrajectory.StateInfo(state=state_begin_np, time=9.8)
        state_dist_weight = np.ones(shape=(5,), dtype=np.float64)
        state_dist_weight[0] = 2
        merge_out = MergeTrajectory.merge_by_nearest(state_begin, seq, state_config=state_config,
                                                     distance_weight_in=state_dist_weight, merge_blend_ratio=1.0)
        self.assertTrue(merge_out is not None)
        merged_state_begin = merge_out.get_at_knot_idx(0)
        merged_begin_dist = float(np.linalg.norm(state_begin.state - merged_state_begin))
        self.assertAlmostEqual(first=merged_begin_dist, second=0.0, delta=1e-4)

    def test_from_one_point(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        time = 10.0
        seq = TimedSequenceArray.from_one_point(data, time)
        self.assertEqual(len(seq), 1)
        np.testing.assert_array_almost_equal(seq.raw_data_points[0, :], data)
        np.testing.assert_array_almost_equal(seq.raw_data_times, [time])
        result = seq.get_at_knot_idx(0)
        np.testing.assert_array_almost_equal(result, data)

    def test_empty_and_single_point_sequence(self):
        data = np.zeros(shape=(1, 5), dtype=np.float64)
        data[0, :] = [1.0, 2.0, 3.0, 4.0, 5.0]
        time = np.array([10.0])
        seq = TimedSequenceArray(data, time)
        self.assertEqual(len(seq), 1)
        t = seq.begin()
        self.assertEqual(t, 10.0)
        t = seq.end()
        self.assertEqual(t, 10.0)
        result = seq.get_at_time(5.0)
        np.testing.assert_array_almost_equal(result, data[0, :])
        result = seq.get_at_time(15.0)
        np.testing.assert_array_almost_equal(result, data[0, :])
        eval_times = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
        out = seq.get_at_times(eval_times)
        self.assertEqual(out.shape[0], 5)
        for i in range(5):
            np.testing.assert_array_almost_equal(out[i, :], data[0, :])

    def test_eval_at_multiple_times_with_discrete_dims(self):
        data = np.zeros(shape=(3, 5), dtype=np.float64)
        data[0, :] = [0.0, 0.0, 0.0, 0.0, 0.0]
        data[1, :] = [5.0, 5.0, 5.0, 5.0, 5.0]
        data[2, :] = [10.0, 10.0, 10.0, 10.0, 10.0]
        time = np.array([10.0, 11.0, 12.0])
        eval_times = np.array([10.0, 10.5, 11.0, 11.5, 12.0])
        discrete_dims = [0, 2, 4]
        out = TimedSequenceArray.eval_at_multiple_times(data, time, eval_times, discrete_state_dims=discrete_dims)
        self.assertIsNotNone(out)
        expected_results = [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [5.0, 2.5, 5.0, 2.5, 5.0],
            [5.0, 5.0, 5.0, 5.0, 5.0],
            [10.0, 7.5, 10.0, 7.5, 10.0],
            [10.0, 10.0, 10.0, 10.0, 10.0]
        ]
        for i in range(len(eval_times)):
            np.testing.assert_array_almost_equal(out[i, :], expected_results[i])

    def test_eval_at_multiple_times_empty(self):
        data = np.zeros(shape=(0, 5), dtype=np.float64)
        time = np.array([])
        eval_times = np.array([10.0, 11.0, 12.0])
        out = TimedSequenceArray.eval_at_multiple_times(data, time, eval_times)
        self.assertIsNone(out)

    def test_merge_by_knot_idx(self):
        state_dim = 3
        data = np.zeros(shape=(3, state_dim), dtype=np.float64)
        data[0, :] = [0.0, 0.0, 0.0]
        data[1, :] = [5.0, 5.0, 5.0]
        data[2, :] = [10.0, 10.0, 10.0]
        time = np.array([10.0, 11.0, 12.0])
        traj = TimedSequenceArray(data, time)
        current_state = MergeTrajectory.StateInfo(state=np.array([1.0, 1.0, 1.0], dtype=np.float64), time=10.5)
        state_config = AsyncInferStateDimensionConfig(state_dim=state_dim, discrete_tool_state_indices=[],
                                                      state_distance_weight=None)
        result = MergeTrajectory.merge_by_knot_idx(current_state, traj, 0, None, 1.0)
        self.assertIsNotNone(result)
        self.assertGreater(len(result), 0)

    def test_merge_by_time2(self):
        state_dim = 3
        data = np.zeros(shape=(3, state_dim), dtype=np.float64)
        data[0, :] = [0.0, 0.0, 0.0]
        data[1, :] = [5.0, 5.0, 5.0]
        data[2, :] = [10.0, 10.0, 10.0]
        time = np.array([10.0, 11.0, 12.0])
        existing_traj = TimedSequenceArray(data, time)
        new_data = np.zeros(shape=(3, state_dim), dtype=np.float64)
        new_data[0, :] = [2.0, 2.0, 2.0]
        new_data[1, :] = [7.0, 7.0, 7.0]
        new_data[2, :] = [12.0, 12.0, 12.0]
        new_time = np.array([10.0, 11.0, 12.0])
        new_traj = TimedSequenceArray(new_data, new_time)
        current_state = MergeTrajectory.StateInfo(state=np.array([1.0, 1.0, 1.0], dtype=np.float64), time=10.5)
        state_config = AsyncInferStateDimensionConfig(state_dim=state_dim, discrete_tool_state_indices=[],
                                                      state_distance_weight=None)
        result = MergeTrajectory.merge_by_time_general(existing_traj, current_state, new_traj, 1.0, state_config)
        self.assertIsNotNone(result)

    def test_resample_trajectory(self):
        state_dim = 3
        data = np.zeros(shape=(5, state_dim), dtype=np.float64)
        for i in range(5):
            data[i, :] = [float(i), float(i), float(i)]
        time = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        traj = TimedSequenceArray(data, time)
        resampled = MergeTrajectory.resample_trajectory(traj, None)
        self.assertEqual(len(resampled), 5)
        resampled_begin = resampled.begin()
        self.assertAlmostEqual(resampled_begin, 10.0)

    def test_make_two_points_trajectory(self):
        start_state = MergeTrajectory.StateInfo(state=np.array([1.0, 2.0, 3.0], dtype=np.float64), time=10.0)
        end_state = np.array([5.0, 6.0, 7.0], dtype=np.float64)
        end_time = 15.0
        result = MergeTrajectory.make_two_points_trajectory(start_state, end_state, end_time)
        self.assertEqual(len(result), 2)
        first_point = result.get_at_knot_idx(0)
        np.testing.assert_array_almost_equal(first_point, start_state.state)
        second_point = result.get_at_knot_idx(1)
        np.testing.assert_array_almost_equal(second_point, end_state)

    def test_blend_start_point_and_trajectory_linear(self):
        state_dim = 3
        data = np.zeros(shape=(3, state_dim), dtype=np.float64)
        data[0, :] = [0.0, 0.0, 0.0]
        data[1, :] = [5.0, 5.0, 5.0]
        data[2, :] = [10.0, 10.0, 10.0]
        time = np.array([10.0, 11.0, 12.0])
        traj = TimedSequenceArray(data, time)
        start_point = MergeTrajectory.StateInfo(state=np.array([2.0, 2.0, 2.0], dtype=np.float64), time=10.5)
        result = MergeTrajectory.blend_start_point_and_trajectory_linear(start_point, traj, None, 1.0)
        self.assertIsNotNone(result)
        self.assertGreater(len(result), 0)

    def test_get_at_times_with_discrete_state_dims(self):
        data = np.zeros(shape=(3, 5), dtype=np.float64)
        data[0, :] = [0.0, 0.0, 0.0, 0.0, 0.0]
        data[1, :] = [5.0, 5.0, 5.0, 5.0, 5.0]
        data[2, :] = [10.0, 10.0, 10.0, 10.0, 10.0]
        time = np.array([10.0, 11.0, 12.0])
        seq = TimedSequenceArray(data, time)
        eval_times = np.array([10.5, 11.0, 11.5])
        discrete_dims = [0, 2, 4]
        out = seq.get_at_times(eval_times, discrete_state_dims=discrete_dims)
        self.assertIsNotNone(out)
        expected_results = [
            [5.0, 2.5, 5.0, 2.5, 5.0],
            [5.0, 5.0, 5.0, 5.0, 5.0],
            [10.0, 7.5, 10.0, 7.5, 10.0]
        ]
        for i in range(len(eval_times)):
            np.testing.assert_array_almost_equal(out[i, :], expected_results[i])

    def test_merge_by_nearest_with_distance_weight(self):
        state_dim = 3
        data = np.zeros(shape=(3, state_dim), dtype=np.float64)
        data[0, :] = [0.0, 0.0, 0.0]
        data[1, :] = [5.0, 5.0, 5.0]
        data[2, :] = [10.0, 10.0, 10.0]
        time = np.array([10.0, 11.0, 12.0])
        seq = TimedSequenceArray(data, time)
        current_state = MergeTrajectory.StateInfo(state=np.array([1.0, 1.0, 1.0], dtype=np.float32), time=10.5)
        state_config = AsyncInferStateDimensionConfig(state_dim=state_dim, discrete_tool_state_indices=[],
                                                      state_distance_weight=None)
        distance_weight = np.array([1.0, 2.0, 1.0], dtype=np.float32)
        result = MergeTrajectory.merge_by_nearest(current_state, seq, state_config=state_config,
                                                  distance_weight_in=distance_weight, merge_blend_ratio=0.5)
        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()
