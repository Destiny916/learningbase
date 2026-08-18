import unittest
import numpy as np
from async_infer.timed_sequence import *


class TestTimedSequence(unittest.TestCase):

    def test_float_basic(self):
        seq0 = TimedSequenceList[float]()
        self._run_test_seq_float_basic(seq0)
        seq1 = TimedSequenceList[float]()
        self._run_test_seq_float_loop(seq1, test_n=100)

    def _run_test_seq_float_basic(self, seq: TimedSequenceInterface[float]):
        assert seq.append(10.0, 0.0)
        assert not seq.append(10.0, 0.0)
        seq.append(11.0, 1.0)
        v = seq.get_at_knot_idx(0)
        assert v is not None and abs(v) < 1e-6
        v = seq.get_at_knot_idx(1)
        assert v is not None and abs(v - 1) < 1e-6
        v = seq.get_at_knot_idx(2)
        assert v is None
        assert len(seq) == 2

        # By time
        v = seq.get_at_time(9.0, True)
        assert v is not None and abs(v) < 1e-6
        v = seq.get_at_time(10.0, True)
        assert v is not None and abs(v) < 1e-6
        v = seq.get_at_time(11.0, True)
        assert v is not None and abs(v - 1) < 1e-6
        v = seq.get_at_time(11.1, True)
        assert v is not None and abs(v - 1) < 1e-6
        v = seq.get_at_time(10.1, True)
        assert v is not None and abs(v - 0.1) < 1e-6
        v = seq.get_at_time(10.7, True)
        assert v is not None and abs(v - 0.7) < 1e-6

        seq.append(12.0, 2.0)
        assert len(seq) == 3
        v = seq.get_at_time(9.0, True)
        assert v is not None and abs(v) < 1e-6
        v = seq.get_at_time(10.0, True)
        assert v is not None and abs(v) < 1e-6
        v = seq.get_at_time(11.0, True)
        assert v is not None and abs(v - 1) < 1e-6
        v = seq.get_at_time(11.1, True)
        assert v is not None and abs(v - 1.1) < 1e-6
        v = seq.get_at_time(10.1, True)
        assert v is not None and abs(v - 0.1) < 1e-6
        v = seq.get_at_time(10.7, True)
        assert v is not None and abs(v - 0.7) < 1e-6

        # Prune
        seq.prune_by_time(11.1)
        assert len(seq) == 2
        v = seq.get_at_time(11.0, True)
        assert v is not None and abs(v - 1) < 1e-6
        v = seq.get_at_time(11.1, True)
        assert v is not None and abs(v - 1.1) < 1e-6

    def _run_test_seq_float_loop(self, seq: TimedSequenceInterface[float], test_n: int = 10):
        offset = 0
        for test_i in range(test_n):
            for i in range(test_n):
                t_i = float(i + offset)
                v_i = 2.0 * t_i + 1.0
                ok = seq.append(t=t_i, v=v_i)
                assert ok

            # Eval the value
            begin_t= seq.begin()
            end_t = seq.end()
            eval_n = 10 * test_n
            duration_between_val = (end_t - begin_t) / float(eval_n - 1)
            for i in range(eval_n):
                eval_i_t = begin_t + float(i) * duration_between_val
                eval_i_v = seq.get_at_time(eval_i_t, interpolate_data=True)
                eval_i_v_expected = 2 * eval_i_t + 1
                self.assertAlmostEqual(eval_i_v, eval_i_v_expected, delta=1e-5)

            # Update offset
            offset = end_t + 0.1

            # Prune
            prune_time = 0.9 * end_t + 0.1 * begin_t
            seq.prune_by_time(oldest_time_to_keep=prune_time)

    def test_timed_sequence_discrete(self):
        seq0 = TimedSequenceList[int]()
        seq0.append(t=0.0, v=1)
        seq0.append(t=1.0, v=2)
        self.assertEqual(seq0.get_at_time(-10), 1)
        self.assertEqual(seq0.get_at_time(-1), 1)
        self.assertEqual(seq0.get_at_time(0.5), 2)
        self.assertEqual(seq0.get_at_time(10), 2)

        append_ok = seq0.append(t=0.5, v=1000)
        self.assertFalse(append_ok)
        append_ok = seq0.append(t=1.5, v=3)
        self.assertTrue(append_ok)
        self.assertEqual(seq0.get_at_time(1.1), 3)
        self.assertEqual(seq0.get_at_time(10), 3)

    def test_empty_sequence(self):
        seq = TimedSequenceList[float]()
        self.assertEqual(len(seq), 0)
        self.assertTrue(seq.empty)
        self.assertIsNone(seq.get_at_knot_idx(0))
        self.assertIsNone(seq.get_at_time(0.0))
        begin_t = seq.begin()
        self.assertEqual(begin_t, -1)
        end_t = seq.end()
        self.assertEqual(end_t, -1)
        idx0, idx1 = seq.find_time_index(0.0)
        self.assertFalse(idx0.is_valid)
        self.assertFalse(idx1.is_valid)

    def test_single_element_sequence(self):
        seq = TimedSequenceList[float]()
        self.assertTrue(seq.append(t=5.0, v=10.0))
        self.assertEqual(len(seq), 1)
        self.assertFalse(seq.empty)
        self.assertIsNotNone(seq.get_at_knot_idx(0))
        self.assertAlmostEqual(seq.get_at_knot_idx(0), 10.0)
        self.assertIsNone(seq.get_at_knot_idx(1))
        self.assertIsNone(seq.get_at_knot_idx(-1))
        begin_t = seq.begin()
        self.assertEqual(begin_t, 5.0)
        end_t = seq.end()
        self.assertEqual(end_t, 5.0)
        self.assertAlmostEqual(seq.get_at_time(0.0), 10.0)
        self.assertAlmostEqual(seq.get_at_time(5.0), 10.0)
        self.assertAlmostEqual(seq.get_at_time(10.0), 10.0)
        seq.prune_by_time(4.0)
        self.assertEqual(len(seq), 1)
        seq.prune_by_time(5.0)
        self.assertEqual(len(seq), 1)
        seq.prune_by_time(6.0)
        self.assertEqual(len(seq), 1)

    def test_find_time_index(self):
        seq = TimedSequenceList[float]()
        seq.append(t=0.0, v=0.0)
        seq.append(t=5.0, v=5.0)
        seq.append(t=10.0, v=10.0)
        idx0, idx1 = seq.find_time_index(0.0)
        self.assertEqual(idx0.knot_idx, 0)
        self.assertEqual(idx1.knot_idx, 0)
        self.assertAlmostEqual(idx0.interpolate_weight, 1.0)
        self.assertAlmostEqual(idx1.interpolate_weight, 0.0)
        idx0, idx1 = seq.find_time_index(2.5)
        self.assertEqual(idx0.knot_idx, 0)
        self.assertEqual(idx1.knot_idx, 1)
        self.assertAlmostEqual(idx0.interpolate_weight, 0.5)
        self.assertAlmostEqual(idx1.interpolate_weight, 0.5)
        idx0, idx1 = seq.find_time_index(7.5)
        self.assertEqual(idx0.knot_idx, 1)
        self.assertEqual(idx1.knot_idx, 2)
        self.assertAlmostEqual(idx0.interpolate_weight, 0.5)
        self.assertAlmostEqual(idx1.interpolate_weight, 0.5)
        idx0, idx1 = seq.find_time_index(-1.0)
        self.assertEqual(idx0.knot_idx, 0)
        self.assertEqual(idx1.knot_idx, 0)
        idx0, idx1 = seq.find_time_index(15.0)
        self.assertEqual(idx0.knot_idx, 2)
        self.assertEqual(idx1.knot_idx, 2)

    def test_get_by_knots(self):
        seq = TimedSequenceList[float]()
        seq.append(t=0.0, v=0.0)
        seq.append(t=10.0, v=10.0)
        idx_valid0 = TimedSequenceKnotIndex(time=0.0, knot_idx=0, interpolate_weight=1.0)
        idx_valid1 = TimedSequenceKnotIndex(time=10.0, knot_idx=1, interpolate_weight=0.0)
        idx_mid0 = TimedSequenceKnotIndex(time=0.0, knot_idx=0, interpolate_weight=0.5)
        idx_mid1 = TimedSequenceKnotIndex(time=10.0, knot_idx=1, interpolate_weight=0.5)
        idx_invalid = TimedSequenceKnotIndex.invalid()
        result = seq.get_by_knots(idx_valid0, idx_valid1, interpolate_data=False)
        self.assertAlmostEqual(result, 10.0)
        result = seq.get_by_knots(idx_mid0, idx_mid1, interpolate_data=True)
        self.assertAlmostEqual(result, 5.0)
        result = seq.get_by_knots(idx_invalid, idx_valid1, interpolate_data=True)
        self.assertIsNone(result)
        result = seq.get_by_knots(idx_valid0, idx_invalid, interpolate_data=True)
        self.assertIsNone(result)

    def test_prune_by_time_edge_cases(self):
        seq = TimedSequenceList[float]()
        seq.append(t=0.0, v=0.0)
        seq.append(t=1.0, v=1.0)
        seq.append(t=2.0, v=2.0)
        seq.append(t=3.0, v=3.0)
        seq.append(t=4.0, v=4.0)
        seq.prune_by_time(2.5)
        self.assertEqual(len(seq), 3)
        t = seq.begin()
        self.assertAlmostEqual(t, 2.0)
        t = seq.end()
        self.assertAlmostEqual(t, 4.0)
        seq2 = TimedSequenceList[float]()
        seq2.append(t=1.0, v=1.0)
        seq2.prune_by_time(0.5)
        self.assertEqual(len(seq2), 1)
        seq2.prune_by_time(1.5)
        self.assertEqual(len(seq2), 1)

    def test_append_reject_duplicate_time(self):
        seq = TimedSequenceList[float]()
        self.assertTrue(seq.append(t=1.0, v=1.0))
        self.assertFalse(seq.append(t=1.0, v=2.0))
        self.assertFalse(seq.append(t=0.9, v=3.0))
        self.assertFalse(seq.append(t=0.9999, v=4.0))
        self.assertTrue(seq.append(t=1.0002, v=5.0))
        self.assertEqual(len(seq), 2)

    def test_get_vector_state_at_time(self):
        seq = TimedSequenceList[np.ndarray]()
        seq.append(t=0.0, v=np.array([0.0, 1.0, 2.0]))
        seq.append(t=5.0, v=np.array([5.0, 6.0, 7.0]))
        seq.append(t=10.0, v=np.array([10.0, 11.0, 12.0]))
        result = TimedSequenceInterface.get_vector_state_at_time(seq, 0.0, None)
        self.assertIsNotNone(result)
        np.testing.assert_array_almost_equal(result, np.array([0.0, 1.0, 2.0]))
        result = TimedSequenceInterface.get_vector_state_at_time(seq, 5.0, None)
        self.assertIsNotNone(result)
        np.testing.assert_array_almost_equal(result, np.array([5.0, 6.0, 7.0]))
        result = TimedSequenceInterface.get_vector_state_at_time(seq, 2.5, None)
        self.assertIsNotNone(result)
        np.testing.assert_array_almost_equal(result, np.array([2.5, 3.5, 4.5]))
        result = TimedSequenceInterface.get_vector_state_at_time(seq, -1.0, None)
        self.assertIsNotNone(result)
        np.testing.assert_array_almost_equal(result, np.array([0.0, 1.0, 2.0]))
        result = TimedSequenceInterface.get_vector_state_at_time(seq, 15.0, None)
        self.assertIsNotNone(result)
        np.testing.assert_array_almost_equal(result, np.array([10.0, 11.0, 12.0]))
        result = TimedSequenceInterface.get_vector_state_at_time(seq, 5.0, [0, 2])
        np.testing.assert_array_almost_equal(result, np.array([5.0, 6.0, 7.0]))
        result = TimedSequenceInterface.get_vector_state_at_time(seq, 2.5, [0, 2])
        np.testing.assert_array_almost_equal(result, np.array([5.0, 3.5, 7.0]))

    def test_timed_sequence_interface_invalid(self):
        idx = TimedSequenceKnotIndex.invalid()
        self.assertFalse(idx.is_valid)
        self.assertEqual(idx.knot_idx, -1)
        self.assertEqual(idx.time, -1.0)

    def test_interpolation_consistency(self):
        seq = TimedSequenceList[float]()
        for i in range(20):
            seq.append(t=float(i), v=float(i * 2))
        for t_test in [0.0, 0.5, 1.0, 1.5, 5.5, 10.25, 15.75, 19.0]:
            result = seq.get_at_time(t_test, interpolate_data=True)
            expected = t_test * 2
            self.assertAlmostEqual(result, expected, places=5)


if __name__ == '__main__':
    unittest.main()
