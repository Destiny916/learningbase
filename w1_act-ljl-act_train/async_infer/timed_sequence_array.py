import numpy as np
from typing import Tuple, Optional, Sequence
from async_infer.async_infer_typedef import (
    NpArray1d, NpArray2d,
    ensure_immutable_numpy
)
from async_infer.timed_sequence import *


class TimedSequenceArray(TimedSequenceInterface[NpArray1d]):

    def __init__(self, data: NpArray2d, time: NpArray1d, ensure_immutable: bool = True, ensure_float64: bool = True):
        super().__init__()
        assert len(data.shape) == 2
        data_in = data.astype(np.float64) if ensure_float64 else data
        self._data = ensure_immutable_numpy(data_in) if ensure_immutable else data_in

        # Rectify the shape
        assert len(time.shape) == 1 or (len(time.shape) == 2 and (time.shape[0] == 1 or time.shape[1] == 1))
        reshaped_time = time.reshape(-1)
        reshaped_time = reshaped_time.astype(np.float64) if ensure_float64 else reshaped_time
        self._time = ensure_immutable_numpy(reshaped_time) if ensure_immutable else reshaped_time
        assert self._time.shape[0] == self._data.shape[0]

        # Check time increasing
        assert self._time.shape[0] == 1 or ((self._time[1:] - self._time[:-1]).min() > 0)

    def __len__(self) -> int:
        return self._data.shape[0]

    @property
    def raw_data_points(self) -> NpArray2d:
        return self._data

    @property
    def raw_data_times(self):
        return self._time

    @staticmethod
    def from_one_point(data: NpArray1d, data_t: float) -> 'TimedSequenceArray':
        assert len(data.shape) == 1 or (len(data.shape) == 2 and (data.shape[0] == 1 or data.shape[1] == 1))
        data = data.reshape(-1)
        n_dim = data.shape[0]
        point = np.zeros(shape=(1, n_dim), dtype=np.float64)
        point[0, :] = data
        point.flags.writeable = False

        # Make
        time_array = np.zeros(shape=(1, 1), dtype=np.float64)
        time_array[0, 0] = data_t
        time_array.flags.writeable = False

        # Done
        return TimedSequenceArray(data=point, time=time_array, ensure_immutable=True)

    def find_time_index(self, t: float) -> Tuple[TimedSequenceKnotIndex, TimedSequenceKnotIndex]:
        # Into impl
        return TimedSequenceList._find_index_impl(self._time, t)

    def get_at_knot_idx(self, knot_idx: int) -> Optional[NpArray1d]:
        if knot_idx < 0 or knot_idx >= self._data.shape[0]:
            return None
        return self._data[knot_idx, :]

    def begin(self) -> float:
        if self._time.shape[0] > 0:
            return self._time[0]
        return -1.0

    def end(self) -> float:
        if self._time.shape[0] > 0:
            return self._time[-1]
        return -1.0

    @staticmethod
    def eval_at_multiple_times(data: NpArray2d, data_time: NpArray1d, eval_times: NpArray1d,
                               discrete_state_dims: Optional[Sequence[int]] = None) -> Optional[NpArray2d]:
        # Empty time
        assert data.shape[0] == data_time.shape[0]
        if data_time.shape[0] == 0:
            return None

        # Get time
        assert len(eval_times.shape) == 1 or (
                len(eval_times.shape) == 2 and (eval_times.shape[0] == 1 or eval_times.shape[1] == 1))
        t_requested = eval_times.reshape(-1)
        n_requested = t_requested.shape[0]
        if data_time.shape[0] == 1:
            output = np.tile(data, (n_requested, 1))
            return output

        # More than one times
        assert data_time.shape[0] >= 2
        index1 = np.searchsorted(a=data_time, v=t_requested, side='left')
        index0 = index1 - 1

        # Apply mask
        t_begin, t_last = data_time[0], data_time[-1]
        begin_mask = t_requested <= t_begin + TimedSequenceList.MIN_TIME
        end_mask = t_requested >= t_last - TimedSequenceList.MIN_TIME
        index1[begin_mask] = 0
        index1[end_mask] = data_time.shape[0] - 1
        index0[begin_mask] = 0
        index0[end_mask] = data_time.shape[0] - 1

        # Get time
        t0 = data_time[index0]
        t1 = data_time[index1]
        w1: NpArray1d = (t_requested - t0) / ((t1 - t0) + TimedSequenceList.DIV_ADD_TIME)
        w0: NpArray1d = 1.0 - w1
        w1[begin_mask] = 0.0
        w0[begin_mask] = 1.0
        w1[end_mask] = 1.0
        w0[end_mask] = 0.0

        # Rectify the shape
        w0 = np.expand_dims(w0, axis=1)
        w1 = np.expand_dims(w1, axis=1)

        # Get data
        data0 = data[index0, :]
        data1 = data[index1, :]

        # Go
        output = w0 * data0 + w1 * data1
        if discrete_state_dims is not None:
            output[..., discrete_state_dims] = data1[..., discrete_state_dims]

        # Done
        return output

    def get_at_times(self, t_in: NpArray1d,
                     discrete_state_dims: Optional[Sequence[int]] = None) -> Optional[NpArray2d]:
        return self.eval_at_multiple_times(data=self._data, data_time=self._time, eval_times=t_in,
                                           discrete_state_dims=discrete_state_dims)


__all__ = [
    'TimedSequenceArray'
]

# sandbox below
# Please refer to the test code in async_infer_tests/test_timed_sequence_array.py
