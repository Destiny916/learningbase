from typing import TypeVar, List, Tuple, Generic, Sequence, Union
from dataclasses import dataclass
from async_infer.async_infer_typedef import NpArray1d
import bisect

# Internal type-var
_T = TypeVar("_T")


# Typedef
@dataclass(frozen=True)
class TimedSequenceKnotIndex(object):
    knot_idx: int
    time: float
    interpolate_weight: float

    @staticmethod
    def invalid() -> 'TimedSequenceKnotIndex':
        return TimedSequenceKnotIndex(knot_idx=-1, time=-1.0, interpolate_weight=0.0)

    @property
    def is_valid(self):
        return self.knot_idx >= 0 and self.time >= 0.0


class TimedSequenceInterface(Generic[_T]):

    def __init__(self):
        pass

    def __len__(self) -> int:
        return 0

    @property
    def empty(self):
        return len(self) == 0

    def get_at_knot_idx(self, knot_idx: int) -> Union[_T, None]:
        return None

    def find_time_index(self, t: float) -> Tuple[TimedSequenceKnotIndex, TimedSequenceKnotIndex]:
        pass

    def get_by_knots(self, idx0: TimedSequenceKnotIndex, idx1: TimedSequenceKnotIndex,
                     interpolate_data: bool = False) -> Union[_T, None]:
        if (not idx0.is_valid) or (not idx1.is_valid):
            return None

        # Not interpolated, get the later one
        if not interpolate_data:
            return self.get_at_knot_idx(knot_idx=idx1.knot_idx)

        # The interpolated value
        return ((self.get_at_knot_idx(knot_idx=idx0.knot_idx) * idx0.interpolate_weight)
                + (self.get_at_knot_idx(knot_idx=idx1.knot_idx) * idx1.interpolate_weight))

    def get_at_time(self, t: float, interpolate_data: bool = False) -> Union[_T, None]:
        idx0, idx1 = self.find_time_index(t)
        return self.get_by_knots(idx0, idx1, interpolate_data)

    def append(self, t: float, v: _T) -> bool:
        return False

    def begin(self) -> float:
        return -1.0

    def end(self) -> float:
        return -1.0

    def prune_by_time(self, oldest_time_to_keep: float) -> bool:
        return False

    @staticmethod
    def get_vector_state_at_time(
            trajectory: 'TimedSequenceInterface[NpArray1d]',
            request_time: float,
            discrete_tool_states_in: Union[Sequence[int], None]) -> Union[NpArray1d, None]:
        idx0, idx1 = trajectory.find_time_index(t=request_time)
        if (not idx0.is_valid) or (not idx1.is_valid):
            return None

        # Run interpolated
        p0: NpArray1d = trajectory.get_at_knot_idx(knot_idx=idx0.knot_idx)
        p1: NpArray1d = trajectory.get_at_knot_idx(knot_idx=idx1.knot_idx)
        p_out: NpArray1d = p0 * idx0.interpolate_weight + p1 * idx1.interpolate_weight

        # The tools
        if discrete_tool_states_in is not None and len(discrete_tool_states_in) > 0:
            p_out[discrete_tool_states_in] = p1[discrete_tool_states_in]

        # Done
        return p_out


class TimedSequenceList(TimedSequenceInterface[_T]):
    # Constants
    MIN_TIME = 1e-4
    DIV_ADD_TIME = 1e-7
    NAIVE_SEARCH_BOUND = 5

    # Typedef
    @dataclass
    class RawKnotIndex(object):
        time: float
        raw_index: int
        interpolate_weight: float

    def __init__(self):
        super().__init__()
        self._data_sequence: List[_T] = list()
        self._time_sequence: List[float] = list()

    def __len__(self) -> int:
        return len(self._time_sequence)

    def find_time_index(self, t: float) -> Tuple[TimedSequenceKnotIndex, TimedSequenceKnotIndex]:
        return self._find_index_impl(self._time_sequence, t)

    def get_at_knot_idx(self, knot_idx: int) -> Union[_T, None]:
        if knot_idx < 0 or knot_idx >= len(self._data_sequence):
            return None
        return self._data_sequence[knot_idx]

    def append(self, t: float, v: _T) -> bool:
        if len(self._time_sequence) > 0 and t <= self._time_sequence[-1] + TimedSequenceList.MIN_TIME:
            return False
        self._data_sequence.append(v)
        self._time_sequence.append(float(t))
        return True

    def begin(self) -> float:
        if len(self._time_sequence) > 0:
            return self._time_sequence[0]
        return -1.0

    def end(self) -> float:
        if len(self._time_sequence) > 0:
            return self._time_sequence[-1]
        return -1.0

    @staticmethod
    def _find_index_impl(
            time_sequence: Sequence[float], t: float) -> Tuple[TimedSequenceKnotIndex, TimedSequenceKnotIndex]:
        # Return null only if empty case
        n_elements = len(time_sequence)
        if n_elements == 0:
            return TimedSequenceKnotIndex.invalid(), TimedSequenceKnotIndex.invalid()

        # Check special case
        n_elements = len(time_sequence)
        if t <= time_sequence[0] + TimedSequenceList.MIN_TIME:
            t0 = time_sequence[0]
            return TimedSequenceKnotIndex(time=t0, knot_idx=0, interpolate_weight=1.0), TimedSequenceKnotIndex(time=t0,
                                                                                                               knot_idx=0,
                                                                                                               interpolate_weight=0.0)
        elif t >= time_sequence[-1] - TimedSequenceList.MIN_TIME:
            t_last = time_sequence[-1]
            last_idx = n_elements - 1
            return TimedSequenceKnotIndex(time=t_last, knot_idx=last_idx,
                                          interpolate_weight=1.0), TimedSequenceKnotIndex(time=t_last,
                                                                                          knot_idx=last_idx,
                                                                                          interpolate_weight=0.0)

        # Using naive or binary search
        assert t > time_sequence[0] + TimedSequenceList.MIN_TIME
        assert t < time_sequence[-1] - TimedSequenceList.MIN_TIME
        if n_elements <= TimedSequenceList.NAIVE_SEARCH_BOUND:
            return TimedSequenceList._find_index_naive_inclusive(time_sequence, t, n_elements)
        return TimedSequenceList._find_index_bisect_inclusive(time_sequence, t, n_elements)

    @staticmethod
    def _find_index_naive_inclusive(
            time_sequence: Sequence[float],
            t: float, n_elements: int
    ) -> Tuple[TimedSequenceKnotIndex, TimedSequenceKnotIndex]:
        assert n_elements >= 2 and t >= time_sequence[0]
        index0 = 0
        index1 = -1
        while index0 < n_elements:
            index1_candidate = index0 + 1
            if t <= time_sequence[index1_candidate] + TimedSequenceList.MIN_TIME:
                index1 = index1_candidate
                break

            # Update index0
            index0 += 1

        assert index1 >= 0
        if index1 < 0:
            index1 = n_elements - 1
        t0: float = time_sequence[index0]
        t1: float = time_sequence[index1]
        # assert t0 <= t <= t1
        t_eval = t
        if t_eval < t0:
            assert t_eval >= t0 - TimedSequenceList.MIN_TIME
            t_eval = t0
        if t_eval > t1:
            assert t_eval <= t1 + TimedSequenceList.MIN_TIME
            t_eval = t1
        w1: float = (t_eval - t0) / (t1 - t0 + TimedSequenceList.DIV_ADD_TIME)
        w0: float = 1.0 - w1
        return TimedSequenceKnotIndex(time=t0, knot_idx=index0, interpolate_weight=w0), TimedSequenceKnotIndex(time=t1,
                                                                                                               knot_idx=index1,
                                                                                                               interpolate_weight=w1)

    @staticmethod
    def _find_index_bisect_inclusive(
            time_sequence: Sequence[float],
            t: float, n_elements: int
    ) -> Tuple[TimedSequenceKnotIndex, TimedSequenceKnotIndex]:
        search_index = bisect.bisect_left(a=time_sequence, x=t)
        index0 = search_index - 1
        index1 = search_index
        assert index1 < n_elements and index0 >= 0
        t0: float = time_sequence[index0]
        t1: float = time_sequence[index1]
        t_eval = t
        if t_eval < t0:
            assert t_eval >= t0 - TimedSequenceList.MIN_TIME
            t_eval = t0
        if t_eval > t1:
            assert t_eval <= t1 + TimedSequenceList.MIN_TIME
            t_eval = t1
        w1: float = (t_eval - t0) / (t1 - t0 + TimedSequenceList.DIV_ADD_TIME)
        w0: float = 1.0 - w1
        return TimedSequenceKnotIndex(time=t0, knot_idx=index0, interpolate_weight=w0), TimedSequenceKnotIndex(time=t1,
                                                                                                               knot_idx=index1,
                                                                                                               interpolate_weight=w1)

    def prune_by_time(self, oldest_time_to_keep: float):
        # Do nothing if empty
        n_existing_elements = len(self._time_sequence)
        if n_existing_elements == 0:
            return

        # Only keep the last
        if n_existing_elements == 1 or oldest_time_to_keep >= self._time_sequence[-1] - TimedSequenceList.MIN_TIME:
            new_data = [self._data_sequence[-1]]
            new_time = [self._time_sequence[-1]]
            self._data_sequence = new_data
            self._time_sequence = new_time
            return

        # Keep at least two elements
        assert oldest_time_to_keep < self._time_sequence[-1] and n_existing_elements >= 2
        keep_begin = -1
        for i in range(n_existing_elements):
            next_idx = i + 1
            if next_idx >= n_existing_elements - 1 or self._time_sequence[next_idx] >= oldest_time_to_keep:
                keep_begin = i
                break

        # Make new data
        assert keep_begin >= 0
        new_time = self._time_sequence[keep_begin:]
        new_data = self._data_sequence[keep_begin:]
        self._data_sequence = new_data
        self._time_sequence = new_time


__all__ = [
    'TimedSequenceKnotIndex',
    'TimedSequenceInterface',
    'TimedSequenceList'
]

# sandbox below
# Please refer to async_infer_tests/test_timed_sequence.py
