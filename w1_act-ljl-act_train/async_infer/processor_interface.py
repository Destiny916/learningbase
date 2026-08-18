from dataclasses import dataclass
from typing import Callable, Tuple, Union
from async_infer.async_infer_typedef import *
from async_infer.timed_sequence_array import TimedSequenceArray
from async_infer.policy_client_interface import *


# For command publish
class CommandStatePublishInterface(object):

    def __init__(self):
        pass

    def __call__(self, t_now: float, cmd_state: NpArray1d, seq_index: int):
        pass


# Combine with callable
CommandStatePublishFunctor = Union[CommandStatePublishInterface, Callable[[float, NpArray1d, int], None]]


# Interface for determine new policy invoke
class CheckShouldStartNewPolicyInvokeInterface(object):

    def __init__(self, invoke_after_trajectory_ratio: float = 1.0, time_before_trajectory_end: float = 0.0):
        invoke_ratio = min(max(invoke_after_trajectory_ratio, 0.01), 1.0)
        self._invoke_after_trajectory_ratio = invoke_ratio
        self._time_before_end = max(time_before_trajectory_end, 0.0)

    def __call__(
            self, observation: RolloutClientObservation,
            active_info_info: PolicyClientInvokeInfo,
            t_now: float, seq_index_now: int,
            cmd_trajectory: TimedSequenceArray) -> bool:
        return CheckShouldStartNewPolicyInvokeInterface.check_by_trajectory_time(
            active_info_info=active_info_info, t_now=t_now, cmd_trajectory=cmd_trajectory,
            invoke_after_trajectory_time_ratio=self._invoke_after_trajectory_ratio,
            time_before_end=self._time_before_end
        )

    @staticmethod
    def check_by_trajectory_time(active_info_info: PolicyClientInvokeInfo,
                                 t_now: float,
                                 cmd_trajectory: TimedSequenceArray,
                                 invoke_after_trajectory_time_ratio: float,
                                 time_before_end: float):
        # Already invoke
        if active_info_info.reqeust_id >= 0:
            return False

        # No trajectory
        if cmd_trajectory is None:
            return True

        # Check by time
        cmd_begin_time = cmd_trajectory.begin()
        cmd_end_time = cmd_trajectory.end()
        invoke_ratio = min(max(invoke_after_trajectory_time_ratio, 0.01), 1.0)
        invoke_time: float = invoke_ratio * cmd_end_time + (1.0 - invoke_ratio) * cmd_begin_time
        return t_now >= invoke_time or t_now >= cmd_end_time - time_before_end


# Combine with functor
ShouldStartNewPolicyInvokeFunctor = Union[
    CheckShouldStartNewPolicyInvokeInterface,
    Callable[[RolloutClientObservation, PolicyClientInvokeInfo, float, int, TimedSequenceArray], bool]
]


# Processor for observation
@dataclass
class ObservationProcessMeta(object):
    rollout_client_t_now: float
    rollout_client_seq_index: int
    is_state_only: bool


class ObservationProcessorInterface(object):

    def __init__(self):
        pass

    def __call__(self,
                 observation: ObservationMap,
                 meta: ObservationProcessMeta) -> Tuple[Union[ObservationMap, None], Union[str, None]]:
        return None, None


# Combine with functor
ObservationProcessFunctor = Union[
    ObservationProcessorInterface,
    Callable[[ObservationMap, ObservationProcessMeta], Tuple[Union[ObservationMap, None], Union[str, None]]]
]

__all__ = [
    'CommandStatePublishInterface',
    'CommandStatePublishFunctor',
    'CheckShouldStartNewPolicyInvokeInterface',
    'ShouldStartNewPolicyInvokeFunctor',
    'ObservationProcessMeta',
    'ObservationProcessorInterface',
    'ObservationProcessFunctor'
]
