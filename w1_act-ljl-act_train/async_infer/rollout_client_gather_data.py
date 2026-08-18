from typing import Dict, Any, List, Tuple, Union
import numpy as np
from enum import Enum
from dataclasses import dataclass
from async_infer.gather_obs_typedef import *
from async_infer.timed_sequence import *
from async_infer.async_infer_typedef import *


class FindSyncTimeType(Enum):
    kSyncToState = 0
    kSyncToOneTensor = 1
    kSyncToImageTensors = 2


@dataclass
class FindSyncTimeOption(object):
    sync_type: FindSyncTimeType = FindSyncTimeType.kSyncToState
    tensor_key: Union[str, None] = None

    @staticmethod
    def default() -> 'FindSyncTimeOption':
        return FindSyncTimeOption(sync_type=FindSyncTimeType.kSyncToState, tensor_key=None)

    @staticmethod
    def sync_to_images() -> 'FindSyncTimeOption':
        return FindSyncTimeOption(sync_type=FindSyncTimeType.kSyncToImageTensors, tensor_key=None)


class GatherObservationDataSequenceBuffer(object):

    def __init__(self, state_dim_config: AsyncInferStateDimensionConfig):
        self._state_dim_config = state_dim_config
        self._state_buffer: TimedSequenceList[NpArray1d] = TimedSequenceList[NpArray1d]()
        self._tensor_buffer: Dict[str, TimedSequenceInterface[np.ndarray]] = dict()
        self._misc_buffer: Dict[str, TimedSequenceInterface[Any]] = dict()

    def initialize(self, gather_fn: GatherDataFunctors, obs_keys: AsyncInferObservationKeys):
        for k in gather_fn.gather_tensor_fn_dict.keys():
            self._tensor_buffer[k] = TimedSequenceList[np.ndarray]()
        for k in gather_fn.gather_misc_fn_dict.keys():
            self._misc_buffer[k] = TimedSequenceList[Any]()

    @staticmethod
    def _append_response_data(
            buffer: TimedSequenceInterface,
            response: Union[GatherDataResponse, List[GatherDataResponse]]):
        # Null
        if response is None:
            return

        # Just one element
        if isinstance(response, GatherDataResponse):
            buffer.append(t=response.obtained_time, v=response.data)
            return

        # Sort by time and append
        sorted_response: List[GatherDataResponse] = sorted(response, key=lambda x: x.obtained_time)
        for elem in sorted_response:
            buffer.append(t=elem.obtained_time, v=elem.data)

    def obtain_data(self, request_time: float, gather_fn: GatherDataFunctors):
        # Obtain the state
        state_response: GatherDataResponse = gather_fn.state_gather_fn(request_time)
        self._append_response_data(self._state_buffer, state_response)

        # Obtain others
        for k, fn in gather_fn.gather_tensor_fn_dict.items():
            assert k in self._tensor_buffer
            response_k: GatherDataResponse = fn(request_time)
            self._append_response_data(self._tensor_buffer[k], response_k)

        # Obtain others
        for k, fn in gather_fn.gather_misc_fn_dict.items():
            assert k in self._misc_buffer
            response_k: GatherDataResponse = fn(request_time)
            self._append_response_data(self._misc_buffer[k], response_k)

    def make_observation_map(self,
                             request_time: float,
                             state_only: bool = False) -> Tuple[ObservationMap, Union[str, None]]:
        # Get the state
        state_out = self._make_state_out(request_time=request_time)
        obs_map = ObservationMap(observation_time=request_time, state=state_out)
        if state_only:
            return obs_map, None

        # Items other than state
        status_str: Union[str, None] = None
        for k, v in self._tensor_buffer.items():
            value = v.get_at_time(t=request_time, interpolate_data=False)
            assert value is None or isinstance(value, np.ndarray)
            obs_map.tensor_dict[k] = value
        for k, v in self._misc_buffer.items():
            value = v.get_at_time(t=request_time, interpolate_data=False)
            obs_map.misc_dict[k] = value

        # Done
        return obs_map, status_str

    def _make_state_out(self, request_time: float) -> Union[NpArray1d, None]:
        return TimedSequenceInterface.get_vector_state_at_time(
            self._state_buffer, request_time, self._state_dim_config.discrete_tool_state_indices)

    def prune_by_time(self, t_oldest_to_keep: float):
        self._state_buffer.prune_by_time(t_oldest_to_keep)
        for k, v in self._tensor_buffer.items():
            v.prune_by_time(t_oldest_to_keep)
        for k, v in self._misc_buffer.items():
            v.prune_by_time(t_oldest_to_keep)

    def find_sync_time(self,
                       obs_keys: AsyncInferObservationKeys,
                       find_option_in: Union[FindSyncTimeOption, None] = None) -> Union[float, None]:
        # Get option
        find_option = find_option_in if find_option_in is not None else FindSyncTimeOption.default()
        if find_option.sync_type == FindSyncTimeType.kSyncToState:
            end_time = self._state_buffer.end()
            return end_time

        # One tensor
        if find_option.sync_type == FindSyncTimeType.kSyncToOneTensor:
            if find_option.tensor_key is None or (find_option.tensor_key not in self._tensor_buffer):
                return None
            buffer = self._tensor_buffer[find_option.tensor_key]
            end_time = buffer.end()
            return end_time

        # Obs tensors
        sync_time: Union[float, None] = None
        for k in obs_keys.rgb_images:
            if k in self._tensor_buffer:
                buffer = self._tensor_buffer[k]
                end_time = buffer.end()
                if sync_time is None:
                    sync_time = end_time
                else:
                    sync_time = min(sync_time, end_time)
            else:
                return None

        for k in obs_keys.depth_images:
            if k in self._tensor_buffer:
                buffer = self._tensor_buffer[k]
                end_time = buffer.end()
                if sync_time is None:
                    sync_time = end_time
                else:
                    sync_time = min(sync_time, end_time)
            else:
                return None

        # Done
        return sync_time if (sync_time is not None) else None


__all__ = ['FindSyncTimeOption', 'GatherObservationDataSequenceBuffer']

# sandbox below
# Please refet to async_infer_tests/rollout_client_sandbox
