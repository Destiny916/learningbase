import copy
from dataclasses import dataclass, field
from typing import Dict, Union, Optional, Tuple
import numpy as np
from async_infer.timed_sequence_array import *
from async_infer.merge_trajectory import *
from async_infer.policy_client_interface import *
from async_infer.async_infer_typedef import *


@dataclass(frozen=True)
class RolloutClientCommandOption(object):
    # Required
    state_dim_config: AsyncInferStateDimensionConfig

    # Optional
    wait_response_timeout: float = -1.0
    merge_option: MergeTrajectoryOption = field(default_factory=MergeTrajectoryOption.default_option)

    # Handy access
    @property
    def merge_type(self) -> MergeTrajectoryType:
        return self.merge_option.merge_type

    @property
    def merge_blend_ratio(self) -> float:
        return self.merge_option.merge_blend_ratio


class RolloutClientBase(object):

    def __init__(self, command_option: RolloutClientCommandOption):
        # Option
        self._cmd_option = command_option

        # Invoke info
        self._policy_invoke_info = PolicyClientInvokeInfo.invalid()

        # Trajectory
        self._command_trajectory: Optional[TimedSequenceArray] = None
        self._loop_seq_index: int = 0

    def loop_once(self, t_now: float):
        # Update seq index
        loop_seq_index = self._loop_seq_index
        self._loop_seq_index += 1
        time_point = TimePointAndSequenceIndex(time=t_now, seq_index=loop_seq_index)

        # Invoke callback
        self._on_loop_begin(timepoint_now=time_point)

        # Take observation
        observation = self._take_observation(timepoint_now=time_point,
                                             is_invoke_active=(self._policy_invoke_info.reqeust_id >= 0))
        if not observation.is_valid:
            status_str = observation.process_obs_status_str if observation.process_obs_status_str is not None else "No status"
            logging_str = f'Invalid observation (t_now: {t_now} and seq_index: {loop_seq_index}), status: {status_str}'
            self._logging(content=logging_str)
            return

        # Init of trajectory
        if self._command_trajectory is None:
            self._command_trajectory = TimedSequenceArray.from_one_point(observation.state, t_now)

        # Maybe invoke the policy
        if self._policy_invoke_info.reqeust_id < 0:
            new_invoke_info, new_invoke_status_str = self._loop_maybe_invoke(timepoint_now=time_point,
                                                                             observation=observation,
                                                                             current_cmd_trajectory=self._command_trajectory,
                                                                             current_invoke_info=self._policy_invoke_info)
            if new_invoke_status_str is not None: self._logging(content=new_invoke_status_str)
            if new_invoke_info is not None: self._policy_invoke_info = new_invoke_info

        # Maybe update cmd
        self._loop_peek_response_update_command_states(observation=observation, timepoint_now=time_point)

        # Send command
        assert self._command_trajectory is not None
        command = TimedSequenceArray.get_vector_state_at_time(self._command_trajectory, t_now,
                                                              self._cmd_option.state_dim_config.discrete_tool_state_indices)
        if command is not None:
            self._loop_publish_command(timepoint_now=time_point, command=command)

        # Invoke callback
        self._on_loop_finish(timepoint_now=time_point)

    # These are interface for derived class
    def _take_observation(self,
                          timepoint_now: TimePointAndSequenceIndex,
                          is_invoke_active: bool) -> RolloutClientObservation:
        return RolloutClientObservation.invalid()

    def _loop_maybe_invoke(
            self,
            timepoint_now: TimePointAndSequenceIndex,
            observation: RolloutClientObservation,
            current_cmd_trajectory: TimedSequenceArray,
            current_invoke_info: PolicyClientInvokeInfo
    ) -> Tuple[Optional[PolicyClientInvokeInfo], Optional[str]]:
        return None, None

    def _loop_peek_response(
            self,
            timepoint_now: TimePointAndSequenceIndex,
            current_invoke_info: PolicyClientInvokeInfo) -> Optional[PolicyClientResponse]:
        return None

    def _loop_publish_command(self, timepoint_now: TimePointAndSequenceIndex, command: np.ndarray):
        pass

    def _logging(self, content: Union[str, bytes]):
        pass

    def _on_loop_begin(self, timepoint_now: TimePointAndSequenceIndex):
        pass

    def _on_loop_finish(self, timepoint_now: TimePointAndSequenceIndex):
        pass

    # These are for internal implementation
    def _loop_peek_response_update_command_states(
            self,
            observation: RolloutClientObservation,
            timepoint_now: TimePointAndSequenceIndex):
        # One is active
        if self._policy_invoke_info.reqeust_id < 0:
            return

        # Get response
        response = self._loop_peek_response(
            timepoint_now=timepoint_now, current_invoke_info=self._policy_invoke_info)
        if response is None:
            # Check timeout
            invoke_time: float = self._policy_invoke_info.meta.reqeust_time
            t_now: float = timepoint_now.time
            wait_timeout: float = self._cmd_option.wait_response_timeout
            if wait_timeout > 0 and (t_now > invoke_time + wait_timeout):
                self._logging(
                    f'Receive timeout for request id {self._policy_invoke_info.reqeust_id}, t_now: {t_now}, invoke_time: {invoke_time}, wait_timeout: {wait_timeout}')
                self._policy_invoke_info = PolicyClientInvokeInfo.invalid()

            # Done
            return

        # A message is obtained, mark the flag as invalid
        assert response is not None
        self._policy_invoke_info = PolicyClientInvokeInfo.invalid()

        # Check validity
        if not response.is_valid:
            error_msg = 'No Message' if response.error_str is None else response.error_str
            self._logging(f'Receive invalid response with error msg: {error_msg}')
            return

        # Perform merging
        assert observation.state is not None and observation.is_valid
        assert self._command_trajectory is not None
        observation_state = MergeTrajectory.StateInfo(state=observation.state, time=observation.sync_time)
        merged: Union[TimedSequenceArray, None] = self._merge_trajectory(observation_state=observation_state,
                                                                         timepoint_now=timepoint_now,
                                                                         response=response)
        if merged is None:
            return

        # Update trajectory
        # assert abs(merged.begin() - timepoint_now.time) < 1e-5
        assert merged.begin() <= timepoint_now.time + 1e-5
        self._command_trajectory = merged

    def _merge_trajectory(
            self,
            observation_state: MergeTrajectory.StateInfo,
            timepoint_now: TimePointAndSequenceIndex,
            response: PolicyClientResponse) -> Union[TimedSequenceArray, None]:
        assert response.is_valid
        policy_sync_time: float = response.request_meta.observation_sync_time
        trajectory_begin_time = response.state_trajectory.begin()
        time_offset: Optional[float] = None
        if abs(trajectory_begin_time) < 1e-3:
            # Need offset
            is_merge_by_append = (self._cmd_option.merge_type == MergeTrajectoryType.MergeByAppend)
            time_offset = self._command_trajectory.end() if is_merge_by_append else policy_sync_time

        # Make new trajectory and apply offset if necessary
        new_trajectory = response.state_trajectory
        if time_offset is not None:
            new_trajectory_points = ensure_immutable_numpy(response.state_trajectory.raw_data_points)
            new_trajectory_times = np.copy(response.state_trajectory.raw_data_times)
            new_trajectory_times += time_offset
            new_trajectory = TimedSequenceArray(data=new_trajectory_points, time=new_trajectory_times)

        # Goto impl
        return MergeTrajectory.run(existing_cmd_trajectory=self._command_trajectory,
                                   observation_state=observation_state,
                                   new_trajectory=new_trajectory, timepoint_now=timepoint_now,
                                   state_config=self._cmd_option.state_dim_config,
                                   merge_option=self._cmd_option.merge_option)


__all__ = [
    "RolloutClientObservation",
    "RolloutClientCommandOption",
    'RolloutClientBase'
]
