import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Union, Optional, Tuple
from async_infer.gather_obs_typedef import GatherDataFunctors
from async_infer.timed_sequence_array import *
from async_infer.async_infer_typedef import *
from async_infer.rollout_client_gather_data import *
from async_infer.processor_interface import *
from async_infer.policy_client_interface import *
from async_infer.rollout_client_base import *


@dataclass(frozen=True)
class RolloutClientOutputOperation(object):
    # Interface to publish command
    publish_cmd_fn: Optional[CommandStatePublishFunctor] = None

    # Logging
    logger: Union[Callable[[str], None], None] = None


@dataclass(frozen=True)
class RolloutClientInvokePolicyOperation(object):
    # Client for invoke the policy
    policy_client: Optional[PolicyClientInterface] = None

    # Time to invoke
    should_start_new_policy_invoke: Optional[ShouldStartNewPolicyInvokeFunctor] = None


# For observation operation
@dataclass(frozen=True)
class RolloutClientObservationOperationBuffered(object):
    # Keys for observation
    observation_keys: AsyncInferObservationKeys

    # Functors for gather data
    gather_data_fn: GatherDataFunctors

    # These are optional
    find_sync_time_option: FindSyncTimeOption = field(default_factory=FindSyncTimeOption.sync_to_images)
    observation_processor: Optional[ObservationProcessFunctor] = None


# Func for async policy invoke
RolloutClientSimpleObservationFunc = Callable[
    # Input argument: (timepoint_now, is_invoke_active)
    [TimePointAndSequenceIndex, bool],
        # Output: observation
    RolloutClientObservation
]

# Operation
RolloutClientObservationOperation = Union[
    RolloutClientObservationOperationBuffered, RolloutClientSimpleObservationFunc, None]


class RolloutClientFunctor(RolloutClientBase):

    def __init__(self,
                 command_option: RolloutClientCommandOption,
                 observation_operation: RolloutClientObservationOperation = None,
                 invoke_operation: Optional[RolloutClientInvokePolicyOperation] = None,
                 output_operation: Optional[RolloutClientOutputOperation] = None):
        super(RolloutClientFunctor, self).__init__(command_option)

        # For observation
        self._observation_operation = observation_operation
        self._gather_buffer: Optional[GatherObservationDataSequenceBuffer] = None
        self._use_simple_fn_observation: bool = False
        if observation_operation is not None:
            if isinstance(observation_operation, RolloutClientObservationOperationBuffered):
                self._gather_buffer = GatherObservationDataSequenceBuffer(
                    state_dim_config=command_option.state_dim_config)
                self._gather_buffer.initialize(self._observation_operation.gather_data_fn,
                                               self._observation_operation.observation_keys)
            else:
                assert callable(observation_operation)
                self._use_simple_fn_observation = True

        # For invoke
        self._invoke_operation = invoke_operation if invoke_operation is not None else RolloutClientInvokePolicyOperation()
        self._output_operation = output_operation if output_operation is not None else RolloutClientOutputOperation()

    # Impl of interface
    def _take_observation(self,
                          timepoint_now: TimePointAndSequenceIndex,
                          is_invoke_active: bool) -> RolloutClientObservation:
        # Depends on case
        if self._gather_buffer is not None:
            return self._take_observation_buffered(timepoint_now, is_invoke_active)
        if self._use_simple_fn_observation:
            return self._observation_operation(timepoint_now, is_invoke_active)

        # Not active
        return RolloutClientObservation.invalid()

    def _loop_maybe_invoke(
            self,
            timepoint_now: TimePointAndSequenceIndex,
            observation: RolloutClientObservation,
            current_cmd_trajectory: TimedSequenceArray,
            current_invoke_info: PolicyClientInvokeInfo
    ) -> Tuple[Optional[PolicyClientInvokeInfo], Optional[str]]:
        # One is active
        should_invoke = self._check_should_invoke_now(timepoint_now=timepoint_now, observation=observation,
                                                      current_cmd_trajectory=current_cmd_trajectory,
                                                      current_invoke_info=current_invoke_info)
        if not should_invoke:
            return None, None

        # Into invoke
        return self._invoke_policy_client(timepoint_now=timepoint_now, observation=observation,
                                          current_cmd_trajectory=current_cmd_trajectory)

    def _loop_peek_response(
            self,
            timepoint_now: TimePointAndSequenceIndex,
            current_invoke_info: PolicyClientInvokeInfo) -> Optional[PolicyClientResponse]:
        # Check status
        if current_invoke_info.reqeust_id < 0 or self._invoke_operation.policy_client is None:
            return PolicyClientResponse(request_meta=current_invoke_info.meta, state_trajectory=None,
                                        error_str='Invalid Policy Client')

        # Take response
        request_id = current_invoke_info.reqeust_id
        response = self._invoke_operation.policy_client.try_take_response(
            request_id=request_id, current_timepoint=timepoint_now)
        return response

    def _loop_publish_command(self, timepoint_now: TimePointAndSequenceIndex, command: np.ndarray):
        if self._output_operation.publish_cmd_fn is not None:
            self._output_operation.publish_cmd_fn(timepoint_now.time, command, timepoint_now.seq_index)

    def _logging(self, content: Union[str, bytes]):
        if self._output_operation.logger is not None:
            self._output_operation.logger(content)

    def _on_loop_finish(self, timepoint_now: TimePointAndSequenceIndex):
        if self._gather_buffer is None:
            return

        # Prune the buffer
        t_prune = max(timepoint_now.time - 0.5, 0.0)
        self._gather_buffer.prune_by_time(t_oldest_to_keep=t_prune)

    # Detailed implementation and/or interface for next level
    def _invoke_policy_client(self,
                              timepoint_now: TimePointAndSequenceIndex,
                              observation: RolloutClientObservation,
                              current_cmd_trajectory: TimedSequenceArray
                              ) -> Tuple[Optional[PolicyClientInvokeInfo], Optional[str]]:
        if self._invoke_operation.policy_client is None:
            return None, "Invalid policy client"

        # Make meta
        meta = PolicyClientRequestMeta(reqeust_time=timepoint_now.time, reqeust_seq_index=timepoint_now.seq_index,
                                       observation_sync_time=observation.sync_time)
        request_id = self._invoke_operation.policy_client.invoke_async(meta=meta, observation=observation,
                                                                       current_cmd_trajectory=current_cmd_trajectory)
        return PolicyClientInvokeInfo(meta=meta, reqeust_id=request_id), None

    def _take_observation_buffered(self,
                                   timepoint_now: TimePointAndSequenceIndex,
                                   is_invoke_active: bool) -> RolloutClientObservation:
        # First gather data
        assert self._gather_buffer is not None
        self._gather_buffer.obtain_data(
            request_time=timepoint_now.time, gather_fn=self._observation_operation.gather_data_fn)
        obs_sync_time = self._gather_buffer.find_sync_time(obs_keys=self._observation_operation.observation_keys,
                                                           find_option_in=self._observation_operation.find_sync_time_option)
        if obs_sync_time is None:
            return RolloutClientObservation(sync_time=-1.0, observation_map=None,
                                            process_obs_status_str='Cannot find legal sync time')

        # Valid sync time
        observation, obs_status_str = self._gather_buffer.make_observation_map(
            request_time=obs_sync_time, state_only=is_invoke_active)

        # Processor
        if self._observation_operation.observation_processor is not None:
            proc_meta = ObservationProcessMeta(rollout_client_t_now=timepoint_now.time,
                                               rollout_client_seq_index=timepoint_now.seq_index,
                                               is_state_only=is_invoke_active)
            observation, process_obs_status_str = self._observation_operation.observation_processor(
                observation=observation,
                meta=proc_meta)
            if process_obs_status_str is not None: obs_status_str = process_obs_status_str

        # Done
        return RolloutClientObservation.from_observation_map(observation_map=observation,
                                                             status_str=obs_status_str)

    def _check_should_invoke_now(self, timepoint_now: TimePointAndSequenceIndex,
                                 observation: RolloutClientObservation,
                                 current_cmd_trajectory: TimedSequenceArray,
                                 current_invoke_info: PolicyClientInvokeInfo) -> bool:
        if self._invoke_operation.should_start_new_policy_invoke is not None:
            return self._invoke_operation.should_start_new_policy_invoke(observation=observation,
                                                                         active_info_info=current_invoke_info,
                                                                         t_now=timepoint_now.time,
                                                                         seq_index_now=timepoint_now.seq_index,
                                                                         cmd_trajectory=current_cmd_trajectory)

        # One is active
        if current_invoke_info.reqeust_id >= 0:
            return False

        # Check trajectory
        return CheckShouldStartNewPolicyInvokeInterface.check_by_trajectory_time(
            active_info_info=current_invoke_info, t_now=timepoint_now.time, cmd_trajectory=current_cmd_trajectory,
            invoke_after_trajectory_time_ratio=1.0, time_before_end=0.0)


__all__ = [
    "RolloutClientOutputOperation",
    "RolloutClientInvokePolicyOperation",
    "RolloutClientObservationOperationBuffered",
    "RolloutClientObservationOperation",
    "RolloutClientFunctor"
]
