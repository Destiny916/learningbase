import numpy as np
import copy
from typing import Union
from async_infer.async_infer_typedef import *
from async_infer.gather_obs_typedef import *
from async_infer.policy_client_interface import *
from async_infer.processor_interface import *
from async_infer.rollout_client_gather_data import *
from async_infer.rollout_client_base import *
from async_infer.rollout_client_functor import *
from async_infer.timed_sequence_array import TimedSequenceArray
from async_infer.merge_trajectory import *


class SimpleMoveSandboxPolicy(PolicyClientInterface):

    def __init__(self):
        super().__init__()
        self._delta_x: float = 0.1
        self._delta_y: float = 0.2
        self._cmd_trajectory_length: int = 20
        self._nominal_time = 1
        self._wait_time = -1

        # Response
        self._response: Union[PolicyClientResponse, None] = None

    def invoke_async(self, meta: PolicyClientRequestMeta, observation: ObservationMap,
                     current_cmd_trajectory: TimedSequenceArray) -> int:
        if observation.state is None:
            return -1

        # Get the state
        last_cmd_state = current_cmd_trajectory.raw_data_points[-1, :]
        assert len(last_cmd_state.shape) == 1 and last_cmd_state.shape[0] == 3
        delta_range = np.arange(self._cmd_trajectory_length).astype(np.float64) / float(self._cmd_trajectory_length - 1)
        delta_x = delta_range * self._delta_x
        delta_y = delta_range * self._delta_y

        # Make cmd
        output_cmd = np.zeros(shape=(self._cmd_trajectory_length, 3), dtype=np.float64)
        output_cmd[:, 0] = float(last_cmd_state[0])
        output_cmd[:, 1] = float(last_cmd_state[1])
        output_cmd[:, 2] = float(last_cmd_state[2])
        output_cmd[:, 0] += delta_x
        output_cmd[:, 1] += delta_y

        # Make time
        last_cmd_time = current_cmd_trajectory.raw_data_times[-1]
        cmd_times = delta_range * self._nominal_time
        # cmd_times[:] += last_cmd_time
        cmd_trajectory = TimedSequenceArray(data=output_cmd, time=cmd_times)

        # Make response
        self._response = PolicyClientResponse(request_meta=copy.deepcopy(meta), state_trajectory=cmd_trajectory)
        print(f'Invoke at {meta.reqeust_time}')
        return 1

    def try_take_response(self, request_id: int,
                          current_timepoint: TimePointAndSequenceIndex) -> Union[PolicyClientResponse, None]:
        if (self._wait_time > 0) and (
                current_timepoint.time - self._response.request_meta.reqeust_time < self._wait_time):
            return None
        if request_id == 1 and self._response is not None:
            response = self._response
            self._response = None
            begin_time = response.state_trajectory.begin()
            end_time = response.state_trajectory.end()
            print(f'Response at {current_timepoint.time}, trajectory time ({begin_time}, {end_time})')
            return response
        return None


class MockRobotService(object):

    def __init__(self, state_dim: int):
        self._state_dim = state_dim
        self._state = np.zeros(self._state_dim, )
        self._state_t = 0

    def peek_latest_state(self, request_time: float) -> GatherDataResponse:
        return GatherDataResponse(data=copy.deepcopy(self._state), obtained_time=self._state_t)

    def publish_cmd(self, cmd: NpArray1d, t_now: float, seq_index: int):
        self._state_t = t_now
        self._state = copy.deepcopy(cmd)
        print(f'Sending command at time {t_now} with seq_idx {seq_index}, command is {cmd}')


def _sandbox_gather_data():
    # Make config
    state_dim = 10
    state_dim_config = AsyncInferStateDimensionConfig(state_dim=state_dim, discrete_tool_state_indices=[9])
    obs_keys = AsyncInferObservationKeys(state_key='state', rgb_images=['img0'], depth_images=['depth0'])

    # Gather data fn
    gather_functors = GatherDataFunctors()

    from typing import Callable
    from async_infer.async_infer_typedef import NpArray1d
    state_gather_fn: Callable[[float], GatherDataResponse[NpArray1d]] = lambda t: GatherDataResponse(
        data=np.random.randn(state_dim, ), obtained_time=t)
    gather_functors.state_gather_fn = state_gather_fn

    gather_functors.gather_tensor_fn_dict['img0'] = lambda t: GatherDataResponse(data=np.random.randn(3, 224, 224),
                                                                                 obtained_time=t + 0.1)
    gather_functors.gather_tensor_fn_dict['depth0'] = lambda t: [GatherDataResponse(data=np.random.randn(224, 224),
                                                                                    obtained_time=t - 0.1),
                                                                 GatherDataResponse(data=np.random.randn(224, 224),
                                                                                    obtained_time=t - 0.2)]

    # Make buffer
    buffer = GatherObservationDataSequenceBuffer(state_dim_config)
    buffer.initialize(gather_functors, obs_keys)
    buffer.obtain_data(0, gather_functors)
    time = buffer.find_sync_time(obs_keys, FindSyncTimeOption.sync_to_images())
    assert time is not None and (abs(time + 0.1) < 1e-5)
    obs_map, _ = buffer.make_observation_map(request_time=time)
    print(obs_map.tensor_dict.keys())


def _sandbox_rollout_client():
    # Make config
    state_dim = 3
    state_dim_config = AsyncInferStateDimensionConfig(state_dim=state_dim, discrete_tool_state_indices=[2])
    rollout_cmd_config = RolloutClientCommandOption(state_dim_config=state_dim_config,
                                                    merge_option=MergeTrajectoryOption(
                                                        merge_type=MergeTrajectoryType.MergeByAppend,
                                                        merge_blend_ratio=0.1))

    # Mock robot srv
    robot_srv = MockRobotService(state_dim)

    # Gather data fn
    obs_keys = AsyncInferObservationKeys(state_key='state', rgb_images=['img0'], depth_images=['depth0'])
    gather_functors = GatherDataFunctors()
    gather_functors.state_gather_fn = lambda t: robot_srv.peek_latest_state(request_time=t)
    gather_functors.gather_tensor_fn_dict['img0'] = lambda t: GatherDataResponse(data=np.random.randn(3, 224, 224),
                                                                                 obtained_time=t)
    gather_functors.gather_tensor_fn_dict['depth0'] = lambda t: GatherDataResponse(data=np.random.randn(224, 224),
                                                                                   obtained_time=t)
    obs_operation = RolloutClientObservationOperationBuffered(observation_keys=obs_keys, gather_data_fn=gather_functors)

    # Publish cmd fn
    publish_cmd = lambda t, cmd, seq: robot_srv.publish_cmd(cmd=cmd, t_now=t, seq_index=seq)
    output_operation = RolloutClientOutputOperation(publish_cmd_fn=publish_cmd)

    # Policy operation
    policy = SimpleMoveSandboxPolicy()
    should_invoke_new = CheckShouldStartNewPolicyInvokeInterface(invoke_after_trajectory_ratio=0.5,
                                                                 time_before_trajectory_end=0.3)
    policy_operation = RolloutClientInvokePolicyOperation(policy_client=policy,
                                                          should_start_new_policy_invoke=should_invoke_new)

    # Make client
    client = RolloutClientFunctor(command_option=rollout_cmd_config, observation_operation=obs_operation,
                                  invoke_operation=policy_operation, output_operation=output_operation)
    interval = 0.1
    for i in range(100):
        t_i = float(i) * interval
        client.loop_once(t_now=t_i)


if __name__ == "__main__":
    # _sandbox_gather_data()
    _sandbox_rollout_client()
