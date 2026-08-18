import numpy as np
import copy
import time
from concurrent.futures import ThreadPoolExecutor
from async_infer.async_infer_typedef import *
from async_infer.gather_obs_typedef import *
from async_infer.policy_client_interface import *
from async_infer.policy_client_async import *
from async_infer.processor_interface import *
from async_infer.rollout_client_base import *
from async_infer.rollout_client_functor import *
from async_infer.timed_sequence_array import TimedSequenceArray
from async_infer.merge_trajectory import *


def now_sec() -> float:
    return time.time()


def simple_move_policy_func(invoke_info: PolicyClientInvokeInfo,
                            observation: ObservationMap,
                            current_cmd_trajectory: TimedSequenceArray) -> PolicyClientResponse:
    """Simple move policy function for PolicyClientByFunctor"""
    if observation.state is None:
        return PolicyClientResponse(
            request_meta=invoke_info.meta,
            state_trajectory=None,
            error_str="No state in observation"
        )

    # Get the state
    obs_state = copy.deepcopy(observation.state)
    assert len(obs_state.shape) == 1 and obs_state.shape[0] == 3

    # Policy parameters
    delta_x: float = 0.1
    delta_y: float = 0.2
    cmd_trajectory_length: int = 20
    nominal_time = 1

    # Generate trajectory
    delta_range = np.arange(cmd_trajectory_length).astype(np.float64) / float(cmd_trajectory_length - 1)
    delta_x_vals = delta_range * delta_x
    delta_y_vals = delta_range * delta_y

    # Make cmd
    output_cmd = np.zeros(shape=(cmd_trajectory_length, 3), dtype=np.float64)
    output_cmd[:, 0] = float(obs_state[0])
    output_cmd[:, 1] = float(obs_state[1])
    output_cmd[:, 2] = float(obs_state[2])
    output_cmd[:, 0] += delta_x_vals
    output_cmd[:, 1] += delta_y_vals

    # Make time
    cmd_times = delta_range * nominal_time
    cmd_trajectory = TimedSequenceArray(data=output_cmd, time=cmd_times)

    # Make response
    return PolicyClientResponse(
        request_meta=invoke_info.meta,
        state_trajectory=cmd_trajectory
    )


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


def _sandbox_rollout_client(use_physical_time: bool = False):
    # Make config
    state_dim = 3
    state_dim_config = AsyncInferStateDimensionConfig(state_dim=state_dim, discrete_tool_state_indices=[2])
    rollout_cmd_config = RolloutClientCommandOption(state_dim_config=state_dim_config,
                                                    merge_option=MergeTrajectoryOption(
                                                        MergeTrajectoryType.MergeByNearest, merge_blend_ratio=0.1))

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

    # Create PolicyClientByFunctor with thread pool executor
    executor = ThreadPoolExecutor(max_workers=1)
    policy = PolicyClientByFunctorOneActive(func=simple_move_policy_func, executor_now_owned=executor)
    should_invoke_new = CheckShouldStartNewPolicyInvokeInterface(invoke_after_trajectory_ratio=0.5)
    policy_operation = RolloutClientInvokePolicyOperation(policy_client=policy,
                                                          should_start_new_policy_invoke=should_invoke_new)

    # Make client
    client = RolloutClientFunctor(command_option=rollout_cmd_config, observation_operation=obs_operation,
                                  invoke_operation=policy_operation, output_operation=output_operation)
    interval = 0.1

    if not use_physical_time:
        for i in range(100):
            t_i = float(i) * interval
            client.loop_once(t_now=t_i)
    else:
        t_offset = now_sec()
        for i in range(100):
            t_i = now_sec() - t_offset
            client.loop_once(t_now=t_i)
            time.sleep(interval)

    # stop
    executor.shutdown(wait=True)


if __name__ == "__main__":
    _sandbox_rollout_client(use_physical_time=False)
