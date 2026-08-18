import numpy as np
from typing import Tuple, Optional, Sequence
from dataclasses import dataclass
from enum import Enum
from async_infer.async_infer_typedef import (
    NpArray1d, NpArray2d,
    AsyncInferStateDimensionConfig,
    ensure_immutable_numpy,
    TimePointAndSequenceIndex
)
from async_infer.timed_sequence import *
from async_infer.timed_sequence_array import *


# Options for merge trajectory
class MergeTrajectoryType(Enum):
    # Merge at a point that is closet using weighted L2 norm
    MergeByNearest = 0

    # Merge by time of the new trajectory, existing trajectory
    # and current time.
    MergeByTime = 1

    # A special case of MergeByTime that always assuming the new trajectory
    # is started from the end of current command trajectory
    MergeByAppend = 2


@dataclass(frozen=True)
class MergeTrajectoryOption(object):
    merge_type: MergeTrajectoryType = MergeTrajectoryType.MergeByTime
    merge_blend_ratio: float = 0.1  # ratio of target trajectory for blending
    prefer_command_state_over_observation_state: bool = True

    @staticmethod
    def default_option() -> 'MergeTrajectoryOption':
        return MergeTrajectoryOption(merge_type=MergeTrajectoryType.MergeByTime, merge_blend_ratio=0.1,
                                     prefer_command_state_over_observation_state=True)


class MergeTrajectory:
    @dataclass(frozen=True)
    class StateInfo(object):
        state: NpArray1d
        time: float

    # Main interface
    @staticmethod
    def run(existing_cmd_trajectory: TimedSequenceArray,
            observation_state: StateInfo,
            new_trajectory: TimedSequenceArray,
            timepoint_now: TimePointAndSequenceIndex,
            state_config: AsyncInferStateDimensionConfig,
            merge_option: MergeTrajectoryOption) -> Optional[TimedSequenceArray]:
        # Merge from state
        merge_from_state = observation_state
        if merge_option.prefer_command_state_over_observation_state:
            next_cmd = TimedSequenceArray.get_vector_state_at_time(existing_cmd_trajectory, timepoint_now.time,
                                                                   state_config.discrete_tool_state_indices)
            merge_from_state = MergeTrajectory.StateInfo(state=next_cmd, time=timepoint_now.time)

        # By time
        if merge_option.merge_type == MergeTrajectoryType.MergeByTime or len(existing_cmd_trajectory) == 1:
            return MergeTrajectory.merge_by_time_general(existing_cmd_trajectory=existing_cmd_trajectory,
                                                         current_state=merge_from_state,
                                                         new_trajectory=new_trajectory,
                                                         merge_blend_ratio=merge_option.merge_blend_ratio,
                                                         state_config=state_config)

        # By append
        if merge_option.merge_type == MergeTrajectoryType.MergeByAppend:
            # Apply offset
            existing_end_time = existing_cmd_trajectory.end()
            shifted_new_trajectory = new_trajectory
            new_trajectory_begin_time: float = new_trajectory.begin()
            if abs(new_trajectory_begin_time - existing_end_time) > 1e-3:
                shifted_times = np.copy(new_trajectory.raw_data_times)
                shifted_times[:] += (existing_end_time - new_trajectory_begin_time)
                shifted_new_trajectory = TimedSequenceArray(data=new_trajectory.raw_data_points, time=shifted_times)

            # Go
            return MergeTrajectory.merge_by_time_general(existing_cmd_trajectory=existing_cmd_trajectory,
                                                         current_state=merge_from_state,
                                                         new_trajectory=shifted_new_trajectory,
                                                         merge_blend_ratio=merge_option.merge_blend_ratio,
                                                         state_config=state_config)

        # Nearest merge
        assert merge_option.merge_type == MergeTrajectoryType.MergeByNearest
        return MergeTrajectory.merge_by_nearest(current_state=merge_from_state,
                                                new_trajectory=new_trajectory,
                                                merge_blend_ratio=merge_option.merge_blend_ratio,
                                                state_config=state_config)

    @staticmethod
    def merge_by_time_general(
            existing_cmd_trajectory: TimedSequenceArray,
            current_state: StateInfo,
            new_trajectory: TimedSequenceArray,
            merge_blend_ratio: float,
            state_config: AsyncInferStateDimensionConfig):
        # Basic check
        if new_trajectory is None or (not isinstance(new_trajectory, TimedSequenceArray)):
            return None

        # Should not be single point trajectory
        n_trajectory_points = len(new_trajectory)
        if n_trajectory_points <= 1:
            return None

        # Get time
        t_now = current_state.time
        # Upon construction: existing_cmd_trajectory.begin() < t_construction, t_now > t_construction
        assert t_now + TimedSequenceList.MIN_TIME >= existing_cmd_trajectory.begin()
        existing_end = existing_cmd_trajectory.end()
        new_begin = new_trajectory.begin()
        n_existing_trajectory_points: int = len(existing_cmd_trajectory)

        # Case 0: not invoke from future state
        if t_now + TimedSequenceList.MIN_TIME >= new_begin or n_existing_trajectory_points == 1:
            return MergeTrajectory.merge_by_time_no_future_sync_time(existing_cmd_trajectory=existing_cmd_trajectory,
                                                                     current_state=current_state,
                                                                     new_trajectory=new_trajectory,
                                                                     merge_blend_ratio=merge_blend_ratio,
                                                                     state_config=state_config)

        # Case 1: future sync time
        return MergeTrajectory.merge_by_time_future_sync_time(existing_cmd_trajectory=existing_cmd_trajectory,
                                                              current_state=current_state,
                                                              new_trajectory=new_trajectory,
                                                              merge_blend_ratio=merge_blend_ratio,
                                                              state_config=state_config)

    @staticmethod
    def merge_by_time_future_sync_time(existing_cmd_trajectory: TimedSequenceArray,
                                       current_state: 'MergeTrajectory.StateInfo',
                                       new_trajectory: TimedSequenceArray,
                                       merge_blend_ratio: float,
                                       state_config: AsyncInferStateDimensionConfig):
        # Basic check
        if new_trajectory is None or (not isinstance(new_trajectory, TimedSequenceArray)):
            return None

        # Should not be single point trajectory
        n_trajectory_points = len(new_trajectory)
        if n_trajectory_points <= 1:
            return None

        # Get time
        t_now = current_state.time
        # Upon construction: existing_cmd_trajectory.begin() < t_construction, t_now > t_construction
        assert t_now + TimedSequenceList.MIN_TIME >= existing_cmd_trajectory.begin()
        existing_end = existing_cmd_trajectory.end()
        new_begin = new_trajectory.begin()
        n_existing_trajectory_points: int = len(existing_cmd_trajectory)

        assert t_now + TimedSequenceList.MIN_TIME < new_begin
        # We already run-out of the existing trajectory
        if existing_end <= t_now + TimedSequenceList.MIN_TIME:
            return MergeTrajectory.blend_start_point_and_trajectory_linear(
                start_point=current_state, trajectory=new_trajectory,
                discrete_state_dims=state_config.discrete_tool_state_indices,
                blend_time_as_ratio_of_trajectory_duration=merge_blend_ratio)

        # Might need split
        split_existing_until = min(new_begin, existing_end)
        assert t_now + TimedSequenceList.MIN_TIME < split_existing_until

        # Compute append_trajectory
        blending_state_start_value = TimedSequenceArray.get_vector_state_at_time(existing_cmd_trajectory,
                                                                                 split_existing_until,
                                                                                 state_config.discrete_tool_state_indices)
        blending_state_start = MergeTrajectory.StateInfo(state=blending_state_start_value, time=split_existing_until)
        blended_append_trajectory = MergeTrajectory.blend_start_point_and_trajectory_linear(
            start_point=blending_state_start, trajectory=new_trajectory,
            discrete_state_dims=state_config.discrete_tool_state_indices,
            blend_time_as_ratio_of_trajectory_duration=merge_blend_ratio)
        if blended_append_trajectory is None:
            return None

        # Re-sample existing trajectory
        n_points_ratio: float = (split_existing_until - t_now) / (
                existing_end - existing_cmd_trajectory.begin() + TimedSequenceList.MIN_TIME)
        n_resampled_points = int(n_points_ratio * float(n_existing_trajectory_points)) + 1
        n_resampled_points = max(n_resampled_points, 2)

        eval_times = np.arange(n_resampled_points).astype(np.float64) / float(n_resampled_points - 1)
        eval_times *= (split_existing_until - t_now)
        eval_times += t_now
        eval_times.flags.writeable = False
        evaluated_points = existing_cmd_trajectory.get_at_times(t_in=eval_times,
                                                                discrete_state_dims=state_config.discrete_tool_state_indices)

        # Append
        appended_data = np.concatenate((evaluated_points[:-1, :], blended_append_trajectory.raw_data_points), axis=0)
        appended_data_times = np.concatenate((eval_times[:-1], blended_append_trajectory.raw_data_times), axis=0)
        appended_data.flags.writeable = False
        appended_data_times.flags.writeable = False
        output = TimedSequenceArray(data=appended_data, time=appended_data_times, ensure_immutable=True)

        # Re-sample if too much
        max_n_points = 5 * len(new_trajectory)
        if len(output) > max_n_points:
            output = MergeTrajectory.resample_trajectory_impl(
                new_num_points_for_non_single_point_trajectory=max_n_points, input_trajectory=output,
                discrete_state_dims=state_config.discrete_tool_state_indices, new_begin_time_in=-1.0)

        # Done
        return output

    @staticmethod
    def merge_by_time_no_future_sync_time(
            existing_cmd_trajectory: TimedSequenceArray,
            current_state: 'MergeTrajectory.StateInfo',
            new_trajectory: TimedSequenceArray,
            merge_blend_ratio: float,
            state_config: AsyncInferStateDimensionConfig):
        # Basic check
        if new_trajectory is None or (not isinstance(new_trajectory, TimedSequenceArray)):
            return None

        # Should not be single point trajectory
        n_trajectory_points = len(new_trajectory)
        if n_trajectory_points <= 1:
            return None

        # Get time
        t_now = current_state.time
        # Upon construction: existing_cmd_trajectory.begin() < t_construction, t_now > t_construction
        # assert t_now + TimedSequenceList.MIN_TIME >= existing_cmd_trajectory.begin()
        existing_end = existing_cmd_trajectory.end()
        new_begin = new_trajectory.begin()
        new_end = new_trajectory.end()

        # We are evaluating the policy earlier
        assert t_now + TimedSequenceList.MIN_TIME >= new_begin or len(existing_cmd_trajectory) == 1

        # Split time
        new_trajectory_split_time = min(t_now, existing_end)
        if new_trajectory_split_time < new_begin + TimedSequenceList.MIN_TIME:
            return MergeTrajectory.blend_start_point_and_trajectory_linear(
                start_point=current_state, trajectory=new_trajectory,
                discrete_state_dims=state_config.discrete_tool_state_indices,
                blend_time_as_ratio_of_trajectory_duration=merge_blend_ratio)

        # Case 2: it is directly the end points
        if new_trajectory_split_time > new_end:
            return None

        # Run evaluate
        eval_times = np.arange(n_trajectory_points).astype(np.float64) / float(n_trajectory_points - 1)
        eval_times *= (new_end - new_trajectory_split_time)
        eval_times += new_trajectory_split_time
        eval_times.flags.writeable = False
        evaluated_points = new_trajectory.get_at_times(t_in=eval_times,
                                                       discrete_state_dims=state_config.discrete_tool_state_indices)
        evaluated_points.flags.writeable = False
        processed_trajectory = TimedSequenceArray(data=evaluated_points, time=eval_times)
        output_traj = MergeTrajectory.blend_start_point_and_trajectory_linear(
            start_point=current_state, trajectory=processed_trajectory,
            discrete_state_dims=state_config.discrete_tool_state_indices,
            blend_time_as_ratio_of_trajectory_duration=merge_blend_ratio)
        # print(f'Merge by time2 t_now: {t_now}, existing [{existing_begin}, {existing_end}], new [{new_begin}, {new_end}], merged [{output_traj.begin_time()[0]}, {output_traj.end_time()[0]}]')
        return output_traj

    @staticmethod
    def merge_by_nearest(
            current_state: StateInfo,
            new_trajectory: TimedSequenceArray,
            merge_blend_ratio: float,
            state_config: AsyncInferStateDimensionConfig,
            distance_weight_in: Optional[NpArray2d] = None
    ) -> Optional[TimedSequenceArray]:
        # Basic check
        if new_trajectory is None or (not isinstance(new_trajectory, TimedSequenceArray)):
            return None
        n_trajectory_points = len(new_trajectory)
        if n_trajectory_points <= 1:
            return None

        # At least two point
        assert n_trajectory_points >= 2
        point_diff = np.expand_dims(current_state.state, axis=0) - new_trajectory.raw_data_points
        distance_weight = distance_weight_in if (distance_weight_in is not None) else state_config.state_distance_weight
        if distance_weight is not None:
            assert len(distance_weight.shape) == 1
            weighted_diff = np.expand_dims(distance_weight, axis=0) * point_diff
            point_diff = weighted_diff

        # Get the norm
        point_diff_norm = np.linalg.norm(point_diff, axis=-1, keepdims=False)
        min_index = np.argmin(point_diff_norm)
        return MergeTrajectory.merge_by_knot_idx(current_state=current_state,
                                                 new_trajectory=new_trajectory,
                                                 merge_at_new_trajectory_knot_index=min_index,
                                                 discrete_state_dims=state_config.discrete_tool_state_indices,
                                                 merge_blend_ratio=merge_blend_ratio)

    @staticmethod
    def merge_by_knot_idx(
            current_state: StateInfo,
            new_trajectory: TimedSequenceArray,
            merge_at_new_trajectory_knot_index: int,
            discrete_state_dims: Optional[Sequence[int]],
            merge_blend_ratio: float) -> Optional[TimedSequenceArray]:
        # Basic check
        if new_trajectory is None or (not isinstance(new_trajectory, TimedSequenceArray)):
            return None

        # Case 1: this point is later than the full trajectory
        # _, begin_knot_index = new_trajectory.begin_time()
        # _, end_knot_index = new_trajectory.end_time()
        begin_knot_index, end_knot_index = 0, len(new_trajectory) - 1
        if merge_at_new_trajectory_knot_index >= end_knot_index:
            # Static point at current state
            return TimedSequenceArray.from_one_point(current_state.state, current_state.time)

        # Case 2: directly blend from start
        assert merge_at_new_trajectory_knot_index < end_knot_index
        if merge_at_new_trajectory_knot_index <= begin_knot_index:
            return MergeTrajectory.blend_start_point_and_trajectory_linear(
                start_point=current_state, trajectory=new_trajectory,
                discrete_state_dims=discrete_state_dims, blend_time_as_ratio_of_trajectory_duration=merge_blend_ratio)

        # Case 3: split and merge
        new_traj_begin = merge_at_new_trajectory_knot_index - begin_knot_index
        new_data_points = new_trajectory.raw_data_points[new_traj_begin:, ...]
        new_data_times = new_trajectory.raw_data_times[new_traj_begin:, ...]
        separated_traj = TimedSequenceArray(data=new_data_points, time=new_data_times)
        return MergeTrajectory.blend_start_point_and_trajectory_linear(
            start_point=current_state, trajectory=separated_traj, discrete_state_dims=discrete_state_dims,
            blend_time_as_ratio_of_trajectory_duration=merge_blend_ratio)

    @staticmethod
    def resample_trajectory(
            input_trajectory: TimedSequenceArray,
            discrete_state_dims: Optional[Sequence[int]],
            new_begin_time_in: float = -1.0
    ) -> TimedSequenceArray:
        return MergeTrajectory.resample_trajectory_impl(new_num_points_for_non_single_point_trajectory=-1,
                                                        input_trajectory=input_trajectory,
                                                        discrete_state_dims=discrete_state_dims,
                                                        new_begin_time_in=new_begin_time_in)

    @staticmethod
    def resample_trajectory_impl(
            new_num_points_for_non_single_point_trajectory: int,
            input_trajectory: TimedSequenceArray,
            discrete_state_dims: Optional[Sequence[int]],
            new_begin_time_in: float
    ) -> TimedSequenceArray:
        # Get input
        begin_time = input_trajectory.begin()
        new_begin_time = begin_time if new_begin_time_in < 0 else new_begin_time_in
        n_trajectory_points = len(input_trajectory)
        assert n_trajectory_points >= 1

        # Special case
        if n_trajectory_points == 1:
            raw_data_points = ensure_immutable_numpy(input_trajectory.raw_data_points)
            raw_data_times = np.copy(input_trajectory.raw_data_times)
            raw_data_times[:] = new_begin_time
            raw_data_times.flags.writeable = False
            return TimedSequenceArray(raw_data_points, raw_data_times, ensure_immutable=True)

        # At least two points
        assert n_trajectory_points >= 2
        end_time = input_trajectory.end()
        assert end_time > begin_time
        duration = end_time - begin_time

        # Maybe new number of points
        n_new_trajectory_points = n_trajectory_points
        if new_num_points_for_non_single_point_trajectory >= 2:
            n_new_trajectory_points = new_num_points_for_non_single_point_trajectory
        n_new_trajectory_points = max(n_new_trajectory_points, 2)

        # Run eval
        eval_times = np.arange(n_new_trajectory_points, dtype=np.float64) * (
                duration / float(n_new_trajectory_points - 1))
        eval_times += begin_time
        resampled_data = input_trajectory.get_at_times(t_in=eval_times, discrete_state_dims=discrete_state_dims)
        resampled_data.flags.writeable = False

        # Mark no-longer writeable and go
        eval_times += (new_begin_time - begin_time)
        eval_times.flags.writeable = False
        return TimedSequenceArray(data=resampled_data, time=eval_times)

    @staticmethod
    def make_two_points_trajectory(start_point: StateInfo, end_point: NpArray1d,
                                   end_point_time: float) -> TimedSequenceArray:
        combined_state = np.zeros(shape=(2, start_point.state.shape[0]), dtype=np.float64)
        combined_state[0, :] = start_point.state
        combined_state[1, :] = end_point

        # Time
        combined_times = np.zeros(shape=(2,), dtype=np.float64)
        combined_times[0] = start_point.time
        combined_times[1] = max(end_point_time, start_point.time + 0.01)
        return TimedSequenceArray(combined_state, combined_times)

    @staticmethod
    def blend_start_point_and_trajectory_linear(
            start_point: StateInfo,
            trajectory: TimedSequenceArray,
            discrete_state_dims: Optional[Sequence[int]],
            blend_time_as_ratio_of_trajectory_duration: float = -1.0) -> TimedSequenceArray:
        # Check status
        n_trajectory_points = len(trajectory)
        assert n_trajectory_points >= 1
        trajectory_begin_time = trajectory.begin()
        if n_trajectory_points == 1:
            # Two points
            return MergeTrajectory.make_two_points_trajectory(start_point,
                                                              trajectory.get_at_knot_idx(0),
                                                              trajectory_begin_time)

        # At least two points
        assert n_trajectory_points >= 2
        blend_ratio = blend_time_as_ratio_of_trajectory_duration if blend_time_as_ratio_of_trajectory_duration >= 0 else 1.0
        blend_ratio = min(blend_ratio, 1.0)
        if blend_ratio < TimedSequenceList.MIN_TIME:
            data_points = ensure_immutable_numpy(trajectory.raw_data_points)
            data_points_times = np.copy(trajectory.raw_data_times)
            data_points_times += (start_point.time - trajectory_begin_time)
            data_points_times.flags.writeable = False
            return TimedSequenceArray(data=data_points, time=data_points_times, ensure_immutable=True)

        # Need blend
        n_blend_knots = int(blend_ratio * n_trajectory_points) + 1
        n_blend_knots = min(n_blend_knots, n_trajectory_points)
        n_blend_knots = max(n_blend_knots, 2)

        # Run blend
        trajectory = MergeTrajectory.resample_trajectory(trajectory, discrete_state_dims)
        raw_data_points = trajectory.raw_data_points
        raw_data_points_before_blend = raw_data_points[:n_blend_knots, ...]
        raw_data_points_after_blend = raw_data_points[n_blend_knots:, ...]
        delta_w_begin = start_point.state - raw_data_points[0, ...]
        delta_w_begin = np.expand_dims(delta_w_begin, axis=0)
        delta_w_begin = np.tile(delta_w_begin, (n_blend_knots, 1))

        delta_scale = 1.0 - np.arange(n_blend_knots, dtype=np.float64) / float(n_blend_knots - 1)
        delta_scale = np.expand_dims(delta_scale, axis=1)
        scaled_delta = delta_scale * delta_w_begin

        # Zero-out state dim
        if discrete_state_dims is not None:
            scaled_delta[..., discrete_state_dims] = 0

        # Apply offset
        points_w_offset = raw_data_points_before_blend + scaled_delta
        blended_data = np.concatenate((points_w_offset, raw_data_points_after_blend), axis=0)
        blended_data.flags.writeable = False
        blended_data = ensure_immutable_numpy(blended_data)

        # Offset the times
        data_times = np.copy(trajectory.raw_data_times)
        data_times += (start_point.time - trajectory_begin_time)
        data_times.flags.writeable = False

        # Make output
        output = TimedSequenceArray(data=blended_data, time=data_times)
        return output


__all__ = [
    'MergeTrajectoryType',
    'MergeTrajectoryOption',
    'MergeTrajectory'
]
