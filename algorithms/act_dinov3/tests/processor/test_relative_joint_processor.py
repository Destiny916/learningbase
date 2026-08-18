"""Tests for the shared seven-dimensional relative joint processors."""

import json

import pytest
import torch

from lerobot.configs import FeatureType, NormalizationMode, PipelineFeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.factory import _reconnect_relative_absolute_steps, make_pre_post_processors
from lerobot.processor import (
    DataProcessorPipeline,
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
    TransitionKey,
)
from lerobot.processor.converters import (
    create_transition,
    identity_transition,
    observation_to_transition,
    policy_action_to_transition,
    transition_to_observation,
    transition_to_policy_action,
)
from lerobot.processor.relative_joint_processor import (
    RelativeJointAbsoluteActionProcessorStep,
    RelativeJointProcessorStep,
    StateNoiseProcessorStep,
    ZeroStateProcessorStep,
)
from lerobot.scripts.lerobot_eval import _reset_rollout_state
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGE,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


def _offline_transition(
    previous: torch.Tensor,
    current: torch.Tensor,
    action: torch.Tensor,
    complementary_data: dict | None = None,
):
    return create_transition(
        observation={OBS_STATE: torch.stack((previous, current), dim=-2)},
        action=action,
        complementary_data=complementary_data,
    )


def test_offline_horizon_two_uses_current_state_for_arm_actions_and_keeps_gripper_absolute():
    previous = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.2]])
    current = torch.tensor([[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 0.7]])
    action = torch.tensor(
        [[[3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 0.1], [4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 0.9]]]
    )

    result = RelativeJointProcessorStep()(_offline_transition(previous, current, action))

    torch.testing.assert_close(
        result[TransitionKey.OBSERVATION][OBS_STATE],
        torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.7]]),
    )
    torch.testing.assert_close(
        result[TransitionKey.ACTION],
        torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.1], [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 0.9]]]),
    )


def test_offline_dual_arm_14d_keeps_both_grippers_absolute():
    previous = torch.arange(14, dtype=torch.float32).unsqueeze(0)
    current = previous + 1.0
    current[..., 6] = 0.02
    current[..., 13] = 0.08
    action = torch.stack((current + 2.0, current + 3.0), dim=1)
    action[..., 6] = torch.tensor([[0.03, 0.04]])
    action[..., 13] = torch.tensor([[0.07, 0.06]])

    result = RelativeJointProcessorStep(
        joint_names=[f"joint_{index}" for index in range(14)], gripper_indices=[6, 13]
    )(_offline_transition(previous, current, action))

    expected_state = current - previous
    expected_state[..., [6, 13]] = current[..., [6, 13]]
    expected_action = action - current.unsqueeze(1)
    expected_action[..., [6, 13]] = action[..., [6, 13]]
    torch.testing.assert_close(result[TransitionKey.OBSERVATION][OBS_STATE], expected_state)
    torch.testing.assert_close(result[TransitionKey.ACTION], expected_action)


def test_17d_state_maps_to_14d_action_and_keeps_bread_delta_in_relative_state():
    state_names = [f"left_joint_{index}" for index in range(6)] + ["left_gripper"] + [
        f"right_joint_{index}" for index in range(6)
    ] + ["right_gripper", "bread_x", "bread_y", "bread_z"]
    action_names = state_names[:14]
    previous = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.10, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 0.20, 0.30, 0.40, 0.50]]
    )
    current = torch.tensor(
        [[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 0.11, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 0.22, 0.35, 0.45, 0.60]]
    )
    action = torch.tensor(
        [[[3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 0.12, 21.0, 24.0, 27.0, 30.0, 33.0, 36.0, 0.24]]]
    )

    result = RelativeJointProcessorStep(
        state_feature_names=state_names,
        action_feature_names=action_names,
        state_gripper_indices=[6, 13],
        action_gripper_indices=[6, 13],
        action_state_indices=list(range(14)),
    )(_offline_transition(previous, current, action))

    expected_state = current - previous
    expected_state[..., [6, 13]] = current[..., [6, 13]]
    expected_action = action - current[..., :14].unsqueeze(1)
    expected_action[..., [6, 13]] = action[..., [6, 13]]
    torch.testing.assert_close(result[TransitionKey.OBSERVATION][OBS_STATE], expected_state)
    torch.testing.assert_close(result[TransitionKey.ACTION], expected_action)


def test_mapped_state_keeps_gripper_and_endpoint_xyz_absolute():
    state_names = [f"joint_{index}" for index in range(6)] + ["left_gripper"] + [
        f"right_joint_{index}" for index in range(6)
    ] + ["right_gripper", "right_endpoint_x", "right_endpoint_y", "left_endpoint_x", "left_endpoint_y"]
    action_names = state_names[:14]
    previous = torch.tensor([[1.0] * 14 + [0.1, 0.2, 0.3, 0.4]])
    current = torch.tensor([[2.0] * 6 + [0.11] + [3.0] * 6 + [0.22] + [0.5, 0.7, 0.9, 1.1]])
    action = current[..., :14].unsqueeze(1) + 1.0

    result = RelativeJointProcessorStep(
        state_feature_names=state_names,
        action_feature_names=action_names,
        state_gripper_indices=[6, 13],
        state_absolute_indices=[14, 15, 16, 17],
        action_gripper_indices=[6, 13],
        action_state_indices=list(range(14)),
    )(_offline_transition(previous, current, action))

    expected_state = current - previous
    expected_state[..., [6, 13, 14, 15, 16, 17]] = current[..., [6, 13, 14, 15, 16, 17]]
    torch.testing.assert_close(result[TransitionKey.OBSERVATION][OBS_STATE], expected_state)


def test_17d_state_reconstructs_14d_absolute_action_from_mapped_anchor():
    state_names = [f"left_joint_{index}" for index in range(6)] + ["left_gripper"] + [
        f"right_joint_{index}" for index in range(6)
    ] + ["right_gripper", "bread_x", "bread_y", "bread_z"]
    action_names = state_names[:14]
    step = RelativeJointProcessorStep(
        execution_horizon=2,
        state_feature_names=state_names,
        action_feature_names=action_names,
        state_gripper_indices=[6, 13],
        action_gripper_indices=[6, 13],
        action_state_indices=list(range(14)),
    )
    state = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.10, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 0.20, 0.3, 0.4, 0.5]]
    )
    step(create_transition(observation={OBS_STATE: state}))
    relative_action = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.30, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 0.40],
          [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.50, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 0.60]]]
    )
    absolute = RelativeJointAbsoluteActionProcessorStep(relative_step=step)(
        create_transition(action=relative_action)
    )
    expected = relative_action.clone()
    expected[..., [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]] += state[..., [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]].unsqueeze(1)
    torch.testing.assert_close(absolute[TransitionKey.ACTION], expected)


def test_online_first_frame_pads_arm_state_then_uses_previous_absolute_state():
    step = RelativeJointProcessorStep()
    first = torch.tensor([[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 0.25]])
    second = torch.tensor([[11.0, 18.0, 33.0, 36.0, 55.0, 54.0, 0.75]])

    first_result = step(create_transition(observation={OBS_STATE: first}))
    second_result = step(create_transition(observation={OBS_STATE: second}))

    torch.testing.assert_close(
        first_result[TransitionKey.OBSERVATION][OBS_STATE], torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25]])
    )
    torch.testing.assert_close(
        second_result[TransitionKey.OBSERVATION][OBS_STATE], torch.tensor([[1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 0.75]])
    )


def test_image_only_keeps_state_for_action_conversion_and_zero_step_removes_model_condition():
    previous = torch.tensor([[9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 0.1]])
    current = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.4]])
    action = torch.tensor([[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 0.8]])

    relative = RelativeJointProcessorStep(condition_on_state=False)(
        _offline_transition(previous, current, action)
    )
    zeroed = ZeroStateProcessorStep()(relative)

    torch.testing.assert_close(
        relative[TransitionKey.OBSERVATION][OBS_STATE], current
    )
    torch.testing.assert_close(
        relative[TransitionKey.ACTION], torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.8]]))
    assert torch.equal(zeroed[TransitionKey.OBSERVATION][OBS_STATE], torch.zeros_like(current))


def test_image_only_zero_state_runs_after_non_symmetric_quantile_normalization():
    current = torch.tensor([[3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.4]])
    normalizer = NormalizerProcessorStep(
        features={OBS_STATE: PolicyFeature(FeatureType.STATE, (7,))},
        norm_map={FeatureType.STATE: NormalizationMode.QUANTILES},
        stats={OBS_STATE: {"q01": [1.0] * 7, "q99": [5.0] * 7}},
    )
    pipeline = PolicyProcessorPipeline(
        steps=[RelativeJointProcessorStep(condition_on_state=False), normalizer, ZeroStateProcessorStep()],
        to_transition=identity_transition,
        to_output=identity_transition,
    )
    raw_zero_normalized = normalizer(create_transition(observation={OBS_STATE: torch.zeros_like(current)}))

    result = pipeline(_offline_transition(torch.full_like(current, 9.0), current, current + 1.0))

    assert not torch.equal(raw_zero_normalized[TransitionKey.OBSERVATION][OBS_STATE], torch.zeros_like(current))
    assert torch.equal(result[TransitionKey.OBSERVATION][OBS_STATE], torch.zeros_like(current))


def test_offline_padded_previous_frame_zeros_only_state_arm_without_changing_action_label():
    previous = torch.tensor(
        [[9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 0.1], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.2]]
    )
    current = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.4], [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 0.5]]
    )
    action = torch.tensor(
        [[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 0.8], [4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 0.9]]
    )

    result = RelativeJointProcessorStep()(
        _offline_transition(
            previous,
            current,
            action,
            complementary_data={"observation.state_is_pad": torch.tensor([[True, False], [False, False]])},
        )
    )

    torch.testing.assert_close(
        result[TransitionKey.OBSERVATION][OBS_STATE],
        torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5]]),
    )
    torch.testing.assert_close(
        result[TransitionKey.ACTION],
        torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.8], [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 0.9]]),
    )


def test_disabled_relative_joint_processor_is_a_complete_no_op():
    step = RelativeJointProcessorStep(enabled=False)
    transition = _offline_transition(
        torch.zeros(1, 7),
        torch.ones(1, 7),
        torch.full((1, 7), 2.0),
    )

    result = step(transition)

    assert result is transition
    assert step.get_cached_absolute_state() is None


def test_offline_action_never_consumes_or_writes_online_cache_and_requires_history():
    step = RelativeJointProcessorStep()
    online = torch.tensor([[9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 0.3]])
    step(create_transition(observation={OBS_STATE: online}))
    cached = step.get_cached_absolute_state().clone()

    previous = torch.zeros(1, 7)
    current = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.4]])
    action = torch.tensor([[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 0.8]])
    result = step(_offline_transition(previous, current, action))

    torch.testing.assert_close(result[TransitionKey.ACTION][..., :6], action[..., :6] - current[..., :6])
    torch.testing.assert_close(step.get_cached_absolute_state(), cached)
    with pytest.raises(ValueError, match="previous and current"):
        step(create_transition(observation={OBS_STATE: current}, action=action))


def test_online_cache_reconstructs_action_chunk_then_releases_its_anchor_without_clipping():
    relative_step = RelativeJointProcessorStep(execution_horizon=2)
    absolute_step = RelativeJointAbsoluteActionProcessorStep(relative_step=relative_step)
    current = torch.tensor([[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 0.4]])
    relative_step(create_transition(observation={OBS_STATE: current}))
    relative_action = torch.tensor(
        [[[100.0, -200.0, 300.0, -400.0, 500.0, -600.0, 0.9], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.1]]]
    )

    reconstructed = absolute_step(create_transition(action=relative_action))

    torch.testing.assert_close(
        reconstructed[TransitionKey.ACTION],
        torch.tensor(
            [[[110.0, -180.0, 330.0, -360.0, 550.0, -540.0, 0.9], [11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 0.1]]]
        ),
    )
    assert relative_step.get_cached_absolute_state() is None
    relative_step.reset()
    assert relative_step.get_cached_absolute_state() is None
    with pytest.raises(RuntimeError, match="no absolute state has been cached"):
        absolute_step(create_transition(action=relative_action))


def test_online_action_anchor_is_consumed_across_execution_horizon_then_replaced():
    relative_step = RelativeJointProcessorStep(execution_horizon=2)
    absolute_step = RelativeJointAbsoluteActionProcessorStep(relative_step=relative_step)
    q0 = torch.tensor([[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 0.4]])
    q1 = torch.tensor([[11.0, 18.0, 33.0, 36.0, 55.0, 54.0, 0.5]])
    q2 = torch.tensor([[12.0, 16.0, 36.0, 32.0, 60.0, 48.0, 0.6]])

    relative_step(create_transition(observation={OBS_STATE: q0}))
    tick0 = absolute_step(create_transition(action=torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.8]])))
    relative_step(create_transition(observation={OBS_STATE: q1}))
    tick1 = absolute_step(create_transition(action=torch.tensor([[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 0.9]])))

    torch.testing.assert_close(
        tick0[TransitionKey.ACTION], torch.tensor([[11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 0.8]])
    )
    torch.testing.assert_close(
        tick1[TransitionKey.ACTION], torch.tensor([[12.0, 24.0, 36.0, 48.0, 60.0, 72.0, 0.9]])
    )
    assert relative_step.get_cached_absolute_state() is None

    relative_step(create_transition(observation={OBS_STATE: q2}))
    tick2 = absolute_step(create_transition(action=torch.tensor([[3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 1.0]])))
    torch.testing.assert_close(
        tick2[TransitionKey.ACTION], torch.tensor([[15.0, 22.0, 45.0, 44.0, 75.0, 66.0, 1.0]])
    )


def test_online_state_uses_immediate_previous_observation_while_action_anchor_is_retained():
    relative_step = RelativeJointProcessorStep(execution_horizon=2)
    q0 = torch.tensor([[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 0.4]])
    q1 = torch.tensor([[11.0, 18.0, 33.0, 36.0, 55.0, 54.0, 0.5]])

    relative_step(create_transition(observation={OBS_STATE: q0}))
    tick1 = relative_step(create_transition(observation={OBS_STATE: q1}))

    torch.testing.assert_close(
        tick1[TransitionKey.OBSERVATION][OBS_STATE], torch.tensor([[1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 0.5]])
    )
    torch.testing.assert_close(relative_step.get_cached_absolute_state(), q0)


def test_rollout_reset_resets_policy_before_processors():
    calls = []

    class Resettable:
        def __init__(self, name):
            self.name = name

        def reset(self):
            calls.append(self.name)

    _reset_rollout_state(Resettable("policy"), Resettable("preprocessor"), Resettable("postprocessor"))

    assert calls == ["policy", "preprocessor", "postprocessor"]


def test_factory_reconnects_shared_relative_joint_steps_without_legacy_relative_step():
    relative_step = RelativeJointProcessorStep()
    absolute_step = RelativeJointAbsoluteActionProcessorStep()
    preprocessor = PolicyProcessorPipeline(
        steps=[relative_step], to_transition=identity_transition, to_output=identity_transition
    )
    postprocessor = PolicyProcessorPipeline(
        steps=[absolute_step], to_transition=identity_transition, to_output=identity_transition
    )

    _reconnect_relative_absolute_steps(preprocessor, postprocessor)

    assert absolute_step.relative_step is relative_step
    current = torch.tensor([[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 0.4]])
    preprocessor(create_transition(observation={OBS_STATE: current}))
    reconstructed = postprocessor(
        create_transition(action=torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.9]]))
    )
    torch.testing.assert_close(
        reconstructed[TransitionKey.ACTION],
        torch.tensor([[11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 0.9]]),
    )


def test_factory_load_reconnects_serialized_relative_joint_processors_for_online_actions(tmp_path):
    preprocessor = PolicyProcessorPipeline(
        steps=[RelativeJointProcessorStep()],
        name=POLICY_PREPROCESSOR_DEFAULT_NAME,
    )
    postprocessor = PolicyProcessorPipeline(
        steps=[RelativeJointAbsoluteActionProcessorStep(relative_step=None)],
        name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
    )
    preprocessor.save_pretrained(tmp_path)
    postprocessor.save_pretrained(tmp_path)

    act_cfg = ACTConfig(
        input_features={
            OBS_IMAGE: PolicyFeature(FeatureType.VISUAL, (3, 96, 96)),
            OBS_STATE: PolicyFeature(FeatureType.STATE, (7,)),
        },
        output_features={ACTION: PolicyFeature(FeatureType.ACTION, (7,))},
    )
    act_cfg.validate_features()

    loaded_preprocessor, loaded_postprocessor = make_pre_post_processors(
        policy_cfg=act_cfg,
        pretrained_path=tmp_path,
        preprocessor_overrides={"rename_observations_processor": {"rename_map": {}}},
    )

    loaded_relative_step = loaded_preprocessor.steps[0]
    loaded_absolute_step = loaded_postprocessor.steps[0]
    assert isinstance(loaded_relative_step, RelativeJointProcessorStep)
    assert isinstance(loaded_absolute_step, RelativeJointAbsoluteActionProcessorStep)
    assert loaded_absolute_step.relative_step is loaded_relative_step

    q_t = torch.tensor([[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 0.4]])
    relative_action = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.9]])
    loaded_preprocessor({OBS_STATE: q_t})
    absolute_action = loaded_postprocessor(relative_action)

    torch.testing.assert_close(
        absolute_action,
        torch.tensor([[11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 0.9]]),
    )


def test_serialized_shared_relative_joint_processors_reconnect_for_online_actions(tmp_path):
    preprocessor = PolicyProcessorPipeline(
        steps=[RelativeJointProcessorStep(condition_on_state=False, execution_horizon=2)],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )
    postprocessor = PolicyProcessorPipeline(
        steps=[RelativeJointAbsoluteActionProcessorStep(relative_step=None)],
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    q = torch.tensor([[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 0.4]])
    preprocessor({OBS_STATE: q})
    preprocessor.save_pretrained(tmp_path, config_filename="preprocessor.json")
    postprocessor.save_pretrained(tmp_path, config_filename="postprocessor.json")
    saved_preprocessor_config = json.loads((tmp_path / "preprocessor.json").read_text())
    loaded_preprocessor = PolicyProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename="preprocessor.json",
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )
    loaded_postprocessor = PolicyProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename="postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )

    loaded_relative_step = loaded_preprocessor.steps[0]
    loaded_absolute_step = loaded_postprocessor.steps[0]
    assert isinstance(loaded_relative_step, RelativeJointProcessorStep)
    assert isinstance(loaded_absolute_step, RelativeJointAbsoluteActionProcessorStep)
    assert loaded_relative_step.get_config() == {
        "enabled": True,
        "condition_on_state": False,
        "execution_horizon": 2,
        "joint_names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"],
    }
    assert "_last_observation_state" not in json.dumps(saved_preprocessor_config)
    assert "_action_anchor_state" not in json.dumps(saved_preprocessor_config)
    assert loaded_absolute_step.relative_step is None

    relative_action = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.9]])
    loaded_preprocessor({OBS_STATE: q})
    with pytest.raises(RuntimeError, match="paired RelativeJointProcessorStep"):
        loaded_postprocessor(relative_action)

    _reconnect_relative_absolute_steps(loaded_preprocessor, loaded_postprocessor)

    assert loaded_absolute_step.relative_step is loaded_relative_step
    absolute_action = loaded_postprocessor(relative_action)
    torch.testing.assert_close(
        absolute_action, torch.tensor([[11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 0.9]])
    )


def test_absolute_step_requires_paired_relative_step():
    with pytest.raises(RuntimeError, match="paired RelativeJointProcessorStep"):
        RelativeJointAbsoluteActionProcessorStep()(create_transition(action=torch.zeros(1, 7)))


def test_config_and_state_are_cache_free_and_features_are_unchanged():
    step = RelativeJointProcessorStep(condition_on_state=False)
    step(create_transition(observation={OBS_STATE: torch.ones(1, 7)}))
    features = {
        PipelineFeatureType.OBSERVATION: {OBS_STATE: PolicyFeature(FeatureType.STATE, (7,))},
        PipelineFeatureType.ACTION: {"action": PolicyFeature(FeatureType.ACTION, (7,))},
    }

    assert step.get_config() == {
        "enabled": True,
        "condition_on_state": False,
        "execution_horizon": 1,
        "joint_names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"],
    }
    assert step.state_dict() == {}
    assert RelativeJointAbsoluteActionProcessorStep(relative_step=step).get_config() == {}
    assert ZeroStateProcessorStep().get_config() == {}
    assert step.transform_features(features) is features
    assert ZeroStateProcessorStep().transform_features(features) is features
    assert RelativeJointAbsoluteActionProcessorStep(relative_step=step).transform_features(features) is features
    assert ProcessorStepRegistry.get("relative_joint_processor") is RelativeJointProcessorStep
    assert ProcessorStepRegistry.get("relative_joint_absolute_action_processor") is RelativeJointAbsoluteActionProcessorStep
    assert ProcessorStepRegistry.get("zero_state_processor") is ZeroStateProcessorStep


@pytest.mark.parametrize(
    "joint_names",
    [
        ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5"],
        ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "gripper", "joint_5"],
    ],
)
def test_joint_names_require_six_arm_joints_and_gripper_at_index_six(joint_names):
    with pytest.raises(ValueError, match="joint_0.*joint_5.*gripper"):
        RelativeJointProcessorStep(joint_names=joint_names)


def test_state_and_action_require_seven_dimensions():
    step = RelativeJointProcessorStep()
    with pytest.raises(ValueError, match="7 dimensions"):
        step(create_transition(observation={OBS_STATE: torch.zeros(1, 6)}))
    with pytest.raises(ValueError, match="7 dimensions"):
        step(
            create_transition(
                observation={OBS_STATE: torch.zeros(1, 2, 7)}, action=torch.zeros(1, 2, 6)
            )
        )
    with pytest.raises(ValueError, match="execution_horizon"):
        RelativeJointProcessorStep(execution_horizon=0)


def test_state_noise_only_applies_in_training_mode_and_never_changes_action():
    step = StateNoiseProcessorStep(joint_std_rad=0.1, gripper_std_m=0.02)
    state = torch.zeros((2, 7))
    action = torch.full((2, 3, 7), 0.5)
    transition = create_transition(observation={OBS_STATE: state}, action=action)

    torch.manual_seed(0)
    step.train()
    noisy = step(transition)

    assert not torch.equal(noisy[TransitionKey.OBSERVATION][OBS_STATE], state)
    torch.testing.assert_close(noisy[TransitionKey.ACTION], action)

    step.eval()
    clean = step(transition)
    torch.testing.assert_close(clean[TransitionKey.OBSERVATION][OBS_STATE], state)


def test_state_noise_disabled_for_image_only_keeps_zero_state_exactly_zero():
    step = StateNoiseProcessorStep(joint_std_rad=0.1, gripper_std_m=0.02, enabled=False)
    zero_state = torch.zeros((2, 7))

    step.train()
    result = ZeroStateProcessorStep()(step(create_transition(observation={OBS_STATE: zero_state})))

    torch.testing.assert_close(result[TransitionKey.OBSERVATION][OBS_STATE], zero_state)


def test_processor_pipeline_disables_state_noise_in_eval_mode():
    pipeline = DataProcessorPipeline(
        steps=[StateNoiseProcessorStep(joint_std_rad=0.1, gripper_std_m=0.02)],
        to_transition=identity_transition,
        to_output=identity_transition,
    )
    transition = create_transition(observation={OBS_STATE: torch.zeros((1, 7))})

    torch.manual_seed(0)
    pipeline.train()
    train_result = pipeline(transition)
    assert not torch.equal(train_result[TransitionKey.OBSERVATION][OBS_STATE], torch.zeros((1, 7)))

    pipeline.eval()
    eval_result = pipeline(transition)
    torch.testing.assert_close(eval_result[TransitionKey.OBSERVATION][OBS_STATE], torch.zeros((1, 7)))
