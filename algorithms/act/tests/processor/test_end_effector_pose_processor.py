"""SE(3) relative pose processor tests for the independent pose10d workflow."""

from __future__ import annotations

import json

import numpy as np
import torch

from lerobot.processor import PolicyProcessorPipeline, identity_transition
from lerobot.processor.converters import create_transition
from lerobot.processor.end_effector_pose_processor import (
    PoseStateNoiseProcessorStep,
    PoseQuantileNormalizerProcessorStep,
    PoseQuantileUnnormalizerProcessorStep,
    RelativePoseAbsoluteActionProcessorStep,
    RelativePoseProcessorStep,
    rot6d_to_matrix,
)
from lerobot.scripts.convert_right_end_effector_pose_to_lerobot_v30 import (
    pose10d_from_end_pose,
    relative_pose10d,
)
from lerobot.types import TransitionKey
from lerobot.utils.constants import ACTION, OBS_STATE


def _pose(*, x: float, y: float, z: float, roll: float, pitch: float, yaw: float, gripper: float) -> np.ndarray:
    return pose10d_from_end_pose(
        {"x": x, "y": y, "z": z, "roll": roll, "pitch": pitch, "yaw": yaw},
        gripper,
    )


def test_offline_processor_uses_se3_relative_state_and_chunk_actions() -> None:
    previous = _pose(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, gripper=0.01)
    current = _pose(x=0.1, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=np.pi / 2, gripper=0.02)
    target = _pose(x=0.1, y=0.1, z=0.0, roll=0.0, pitch=0.1, yaw=np.pi / 2, gripper=0.03)
    transition = create_transition(
        observation={OBS_STATE: torch.tensor([[previous, current]])},
        action=torch.tensor([[target]]),
    )

    result = RelativePoseProcessorStep()(transition)

    np.testing.assert_allclose(
        result[TransitionKey.OBSERVATION][OBS_STATE].numpy()[0],
        relative_pose10d(previous, current),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result[TransitionKey.ACTION].numpy()[0], relative_pose10d(current, target)[None], atol=1e-6
    )


def test_offline_processor_uses_identity_arm_pose_for_padded_previous_frame() -> None:
    current = _pose(x=0.1, y=0.2, z=0.3, roll=0.1, pitch=0.2, yaw=0.3, gripper=0.06)
    transition = create_transition(
        observation={OBS_STATE: torch.tensor([[current, current]])},
        action=torch.tensor([[current]]),
        complementary_data={f"{OBS_STATE}_is_pad": torch.tensor([[True, False]])},
    )

    result = RelativePoseProcessorStep()(transition)

    expected = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.06], dtype=np.float32)
    np.testing.assert_allclose(result[TransitionKey.OBSERVATION][OBS_STATE].numpy()[0], expected, atol=1e-6)


def test_online_postprocessor_restores_absolute_pose_and_absolute_gripper() -> None:
    current = _pose(x=0.1, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=np.pi / 2, gripper=0.02)
    target = _pose(x=0.1, y=0.1, z=0.0, roll=0.0, pitch=0.1, yaw=np.pi / 2, gripper=0.08)
    relative = relative_pose10d(current, target)
    relative_step = RelativePoseProcessorStep(execution_horizon=1)

    preprocessed = relative_step(create_transition(observation={OBS_STATE: torch.tensor([current])}))
    np.testing.assert_allclose(preprocessed[TransitionKey.OBSERVATION][OBS_STATE].numpy()[0, :3], 0.0, atol=1e-6)
    output = RelativePoseAbsoluteActionProcessorStep(relative_step=relative_step)(
        create_transition(action=torch.tensor([[relative]]))
    )

    np.testing.assert_allclose(output[TransitionKey.ACTION].numpy()[0, 0], target, atol=1e-6)


def test_pose_quantiles_scale_only_translation_and_gripper() -> None:
    state = torch.tensor([[0.05, -0.05, 0.10, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.06]])
    action = state.unsqueeze(1).clone()
    stats = {
        OBS_STATE: {"q01": [-0.1, -0.1, -0.1, 0, 0, 0, 0, 0, 0, 0.02], "q99": [0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0.10]},
        ACTION: {"q01": [-0.2, -0.2, -0.2, 0, 0, 0, 0, 0, 0, 0.02], "q99": [0.2, 0.2, 0.2, 0, 0, 0, 0, 0, 0, 0.10]},
    }
    transition = create_transition(observation={OBS_STATE: state}, action=action)

    normalized = PoseQuantileNormalizerProcessorStep(stats=stats)(transition)
    normalized_state = normalized[TransitionKey.OBSERVATION][OBS_STATE]
    assert torch.allclose(normalized_state[:, [0, 1, 2, 9]], torch.tensor([[0.5, -0.5, 1.0, 0.0]]))
    assert torch.equal(normalized_state[:, 3:9], state[:, 3:9])
    restored = PoseQuantileUnnormalizerProcessorStep(stats=stats)(normalized)
    torch.testing.assert_close(restored[TransitionKey.ACTION], action)


def test_pose_state_noise_changes_training_state_without_changing_action() -> None:
    torch.manual_seed(7)
    state = torch.tensor([[_pose(x=0.1, y=-0.2, z=0.3, roll=0.1, pitch=-0.2, yaw=0.3, gripper=0.06)]])
    action = state.clone().unsqueeze(1)
    transition = create_transition(observation={OBS_STATE: state}, action=action)
    step = PoseStateNoiseProcessorStep(position_std_m=0.003, rotation_std_rad=0.003, gripper_std_m=0.001)
    step.train()

    result = step(transition)
    noisy_state = result[TransitionKey.OBSERVATION][OBS_STATE]

    assert not torch.equal(noisy_state, state)
    torch.testing.assert_close(result[TransitionKey.ACTION], action)
    rotation = rot6d_to_matrix(noisy_state[..., 3:9])
    identity = torch.eye(3).expand_as(rotation)
    torch.testing.assert_close(rotation.transpose(-1, -2) @ rotation, identity, atol=1e-5, rtol=1e-5)


def test_pose_state_noise_is_disabled_in_eval_mode() -> None:
    state = torch.tensor([[_pose(x=0.1, y=-0.2, z=0.3, roll=0.1, pitch=-0.2, yaw=0.3, gripper=0.06)]])
    action = state.clone().unsqueeze(1)
    transition = create_transition(observation={OBS_STATE: state}, action=action)
    step = PoseStateNoiseProcessorStep(position_std_m=0.003, rotation_std_rad=0.003, gripper_std_m=0.001)
    step.eval()

    result = step(transition)

    torch.testing.assert_close(result[TransitionKey.OBSERVATION][OBS_STATE], state)
    torch.testing.assert_close(result[TransitionKey.ACTION], action)


def test_pose_quantile_processors_save_numpy_stats_as_json(tmp_path) -> None:
    stats = {
        OBS_STATE: {
            "q01": np.array([-0.1, -0.1, -0.1, 0, 0, 0, 0, 0, 0, 0.02], dtype=np.float32),
            "q99": np.array([0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0.10], dtype=np.float32),
        },
        ACTION: {
            "q01": np.array([-0.2, -0.2, -0.2, 0, 0, 0, 0, 0, 0, 0.02], dtype=np.float32),
            "q99": np.array([0.2, 0.2, 0.2, 0, 0, 0, 0, 0, 0, 0.10], dtype=np.float32),
        },
    }
    pipeline = PolicyProcessorPipeline(
        steps=[PoseQuantileNormalizerProcessorStep(stats=stats), PoseQuantileUnnormalizerProcessorStep(stats=stats)],
        name="pose_quantiles",
        to_transition=identity_transition,
        to_output=identity_transition,
    )

    pipeline.save_pretrained(tmp_path)

    saved_config = json.loads((tmp_path / "pose_quantiles.json").read_text())
    assert saved_config["steps"][0]["config"]["stats"][OBS_STATE]["q01"] == stats[OBS_STATE]["q01"].tolist()
    loaded = PolicyProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename="pose_quantiles.json",
        to_transition=identity_transition,
        to_output=identity_transition,
    )
    assert isinstance(loaded.steps[0], PoseQuantileNormalizerProcessorStep)
    assert isinstance(loaded.steps[1], PoseQuantileUnnormalizerProcessorStep)
