"""ACT and PI05 pose10d processor pipeline selection tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.datasets.end_effector_pose_stats import compute_relative_pose_stats_from_episodes, save_relative_pose_stats
from lerobot.datasets.absolute_action_stats import compute_absolute_action_stats_from_episodes, save_absolute_action_stats
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep, make_pi05_pre_post_processors
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    PoseQuantileNormalizerProcessorStep,
    PoseQuantileUnnormalizerProcessorStep,
    RelativePoseAbsoluteActionProcessorStep,
    RelativePoseProcessorStep,
    TokenizerProcessorStep,
    ZeroStateProcessorStep,
)
from lerobot.processor.end_effector_pose_processor import PoseStateNoiseProcessorStep
from lerobot.scripts.convert_right_end_effector_pose_to_lerobot_v30 import pose10d_from_end_pose
from lerobot.utils.constants import ACTION, OBS_STATE


def _stats_paths(tmp_path, horizon: int):
    identity = pose10d_from_end_pose(
        {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}, 0.02
    )
    moved = pose10d_from_end_pose(
        {"x": 0.1, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.1}, 0.04
    )
    bundle = compute_relative_pose_stats_from_episodes([torch.tensor([identity, moved, moved])], horizons=[horizon])
    save_relative_pose_stats(bundle, tmp_path)
    return tmp_path / "relative_pose_state_q01_q99.json", tmp_path / f"relative_pose_action_chunk{horizon}_q01_q99.json"


def _features():
    return (
        {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,)), "observation.images.camera": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32))},
        {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(10,))},
    )


def _absolute_stats_paths(tmp_path, horizon: int):
    identity = pose10d_from_end_pose(
        {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}, 0.02
    )
    moved = pose10d_from_end_pose(
        {"x": 0.1, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.1}, 0.04
    )
    bundle = compute_absolute_action_stats_from_episodes(
        [torch.tensor([identity, moved, moved])],
        horizons=[horizon],
        feature_names=["x", "y", "z", "rot6d_0", "rot6d_1", "rot6d_2", "rot6d_3", "rot6d_4", "rot6d_5", "gripper"],
        scaled_indices=[0, 1, 2, 9],
    )
    save_absolute_action_stats(bundle, tmp_path)
    return tmp_path / "absolute_state_q01_q99.json", tmp_path / f"absolute_action_chunk{horizon}_q01_q99.json"


def test_act_uses_pose_specific_relative_pipeline(tmp_path) -> None:
    state_path, action_path = _stats_paths(tmp_path, 2)
    inputs, outputs = _features()
    config = ACTConfig(
        device="cpu", chunk_size=2, n_action_steps=2, input_features=inputs, output_features=outputs,
        end_effector_pose_representation="relative", pose_state_stats_path=str(state_path), pose_action_stats_path=str(action_path),
        condition_on_state=False,
    )

    preprocessor, postprocessor = make_act_pre_post_processors(config)

    assert [type(step) for step in preprocessor.steps] == [
        AddBatchDimensionProcessorStep, RelativePoseProcessorStep, PoseQuantileNormalizerProcessorStep,
        ZeroStateProcessorStep, DeviceProcessorStep,
    ]
    assert [type(step) for step in postprocessor.steps] == [
        PoseQuantileUnnormalizerProcessorStep, RelativePoseAbsoluteActionProcessorStep, DeviceProcessorStep,
    ]


def test_pi05_uses_pose_specific_relative_pipeline(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer_module = pytest.importorskip("lerobot.processor.tokenizer_processor")
    monkeypatch.setattr(tokenizer_module, "_transformers_available", True)
    monkeypatch.setattr(tokenizer_module, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: object()))
    state_path, action_path = _stats_paths(tmp_path, 2)
    inputs, outputs = _features()
    config = PI05Config(
        device="cpu", chunk_size=2, n_action_steps=2, input_features=inputs, output_features=outputs,
        end_effector_pose_representation="relative", pose_state_stats_path=str(state_path), pose_action_stats_path=str(action_path),
        condition_on_state=False, joint_gripper_indices=[9],
    )

    preprocessor, postprocessor = make_pi05_pre_post_processors(config)

    assert [type(step) for step in preprocessor.steps] == [
        AddBatchDimensionProcessorStep, RelativePoseProcessorStep, PoseQuantileNormalizerProcessorStep,
        ZeroStateProcessorStep, Pi05PrepareStateTokenizerProcessorStep, TokenizerProcessorStep, DeviceProcessorStep,
    ]
    assert [type(step) for step in postprocessor.steps] == [
        PoseQuantileUnnormalizerProcessorStep, RelativePoseAbsoluteActionProcessorStep, DeviceProcessorStep,
    ]


def test_pi05_uses_pose_specific_absolute_pipeline(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer_module = pytest.importorskip("lerobot.processor.tokenizer_processor")
    monkeypatch.setattr(tokenizer_module, "_transformers_available", True)
    monkeypatch.setattr(tokenizer_module, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: object()))
    state_path, action_path = _absolute_stats_paths(tmp_path, 2)
    inputs, outputs = _features()
    config = PI05Config(
        device="cpu", chunk_size=2, n_action_steps=2, input_features=inputs, output_features=outputs,
        end_effector_pose_representation="absolute", absolute_state_stats_path=str(state_path),
        absolute_action_stats_path=str(action_path), condition_on_state=False, joint_gripper_indices=[9],
        action_feature_names=["x", "y", "z", "rot6d_0", "rot6d_1", "rot6d_2", "rot6d_3", "rot6d_4", "rot6d_5", "gripper"],
    )

    preprocessor, postprocessor = make_pi05_pre_post_processors(config)

    assert [type(step) for step in preprocessor.steps] == [
        AddBatchDimensionProcessorStep, PoseQuantileNormalizerProcessorStep,
        ZeroStateProcessorStep, Pi05PrepareStateTokenizerProcessorStep, TokenizerProcessorStep, DeviceProcessorStep,
    ]
    assert [type(step) for step in postprocessor.steps] == [
        PoseQuantileUnnormalizerProcessorStep, DeviceProcessorStep,
    ]


@pytest.mark.parametrize("representation", ["relative", "absolute"])
def test_pi05_pose_pipeline_includes_state_noise_when_configured(
    tmp_path, monkeypatch: pytest.MonkeyPatch, representation: str
) -> None:
    tokenizer_module = pytest.importorskip("lerobot.processor.tokenizer_processor")
    monkeypatch.setattr(tokenizer_module, "_transformers_available", True)
    monkeypatch.setattr(tokenizer_module, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: object()))
    state_path, action_path = _stats_paths(tmp_path, 2) if representation == "relative" else _absolute_stats_paths(tmp_path, 2)
    inputs, outputs = _features()
    config_kwargs = {
        "end_effector_pose_representation": representation,
        "joint_gripper_indices": [9],
        "state_position_noise_std_m": 0.003,
        "state_noise_std_rad": 0.003,
        "gripper_noise_std_m": 0.001,
    }
    if representation == "relative":
        config_kwargs.update(pose_state_stats_path=str(state_path), pose_action_stats_path=str(action_path))
    else:
        config_kwargs.update(absolute_state_stats_path=str(state_path), absolute_action_stats_path=str(action_path))
    config = PI05Config(
        device="cpu", chunk_size=2, n_action_steps=2, input_features=inputs, output_features=outputs, **config_kwargs
    )

    preprocessor, _ = make_pi05_pre_post_processors(config)

    noise_step = next(step for step in preprocessor.steps if isinstance(step, PoseStateNoiseProcessorStep))
    assert noise_step.position_std_m == 0.003
    assert noise_step.rotation_std_rad == 0.003
    assert noise_step.gripper_std_m == 0.001
