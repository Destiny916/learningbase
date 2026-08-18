#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for ACT policy processor."""

import json
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.configs import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.datasets.relative_joint_stats import (
    compute_relative_joint_stats_from_episodes,
    save_relative_joint_stats,
)
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACT, ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DataProcessorPipeline,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    RelativeJointAbsoluteActionProcessorStep,
    RelativeJointProcessorStep,
    RenameObservationsProcessorStep,
    StateNoiseProcessorStep,
    TransitionKey,
    UnnormalizerProcessorStep,
    ZeroStateProcessorStep,
)
from lerobot.processor.converters import create_transition, transition_to_batch
from lerobot.utils.constants import (
    ACTION,
    OBS_ENV_STATE,
    OBS_IMAGES,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

JOINT_NAMES_7D = [f"joint_{index}" for index in range(6)] + ["gripper"]
JOINT_NAMES_14D = [f"left_joint_{index}" for index in range(6)] + ["left_gripper"] + [
    f"right_joint_{index}" for index in range(6)
] + ["right_gripper"]


def _save_act_relative_stats(tmp_path: Path) -> tuple[Path, Path]:
    episodes = [torch.arange(21, dtype=torch.float32).reshape(3, 7)]
    bundle = compute_relative_joint_stats_from_episodes(
        episodes,
        gripper_indices=[6],
        horizons=[16, 50],
        feature_names=JOINT_NAMES_7D,
        source_manifest_sha256="a" * 64,
    )
    save_relative_joint_stats(bundle, tmp_path)
    return (
        tmp_path / "relative_state_q01_q99.json",
        tmp_path / "relative_action_chunk16_q01_q99.json",
    )


def _act_relative_config(tmp_path: Path, **overrides) -> ACTConfig:
    state_path, action_path = _save_act_relative_stats(tmp_path)
    kwargs = {
        "joint_representation": "relative",
        "device": "cpu",
        "chunk_size": 16,
        "n_action_steps": 16,
        "relative_state_stats_path": str(state_path),
        "relative_action_stats_path": str(action_path),
        "input_features": {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            "observation.images.camera": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
        },
        "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        "action_feature_names": JOINT_NAMES_7D,
    }
    kwargs.update(overrides)
    return ACTConfig(**kwargs)


def test_act_dual_arm_relative_config_loads_separate_14d_quantiles(tmp_path: Path) -> None:
    bundle = compute_relative_joint_stats_from_episodes(
        [torch.arange(70, dtype=torch.float32).reshape(5, 14)],
        gripper_indices=[6, 13],
        horizons=[16, 50],
        feature_names=JOINT_NAMES_14D,
        source_manifest_sha256="a" * 64,
    )
    save_relative_joint_stats(bundle, tmp_path)
    config = ACTConfig(
        joint_representation="relative",
        chunk_size=16,
        n_action_steps=16,
        gripper_indices=[6, 13],
        relative_state_stats_path=str(tmp_path / "relative_state_q01_q99.json"),
        relative_action_stats_path=str(tmp_path / "relative_action_chunk16_q01_q99.json"),
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(14,)),
            "observation.images.camera": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(14,))},
        action_feature_names=JOINT_NAMES_14D,
    )

    config.validate_features()

    assert config.relative_joint_stats.state.q01.shape == (14,)
    assert config.relative_joint_stats.gripper_indices == [6, 13]


@pytest.mark.parametrize("condition_on_state", [True, False])
def test_act_raw_relative_config_contract_is_identical_for_both_conditioning_modes(
    tmp_path: Path, condition_on_state: bool
):
    cfg = _act_relative_config(tmp_path, condition_on_state=condition_on_state)

    cfg.validate_features()

    assert cfg.observation_delta_indices == [-1, 0]
    assert cfg.action_delta_indices == list(range(16))
    assert OBS_STATE in cfg.input_features
    assert cfg.relative_joint_stats.actions[16].q01.shape == (7,)
    assert cfg.normalization_mapping["STATE"] is NormalizationMode.QUANTILES
    assert cfg.normalization_mapping["ACTION"] is NormalizationMode.QUANTILES
    assert json.dumps(asdict(cfg))


@pytest.mark.parametrize(("state_shape", "action_shape"), [((14,), (7,)), ((7,), (14,))])
def test_act_relative_config_rejects_mismatched_feature_dimensions(
    tmp_path: Path, state_shape: tuple[int, ...], action_shape: tuple[int, ...]
):
    state_path, action_path = _save_act_relative_stats(tmp_path)

    cfg = ACTConfig(
        joint_representation="relative",
        chunk_size=16,
        n_action_steps=16,
        relative_state_stats_path=str(state_path),
        relative_action_stats_path=str(action_path),
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=state_shape),
            "observation.images.camera": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=action_shape)},
        action_feature_names=JOINT_NAMES_7D,
    )

    with pytest.raises(ValueError, match="matching"):
        cfg.validate_features()


def test_act_from_pretrained_defers_relative_stats_io_until_feature_validation(tmp_path: Path):
    stats_dir = tmp_path / "original_stats"
    checkpoint_dir = tmp_path / "checkpoint"
    cfg = _act_relative_config(stats_dir)
    cfg.save_pretrained(checkpoint_dir)
    shutil.rmtree(stats_dir)

    loaded = PreTrainedConfig.from_pretrained(checkpoint_dir)
    assert isinstance(loaded, ACTConfig)

    with pytest.raises(ValueError, match="does not exist"):
        loaded.validate_features()

    state_path, action_path = _save_act_relative_stats(tmp_path / "local_stats")
    loaded.relative_state_stats_path = str(state_path)
    loaded.relative_action_stats_path = str(action_path)
    loaded.validate_features()
    assert loaded.relative_joint_stats.actions[16].q01.shape == (7,)


def test_act_relative_stats_cache_clears_on_failure_and_absolute_mode(tmp_path: Path):
    cfg = _act_relative_config(tmp_path / "stats")
    valid_state_path = cfg.relative_state_stats_path
    cfg.validate_features()
    assert cfg.relative_joint_stats is not None

    cfg.relative_state_stats_path = str(tmp_path / "missing.json")
    with pytest.raises(ValueError, match="does not exist"):
        cfg.validate_features()
    assert cfg.relative_joint_stats is None

    cfg.relative_state_stats_path = valid_state_path
    cfg.validate_features()
    assert cfg.relative_joint_stats is not None
    cfg.joint_representation = "absolute"
    cfg.validate_features()
    assert cfg.relative_joint_stats is None


def test_act_validate_features_clears_stats_before_public_feature_checks(tmp_path: Path):
    cfg = _act_relative_config(tmp_path / "stats")
    valid_input_features = cfg.input_features.copy()
    cfg.validate_features()
    assert cfg.relative_joint_stats is not None

    cfg.input_features = {OBS_STATE: valid_input_features[OBS_STATE]}
    with pytest.raises(ValueError, match="image or the environment state"):
        cfg.validate_features()
    assert cfg.relative_joint_stats is None

    cfg.input_features = valid_input_features
    cfg.validate_features()
    assert cfg.relative_joint_stats is not None
    cfg.input_features = {
        key: feature for key, feature in valid_input_features.items() if key != OBS_STATE
    }
    with pytest.raises(ValueError, match="observation.state"):
        cfg.validate_features()
    assert cfg.relative_joint_stats is None


def test_act_relative_normalization_does_not_mutate_supplied_absolute_defaults(tmp_path: Path):
    normalization_mapping = {
        "VISUAL": NormalizationMode.MEAN_STD,
        "STATE": NormalizationMode.MEAN_STD,
        "ACTION": NormalizationMode.MEAN_STD,
    }

    cfg = _act_relative_config(tmp_path, normalization_mapping=normalization_mapping)

    assert normalization_mapping["STATE"] is NormalizationMode.MEAN_STD
    assert normalization_mapping["ACTION"] is NormalizationMode.MEAN_STD
    assert cfg.normalization_mapping is not normalization_mapping


@pytest.mark.parametrize("condition_on_state", [True, False])
def test_act_relative_processor_order_stats_clip_and_real_action_anchor(
    tmp_path: Path, condition_on_state: bool
):
    cfg = _act_relative_config(tmp_path, condition_on_state=condition_on_state)
    cfg.validate_features()

    preprocessor, postprocessor = make_act_pre_post_processors(cfg)

    expected_pre_types = [
        AddBatchDimensionProcessorStep,
        RelativeJointProcessorStep,
    ]
    if condition_on_state:
        expected_pre_types.append(StateNoiseProcessorStep)
    expected_pre_types.append(NormalizerProcessorStep)
    if not condition_on_state:
        expected_pre_types.append(ZeroStateProcessorStep)
    expected_pre_types.append(DeviceProcessorStep)
    assert [type(step) for step in preprocessor.steps] == expected_pre_types
    assert [type(step) for step in postprocessor.steps] == [
        UnnormalizerProcessorStep,
        RelativeJointAbsoluteActionProcessorStep,
        DeviceProcessorStep,
    ]

    normalizer = next(step for step in preprocessor.steps if isinstance(step, NormalizerProcessorStep))
    assert normalizer.clip_quantiles is True
    torch.testing.assert_close(
        normalizer._tensor_stats[OBS_STATE]["q01"],
        torch.from_numpy(cfg.relative_joint_stats.state.q01.copy()).float(),
    )
    torch.testing.assert_close(
        normalizer._tensor_stats[ACTION]["q99"],
        torch.from_numpy(cfg.relative_joint_stats.actions[16].q99.copy()).float(),
    )

    previous = torch.arange(7, dtype=torch.float32)
    current = previous + torch.tensor([10, 20, 30, 40, 50, 60, 0.25])
    absolute_actions = current + torch.arange(16, dtype=torch.float32).unsqueeze(-1)
    absolute_actions[:, 6] = torch.linspace(-10, 10, 16)
    batch = {
        OBS_STATE: torch.stack([previous, current]).unsqueeze(0),
        "observation.images.camera": torch.zeros(1, 3, 32, 32),
        ACTION: absolute_actions.unsqueeze(0),
    }

    processed = preprocessor(batch)
    assert processed[OBS_STATE].shape == (1, 7)
    if condition_on_state:
        assert torch.count_nonzero(processed[OBS_STATE]).item() > 0
    else:
        torch.testing.assert_close(processed[OBS_STATE], torch.zeros_like(processed[OBS_STATE]))
    assert processed[ACTION].amin().item() >= -1.0
    assert processed[ACTION].amax().item() <= 1.0

    relative_action = absolute_actions.clone()
    relative_action[:, :6] -= current[:6]
    q01 = torch.from_numpy(cfg.relative_joint_stats.actions[16].q01.copy()).float()
    q99 = torch.from_numpy(cfg.relative_joint_stats.actions[16].q99.copy()).float()
    expected_normalized = (2 * (relative_action - q01) / (q99 - q01).clamp_min(1e-8) - 1).clamp(-1, 1)
    torch.testing.assert_close(processed[ACTION][0], expected_normalized)

    online_state = current.unsqueeze(0)
    preprocessor.reset()
    preprocessor({OBS_STATE: online_state, "observation.images.camera": torch.zeros(1, 3, 32, 32)})
    reconstructed = postprocessor(torch.zeros(1, 16, 7))
    expected_arm = current[:6] + (q01[:6] + q99[:6]) / 2
    expected_gripper = torch.full((1, 16), (q01[6] + q99[6]) / 2)
    torch.testing.assert_close(reconstructed[..., :6], expected_arm.expand(1, 16, 6))
    torch.testing.assert_close(reconstructed[..., 6], expected_gripper)


def test_act_image_only_relative_step_computes_delta_before_normalize_and_zero(tmp_path: Path):
    cfg = _act_relative_config(tmp_path, condition_on_state=False)
    preprocessor, _ = make_act_pre_post_processors(cfg)
    relative_step = preprocessor.steps[1]
    assert isinstance(relative_step, RelativeJointProcessorStep)
    assert relative_step.condition_on_state is True
    normalizer = preprocessor.steps[2]
    assert isinstance(normalizer, NormalizerProcessorStep)
    state_q01 = normalizer._tensor_stats[OBS_STATE]["q01"]
    state_q99 = normalizer._tensor_stats[OBS_STATE]["q99"]
    assert not torch.allclose(state_q01, -state_q99)

    intermediate = {}

    def capture_relative_output(step_index, transition):
        if step_index == 1:
            intermediate[OBS_STATE] = transition[TransitionKey.OBSERVATION][OBS_STATE].clone()
            intermediate[ACTION] = transition[TransitionKey.ACTION].clone()
        elif step_index == 2:
            intermediate["normalized_state"] = transition[TransitionKey.OBSERVATION][OBS_STATE].clone()

    preprocessor.register_after_step_hook(capture_relative_output)
    previous = torch.tensor([100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 0.2])
    state_delta = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.6])
    current = previous + state_delta
    absolute_action = current.expand(16, 7).clone()
    absolute_action[:, :6] += torch.arange(16, dtype=torch.float32).unsqueeze(-1)
    absolute_action[:, 6] = torch.linspace(0.3, 0.9, 16)

    processed = preprocessor(
        {
            OBS_STATE: torch.stack([previous, current]).unsqueeze(0),
            "observation.images.camera": torch.zeros(1, 3, 32, 32),
            ACTION: absolute_action.unsqueeze(0),
        }
    )

    expected_relative_state = current.clone()
    expected_relative_state[:6] -= previous[:6]
    torch.testing.assert_close(intermediate[OBS_STATE][0], expected_relative_state)
    assert not torch.equal(intermediate[OBS_STATE][0, :6], current[:6])
    expected_relative_action = absolute_action.clone()
    expected_relative_action[:, :6] -= current[:6]
    torch.testing.assert_close(intermediate[ACTION][0], expected_relative_action)
    expected_normalized_state = (
        2 * (expected_relative_state - state_q01) / (state_q99 - state_q01).clamp_min(1e-8) - 1
    ).clamp(-1, 1)
    torch.testing.assert_close(intermediate["normalized_state"][0], expected_normalized_state)
    torch.testing.assert_close(processed[OBS_STATE], torch.zeros(1, 7))


def test_act_factory_checkpoint_reconnects_relative_steps_and_reset_rebuilds_anchor(tmp_path: Path):
    stats_dir = tmp_path / "stats"
    checkpoint_dir = tmp_path / "checkpoint"
    cfg = _act_relative_config(stats_dir, condition_on_state=False)
    preprocessor, postprocessor = make_act_pre_post_processors(cfg)
    preprocessor.save_pretrained(
        checkpoint_dir,
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
    )
    postprocessor.save_pretrained(
        checkpoint_dir,
        config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
    )

    loaded_preprocessor, loaded_postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(checkpoint_dir),
    )
    loaded_relative_step = next(
        step for step in loaded_preprocessor.steps if isinstance(step, RelativeJointProcessorStep)
    )
    loaded_absolute_step = next(
        step
        for step in loaded_postprocessor.steps
        if isinstance(step, RelativeJointAbsoluteActionProcessorStep)
    )
    assert loaded_absolute_step.relative_step is loaded_relative_step

    first_anchor = torch.tensor([[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 0.4]])
    loaded_preprocessor(
        {OBS_STATE: first_anchor, "observation.images.camera": torch.zeros(1, 3, 32, 32)}
    )
    torch.testing.assert_close(loaded_relative_step.get_cached_absolute_state(), first_anchor)
    loaded_preprocessor.reset()
    assert loaded_relative_step.get_cached_absolute_state() is None

    second_anchor = first_anchor + 5
    loaded_preprocessor(
        {OBS_STATE: second_anchor, "observation.images.camera": torch.zeros(1, 3, 32, 32)}
    )
    action_stats = cfg.relative_joint_stats.actions[16]
    q01 = torch.from_numpy(action_stats.q01.copy()).float()
    q99 = torch.from_numpy(action_stats.q99.copy()).float()
    normalized_physical_zero = 2 * (torch.zeros(7) - q01) / (q99 - q01) - 1
    reconstructed = loaded_postprocessor(normalized_physical_zero.expand(1, 16, 7).clone())
    torch.testing.assert_close(reconstructed[..., :6], second_anchor[..., :6].unsqueeze(1).expand(-1, 16, -1))
    torch.testing.assert_close(reconstructed[..., 6], torch.zeros(1, 16))
    assert loaded_relative_step.get_cached_absolute_state() is None

    third_anchor = first_anchor + 9
    loaded_preprocessor(
        {OBS_STATE: third_anchor, "observation.images.camera": torch.zeros(1, 3, 32, 32)}
    )
    torch.testing.assert_close(loaded_relative_step.get_cached_absolute_state(), third_anchor)


def test_act_absolute_quantile_pipeline_preserves_legacy_order_without_clip():
    config = create_default_config()
    config.normalization_mapping = {
        FeatureType.STATE: NormalizationMode.QUANTILES,
        FeatureType.ACTION: NormalizationMode.QUANTILES,
    }
    stats = {
        OBS_STATE: {"q01": torch.zeros(7), "q99": torch.ones(7)},
        ACTION: {"q01": torch.zeros(4), "q99": torch.ones(4)},
    }

    preprocessor, postprocessor = make_act_pre_post_processors(config, stats)

    assert [type(step) for step in preprocessor.steps] == [
        RenameObservationsProcessorStep,
        AddBatchDimensionProcessorStep,
        DeviceProcessorStep,
        NormalizerProcessorStep,
    ]
    normalizer = preprocessor.steps[-1]
    assert isinstance(normalizer, NormalizerProcessorStep)
    assert normalizer.clip_quantiles is False
    processed = preprocessor({OBS_STATE: torch.full((7,), 2.0), ACTION: torch.full((4,), 2.0)})
    torch.testing.assert_close(processed[OBS_STATE], torch.full((1, 7), 3.0))
    assert [type(step) for step in postprocessor.steps] == [
        UnnormalizerProcessorStep,
        DeviceProcessorStep,
    ]


class _FixedACTModel(nn.Module):
    def __init__(
        self,
        actions_hat: torch.Tensor,
        mu: torch.Tensor | None = None,
        log_sigma_x2: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.actions_hat = actions_hat
        self.mu = mu
        self.log_sigma_x2 = log_sigma_x2
        self.last_batch = None

    def forward(self, batch):
        self.last_batch = batch
        return self.actions_hat, (self.mu, self.log_sigma_x2)


def _fixed_act_policy(
    actions_hat: torch.Tensor,
    *,
    gripper_indices: list[int],
    joint_representation: str = "relative",
    use_vae: bool = False,
    kl_weight: float = 10.0,
    mu: torch.Tensor | None = None,
    log_sigma_x2: torch.Tensor | None = None,
) -> ACTPolicy:
    policy = ACTPolicy.__new__(ACTPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        image_features={},
        robot_state_feature=None,
        env_state_feature=object(),
        condition_on_state=True,
        gripper_indices=gripper_indices,
        joint_representation=joint_representation,
        use_vae=use_vae,
        kl_weight=kl_weight,
    )
    policy.model = _FixedACTModel(actions_hat, mu, log_sigma_x2)
    return policy


def test_act_forward_padding_multi_gripper_all_padding_and_reduction_none():
    actions_hat = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [100.0, 100.0, 100.0, 100.0], [5.0, 6.0, 7.0, 8.0]],
            [[9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0], [17.0, 18.0, 19.0, 20.0]],
        ]
    )
    policy = _fixed_act_policy(actions_hat, gripper_indices=[1, 3])
    batch = {
        OBS_ENV_STATE: torch.zeros(2, 3),
        ACTION: torch.zeros_like(actions_hat),
        "action_is_pad": torch.tensor([[False, True, False], [True, True, True]]),
        "discard_me": torch.ones(1),
    }

    per_sample, details = policy.forward(batch, reduction="none")

    torch.testing.assert_close(per_sample, torch.tensor([4.5, 0.0]))
    torch.testing.assert_close(details["loss_sum_per_sample"], torch.tensor([36.0, 0.0]))
    torch.testing.assert_close(details["loss_count_per_sample"], torch.tensor([8, 0]))
    torch.testing.assert_close(details["gripper_loss_sum_per_sample"], torch.tensor([20.0, 0.0]))
    torch.testing.assert_close(details["gripper_loss_count_per_sample"], torch.tensor([4, 0]))
    torch.testing.assert_close(details["gripper_loss_per_sample"], torch.tensor([5.0, 0.0]))
    assert "discard_me" not in policy.model.last_batch

    mean_loss, mean_details = policy.forward(batch)
    torch.testing.assert_close(mean_loss, torch.tensor(4.5))
    assert mean_details["gripper_loss"] == pytest.approx(5.0)


def test_act_forward_all_padding_with_vae_returns_only_kl_and_finite_zero_metrics():
    actions_hat = torch.full((2, 3, 2), 100.0)
    mu = torch.tensor([[2.0], [2.0]])
    log_sigma_x2 = torch.zeros_like(mu)
    policy = _fixed_act_policy(
        actions_hat,
        gripper_indices=[1],
        use_vae=True,
        kl_weight=3.0,
        mu=mu,
        log_sigma_x2=log_sigma_x2,
    )
    batch = {
        OBS_ENV_STATE: torch.zeros(2, 3),
        ACTION: torch.zeros_like(actions_hat),
        "action_is_pad": torch.ones(2, 3, dtype=torch.bool),
    }

    loss, details = policy.forward(batch)
    per_sample, _ = policy.forward(batch, reduction="none")

    torch.testing.assert_close(details["loss_sum_per_sample"], torch.zeros(2))
    torch.testing.assert_close(details["loss_count_per_sample"], torch.zeros(2, dtype=torch.long))
    torch.testing.assert_close(per_sample, torch.zeros(2))
    torch.testing.assert_close(loss, torch.tensor(6.0))
    assert details["gripper_loss"] == 0.0
    torch.testing.assert_close(details["gripper_loss_per_sample"], torch.zeros(2))
    assert torch.isfinite(loss)
    assert torch.isfinite(details["gripper_loss_per_sample"]).all()


def test_act_prepare_model_batch_keeps_exact_declared_and_internal_training_keys():
    policy = ACTPolicy.__new__(ACTPolicy)
    nn.Module.__init__(policy)
    camera_key = "observation.images.camera"
    policy.config = SimpleNamespace(
        image_features={camera_key: object()},
        robot_state_feature=object(),
        env_state_feature=object(),
        condition_on_state=False,
    )
    camera = torch.zeros(1, 3, 8, 8)
    state = torch.zeros(1, 7)
    env_state = torch.ones(1, 3)
    action = torch.ones(1, 2, 7)
    action_is_pad = torch.zeros(1, 2, dtype=torch.bool)

    prepared = policy._prepare_model_batch(
        {
            camera_key: camera,
            OBS_STATE: state,
            OBS_ENV_STATE: env_state,
            ACTION: action,
            "action_is_pad": action_is_pad,
            OBS_IMAGES: [torch.full_like(camera, 9)],
            "discard_me": torch.ones(1),
        },
        include_targets=True,
    )

    assert set(prepared) == {
        camera_key,
        OBS_STATE,
        OBS_ENV_STATE,
        ACTION,
        "action_is_pad",
        OBS_IMAGES,
    }
    assert prepared[camera_key].shape == (1, 3, 224, 224)
    assert prepared[OBS_STATE] is state
    assert prepared[OBS_ENV_STATE] is env_state
    assert prepared[ACTION] is action
    assert prepared["action_is_pad"] is action_is_pad
    assert len(prepared[OBS_IMAGES]) == 1
    assert prepared[OBS_IMAGES][0].shape == (1, 3, 224, 224)
    torch.testing.assert_close(prepared[OBS_STATE], torch.zeros(1, 7))


@pytest.mark.parametrize(
    "packed_images",
    [None, [], (), torch.zeros(1, 3, 8, 8), [torch.zeros(1, 3, 8, 8), "not-a-tensor"]],
)
def test_act_prepare_model_batch_rejects_invalid_prepacked_images(packed_images):
    policy = ACTPolicy.__new__(ACTPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        image_features={"observation.images.camera": object()},
        robot_state_feature=None,
        env_state_feature=None,
        condition_on_state=True,
    )

    with pytest.raises(ValueError, match="observation.images"):
        policy._prepare_model_batch({OBS_IMAGES: packed_images}, include_targets=False)


def test_act_prepare_model_batch_accepts_nonempty_tensor_tuple_for_prepacked_images():
    policy = ACTPolicy.__new__(ACTPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        image_features={"observation.images.camera": object()},
        robot_state_feature=None,
        env_state_feature=None,
        condition_on_state=True,
    )
    images = (torch.zeros(1, 3, 8, 8),)

    prepared = policy._prepare_model_batch({OBS_IMAGES: images}, include_targets=False)

    assert isinstance(prepared[OBS_IMAGES], list)
    assert prepared[OBS_IMAGES][0].shape == (1, 3, 224, 224)


def test_act_prepare_model_batch_selects_current_image_from_relative_state_history():
    policy = ACTPolicy.__new__(ACTPolicy)
    nn.Module.__init__(policy)
    camera_key = "observation.images.camera"
    policy.config = SimpleNamespace(
        image_features={camera_key: object()},
        robot_state_feature=object(),
        env_state_feature=None,
        condition_on_state=True,
    )
    previous = torch.zeros(2, 3, 8, 8)
    current = torch.ones(2, 3, 8, 8)
    image_history = torch.stack((previous, current), dim=1)
    paired_state = torch.zeros(2, 2, 7)

    prepared = policy._prepare_model_batch(
        {camera_key: image_history, OBS_STATE: paired_state},
        include_targets=False,
    )

    torch.testing.assert_close(prepared[OBS_IMAGES][0], torch.ones(2, 3, 224, 224))
    assert prepared[OBS_STATE] is paired_state


def test_act_predict_action_chunk_does_not_pass_training_targets_to_model():
    actions_hat = torch.zeros(1, 2, 3)
    policy = _fixed_act_policy(actions_hat, gripper_indices=[2])
    batch = {
        OBS_ENV_STATE: torch.zeros(1, 3),
        ACTION: torch.ones_like(actions_hat),
        "action_is_pad": torch.zeros(1, 2, dtype=torch.bool),
    }

    predicted = policy.predict_action_chunk(batch)

    torch.testing.assert_close(predicted, actions_hat)
    assert ACTION not in policy.model.last_batch
    assert "action_is_pad" not in policy.model.last_batch


def test_act_forward_can_return_action_chunk_through_distributed_wrapper_path():
    actions_hat = torch.zeros(1, 2, 3)
    policy = _fixed_act_policy(actions_hat, gripper_indices=[2])
    policy.eval()

    predicted = policy(
        {OBS_ENV_STATE: torch.zeros(1, 3)},
        return_action_chunk=True,
    )

    torch.testing.assert_close(predicted, actions_hat)
    assert ACTION not in policy.model.last_batch
    assert "action_is_pad" not in policy.model.last_batch


def test_act_image_only_prepare_does_not_inspect_state_values(monkeypatch):
    policy = ACTPolicy.__new__(ACTPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        image_features={},
        robot_state_feature=object(),
        env_state_feature=None,
        condition_on_state=False,
    )
    state = torch.ones(1, 7)

    def fail_count_nonzero(*args, **kwargs):
        raise AssertionError("torch.count_nonzero must not be called while preparing the model batch")

    monkeypatch.setattr(torch, "count_nonzero", fail_count_nonzero)
    prepared = policy._prepare_model_batch({OBS_STATE: state}, include_targets=False)

    assert prepared[OBS_STATE] is state


def test_act_total_loss_contains_kld_but_not_gripper_metric():
    actions_hat = torch.tensor([[[1.0, 100.0]]])
    mu = torch.tensor([[2.0]])
    log_sigma_x2 = torch.zeros_like(mu)
    policy = _fixed_act_policy(
        actions_hat,
        gripper_indices=[1],
        use_vae=True,
        kl_weight=3.0,
        mu=mu,
        log_sigma_x2=log_sigma_x2,
    )
    batch = {
        OBS_ENV_STATE: torch.zeros(1, 3),
        ACTION: torch.zeros_like(actions_hat),
        "action_is_pad": torch.zeros(1, 1, dtype=torch.bool),
    }

    loss, details = policy.forward(batch)
    per_sample, _ = policy.forward(batch, reduction="none")

    torch.testing.assert_close(loss, torch.tensor(56.5))
    torch.testing.assert_close(per_sample, torch.tensor([50.5]))
    assert details["gripper_loss"] == pytest.approx(100.0)


@pytest.mark.parametrize("gripper_indices", [[], [1, 1], [-1], [2], [True]])
def test_act_relative_forward_rejects_invalid_gripper_indices(gripper_indices: list[int]):
    policy = _fixed_act_policy(torch.zeros(1, 1, 2), gripper_indices=gripper_indices)
    batch = {
        OBS_ENV_STATE: torch.zeros(1, 3),
        ACTION: torch.zeros(1, 1, 2),
        "action_is_pad": torch.zeros(1, 1, dtype=torch.bool),
    }

    with pytest.raises(ValueError, match="gripper indices"):
        policy.forward(batch)


def test_act_absolute_forward_uses_only_valid_unique_gripper_indices():
    actions_hat = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]])
    policy = _fixed_act_policy(
        actions_hat,
        gripper_indices=[1, 1, -1, 5, True, 3],
        joint_representation="absolute",
    )
    batch = {
        OBS_ENV_STATE: torch.zeros(1, 3),
        ACTION: torch.zeros_like(actions_hat),
        "action_is_pad": torch.zeros(1, 1, dtype=torch.bool),
    }

    loss, details = policy.forward(batch)

    torch.testing.assert_close(loss, torch.tensor(3.0))
    assert details["gripper_loss"] == pytest.approx(3.0)
    torch.testing.assert_close(details["gripper_loss_sum_per_sample"], torch.tensor([6.0]))
    torch.testing.assert_close(details["gripper_loss_count_per_sample"], torch.tensor([2]))
    torch.testing.assert_close(details["gripper_loss_per_sample"], torch.tensor([3.0]))


def test_act_absolute_5d_default_gripper_index_produces_finite_zero_metrics():
    actions_hat = torch.ones(2, 3, 5)
    policy = _fixed_act_policy(
        actions_hat,
        gripper_indices=[6],
        joint_representation="absolute",
    )
    batch = {
        OBS_ENV_STATE: torch.zeros(2, 3),
        ACTION: torch.zeros_like(actions_hat),
        "action_is_pad": torch.zeros(2, 3, dtype=torch.bool),
    }

    loss, details = policy.forward(batch)

    torch.testing.assert_close(loss, torch.tensor(1.0))
    assert details["gripper_loss"] == 0.0
    torch.testing.assert_close(details["gripper_loss_sum_per_sample"], torch.zeros(2))
    torch.testing.assert_close(details["gripper_loss_count_per_sample"], torch.zeros(2, dtype=torch.long))
    torch.testing.assert_close(details["gripper_loss_per_sample"], torch.zeros(2))
    assert torch.isfinite(details["gripper_loss_per_sample"]).all()


def test_act_forward_rejects_broadcastable_action_shape_mismatch():
    policy = _fixed_act_policy(torch.zeros(1, 2, 3), gripper_indices=[2])
    batch = {
        OBS_ENV_STATE: torch.zeros(1, 3),
        ACTION: torch.zeros(1, 1, 3),
        "action_is_pad": torch.zeros(1, 2, dtype=torch.bool),
    }

    with pytest.raises(ValueError, match="action.*shape"):
        policy.forward(batch)


def test_act_env_only_model_uses_available_batch_reference_and_eval_is_deterministic():
    config = ACTConfig(
        input_features={OBS_ENV_STATE: PolicyFeature(FeatureType.ENV, (3,))},
        output_features={ACTION: PolicyFeature(FeatureType.ACTION, (2,))},
        gripper_indices=[1],
        device="cpu",
        chunk_size=2,
        n_action_steps=2,
        dim_model=16,
        n_heads=4,
        dim_feedforward=32,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_vae_encoder_layers=1,
        latent_dim=4,
        dropout=0.0,
    )
    model = ACT(config).eval()
    batch = {OBS_ENV_STATE: torch.zeros(2, 3)}

    first = model(batch)[0]
    second = model(batch)[0]

    torch.testing.assert_close(first, second)


def test_act_image_only_prepare_keeps_exact_zero_state_and_declared_projection(tmp_path: Path):
    config = _act_relative_config(
        tmp_path,
        condition_on_state=False,
        pretrained_backbone_weights=None,
        dim_model=16,
        n_heads=4,
        dim_feedforward=32,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_vae_encoder_layers=1,
        latent_dim=4,
        dropout=0.0,
    )
    preprocessor, _ = make_act_pre_post_processors(config)
    policy = ACTPolicy(config)
    assert policy.model.encoder_robot_state_input_proj.in_features == 7

    previous = torch.zeros(7)
    current = torch.arange(7, dtype=torch.float32) + 1
    processed = preprocessor(
        {
            OBS_STATE: torch.stack([previous, current]).unsqueeze(0),
            "observation.images.camera": torch.zeros(1, 3, 32, 32),
            ACTION: current.expand(1, 16, 7).clone(),
            "action_is_pad": torch.zeros(1, 16, dtype=torch.bool),
        }
    )
    fixed_model = _FixedACTModel(torch.zeros_like(processed[ACTION]))
    policy.model = fixed_model

    policy.forward(processed)

    assert OBS_STATE in fixed_model.last_batch
    torch.testing.assert_close(fixed_model.last_batch[OBS_STATE], torch.zeros(1, 7))
    assert OBS_IMAGES in fixed_model.last_batch


def test_act_relative_7d_config_rejects_wrong_path_horizon_gripper_and_names(tmp_path: Path):
    state_path, action_path = _save_act_relative_stats(tmp_path)
    common = {
        "joint_representation": "relative",
        "chunk_size": 16,
        "n_action_steps": 16,
        "input_features": {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            "observation.images.camera": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
        },
        "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        "action_feature_names": JOINT_NAMES_7D,
    }

    cfg = ACTConfig(**common)
    with pytest.raises(ValueError, match="stats path"):
        cfg.validate_features()
    cfg = ACTConfig(
        **common,
        relative_state_stats_path=str(state_path),
        relative_action_stats_path=str(tmp_path / "missing.json"),
    )
    with pytest.raises(ValueError, match="does not exist"):
        cfg.validate_features()
    cfg = ACTConfig(
        **common,
        relative_state_stats_path=str(state_path),
        relative_action_stats_path=str(tmp_path / "relative_action_chunk50_q01_q99.json"),
    )
    with pytest.raises(ValueError, match="horizon"):
        cfg.validate_features()
    cfg = ACTConfig(
        **(common | {"gripper_indices": [5]}),
        relative_state_stats_path=str(state_path),
        relative_action_stats_path=str(action_path),
    )
    with pytest.raises(ValueError, match="gripper"):
        cfg.validate_features()
    cfg = ACTConfig(
        **(common | {"action_feature_names": [*JOINT_NAMES_7D[:-1], "claw"]}),
        relative_state_stats_path=str(state_path),
        relative_action_stats_path=str(action_path),
    )
    with pytest.raises(ValueError, match="feature names"):
        cfg.validate_features()


def create_default_config():
    """Create a default ACT configuration for testing."""
    config = ACTConfig()
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
    }
    config.normalization_mapping = {
        FeatureType.STATE: NormalizationMode.MEAN_STD,
        FeatureType.ACTION: NormalizationMode.MEAN_STD,
    }
    config.device = "cpu"
    return config


def create_default_stats():
    """Create default dataset statistics for testing."""
    return {
        OBS_STATE: {"mean": torch.zeros(7), "std": torch.ones(7)},
        ACTION: {"mean": torch.zeros(4), "std": torch.ones(4)},
    }


def test_make_act_processor_basic():
    """Test basic creation of ACT processor."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_act_pre_post_processors(config, stats)

    # Check processor names
    assert preprocessor.name == "policy_preprocessor"
    assert postprocessor.name == "policy_postprocessor"

    # Check steps in preprocessor
    assert len(preprocessor.steps) == 4
    assert isinstance(preprocessor.steps[0], RenameObservationsProcessorStep)
    assert isinstance(preprocessor.steps[1], AddBatchDimensionProcessorStep)
    assert isinstance(preprocessor.steps[2], DeviceProcessorStep)
    assert isinstance(preprocessor.steps[3], NormalizerProcessorStep)

    # Check steps in postprocessor
    assert len(postprocessor.steps) == 2
    assert isinstance(postprocessor.steps[0], UnnormalizerProcessorStep)
    assert isinstance(postprocessor.steps[1], DeviceProcessorStep)


def test_act_processor_normalization():
    """Test that ACT processor correctly normalizes and unnormalizes data."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_act_pre_post_processors(
        config,
        stats,
    )

    # Create test data
    observation = {OBS_STATE: torch.randn(7)}
    action = torch.randn(4)
    transition = create_transition(observation, action)
    batch = transition_to_batch(transition)

    # Process through preprocessor
    processed = preprocessor(batch)

    # Check that data is normalized and batched
    assert processed[OBS_STATE].shape == (1, 7)
    assert processed[TransitionKey.ACTION.value].shape == (1, 4)

    # Process action through postprocessor
    postprocessed = postprocessor(processed[TransitionKey.ACTION.value])

    # Check that action is unnormalized
    assert postprocessed.shape == (1, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_act_processor_cuda():
    """Test ACT processor with CUDA device."""
    config = create_default_config()
    config.device = "cuda"
    stats = create_default_stats()

    preprocessor, postprocessor = make_act_pre_post_processors(
        config,
        stats,
    )

    # Create CPU data
    observation = {OBS_STATE: torch.randn(7)}
    action = torch.randn(4)
    transition = create_transition(observation, action)
    batch = transition_to_batch(transition)

    # Process through preprocessor
    processed = preprocessor(batch)

    # Check that data is on CUDA
    assert processed[OBS_STATE].device.type == "cuda"
    assert processed[TransitionKey.ACTION.value].device.type == "cuda"

    # Process through postprocessor
    postprocessed = postprocessor(processed[TransitionKey.ACTION.value])

    # Check that action is back on CPU
    assert postprocessed.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_act_processor_accelerate_scenario():
    """Test ACT processor in simulated Accelerate scenario (data already on GPU)."""
    config = create_default_config()
    config.device = "cuda:0"
    stats = create_default_stats()

    preprocessor, postprocessor = make_act_pre_post_processors(
        config,
        stats,
    )

    # Simulate Accelerate: data already on GPU
    device = torch.device("cuda:0")
    observation = {OBS_STATE: torch.randn(1, 7).to(device)}  # Already batched and on GPU
    action = torch.randn(1, 4).to(device)
    transition = create_transition(observation, action)
    batch = transition_to_batch(transition)

    # Process through preprocessor
    processed = preprocessor(batch)

    # Check that data stays on same GPU (not moved unnecessarily)
    assert processed[OBS_STATE].device == device
    assert processed[TransitionKey.ACTION.value].device == device


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
def test_act_processor_multi_gpu():
    """Test ACT processor with multi-GPU setup."""
    config = create_default_config()
    config.device = "cuda:0"
    stats = create_default_stats()

    preprocessor, postprocessor = make_act_pre_post_processors(
        config,
        stats,
    )

    # Simulate data on different GPU (like in multi-GPU training)
    device = torch.device("cuda:1")
    observation = {OBS_STATE: torch.randn(1, 7).to(device)}
    action = torch.randn(1, 4).to(device)
    transition = create_transition(observation, action)
    batch = transition_to_batch(transition)

    # Process through preprocessor
    processed = preprocessor(batch)

    # Check that data stays on cuda:1 (not moved to cuda:0)
    assert processed[OBS_STATE].device == device
    assert processed[TransitionKey.ACTION.value].device == device


def test_act_processor_without_stats():
    """Test ACT processor creation without dataset statistics."""
    config = create_default_config()

    preprocessor, postprocessor = make_act_pre_post_processors(
        config,
        dataset_stats=None,
    )

    # Should still create processors, but normalization won't have stats
    assert preprocessor is not None
    assert postprocessor is not None

    # Process should still work (but won't normalize without stats)
    observation = {OBS_STATE: torch.randn(7)}
    action = torch.randn(4)
    transition = create_transition(observation, action)
    batch = transition_to_batch(transition)

    processed = preprocessor(batch)
    assert processed is not None


def test_act_processor_save_and_load():
    """Test saving and loading ACT processor."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_act_pre_post_processors(
        config,
        stats,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save preprocessor
        preprocessor.save_pretrained(tmpdir)

        # Load preprocessor
        loaded_preprocessor = DataProcessorPipeline.from_pretrained(
            tmpdir, config_filename="policy_preprocessor.json"
        )

        # Test that loaded processor works
        observation = {OBS_STATE: torch.randn(7)}
        action = torch.randn(4)
        transition = create_transition(observation, action)
        batch = transition_to_batch(transition)

        processed = loaded_preprocessor(batch)
        assert processed[OBS_STATE].shape == (1, 7)
        assert processed[TransitionKey.ACTION.value].shape == (1, 4)


def test_act_processor_device_placement_preservation():
    """Test that ACT processor preserves device placement correctly."""
    config = create_default_config()
    stats = create_default_stats()

    # Test with CPU config
    config.device = "cpu"
    preprocessor, _ = make_act_pre_post_processors(
        config,
        stats,
    )

    # Process CPU data
    observation = {OBS_STATE: torch.randn(7)}
    action = torch.randn(4)
    transition = create_transition(observation, action)
    batch = transition_to_batch(transition)

    processed = preprocessor(batch)
    assert processed[OBS_STATE].device.type == "cpu"
    assert processed[TransitionKey.ACTION.value].device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_act_processor_mixed_precision():
    """Test ACT processor with mixed precision (float16)."""
    config = create_default_config()
    config.device = "cuda"
    stats = create_default_stats()

    # Modify the device processor to use float16
    preprocessor, postprocessor = make_act_pre_post_processors(
        config,
        stats,
    )

    # Replace DeviceProcessorStep with one that uses float16
    modified_steps = []
    for step in preprocessor.steps:
        if isinstance(step, DeviceProcessorStep):
            modified_steps.append(DeviceProcessorStep(device=config.device, float_dtype="float16"))
        elif isinstance(step, NormalizerProcessorStep):
            # Update normalizer to use the same device as the device processor
            norm_step = step  # Now type checker knows this is NormalizerProcessorStep
            modified_steps.append(
                NormalizerProcessorStep(
                    features=norm_step.features,
                    norm_map=norm_step.norm_map,
                    stats=norm_step.stats,
                    device=config.device,
                    dtype=torch.float16,  # Match the float16 dtype
                )
            )
        else:
            modified_steps.append(step)
    preprocessor.steps = modified_steps

    # Create test data
    observation = {OBS_STATE: torch.randn(7, dtype=torch.float32)}
    action = torch.randn(4, dtype=torch.float32)
    transition = create_transition(observation, action)
    batch = transition_to_batch(transition)

    # Process through preprocessor
    processed = preprocessor(batch)

    # Check that data is converted to float16
    assert processed[OBS_STATE].dtype == torch.float16
    assert processed[TransitionKey.ACTION.value].dtype == torch.float16


def test_act_processor_batch_consistency():
    """Test that ACT processor handles different batch sizes correctly."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_act_pre_post_processors(
        config,
        stats,
    )

    # Test single sample (unbatched)
    observation = {OBS_STATE: torch.randn(7)}
    action = torch.randn(4)
    transition = create_transition(observation, action)
    batch = transition_to_batch(transition)

    processed = preprocessor(batch)
    assert processed[OBS_STATE].shape[0] == 1  # Batched

    # Test already batched data
    observation_batched = {OBS_STATE: torch.randn(8, 7)}  # Batch of 8
    action_batched = torch.randn(8, 4)
    transition_batched = create_transition(observation_batched, action_batched)
    batch_batched = transition_to_batch(transition_batched)

    processed_batched = preprocessor(batch_batched)
    assert processed_batched[OBS_STATE].shape[0] == 8
    assert processed_batched[TransitionKey.ACTION.value].shape[0] == 8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_act_processor_bfloat16_device_float32_normalizer():
    """Test: DeviceProcessor(bfloat16) + NormalizerProcessor(float32) → output bfloat16 via automatic adaptation"""
    config = create_default_config()
    config.device = "cuda"
    stats = create_default_stats()

    preprocessor, _ = make_act_pre_post_processors(
        config,
        stats,
    )

    # Modify the pipeline to use bfloat16 device processor with float32 normalizer
    modified_steps = []
    for step in preprocessor.steps:
        if isinstance(step, DeviceProcessorStep):
            # Device processor converts to bfloat16
            modified_steps.append(DeviceProcessorStep(device=config.device, float_dtype="bfloat16"))
        elif isinstance(step, NormalizerProcessorStep):
            # Normalizer stays configured as float32 (will auto-adapt to bfloat16)
            norm_step = step  # Now type checker knows this is NormalizerProcessorStep
            modified_steps.append(
                NormalizerProcessorStep(
                    features=norm_step.features,
                    norm_map=norm_step.norm_map,
                    stats=norm_step.stats,
                    device=config.device,
                    dtype=torch.float32,  # Deliberately configured as float32
                )
            )
        else:
            modified_steps.append(step)
    preprocessor.steps = modified_steps

    # Verify initial normalizer configuration
    normalizer_step = preprocessor.steps[3]  # NormalizerProcessorStep
    assert normalizer_step.dtype == torch.float32

    # Create test data
    observation = {OBS_STATE: torch.randn(7, dtype=torch.float32)}  # Start with float32
    action = torch.randn(4, dtype=torch.float32)
    transition = create_transition(observation, action)
    batch = transition_to_batch(transition)

    # Process through full pipeline
    processed = preprocessor(batch)

    # Verify: DeviceProcessor → bfloat16, NormalizerProcessor adapts → final output is bfloat16
    assert processed[OBS_STATE].dtype == torch.bfloat16
    assert processed[TransitionKey.ACTION.value].dtype == torch.bfloat16

    # Verify normalizer automatically adapted its internal state
    assert normalizer_step.dtype == torch.bfloat16
    for stat_tensor in normalizer_step._tensor_stats[OBS_STATE].values():
        assert stat_tensor.dtype == torch.bfloat16
