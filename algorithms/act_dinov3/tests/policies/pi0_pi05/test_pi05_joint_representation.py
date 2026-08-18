import json
import shutil
import warnings
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.absolute_action_stats import (
    compute_absolute_action_stats_from_episodes,
    save_absolute_action_stats,
)
from lerobot.datasets.relative_joint_stats import (
    QuantileStats,
    RelativeJointStatsBundle,
    compute_relative_joint_stats_from_episodes,
    load_relative_joint_stats_paths,
    save_relative_joint_stats,
)
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
from lerobot.policies.pi05.joint_representation import (
    Pi05AbsoluteActionProcessorStep,
    Pi05JointRepresentationProcessorStep,
    build_arm_mask,
    can_use_pi05_joint_stats,
    make_pi05_joint_stats,
    merge_pi05_joint_stats,
)
from lerobot.policies import factory as policy_factory
from lerobot.policies.pi05 import modeling_pi05 as pi05_modeling
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch, PaliGemmaWithExpertModel
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
    RelativeJointProcessorStep,
    RelativeActionsProcessorStep,
    TokenizerProcessorStep,
    TransitionKey,
    batch_to_transition,
    identity_transition,
)
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

ACTION_NAMES = [f"left_joint_{i}" for i in range(7)] + [f"right_joint_{i}" for i in range(7)]
GRIPPER_INDICES = [6, 13]
JOINT_NAMES_7D = [f"joint_{index}" for index in range(6)] + ["gripper"]
JOINT_NAMES_14D = [f"left_joint_{index}" for index in range(6)] + ["left_gripper"] + [
    f"right_joint_{index}" for index in range(6)
] + ["right_gripper"]
DUAL_ARM_20D_STATE_NAMES = [f"left_joint_{index}" for index in range(6)] + [
    "left_endpoint_x",
    "left_endpoint_y",
    "left_endpoint_z",
    "left_gripper",
] + [f"right_joint_{index}" for index in range(6)] + [
    "right_endpoint_x",
    "right_endpoint_y",
    "right_endpoint_z",
    "right_gripper",
]


def _save_relative_stats(tmp_path: Path) -> tuple[Path, Path]:
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
        tmp_path / "relative_action_chunk50_q01_q99.json",
    )


def _save_absolute_stats(tmp_path: Path) -> tuple[Path, Path]:
    bundle = compute_absolute_action_stats_from_episodes(
        [torch.arange(35, dtype=torch.float32).reshape(5, 7)],
        horizons=[2],
        feature_names=JOINT_NAMES_7D,
        scaled_indices=list(range(7)),
    )
    save_absolute_action_stats(bundle, tmp_path)
    return tmp_path / "absolute_state_q01_q99.json", tmp_path / "absolute_action_chunk2_q01_q99.json"


def test_absolute_7d_config_uses_explicit_state_and_action_quantiles(tmp_path: Path) -> None:
    state_path, action_path = _save_absolute_stats(tmp_path)
    config = PI05Config(
        joint_representation="absolute",
        joint_gripper_indices=[6],
        chunk_size=50,
        n_action_steps=50,
        absolute_state_stats_path=str(state_path),
        absolute_action_stats_path=str(action_path),
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        action_feature_names=JOINT_NAMES_7D,
    )

    config.validate_features()

    stats = merge_pi05_joint_stats(config, dataset_stats=None)
    assert stats is not None
    assert torch.allclose(stats[OBS_STATE]["q01"], torch.tensor(config.absolute_action_stats.state.q01, dtype=torch.float32))
    assert torch.allclose(stats[ACTION]["q99"], torch.tensor(config.absolute_action_stats.actions[2].q99, dtype=torch.float32))


def test_absolute_7d_pipeline_honors_quantile_clipping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer_module = pytest.importorskip("lerobot.processor.tokenizer_processor")
    monkeypatch.setattr(tokenizer_module, "_transformers_available", True)
    monkeypatch.setattr(tokenizer_module, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: object()))
    state_path, action_path = _save_absolute_stats(tmp_path)
    config = PI05Config(
        device="cpu",
        joint_representation="absolute",
        joint_gripper_indices=[6],
        chunk_size=2,
        n_action_steps=2,
        absolute_state_stats_path=str(state_path),
        absolute_action_stats_path=str(action_path),
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        action_feature_names=JOINT_NAMES_7D,
        clip_quantiles=True,
    )
    config.validate_features()

    preprocessor, _ = make_pi05_pre_post_processors(config, dataset_stats=None)

    normalizer = next(step for step in preprocessor.steps if isinstance(step, NormalizerProcessorStep))
    assert normalizer.clip_quantiles is True


def test_libero_postprocessor_keeps_native_delta_gripper_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer_module = pytest.importorskip("lerobot.processor.tokenizer_processor")
    monkeypatch.setattr(tokenizer_module, "_transformers_available", True)
    monkeypatch.setattr(tokenizer_module, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: object()))
    config = PI05Config(
        device="cpu",
        apply_action_limits=False,
        normalization_mapping={
            "ACTION": NormalizationMode.QUANTILES,
            "STATE": NormalizationMode.QUANTILES,
            "VISUAL": NormalizationMode.IDENTITY,
        },
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    dataset_stats = {
        OBS_STATE: {"q01": torch.full((8,), -1.0), "q99": torch.ones(8)},
        ACTION: {"q01": torch.full((7,), -1.0), "q99": torch.ones(7)},
    }

    _, postprocessor = make_pi05_pre_post_processors(config, dataset_stats=dataset_stats)
    result = postprocessor(torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]]))

    assert not any(isinstance(step, Pi05AbsoluteActionProcessorStep) for step in postprocessor.steps)
    torch.testing.assert_close(result[..., 6], torch.tensor([-1.0]))


def _pi05_relative_config(tmp_path: Path, **overrides) -> PI05Config:
    state_path, action_path = _save_relative_stats(tmp_path)
    kwargs = {
        "joint_representation": "relative",
        "joint_gripper_indices": [6],
        "chunk_size": 50,
        "n_action_steps": 50,
        "relative_state_stats_path": str(state_path),
        "relative_action_stats_path": str(action_path),
        "input_features": {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,))},
        "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        "action_feature_names": JOINT_NAMES_7D,
    }
    kwargs.update(overrides)
    return PI05Config(**kwargs)


def test_pi05_dual_arm_relative_config_loads_separate_14d_quantiles(tmp_path: Path) -> None:
    bundle = compute_relative_joint_stats_from_episodes(
        [torch.arange(70, dtype=torch.float32).reshape(5, 14)],
        gripper_indices=GRIPPER_INDICES,
        horizons=[16, 50],
        feature_names=JOINT_NAMES_14D,
        source_manifest_sha256="a" * 64,
    )
    save_relative_joint_stats(bundle, tmp_path)
    config = PI05Config(
        joint_representation="relative",
        joint_gripper_indices=GRIPPER_INDICES,
        chunk_size=50,
        n_action_steps=50,
        relative_state_stats_path=str(tmp_path / "relative_state_q01_q99.json"),
        relative_action_stats_path=str(tmp_path / "relative_action_chunk50_q01_q99.json"),
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(14,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(14,))},
        action_feature_names=JOINT_NAMES_14D,
    )

    config.validate_features()

    assert config.relative_joint_stats.state.q01.shape == (14,)
    assert config.relative_joint_stats.actions[50].q99.shape == (14,)


def test_pi05_dual_arm_relative_processor_uses_quantile_stats(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tokenizer_module = pytest.importorskip("lerobot.processor.tokenizer_processor")
    monkeypatch.setattr(tokenizer_module, "_transformers_available", True)
    monkeypatch.setattr(
        tokenizer_module,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: object()),
    )
    bundle = compute_relative_joint_stats_from_episodes(
        [torch.arange(70, dtype=torch.float32).reshape(5, 14)],
        gripper_indices=GRIPPER_INDICES,
        horizons=[50],
        feature_names=JOINT_NAMES_14D,
        source_manifest_sha256="a" * 64,
    )
    save_relative_joint_stats(bundle, tmp_path)
    config = PI05Config(
        joint_representation="relative",
        joint_gripper_indices=GRIPPER_INDICES,
        chunk_size=50,
        n_action_steps=50,
        relative_state_stats_path=str(tmp_path / "relative_state_q01_q99.json"),
        relative_action_stats_path=str(tmp_path / "relative_action_chunk50_q01_q99.json"),
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(14,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(14,))},
        action_feature_names=JOINT_NAMES_14D,
    )
    config.validate_features()

    preprocessor, _ = make_pi05_pre_post_processors(config)
    normalizer = next(step for step in preprocessor.steps if isinstance(step, NormalizerProcessorStep))

    assert normalizer.norm_map[FeatureType.STATE] is NormalizationMode.QUANTILES
    assert normalizer.norm_map[FeatureType.ACTION] is NormalizationMode.QUANTILES


def test_pi05_20d_relative_pipeline_keeps_endpoint_and_gripper_state_absolute(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tokenizer_module = pytest.importorskip("lerobot.processor.tokenizer_processor")
    monkeypatch.setattr(tokenizer_module, "_transformers_available", True)
    monkeypatch.setattr(
        tokenizer_module,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: object()),
    )
    state = torch.arange(80, dtype=torch.float32).reshape(4, 20) / 100
    action = state[:, [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 19]]
    bundle = compute_relative_joint_stats_from_episodes(
        [state],
        action_episodes=[action],
        state_feature_names=DUAL_ARM_20D_STATE_NAMES,
        action_feature_names=JOINT_NAMES_14D,
        state_gripper_indices=[9, 19],
        action_gripper_indices=GRIPPER_INDICES,
        action_state_indices=[0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 19],
        state_absolute_indices=[6, 7, 8, 9, 16, 17, 18, 19],
        horizons=[2],
        source_manifest_sha256="a" * 64,
    )
    save_relative_joint_stats(bundle, tmp_path)
    config = PI05Config(
        device="cpu",
        joint_representation="relative",
        joint_gripper_indices=GRIPPER_INDICES,
        state_gripper_indices=[9, 19],
        state_absolute_indices=[6, 7, 8, 9, 16, 17, 18, 19],
        state_feature_names=DUAL_ARM_20D_STATE_NAMES,
        action_feature_names=JOINT_NAMES_14D,
        chunk_size=2,
        n_action_steps=2,
        relative_state_stats_path=str(tmp_path / "relative_state_q01_q99.json"),
        relative_action_stats_path=str(tmp_path / "relative_action_chunk2_q01_q99.json"),
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(20,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(14,))},
    )
    config.validate_features()

    preprocessor, _ = make_pi05_pre_post_processors(config)
    joint_step = next(step for step in preprocessor.steps if isinstance(step, RelativeJointProcessorStep))
    transition = batch_to_transition(
        {
            OBS_STATE: torch.stack([state[0], state[1]]).unsqueeze(0),
            ACTION: torch.stack([action[2], action[3]]).unsqueeze(0),
        }
    )

    result = joint_step(transition)
    converted_state = result[TransitionKey.OBSERVATION][OBS_STATE]
    converted_action = result[TransitionKey.ACTION]
    absolute_state_indices = [6, 7, 8, 9, 16, 17, 18, 19]
    relative_state_indices = [index for index in range(20) if index not in absolute_state_indices]

    torch.testing.assert_close(converted_state[..., absolute_state_indices], state[1, absolute_state_indices][None])
    torch.testing.assert_close(
        converted_state[..., relative_state_indices],
        (state[1, relative_state_indices] - state[0, relative_state_indices])[None],
    )
    torch.testing.assert_close(converted_action[..., GRIPPER_INDICES], action[None, 2:, GRIPPER_INDICES])


def test_pi05_20d_relative_pipeline_uses_dual_arm_names_when_metadata_is_not_injected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tokenizer_module = pytest.importorskip("lerobot.processor.tokenizer_processor")
    monkeypatch.setattr(tokenizer_module, "_transformers_available", True)
    monkeypatch.setattr(
        tokenizer_module,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: object()),
    )
    state = torch.arange(80, dtype=torch.float32).reshape(4, 20) / 100
    action = state[:, [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 19]]
    bundle = compute_relative_joint_stats_from_episodes(
        [state],
        action_episodes=[action],
        state_feature_names=DUAL_ARM_20D_STATE_NAMES,
        action_feature_names=JOINT_NAMES_14D,
        state_gripper_indices=[9, 19],
        action_gripper_indices=GRIPPER_INDICES,
        action_state_indices=[0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 19],
        state_absolute_indices=[6, 7, 8, 9, 16, 17, 18, 19],
        horizons=[2],
        source_manifest_sha256="a" * 64,
    )
    save_relative_joint_stats(bundle, tmp_path)
    config = PI05Config(
        device="cpu",
        joint_representation="relative",
        joint_gripper_indices=GRIPPER_INDICES,
        state_gripper_indices=[9, 19],
        state_absolute_indices=[6, 7, 8, 9, 16, 17, 18, 19],
        chunk_size=2,
        n_action_steps=2,
        relative_state_stats_path=str(tmp_path / "relative_state_q01_q99.json"),
        relative_action_stats_path=str(tmp_path / "relative_action_chunk2_q01_q99.json"),
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(20,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(14,))},
    )
    config.validate_features()

    preprocessor, _ = make_pi05_pre_post_processors(config)
    joint_step = next(step for step in preprocessor.steps if isinstance(step, RelativeJointProcessorStep))

    assert joint_step._state_names == DUAL_ARM_20D_STATE_NAMES
    assert joint_step._action_names == JOINT_NAMES_14D


def test_pi05_processor_uses_configured_tokenizer_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tokenizer_module = pytest.importorskip("lerobot.processor.tokenizer_processor")
    monkeypatch.setattr(tokenizer_module, "_transformers_available", True)
    monkeypatch.setattr(
        tokenizer_module,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: object()),
    )
    state_path, action_path = _save_relative_stats(tmp_path)
    config = PI05Config(
        device="cpu",
        joint_representation="relative",
        joint_gripper_indices=[6],
        chunk_size=50,
        n_action_steps=50,
        relative_state_stats_path=str(state_path),
        relative_action_stats_path=str(action_path),
        tokenizer_name="/data/jianan/weight/paligemma-3b-pt-224",
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        action_feature_names=JOINT_NAMES_7D,
    )
    config.validate_features()

    preprocessor, _ = make_pi05_pre_post_processors(config)
    tokenizer_step = next(step for step in preprocessor.steps if isinstance(step, TokenizerProcessorStep))

    assert tokenizer_step.tokenizer_name == "/data/jianan/weight/paligemma-3b-pt-224"


def test_pi05_dual_arm_absolute_state_relative_action_uses_separate_quantiles(tmp_path: Path) -> None:
    absolute_dir = tmp_path / "absolute"
    relative_dir = tmp_path / "relative"
    episode = torch.arange(84, dtype=torch.float32).reshape(6, 14)
    absolute_bundle = compute_absolute_action_stats_from_episodes(
        [episode],
        horizons=[50],
        feature_names=JOINT_NAMES_14D,
        scaled_indices=list(range(14)),
    )
    save_absolute_action_stats(absolute_bundle, absolute_dir)
    relative_bundle = compute_relative_joint_stats_from_episodes(
        [episode],
        gripper_indices=GRIPPER_INDICES,
        horizons=[50],
        feature_names=JOINT_NAMES_14D,
        source_manifest_sha256="a" * 64,
    )
    save_relative_joint_stats(relative_bundle, relative_dir)
    config = PI05Config(
        joint_representation="absolute",
        use_relative_actions=True,
        relative_exclude_joints=["gripper"],
        joint_gripper_indices=GRIPPER_INDICES,
        chunk_size=50,
        n_action_steps=50,
        absolute_state_stats_path=str(absolute_dir / "absolute_state_q01_q99.json"),
        relative_action_stats_path=str(relative_dir / "relative_action_chunk50_q01_q99.json"),
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(14,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(14,))},
        action_feature_names=JOINT_NAMES_14D,
        clip_quantiles=False,
    )

    config.validate_features()

    stats = merge_pi05_joint_stats(config, dataset_stats=None)
    assert stats is not None
    torch.testing.assert_close(
        stats[OBS_STATE]["q01"], torch.tensor(absolute_bundle.state.q01, dtype=torch.float32)
    )
    torch.testing.assert_close(
        stats[ACTION]["q99"], torch.tensor(relative_bundle.actions[50].q99, dtype=torch.float32)
    )
    assert not torch.equal(stats[OBS_STATE]["q01"], stats[ACTION]["q01"])


def test_pi05_dual_arm_absolute_state_relative_action_pipeline_matches_openpi_delta_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer_module = pytest.importorskip("lerobot.processor.tokenizer_processor")
    monkeypatch.setattr(tokenizer_module, "_transformers_available", True)
    monkeypatch.setattr(
        tokenizer_module,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: object()),
    )
    absolute_dir = tmp_path / "absolute"
    relative_dir = tmp_path / "relative"
    episode = torch.arange(112, dtype=torch.float32).reshape(8, 14) / 100
    absolute_bundle = compute_absolute_action_stats_from_episodes(
        [episode], horizons=[2], feature_names=JOINT_NAMES_14D, scaled_indices=list(range(14))
    )
    save_absolute_action_stats(absolute_bundle, absolute_dir)
    relative_bundle = compute_relative_joint_stats_from_episodes(
        [episode],
        gripper_indices=GRIPPER_INDICES,
        horizons=[2],
        feature_names=JOINT_NAMES_14D,
        source_manifest_sha256="a" * 64,
    )
    save_relative_joint_stats(relative_bundle, relative_dir)
    config = PI05Config(
        device="cpu",
        joint_representation="absolute",
        use_relative_actions=True,
        relative_exclude_joints=["gripper"],
        joint_gripper_indices=GRIPPER_INDICES,
        chunk_size=2,
        n_action_steps=2,
        absolute_state_stats_path=str(absolute_dir / "absolute_state_q01_q99.json"),
        relative_action_stats_path=str(relative_dir / "relative_action_chunk2_q01_q99.json"),
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(14,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(14,))},
        action_feature_names=JOINT_NAMES_14D,
        clip_quantiles=False,
        apply_action_limits=False,
    )
    config.validate_features()
    preprocessor, postprocessor = make_pi05_pre_post_processors(config, dataset_stats=None)

    relative_step = next(step for step in preprocessor.steps if isinstance(step, RelativeActionsProcessorStep))
    normalizer = next(step for step in preprocessor.steps if isinstance(step, NormalizerProcessorStep))
    absolute_step = next(step for step in postprocessor.steps if isinstance(step, AbsoluteActionsProcessorStep))
    assert preprocessor.steps.index(relative_step) < preprocessor.steps.index(normalizer)
    assert postprocessor.steps.index(absolute_step) > 0
    assert normalizer.clip_quantiles is False

    current_state = episode[2].unsqueeze(0)
    absolute_actions = torch.stack([episode[3], episode[4]], dim=0).unsqueeze(0)
    transition = batch_to_transition({OBS_STATE: current_state, ACTION: absolute_actions})
    relative_transition = relative_step(transition)
    relative_actions = relative_transition[TransitionKey.ACTION]
    expected_relative = absolute_actions.clone()
    arm_mask = build_arm_mask(14, gripper_indices=GRIPPER_INDICES)
    expected_relative[..., arm_mask] -= current_state[:, None, arm_mask]
    torch.testing.assert_close(relative_transition[TransitionKey.OBSERVATION][OBS_STATE], current_state)
    torch.testing.assert_close(relative_actions, expected_relative)
    torch.testing.assert_close(relative_actions[..., GRIPPER_INDICES], absolute_actions[..., GRIPPER_INDICES])

    normalized_transition = normalizer(relative_transition)
    assert not torch.equal(normalized_transition[TransitionKey.OBSERVATION][OBS_STATE], relative_actions[:, 0])
    restored_relative = postprocessor.steps[0](
        batch_to_transition({ACTION: normalized_transition[TransitionKey.ACTION]})
    )
    restored_absolute = absolute_step(restored_relative)
    torch.testing.assert_close(restored_absolute[TransitionKey.ACTION], absolute_actions)


def _raw_state() -> torch.Tensor:
    return torch.tensor(
        [
            [
                [0.0, 1.0, -1.0, 0.1, 0.2, -0.2, 0.03, 0.0, 1.1, -1.1, -0.1, 0.3, -0.3, 0.04],
                [0.2, 1.3, -1.4, 0.4, 0.6, -0.7, 0.05, -0.3, 1.4, -1.5, 0.2, 0.7, -0.8, 0.06],
            ]
        ],
        dtype=torch.float32,
    )


def _raw_action_chunk() -> torch.Tensor:
    return torch.tensor(
        [
            [
                [0.3, 1.4, -1.6, 0.5, 0.7, -0.8, 0.07, -0.4, 1.5, -1.6, 0.3, 0.8, -0.9, 0.08],
                [0.4, 1.5, -1.8, 0.6, 0.8, -0.9, 0.08, -0.5, 1.6, -1.7, 0.4, 0.9, -1.0, 0.09],
            ]
        ],
        dtype=torch.float32,
    )


def test_piper_pika_profile_marks_joint6_names_as_grippers():
    mask = build_arm_mask(14, gripper_indices=GRIPPER_INDICES, action_names=ACTION_NAMES)
    assert mask.tolist() == [
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    ]


def test_absolute_mode_leaves_state_and_action_absolute():
    state = _raw_state()[:, 1]
    action = _raw_action_chunk()
    transition = batch_to_transition({OBS_STATE: state.clone(), ACTION: action.clone()})

    step = Pi05JointRepresentationProcessorStep(
        joint_representation="absolute",
        gripper_indices=GRIPPER_INDICES,
        action_names=ACTION_NAMES,
    )
    result = step(transition)

    torch.testing.assert_close(result[TransitionKey.OBSERVATION][OBS_STATE], state)
    torch.testing.assert_close(result[TransitionKey.ACTION], action)


def test_relative_mode_converts_state_arm_deltas_and_keeps_grippers_absolute():
    state = _raw_state()
    transition = batch_to_transition({OBS_STATE: state.clone()})

    step = Pi05JointRepresentationProcessorStep(
        joint_representation="relative",
        gripper_indices=GRIPPER_INDICES,
        action_names=ACTION_NAMES,
    )
    result = step(transition)

    expected = state[:, 1].clone()
    expected[:, :6] = state[:, 1, :6] - state[:, 0, :6]
    expected[:, 7:13] = state[:, 1, 7:13] - state[:, 0, 7:13]
    expected[:, GRIPPER_INDICES] = state[:, 1, GRIPPER_INDICES]
    torch.testing.assert_close(result[TransitionKey.OBSERVATION][OBS_STATE], expected)


def test_relative_mode_zeros_first_frame_arm_state_and_keeps_sample():
    state = _raw_state()[:, 1].clone()
    transition = batch_to_transition(
        {
            OBS_STATE: state,
            "observation.state_is_pad": torch.tensor([[True, False]]),
        }
    )

    step = Pi05JointRepresentationProcessorStep(
        joint_representation="relative",
        gripper_indices=GRIPPER_INDICES,
        action_names=ACTION_NAMES,
    )
    result = step(transition)

    expected = state.clone()
    expected[:, :6] = 0.0
    expected[:, 7:13] = 0.0
    torch.testing.assert_close(result[TransitionKey.OBSERVATION][OBS_STATE], expected)


def test_relative_mode_uses_cached_previous_state_for_online_inference():
    states = _raw_state()
    step = Pi05JointRepresentationProcessorStep(
        joint_representation="relative",
        gripper_indices=GRIPPER_INDICES,
        action_names=ACTION_NAMES,
    )

    first = step(batch_to_transition({OBS_STATE: states[:, 0]}))
    second = step(batch_to_transition({OBS_STATE: states[:, 1]}))

    expected_first = states[:, 0].clone()
    expected_first[:, :6] = 0.0
    expected_first[:, 7:13] = 0.0
    torch.testing.assert_close(first[TransitionKey.OBSERVATION][OBS_STATE], expected_first)

    expected_second = states[:, 1].clone()
    expected_second[:, :6] = states[:, 1, :6] - states[:, 0, :6]
    expected_second[:, 7:13] = states[:, 1, 7:13] - states[:, 0, 7:13]
    torch.testing.assert_close(second[TransitionKey.OBSERVATION][OBS_STATE], expected_second)


def test_relative_mode_converts_action_arm_deltas_and_keeps_grippers_absolute():
    state = _raw_state()
    current_state = state[:, 1]
    action = _raw_action_chunk()
    transition = batch_to_transition({OBS_STATE: state.clone(), ACTION: action.clone()})

    step = Pi05JointRepresentationProcessorStep(
        joint_representation="relative",
        gripper_indices=GRIPPER_INDICES,
        action_names=ACTION_NAMES,
    )
    result = step(transition)

    expected = action.clone()
    expected[..., :6] = action[..., :6] - current_state[:, None, :6]
    expected[..., 7:13] = action[..., 7:13] - current_state[:, None, 7:13]
    expected[..., GRIPPER_INDICES] = action[..., GRIPPER_INDICES]
    torch.testing.assert_close(result[TransitionKey.ACTION], expected)


def test_hardware_limit_stats_use_relative_arm_range_and_absolute_gripper_range():
    stats = make_pi05_joint_stats("relative", gripper_indices=GRIPPER_INDICES)

    torch.testing.assert_close(stats[OBS_STATE]["min"][0], torch.tensor(-5.236))
    torch.testing.assert_close(stats[OBS_STATE]["max"][0], torch.tensor(5.236))
    torch.testing.assert_close(stats[ACTION]["min"][6], torch.tensor(0.0))
    torch.testing.assert_close(stats[ACTION]["max"][6], torch.tensor(0.10))
    torch.testing.assert_close(stats[ACTION]["min"][13], torch.tensor(0.0))
    torch.testing.assert_close(stats[ACTION]["max"][13], torch.tensor(0.10))


def test_hardware_limit_stats_normalize_absolute_limits_to_unit_range():
    stats = make_pi05_joint_stats("absolute", gripper_indices=GRIPPER_INDICES)
    features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(14,)),
    }
    normalizer = NormalizerProcessorStep(
        features=features,
        norm_map={FeatureType.STATE: NormalizationMode.MIN_MAX, FeatureType.ACTION: NormalizationMode.MIN_MAX},
        stats=stats,
    )

    transition = batch_to_transition(
        {OBS_STATE: stats[OBS_STATE]["min"].unsqueeze(0), ACTION: stats[ACTION]["max"].view(1, 1, 14)}
    )
    result = normalizer(transition)

    torch.testing.assert_close(result[TransitionKey.OBSERVATION][OBS_STATE], torch.full((1, 14), -1.0))
    torch.testing.assert_close(result[TransitionKey.ACTION], torch.full((1, 1, 14), 1.0))


def test_relative_postprocess_reconstructs_absolute_arm_actions_and_clips():
    state = _raw_state()
    relative_actions = torch.zeros(1, 2, 14)
    relative_actions[..., 0] = 99.0
    relative_actions[..., 6] = 0.12
    relative_actions[..., 13] = -0.02
    transition = batch_to_transition({OBS_STATE: state.clone()})

    pre_step = Pi05JointRepresentationProcessorStep(
        joint_representation="relative",
        gripper_indices=GRIPPER_INDICES,
        action_names=ACTION_NAMES,
    )
    pre_step(transition)
    post_step = Pi05AbsoluteActionProcessorStep(
        joint_representation="relative",
        gripper_indices=GRIPPER_INDICES,
        relative_step=pre_step,
    )
    result = post_step(batch_to_transition({ACTION: relative_actions}))

    absolute = result[TransitionKey.ACTION]
    torch.testing.assert_close(absolute[..., 0], torch.full((1, 2), 2.618))
    torch.testing.assert_close(absolute[..., 6], torch.full((1, 2), 0.10))
    torch.testing.assert_close(absolute[..., 13], torch.full((1, 2), 0.0))


def test_pi05_relative_config_requests_previous_current_state_and_future_t1_to_t10_actions():
    cfg = PI05Config(joint_representation="relative", chunk_size=10, n_action_steps=10)
    assert cfg.observation_delta_indices == [-1, 0]
    assert cfg.action_delta_indices == list(range(10))


@pytest.mark.parametrize("condition_on_state", [True, False])
def test_pi05_raw_relative_config_contract_is_identical_for_both_conditioning_modes(
    tmp_path: Path, condition_on_state: bool
):
    cfg = _pi05_relative_config(tmp_path, condition_on_state=condition_on_state)

    cfg.validate_features()

    assert cfg.observation_delta_indices == [-1, 0]
    assert cfg.action_delta_indices == list(range(50))
    assert OBS_STATE in cfg.input_features
    assert cfg.relative_joint_stats.actions[50].q01.shape == (7,)
    assert json.dumps(asdict(cfg))


@pytest.mark.parametrize(("state_shape", "action_shape"), [((14,), (7,)), ((7,), (14,))])
def test_pi05_relative_config_rejects_mismatched_feature_dimensions(
    tmp_path: Path, state_shape: tuple[int, ...], action_shape: tuple[int, ...]
):
    state_path, action_path = _save_relative_stats(tmp_path)

    cfg = PI05Config(
        joint_representation="relative",
        joint_gripper_indices=[6],
        chunk_size=50,
        n_action_steps=50,
        relative_state_stats_path=str(state_path),
        relative_action_stats_path=str(action_path),
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=state_shape)},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=action_shape)},
        action_feature_names=JOINT_NAMES_7D,
    )

    with pytest.raises(ValueError, match="matching"):
        cfg.validate_features()


def test_pi05_relative_config_loads_independent_17d_state_and_14d_action_stats(tmp_path: Path):
    state_names = [f"left_joint_{index}" for index in range(6)] + ["left_gripper"] + [
        f"right_joint_{index}" for index in range(6)
    ] + ["right_gripper", "bread_x", "bread_y", "bread_z"]
    action_names = state_names[:14]
    states = [torch.arange(51, dtype=torch.float32).reshape(3, 17)]
    actions = [torch.cat((torch.full((1, 14), -1.0), states[0][1:, :14]), dim=0)]
    bundle = compute_relative_joint_stats_from_episodes(
        states,
        action_episodes=actions,
        state_gripper_indices=[6, 13],
        action_gripper_indices=[6, 13],
        action_state_indices=list(range(14)),
        horizons=[16, 50],
        state_feature_names=state_names,
        action_feature_names=action_names,
        source_manifest_sha256="b" * 64,
    )
    save_relative_joint_stats(bundle, tmp_path)

    cfg = PI05Config(
        joint_representation="relative",
        joint_gripper_indices=[6, 13],
        state_gripper_indices=[6, 13],
        state_feature_names=state_names,
        action_feature_names=action_names,
        chunk_size=50,
        n_action_steps=50,
        relative_state_stats_path=str(tmp_path / "relative_state_q01_q99.json"),
        relative_action_stats_path=str(tmp_path / "relative_action_chunk50_q01_q99.json"),
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(17,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(14,))},
    )
    cfg.validate_features()
    assert cfg.relative_joint_stats.state.q01.shape == (17,)
    assert cfg.relative_joint_stats.actions[50].q99.shape == (14,)


def test_relative_stats_paths_rejects_malformed_source_manifest_sha256(tmp_path: Path):
    state_path, action_path = _save_relative_stats(tmp_path)
    manifest_path = tmp_path / "relative_stats_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_manifest_sha256"] = ""
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA256"):
        load_relative_joint_stats_paths(
            state_path,
            action_path,
            expected_horizon=50,
            expected_feature_names=JOINT_NAMES_7D,
            expected_gripper_indices=[6],
        )


def test_pi05_from_pretrained_defers_relative_stats_io_until_feature_validation(tmp_path: Path):
    stats_dir = tmp_path / "original_stats"
    checkpoint_dir = tmp_path / "checkpoint"
    cfg = _pi05_relative_config(stats_dir)
    cfg.save_pretrained(checkpoint_dir)
    shutil.rmtree(stats_dir)

    loaded = PreTrainedConfig.from_pretrained(checkpoint_dir)
    assert isinstance(loaded, PI05Config)

    with pytest.raises(ValueError, match="does not exist"):
        loaded.validate_features()

    state_path, action_path = _save_relative_stats(tmp_path / "local_stats")
    loaded.relative_state_stats_path = str(state_path)
    loaded.relative_action_stats_path = str(action_path)
    loaded.validate_features()
    assert loaded.relative_joint_stats.actions[50].q01.shape == (7,)


def test_pi05_relative_stats_cache_clears_on_failure_and_inactive_modes(tmp_path: Path):
    cfg = _pi05_relative_config(tmp_path / "stats")
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

    cfg.joint_representation = "relative"
    cfg.validate_features()
    assert cfg.relative_joint_stats is not None
    cfg.precomputed_relative_chunk = True
    cfg.validate_features()
    assert cfg.relative_joint_stats is None


def test_pi05_relative_7d_config_rejects_missing_or_incompatible_stats(tmp_path: Path):
    state_path, action_path = _save_relative_stats(tmp_path)
    common = {
        "joint_representation": "relative",
        "joint_gripper_indices": [6],
        "chunk_size": 50,
        "n_action_steps": 50,
        "input_features": {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,))},
        "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        "action_feature_names": JOINT_NAMES_7D,
    }

    cfg = PI05Config(**common)
    with pytest.raises(ValueError, match="stats path"):
        cfg.validate_features()
    cfg = PI05Config(
        **common,
        relative_state_stats_path=str(tmp_path / "missing.json"),
        relative_action_stats_path=str(action_path),
    )
    with pytest.raises(ValueError, match="does not exist"):
        cfg.validate_features()
    cfg = PI05Config(
        **(common | {"chunk_size": 16, "n_action_steps": 16}),
        relative_state_stats_path=str(state_path),
        relative_action_stats_path=str(action_path),
    )
    with pytest.raises(ValueError, match="horizon"):
        cfg.validate_features()
    cfg = PI05Config(
        **(common | {"joint_gripper_indices": [5]}),
        relative_state_stats_path=str(state_path),
        relative_action_stats_path=str(action_path),
    )
    with pytest.raises(ValueError, match="gripper"):
        cfg.validate_features()
    cfg = PI05Config(
        **(common | {"action_feature_names": [*JOINT_NAMES_7D[:-1], "claw"]}),
        relative_state_stats_path=str(state_path),
        relative_action_stats_path=str(action_path),
    )
    with pytest.raises(ValueError, match="feature names"):
        cfg.validate_features()


def test_pi05_defaults_freeze_only_language_for_planned_experiments():
    cfg = PI05Config()

    assert cfg.freeze_language_model is True
    assert cfg.freeze_vision_encoder is False
    assert cfg.train_expert_only is False
    assert cfg.condition_on_state is True
    assert cfg.clip_quantiles is True


def test_pi05_relative_dataset_deltas_apply_only_to_state_not_images():
    cfg = PI05Config(joint_representation="relative", chunk_size=10, n_action_steps=10)
    ds_meta = SimpleNamespace(
        fps=50,
        features={
            OBS_STATE: {},
            "observation.images.left_wrist": {},
            "observation.images.right_wrist": {},
            ACTION: {},
        },
    )

    delta_timestamps = resolve_delta_timestamps(cfg, ds_meta)

    assert delta_timestamps[OBS_STATE] == [-0.02, 0.0]
    assert delta_timestamps[ACTION] == [i / 50 for i in range(10)]
    assert "observation.images.left_wrist" not in delta_timestamps
    assert "observation.images.right_wrist" not in delta_timestamps


def test_pi05_precomputed_relative_config_uses_dataset_state_and_action_chunks_directly():
    cfg = PI05Config(
        joint_representation="relative",
        precomputed_relative_chunk=True,
        chunk_size=20,
        n_action_steps=20,
    )
    ds_meta = SimpleNamespace(
        fps=50,
        features={
            OBS_STATE: {},
            "observation.images.left_wrist": {},
            "observation.images.right_wrist": {},
            ACTION: {},
        },
    )

    assert cfg.observation_delta_indices is None
    assert cfg.action_delta_indices is None
    assert resolve_delta_timestamps(cfg, ds_meta) is None


def test_precomputed_relative_training_batch_is_not_converted_twice():
    relative_state = torch.tensor(
        [[0.1, 0.2, -0.3, 0.4, -0.5, 0.6, 0.03, -0.1, 0.2, -0.3, 0.4, -0.5, 0.6, 0.04]],
        dtype=torch.float32,
    )
    relative_action = torch.full((1, 20, 14), 0.25, dtype=torch.float32)
    relative_action[..., GRIPPER_INDICES] = torch.tensor([0.05, 0.06])
    transition = batch_to_transition({OBS_STATE: relative_state.clone(), ACTION: relative_action.clone()})

    step = Pi05JointRepresentationProcessorStep(
        joint_representation="relative",
        gripper_indices=GRIPPER_INDICES,
        action_names=ACTION_NAMES,
        precomputed_relative_chunk=True,
    )
    result = step(transition)

    torch.testing.assert_close(result[TransitionKey.OBSERVATION][OBS_STATE], relative_state)
    torch.testing.assert_close(result[TransitionKey.ACTION], relative_action)
    assert step.get_cached_absolute_state() is None


def test_precomputed_relative_mode_converts_online_absolute_state():
    states = _raw_state()
    step = Pi05JointRepresentationProcessorStep(
        joint_representation="relative",
        gripper_indices=GRIPPER_INDICES,
        action_names=ACTION_NAMES,
        precomputed_relative_chunk=True,
    )

    first = step(batch_to_transition({OBS_STATE: states[:, 0]}))
    second = step(batch_to_transition({OBS_STATE: states[:, 1]}))

    expected_first = states[:, 0].clone()
    expected_first[:, :6] = 0.0
    expected_first[:, 7:13] = 0.0
    torch.testing.assert_close(first[TransitionKey.OBSERVATION][OBS_STATE], expected_first)

    expected_second = states[:, 1].clone()
    expected_second[:, :6] = states[:, 1, :6] - states[:, 0, :6]
    expected_second[:, 7:13] = states[:, 1, 7:13] - states[:, 0, 7:13]
    torch.testing.assert_close(second[TransitionKey.OBSERVATION][OBS_STATE], expected_second)


def test_pi05_hardware_stats_accept_precomputed_action_chunks():
    config = SimpleNamespace(
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(14,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(20, 14))},
    )

    assert can_use_pi05_joint_stats(config)


def test_freeze_language_model_keeps_vision_encoder_and_projector_trainable():
    class FakePaliGemma(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.language_model = torch.nn.Linear(2, 2)
            self.model.vision_tower = torch.nn.Linear(2, 2)
            self.model.multi_modal_projector = torch.nn.Linear(2, 2)

    model = object.__new__(PaliGemmaWithExpertModel)
    torch.nn.Module.__init__(model)
    model.paligemma = FakePaliGemma()
    model.freeze_vision_encoder = False
    model.freeze_language_model = True
    model.train_expert_only = False

    PaliGemmaWithExpertModel._set_requires_grad(model)

    assert not any(p.requires_grad for p in model.paligemma.model.language_model.parameters())
    assert all(p.requires_grad for p in model.paligemma.model.vision_tower.parameters())
    assert all(p.requires_grad for p in model.paligemma.model.multi_modal_projector.parameters())


def test_pi05_gradient_checkpointing_matches_official_base_policy():

    model = object.__new__(PI05Pytorch)
    torch.nn.Module.__init__(model)
    model.gradient_checkpointing_enabled = False
    model.paligemma_with_expert = SimpleNamespace(
        paligemma=SimpleNamespace(
            model=SimpleNamespace(
                language_model=SimpleNamespace(gradient_checkpointing=False),
                vision_tower=SimpleNamespace(gradient_checkpointing=False),
            )
        ),
        gemma_expert=SimpleNamespace(model=SimpleNamespace(gradient_checkpointing=False)),
    )

    PI05Pytorch.gradient_checkpointing_enable(model)

    assert model.gradient_checkpointing_enabled is True
    assert model.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing is True
    assert model.paligemma_with_expert.gemma_expert.model.gradient_checkpointing is True
    assert model.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing is True


def test_pi05_image_embedding_matches_official_float32_vision_input(
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeImageModel:
        def get_image_features(self, image):
            assert image.dtype is torch.float32
            return SimpleNamespace(pooler_output=torch.ones((1, 2, 3), dtype=torch.float32))

    def unexpected_autocast(*_args, **_kwargs):
        raise AssertionError("official PI05 embed_image does not open an explicit autocast context")

    monkeypatch.setattr(torch, "autocast", unexpected_autocast)
    model = SimpleNamespace(paligemma=SimpleNamespace(model=FakeImageModel()))

    result = PaliGemmaWithExpertModel.embed_image(model, torch.ones((1, 3, 2, 2), dtype=torch.bfloat16))

    assert result.dtype is torch.bfloat16


def test_pi05_config_reorders_real_image_features_before_training():
    top = "observation.images.top"
    left = "observation.images.gripper_left"
    right = "observation.images.gripper_right"
    cfg = PI05Config(
        input_features={
            top: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 405, 720)),
            right: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
            left: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(14,))},
        image_feature_order=[top, left, right],
    )

    cfg.validate_features()

    assert list(cfg.image_features) == [top, left, right]


@pytest.mark.parametrize("condition_on_state", [True, False])
def test_pi05_both_conditioning_modes_freeze_only_language_model(condition_on_state: bool):
    cfg = PI05Config(condition_on_state=condition_on_state)

    assert cfg.freeze_language_model is True
    assert cfg.freeze_vision_encoder is False
    assert cfg.train_expert_only is False


def test_pi05_legacy_joint_registry_wrapper_delegates_7d_relative_conversion():
    step_cls = ProcessorStepRegistry.get("pi05_joint_representation_processor")
    assert step_cls is Pi05JointRepresentationProcessorStep
    step = step_cls(
        joint_representation="relative",
        gripper_indices=[6],
        action_names=JOINT_NAMES_7D,
    )
    state = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.03], [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 0.04]]]
    )
    action = torch.tensor([[[3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 0.05]]])

    result = step(batch_to_transition({OBS_STATE: state, ACTION: action}))
    assert isinstance(step._shared_step, RelativeJointProcessorStep)  # noqa: SLF001
    shared = RelativeJointProcessorStep(condition_on_state=True)
    expected = shared(batch_to_transition({OBS_STATE: state, ACTION: action}))

    torch.testing.assert_close(
        result[TransitionKey.OBSERVATION][OBS_STATE], expected[TransitionKey.OBSERVATION][OBS_STATE]
    )
    torch.testing.assert_close(result[TransitionKey.ACTION], expected[TransitionKey.ACTION])


def _assert_pi05_wrapper_uses_each_online_request_as_action_anchor(
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
) -> None:
    q_t1 = torch.tensor([[0.1, 0.2, -0.3, 0.1, 0.2, -0.2, 0.03]])
    q_t2 = torch.tensor([[0.2, 0.4, -0.5, 0.3, 0.4, -0.4, 0.04]])
    relative_chunk = torch.zeros(1, 2, 7)
    relative_chunk[..., 6] = 0.05

    preprocessor(batch_to_transition({OBS_STATE: q_t1}))
    first = postprocessor(batch_to_transition({ACTION: relative_chunk}))
    preprocessor(batch_to_transition({OBS_STATE: q_t2}))
    second = postprocessor(batch_to_transition({ACTION: relative_chunk}))

    expected_first = q_t1[:, None].expand(-1, 2, -1).clone()
    expected_second = q_t2[:, None].expand(-1, 2, -1).clone()
    expected_first[..., 6] = 0.05
    expected_second[..., 6] = 0.05
    torch.testing.assert_close(first[TransitionKey.ACTION], expected_first)
    torch.testing.assert_close(second[TransitionKey.ACTION], expected_second)


def _pi05_wrapper_pipelines() -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    relative_step = Pi05JointRepresentationProcessorStep(
        joint_representation="relative",
        gripper_indices=[6],
        action_names=JOINT_NAMES_7D,
        execution_horizon=2,
    )
    absolute_step = Pi05AbsoluteActionProcessorStep(
        joint_representation="relative",
        gripper_indices=[6],
        relative_step=relative_step,
    )
    return (
        PolicyProcessorPipeline(
            steps=[relative_step], to_transition=identity_transition, to_output=identity_transition
        ),
        PolicyProcessorPipeline(
            steps=[absolute_step], to_transition=identity_transition, to_output=identity_transition
        ),
    )


def test_pi05_wrapper_consumes_7d_online_anchor_between_action_chunks():
    _assert_pi05_wrapper_uses_each_online_request_as_action_anchor(*_pi05_wrapper_pipelines())


def test_pi05_wrapper_consumes_7d_online_anchor_after_save_load_and_rebind(tmp_path: Path):
    preprocessor, postprocessor = _pi05_wrapper_pipelines()
    preprocessor.save_pretrained(tmp_path, config_filename="preprocessor.json")
    postprocessor.save_pretrained(tmp_path, config_filename="postprocessor.json")
    loaded_preprocessor = PolicyProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename="preprocessor.json",
        to_transition=identity_transition,
        to_output=identity_transition,
    )
    loaded_postprocessor = PolicyProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename="postprocessor.json",
        to_transition=identity_transition,
        to_output=identity_transition,
    )

    policy_factory._reconnect_relative_absolute_steps(loaded_preprocessor, loaded_postprocessor)

    _assert_pi05_wrapper_uses_each_online_request_as_action_anchor(
        loaded_preprocessor, loaded_postprocessor
    )


def _pi05_14d_online_states() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(
            [[0.1, 0.5, -0.5, 0.1, 0.1, -0.1, 0.03, -0.1, 0.6, -0.6, -0.1, 0.2, -0.2, 0.04]]
        ),
        torch.tensor(
            [[0.2, 0.7, -0.7, 0.2, 0.3, -0.2, 0.04, -0.2, 0.8, -0.8, 0.1, 0.4, -0.3, 0.05]]
        ),
        torch.tensor(
            [[0.3, 0.9, -0.9, 0.3, 0.5, -0.3, 0.05, -0.3, 1.0, -1.0, 0.2, 0.6, -0.4, 0.06]]
        ),
    )


def _pi05_14d_wrapper_pipelines(
    *, precomputed_relative_chunk: bool
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    relative_step = Pi05JointRepresentationProcessorStep(
        joint_representation="relative",
        gripper_indices=GRIPPER_INDICES,
        action_names=ACTION_NAMES,
        precomputed_relative_chunk=precomputed_relative_chunk,
        execution_horizon=2,
    )
    absolute_step = Pi05AbsoluteActionProcessorStep(
        joint_representation="relative",
        gripper_indices=GRIPPER_INDICES,
        relative_step=relative_step,
    )
    return (
        PolicyProcessorPipeline(
            steps=[relative_step], to_transition=identity_transition, to_output=identity_transition
        ),
        PolicyProcessorPipeline(
            steps=[absolute_step], to_transition=identity_transition, to_output=identity_transition
        ),
    )


def _assert_pi05_14d_select_action_anchor_lifecycle(
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
) -> None:
    q0, q1, q2 = _pi05_14d_online_states()
    relative_action = torch.zeros(1, 14)
    relative_action[:, 6] = 0.05
    relative_action[:, 13] = 0.06

    first_state = preprocessor(batch_to_transition({OBS_STATE: q0}))
    first_action = postprocessor(batch_to_transition({ACTION: relative_action}))
    second_state = preprocessor(batch_to_transition({OBS_STATE: q1}))
    second_action = postprocessor(batch_to_transition({ACTION: relative_action}))
    third_state = preprocessor(batch_to_transition({OBS_STATE: q2}))
    third_action = postprocessor(batch_to_transition({ACTION: relative_action}))

    arm_mask = build_arm_mask(14, gripper_indices=GRIPPER_INDICES)
    expected_first_state = q0.clone()
    expected_first_state[:, arm_mask] = 0
    expected_second_state = q1.clone()
    expected_second_state[:, arm_mask] = q1[:, arm_mask] - q0[:, arm_mask]
    expected_third_state = q2.clone()
    expected_third_state[:, arm_mask] = q2[:, arm_mask] - q1[:, arm_mask]
    torch.testing.assert_close(first_state[TransitionKey.OBSERVATION][OBS_STATE], expected_first_state)
    torch.testing.assert_close(second_state[TransitionKey.OBSERVATION][OBS_STATE], expected_second_state)
    torch.testing.assert_close(third_state[TransitionKey.OBSERVATION][OBS_STATE], expected_third_state)

    expected_q0_action = q0.clone()
    expected_q0_action[:, GRIPPER_INDICES] = relative_action[:, GRIPPER_INDICES]
    expected_q2_action = q2.clone()
    expected_q2_action[:, GRIPPER_INDICES] = relative_action[:, GRIPPER_INDICES]
    torch.testing.assert_close(first_action[TransitionKey.ACTION], expected_q0_action)
    torch.testing.assert_close(second_action[TransitionKey.ACTION], expected_q0_action)
    torch.testing.assert_close(third_action[TransitionKey.ACTION], expected_q2_action)


@pytest.mark.parametrize("precomputed_relative_chunk", [False, True])
def test_pi05_14d_wrapper_keeps_chunk_anchor_while_previous_observation_advances(
    precomputed_relative_chunk: bool,
):
    _assert_pi05_14d_select_action_anchor_lifecycle(
        *_pi05_14d_wrapper_pipelines(precomputed_relative_chunk=precomputed_relative_chunk)
    )


@pytest.mark.parametrize("precomputed_relative_chunk", [False, True])
def test_pi05_14d_wrapper_anchor_lifecycle_survives_save_load_and_rebind(
    tmp_path: Path,
    precomputed_relative_chunk: bool,
):
    preprocessor, postprocessor = _pi05_14d_wrapper_pipelines(
        precomputed_relative_chunk=precomputed_relative_chunk
    )
    q0, _, _ = _pi05_14d_online_states()
    preprocessor(batch_to_transition({OBS_STATE: q0}))
    preprocessor.save_pretrained(tmp_path, config_filename="preprocessor.json")
    postprocessor.save_pretrained(tmp_path, config_filename="postprocessor.json")
    serialized = (tmp_path / "preprocessor.json").read_text(encoding="utf-8")
    assert "_last_observation_state" not in serialized
    assert "_action_anchor_state" not in serialized
    assert "_action_steps_processed" not in serialized

    loaded_preprocessor = PolicyProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename="preprocessor.json",
        to_transition=identity_transition,
        to_output=identity_transition,
    )
    loaded_postprocessor = PolicyProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename="postprocessor.json",
        to_transition=identity_transition,
        to_output=identity_transition,
    )
    policy_factory._reconnect_relative_absolute_steps(loaded_preprocessor, loaded_postprocessor)

    _assert_pi05_14d_select_action_anchor_lifecycle(loaded_preprocessor, loaded_postprocessor)


@pytest.mark.parametrize("precomputed_relative_chunk", [False, True])
def test_pi05_14d_offline_training_does_not_read_or_write_online_caches(
    precomputed_relative_chunk: bool,
):
    q0, q1, q2 = _pi05_14d_online_states()
    step = Pi05JointRepresentationProcessorStep(
        joint_representation="relative",
        gripper_indices=GRIPPER_INDICES,
        action_names=ACTION_NAMES,
        precomputed_relative_chunk=precomputed_relative_chunk,
        execution_horizon=2,
    )
    offline_action = q1[:, None].expand(-1, 2, -1).clone()

    step(batch_to_transition({OBS_STATE: torch.stack([q0, q1], dim=1), ACTION: offline_action}))

    assert step.get_cached_absolute_state() is None
    online = step(batch_to_transition({OBS_STATE: q2}))
    expected = q2.clone()
    expected[:, build_arm_mask(14, gripper_indices=GRIPPER_INDICES)] = 0
    torch.testing.assert_close(online[TransitionKey.OBSERVATION][OBS_STATE], expected)

    step.reset()
    step(batch_to_transition({OBS_STATE: q0}))
    anchor_before_offline = step.get_cached_absolute_state().clone()
    step(batch_to_transition({OBS_STATE: torch.stack([q1, q2], dim=1), ACTION: offline_action}))
    torch.testing.assert_close(step.get_cached_absolute_state(), anchor_before_offline)

    after_offline = step(batch_to_transition({OBS_STATE: q1}))
    expected_after_offline = q1.clone()
    arm_mask = build_arm_mask(14, gripper_indices=GRIPPER_INDICES)
    expected_after_offline[:, arm_mask] = q1[:, arm_mask] - q0[:, arm_mask]
    torch.testing.assert_close(
        after_offline[TransitionKey.OBSERVATION][OBS_STATE], expected_after_offline
    )

    step.reset()
    assert step.get_cached_absolute_state() is None
    after_reset = step(batch_to_transition({OBS_STATE: q2}))
    expected_after_reset = q2.clone()
    expected_after_reset[:, arm_mask] = 0
    torch.testing.assert_close(after_reset[TransitionKey.OBSERVATION][OBS_STATE], expected_after_reset)


def test_merge_pi05_joint_stats_copies_real_read_only_quantile_storage():
    bundle = RelativeJointStatsBundle(
        state=QuantileStats(
            q01=np.arange(7, dtype=np.float64),
            q99=np.arange(7, dtype=np.float64) + 10,
            count=5,
        ),
        actions={
            2: QuantileStats(
                q01=np.arange(7, dtype=np.float64) + 20,
                q99=np.arange(7, dtype=np.float64) + 30,
                count=10,
            )
        },
        feature_names=JOINT_NAMES_7D,
        gripper_indices=[6],
        source_manifest_sha256="a" * 64,
    )
    config = SimpleNamespace(relative_joint_stats=bundle, chunk_size=2)
    original_state_q01 = bundle.state.q01.copy()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        merged = merge_pi05_joint_stats(config, dataset_stats=None)

    assert not any("not writable" in str(item.message) for item in caught)
    merged[OBS_STATE]["q01"][0] = -99
    np.testing.assert_array_equal(bundle.state.q01, original_state_q01)


def test_pi05_forward_masks_padding_and_reports_exact_aggregation_metrics():
    raw_losses = torch.arange(42, dtype=torch.float32).reshape(2, 3, 7).requires_grad_()
    noise = torch.full((2, 3, 7), 0.25)
    time = torch.tensor([0.2, 0.8])

    class FakeFlowModel:
        def sample_noise(self, *_args, **_kwargs):
            raise AssertionError("explicit noise must bypass random sampling")

        def sample_time(self, *_args, **_kwargs):
            raise AssertionError("explicit time must bypass random sampling")

        def forward(self, _images, _img_masks, _tokens, _masks, _actions, actual_noise, actual_time):
            assert actual_noise is noise
            assert actual_time is time
            return raw_losses

    policy = SimpleNamespace(
        config=SimpleNamespace(
            max_action_dim=7,
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
            joint_gripper_indices=[6],
        ),
        model=FakeFlowModel(),
        _preprocess_images=lambda _batch: ([], []),
        prepare_action=lambda batch: batch[ACTION],
    )
    action_is_pad = torch.tensor([[False, True, False], [True, True, True]])
    batch = {
        ACTION: torch.zeros(2, 3, 7),
        "action_is_pad": action_is_pad,
        OBS_LANGUAGE_TOKENS: torch.zeros(2, 1, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 1, dtype=torch.bool),
    }

    loss, metrics = PI05Policy.forward(policy, batch, noise=noise, time=time)

    valid_losses = torch.cat([raw_losses[0, 0], raw_losses[0, 2]])
    torch.testing.assert_close(loss, valid_losses.mean())
    torch.testing.assert_close(metrics["loss_sum_per_sample"], torch.tensor([140.0, 0.0]))
    torch.testing.assert_close(metrics["loss_count_per_sample"], torch.tensor([14, 0]))
    torch.testing.assert_close(metrics["gripper_loss_sum_per_sample"], torch.tensor([26.0, 0.0]))
    torch.testing.assert_close(metrics["gripper_loss_count_per_sample"], torch.tensor([2, 0]))
    torch.testing.assert_close(metrics["gripper_loss_per_sample"], torch.tensor([13.0, 0.0]))
    assert metrics["gripper_loss"] == pytest.approx(13.0)

    loss.backward()
    expected_grad = torch.zeros_like(raw_losses)
    expected_grad[0, 0] = 1 / 14
    expected_grad[0, 2] = 1 / 14
    torch.testing.assert_close(raw_losses.grad, expected_grad)


def test_pi05_forward_aggregates_all_configured_gripper_dimensions():
    raw_losses = torch.arange(84, dtype=torch.float32).reshape(2, 3, 14).requires_grad_()

    class FakeFlowModel:
        def forward(self, *_args):
            return raw_losses

    policy = SimpleNamespace(
        config=SimpleNamespace(
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(14,))},
            joint_gripper_indices=[6, 13],
        ),
        model=FakeFlowModel(),
        _preprocess_images=lambda _batch: ([], []),
        prepare_action=lambda batch: batch[ACTION],
    )
    batch = {
        ACTION: torch.zeros(2, 3, 14),
        "action_is_pad": torch.tensor([[False, True, False], [False, False, True]]),
        OBS_LANGUAGE_TOKENS: torch.zeros(2, 1, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 1, dtype=torch.bool),
    }

    _, metrics = PI05Policy.forward(
        policy,
        batch,
        noise=torch.zeros_like(batch[ACTION]),
        time=torch.tensor([0.2, 0.8]),
    )

    torch.testing.assert_close(metrics["gripper_loss_sum_per_sample"], torch.tensor([94.0, 234.0]))
    torch.testing.assert_close(metrics["gripper_loss_count_per_sample"], torch.tensor([4, 4]))
    torch.testing.assert_close(metrics["gripper_loss_per_sample"], torch.tensor([23.5, 58.5]))
    assert metrics["gripper_loss"] == pytest.approx(41.0)


def test_pi05_forward_excludes_padding_and_rtc_prefix_from_all_metrics(monkeypatch):
    raw_losses = torch.tensor(
        [[[100.0, 100.0], [1.0, 3.0], [5.0, 7.0], [9.0, 11.0]]],
        requires_grad=True,
    )
    prefix_mask = torch.tensor([[True, False, False, False]])

    class FakeFlowModel:
        def forward(self, *_args):
            assert _args[-1] is prefix_mask
            return raw_losses

    monkeypatch.setattr(
        "lerobot.policies.pi05.modeling_pi05._sample_training_rtc_prefix_mask",
        lambda *_args: prefix_mask,
    )
    policy = SimpleNamespace(
        config=SimpleNamespace(
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
            joint_gripper_indices=[1],
            rtc_training_max_delay=1,
        ),
        model=FakeFlowModel(),
        _preprocess_images=lambda _batch: ([], []),
        prepare_action=lambda batch: batch[ACTION],
    )
    batch = {
        ACTION: torch.zeros(1, 4, 2),
        "action_is_pad": torch.tensor([[False, False, True, False]]),
        OBS_LANGUAGE_TOKENS: torch.zeros(1, 1, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 1, dtype=torch.bool),
    }

    loss, metrics = PI05Policy.forward(
        policy,
        batch,
        noise=torch.zeros_like(batch[ACTION]),
        time=torch.tensor([0.5]),
    )

    assert loss.item() == pytest.approx(6.0)
    assert metrics["gripper_loss"] == pytest.approx(7.0)
    torch.testing.assert_close(metrics["loss_count_per_sample"], torch.tensor([4]))
    torch.testing.assert_close(metrics["gripper_loss_count_per_sample"], torch.tensor([2]))


def test_pi05_forward_rejects_any_out_of_range_gripper_index():
    raw_losses = torch.zeros(1, 2, 14, requires_grad=True)

    class FakeFlowModel:
        def forward(self, *_args):
            return raw_losses

    policy = SimpleNamespace(
        config=SimpleNamespace(
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(14,))},
            joint_gripper_indices=[6, 14],
        ),
        model=FakeFlowModel(),
        _preprocess_images=lambda _batch: ([], []),
        prepare_action=lambda batch: batch[ACTION],
    )
    batch = {
        ACTION: torch.zeros(1, 2, 14),
        OBS_LANGUAGE_TOKENS: torch.zeros(1, 1, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 1, dtype=torch.bool),
    }

    with pytest.raises(ValueError, match="gripper indices"):
        PI05Policy.forward(
            policy,
            batch,
            noise=torch.zeros_like(batch[ACTION]),
            time=torch.tensor([0.5]),
        )


def test_pi05_action_select_types_separate_explicit_noise_from_rtc_kwargs():
    assert hasattr(pi05_modeling, "RTCActionSelectKwargs")
    assert "noise" not in pi05_modeling.RTCActionSelectKwargs.__annotations__
    assert "noise" in pi05_modeling.ActionSelectKwargs.__annotations__
    assert "RTCActionSelectKwargs" in str(
        pi05_modeling.PI05Pytorch.sample_actions.__annotations__["kwargs"]
    )


def test_pi05_predict_action_chunk_forwards_explicit_noise():
    noise = torch.randn(1, 2, 7)

    class FakeFlowModel:
        def sample_actions(self, _images, _img_masks, _tokens, _masks, **kwargs):
            assert kwargs["noise"] is noise
            return torch.zeros(1, 2, 7)

    policy = SimpleNamespace(
        config=SimpleNamespace(
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
        ),
        model=FakeFlowModel(),
        eval=lambda: None,
        _preprocess_images=lambda _batch: ([], []),
    )
    batch = {
        OBS_LANGUAGE_TOKENS: torch.zeros(1, 1, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 1, dtype=torch.bool),
    }

    actions = PI05Policy.predict_action_chunk(policy, batch, noise=noise)

    assert actions.shape == (1, 2, 7)


def test_pi05_forward_can_return_action_chunk_through_distributed_wrapper_path():
    noise = torch.randn(1, 2, 7)
    expected = torch.zeros(1, 2, 7)

    def predict_action_chunk(batch, *, noise):
        assert ACTION not in batch
        assert noise is not None
        return expected

    policy = SimpleNamespace(
        training=False,
        predict_action_chunk=predict_action_chunk,
    )

    actions = PI05Policy.forward(
        policy,
        {OBS_STATE: torch.zeros(1, 7)},
        noise=noise,
        return_action_chunk=True,
    )

    assert actions is expected


def test_pi05_forward_rejects_action_chunk_branch_in_training_mode():
    policy = SimpleNamespace(training=True)

    with pytest.raises(RuntimeError, match="eval mode"):
        PI05Policy.forward(policy, {OBS_STATE: torch.zeros(1, 7)}, return_action_chunk=True)


def test_pi05_config_accepts_visual_pretrained_path_and_language_freeze():
    cfg = PI05Config(
        freeze_language_model=True,
        freeze_vision_encoder=False,
        train_expert_only=False,
        visual_pretrained_path="/data/wengyikun/models/TeleEmbodied_VISTA/pretrained_model/model.safetensors",
    )
    assert cfg.freeze_language_model is True
    assert cfg.freeze_vision_encoder is False
    assert cfg.visual_pretrained_include_projector is True


def test_pi05_image_preprocessing_uses_current_frame_from_relative_observation_pair():
    image_key = "observation.images.right_fisheye"
    policy = SimpleNamespace(
        config=SimpleNamespace(image_features={image_key: object()}, image_resolution=(224, 224)),
        parameters=lambda: iter((torch.empty(0),)),
    )
    previous = torch.zeros(2, 3, 480, 640)
    current = torch.ones(2, 3, 480, 640)

    images, masks = PI05Policy._preprocess_images(policy, {image_key: torch.stack((previous, current), dim=1)})

    assert len(images) == 1
    assert images[0].shape == (2, 3, 224, 224)
    torch.testing.assert_close(images[0], torch.ones_like(images[0]))
    assert torch.equal(masks[0], torch.ones(2, dtype=torch.bool))
