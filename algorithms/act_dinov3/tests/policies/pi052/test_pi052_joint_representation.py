from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.datasets.absolute_action_stats import (
    compute_absolute_action_stats_from_episodes,
    save_absolute_action_stats,
)
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.relative_joint_stats import (
    compute_relative_joint_stats_from_episodes,
    save_relative_joint_stats,
)
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.joint_representation import (
    Pi05AbsoluteActionProcessorStep,
    Pi05JointRepresentationProcessorStep,
)
from lerobot.policies.pi052.configuration_pi052 import PI052Config
from lerobot.policies.pi052.processor_pi052 import make_pi052_pre_post_processors
from lerobot.policies.pi052.text_processor_pi052 import PI052TextTokenizerStep
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    ActionTokenizerProcessorStep,
    NormalizerProcessorStep,
    RelativeActionsProcessorStep,
    TransitionKey,
    batch_to_transition,
)
from lerobot.processor.render_messages_processor import RenderMessagesStep
from lerobot.utils.constants import ACTION, OBS_STATE

JOINT_NAMES_14D = (
    [f"left_joint_{index}" for index in range(6)]
    + ["left_gripper"]
    + [f"right_joint_{index}" for index in range(6)]
    + ["right_gripper"]
)
GRIPPER_INDICES = [6, 13]


def test_pi052_relative_dataset_deltas_apply_only_to_state_not_images() -> None:
    config = PI052Config(joint_representation="relative", chunk_size=10, n_action_steps=10)
    metadata = SimpleNamespace(
        fps=50,
        features={
            OBS_STATE: {},
            "observation.images.top": {},
            "observation.images.gripper_left": {},
            "observation.images.gripper_right": {},
            ACTION: {},
        },
    )

    delta_timestamps = resolve_delta_timestamps(config, metadata)

    assert delta_timestamps[OBS_STATE] == [-0.02, 0.0]
    assert delta_timestamps[ACTION] == [index / 50 for index in range(10)]
    assert "observation.images.top" not in delta_timestamps
    assert "observation.images.gripper_left" not in delta_timestamps
    assert "observation.images.gripper_right" not in delta_timestamps


def _save_stats(root: Path, horizon: int = 2) -> tuple[Path, Path, Path]:
    episode = torch.arange(112, dtype=torch.float32).reshape(8, 14) / 100
    relative_dir = root / "relative"
    relative_bundle = compute_relative_joint_stats_from_episodes(
        [episode],
        gripper_indices=GRIPPER_INDICES,
        horizons=[horizon],
        feature_names=JOINT_NAMES_14D,
        source_manifest_sha256="a" * 64,
    )
    save_relative_joint_stats(relative_bundle, relative_dir)

    absolute_dir = root / "absolute"
    absolute_bundle = compute_absolute_action_stats_from_episodes(
        [episode],
        horizons=[horizon],
        feature_names=JOINT_NAMES_14D,
        scaled_indices=list(range(14)),
    )
    save_absolute_action_stats(absolute_bundle, absolute_dir)
    return (
        relative_dir / "relative_state_q01_q99.json",
        relative_dir / f"relative_action_chunk{horizon}_q01_q99.json",
        absolute_dir / "absolute_state_q01_q99.json",
    )


def _config(tmp_path: Path, representation: str) -> PI052Config:
    relative_state, relative_action, absolute_state = _save_stats(tmp_path)
    kwargs = {
        "device": "cpu",
        "joint_representation": representation,
        "use_relative_actions": representation == "absolute",
        "relative_exclude_joints": ["gripper"],
        "joint_gripper_indices": GRIPPER_INDICES,
        "chunk_size": 2,
        "n_action_steps": 2,
        "input_features": {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(14,))},
        "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(14,))},
        "action_feature_names": JOINT_NAMES_14D,
        "clip_quantiles": False,
        "apply_action_limits": True,
        "enable_fast_action_loss": False,
    }
    if representation == "relative":
        kwargs.update(
            relative_state_stats_path=str(relative_state),
            relative_action_stats_path=str(relative_action),
        )
    else:
        kwargs.update(
            absolute_state_stats_path=str(absolute_state),
            relative_action_stats_path=str(relative_action),
        )
    config = PI052Config(**kwargs)
    config.validate_features()
    return config


@pytest.mark.parametrize("representation", ["relative", "absolute"])
def test_pi052_factory_dispatches_recipe_pipeline_with_custom_joint_semantics(
    tmp_path: Path, representation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, representation)
    monkeypatch.setattr(
        "lerobot.processor.tokenizer_processor.AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: object()),
    )

    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=None)

    joint_step = next(
        step for step in preprocessor.steps if isinstance(step, Pi05JointRepresentationProcessorStep)
    )
    normalizer = next(step for step in preprocessor.steps if isinstance(step, NormalizerProcessorStep))
    render_step = next(step for step in preprocessor.steps if isinstance(step, RenderMessagesStep))
    text_step = next(step for step in preprocessor.steps if isinstance(step, PI052TextTokenizerStep))
    assert joint_step.joint_representation == representation
    assert preprocessor.steps.index(joint_step) < preprocessor.steps.index(normalizer)
    assert preprocessor.steps.index(normalizer) < preprocessor.steps.index(render_step)
    assert preprocessor.steps.index(render_step) < preprocessor.steps.index(text_step)
    assert normalizer.norm_map[FeatureType.STATE] is NormalizationMode.QUANTILES
    assert normalizer.norm_map[FeatureType.ACTION] is NormalizationMode.QUANTILES
    assert not torch.equal(normalizer.stats[OBS_STATE]["q01"], normalizer.stats[ACTION]["q01"])

    if representation == "relative":
        assert not any(isinstance(step, RelativeActionsProcessorStep) for step in preprocessor.steps)
    else:
        relative_step = next(
            step for step in preprocessor.steps if isinstance(step, RelativeActionsProcessorStep)
        )
        absolute_step = next(
            step for step in postprocessor.steps if isinstance(step, AbsoluteActionsProcessorStep)
        )
        assert preprocessor.steps.index(joint_step) < preprocessor.steps.index(relative_step)
        assert preprocessor.steps.index(relative_step) < preprocessor.steps.index(normalizer)
        assert postprocessor.steps.index(absolute_step) > 0

    assert any(isinstance(step, Pi05AbsoluteActionProcessorStep) for step in postprocessor.steps)


def test_pi052_fast_tokenizer_receives_normalized_relative_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, "relative")
    config.enable_fast_action_loss = True
    config.action_tokenizer_name = str(tmp_path / "action_tokenizer")
    config.tokenizer_name = str(tmp_path / "paligemma_tokenizer")
    monkeypatch.setattr(
        "lerobot.processor.tokenizer_processor.AutoProcessor",
        SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: object()),
    )
    monkeypatch.setattr(
        "lerobot.processor.tokenizer_processor.AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: SimpleNamespace(vocab_size=257152)),
    )

    preprocessor, _ = make_pi052_pre_post_processors(config, dataset_stats=None)

    joint_step = next(
        step for step in preprocessor.steps if isinstance(step, Pi05JointRepresentationProcessorStep)
    )
    normalizer = next(step for step in preprocessor.steps if isinstance(step, NormalizerProcessorStep))
    text_step = next(step for step in preprocessor.steps if isinstance(step, PI052TextTokenizerStep))
    fast_step = next(step for step in preprocessor.steps if isinstance(step, ActionTokenizerProcessorStep))
    assert preprocessor.steps.index(joint_step) < preprocessor.steps.index(normalizer)
    assert preprocessor.steps.index(normalizer) < preprocessor.steps.index(fast_step)
    assert text_step.tokenizer_name == config.tokenizer_name
    assert fast_step.paligemma_tokenizer_name == config.tokenizer_name


@pytest.mark.parametrize("representation", ["relative", "absolute"])
def test_pi052_joint_steps_produce_requested_state_and_shared_relative_action(
    tmp_path: Path, representation: str
) -> None:
    config = _config(tmp_path, representation)
    preprocessor, _ = make_pi052_pre_post_processors(config, dataset_stats=None)
    joint_step = next(
        step for step in preprocessor.steps if isinstance(step, Pi05JointRepresentationProcessorStep)
    )
    relative_step = next(
        (step for step in preprocessor.steps if isinstance(step, RelativeActionsProcessorStep)),
        None,
    )

    previous = torch.arange(14, dtype=torch.float32) / 10
    current = previous + 0.2
    future = torch.stack([current + 0.1, current + 0.3])
    current[GRIPPER_INDICES] = torch.tensor([0.04, 0.05])
    future[:, GRIPPER_INDICES] = torch.tensor([[0.06, 0.07], [0.08, 0.09]])
    transition = batch_to_transition(
        {
            OBS_STATE: torch.stack([previous, current]).unsqueeze(0),
            ACTION: future.unsqueeze(0),
        }
    )

    transformed = joint_step(transition)
    if relative_step is not None:
        transformed = relative_step(transformed)

    state = transformed[TransitionKey.OBSERVATION][OBS_STATE]
    action = transformed[TransitionKey.ACTION]
    arm_indices = [index for index in range(14) if index not in GRIPPER_INDICES]
    expected_state = current.unsqueeze(0)
    if representation == "relative":
        expected_state = expected_state.clone()
        expected_state[:, arm_indices] = current[arm_indices] - previous[arm_indices]
    expected_action = future.unsqueeze(0).clone()
    expected_action[..., arm_indices] -= current[arm_indices]

    torch.testing.assert_close(state, expected_state)
    torch.testing.assert_close(action, expected_action)
    torch.testing.assert_close(action[..., GRIPPER_INDICES], future[None, ..., GRIPPER_INDICES])
