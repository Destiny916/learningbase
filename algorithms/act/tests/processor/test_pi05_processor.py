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

"""Compare the PI0.5 processor pipeline against the vendored OpenPI reference processors."""

import os
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature  # noqa: E402
from lerobot.policies.pi05 import PI05Policy  # noqa: E402
from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: E402
from lerobot.policies.pi05.processor_pi05 import (  # noqa: E402
    Pi05PrepareStateTokenizerProcessorStep,
    make_pi05_pre_post_processors,
)
from lerobot.processor import (  # noqa: E402
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    RelativeJointAbsoluteActionProcessorStep,
    RelativeJointProcessorStep,
    StateNoiseProcessorStep,
    TokenizerProcessorStep,
    TransitionKey,
    UnnormalizerProcessorStep,
    ZeroStateProcessorStep,
    batch_to_transition,
)
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402
RUNNING_CI = os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true"
TRANSFORMERS_AVAILABLE = find_spec("transformers") is not None

DUMMY_ACTION_DIM = 32
DUMMY_STATE_DIM = 32
DUMMY_ACTION_HORIZON = 50
DUMMY_MAX_TOKEN_LEN = 200
DEVICE = torch.device("cpu")
IMAGE_KEYS = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]

DUMMY_DATASET_STATS = {
    OBS_STATE: {
        "mean": torch.zeros(DUMMY_STATE_DIM),
        "std": torch.ones(DUMMY_STATE_DIM),
        "q01": torch.zeros(DUMMY_STATE_DIM),
        "q99": torch.ones(DUMMY_STATE_DIM),
    },
    ACTION: {
        "mean": torch.zeros(DUMMY_ACTION_DIM),
        "std": torch.ones(DUMMY_ACTION_DIM),
        "q01": torch.zeros(DUMMY_ACTION_DIM),
        "q99": torch.ones(DUMMY_ACTION_DIM),
    },
    "images": {
        key: {
            "mean": torch.zeros(3, 224, 224),
            "std": torch.ones(3, 224, 224),
            "q01": torch.zeros(3, 224, 224),
            "q99": torch.ones(3, 224, 224),
        }
        for key in IMAGE_KEYS
    },
}


class PI05PolicyInputAdapter(torch.nn.Module):
    """Minimal adapter exposing PI0.5 policy image preparation without loading model weights."""

    _preprocess_images = PI05Policy._preprocess_images

    def __init__(self, config: PI05Config) -> None:
        super().__init__()
        self.config = config
        self._device_anchor = torch.nn.Parameter(torch.empty((), device=config.device), requires_grad=False)


def create_pi05_config() -> PI05Config:
    config = PI05Config(device=str(DEVICE))
    config.max_state_dim = DUMMY_STATE_DIM
    config.max_action_dim = DUMMY_ACTION_DIM
    config.chunk_size = DUMMY_ACTION_HORIZON
    config.n_action_steps = DUMMY_ACTION_HORIZON
    config.tokenizer_max_length = DUMMY_MAX_TOKEN_LEN
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(DUMMY_STATE_DIM,)),
        **{
            f"observation.images.{key}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))
            for key in IMAGE_KEYS
        },
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(DUMMY_ACTION_DIM,)),
    }
    return config


def create_dummy_data() -> dict:
    batch_size = 2
    prompt = "Pick up the red block and place it in the bin"
    return {
        OBS_STATE: torch.randn(batch_size, DUMMY_STATE_DIM, dtype=torch.float32, device=DEVICE),
        ACTION: torch.randn(
            batch_size, DUMMY_ACTION_HORIZON, DUMMY_ACTION_DIM, dtype=torch.float32, device=DEVICE
        ),
        **{
            f"observation.images.{key}": torch.rand(
                batch_size, 3, 224, 224, dtype=torch.float32, device=DEVICE
            )
            for key in IMAGE_KEYS
        },
        "task": [prompt for _ in range(batch_size)],
    }


@pytest.mark.skipif(
    RUNNING_CI or not TRANSFORMERS_AVAILABLE,
    reason="OpenPI parity requires transformers and runs manually outside CI",
)
def test_pi05_processor_inputs_match_openpi_reference():
    from tests.policies.pi0_pi05.utils.openpi_parity import (
        assert_processor_inputs_match_lerobot,
        clone_batch,
        make_openpi_observation_from_raw,
        openpi_model_actions_from_raw,
    )

    torch.manual_seed(0)
    config = create_pi05_config()
    preprocessor, _ = make_pi05_pre_post_processors(config=config, dataset_stats=DUMMY_DATASET_STATS)

    raw_batch = create_dummy_data()
    lerobot_batch = preprocessor(clone_batch(raw_batch))
    openpi_observation = make_openpi_observation_from_raw(
        raw_batch,
        action_dim=DUMMY_ACTION_DIM,
        max_token_len=DUMMY_MAX_TOKEN_LEN,
        dataset_stats=DUMMY_DATASET_STATS,
        pi05=True,
    )

    assert_processor_inputs_match_lerobot(
        PI05PolicyInputAdapter(config),
        lerobot_batch,
        openpi_observation,
        compare_state=False,
    )
    torch.testing.assert_close(
        lerobot_batch[ACTION],
        openpi_model_actions_from_raw(
            raw_batch,
            action_dim=DUMMY_ACTION_DIM,
            dataset_stats=DUMMY_DATASET_STATS,
            pi05=True,
        ),
        rtol=0,
        atol=0,
    )


def _relative_7d_config(*, condition_on_state: bool) -> PI05Config:
    config = PI05Config(
        device=str(DEVICE),
        joint_representation="relative",
        condition_on_state=condition_on_state,
        joint_gripper_indices=[6],
        chunk_size=2,
        n_action_steps=2,
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        action_feature_names=[f"joint_{index}" for index in range(6)] + ["gripper"],
    )
    config._relative_joint_stats = SimpleNamespace(  # noqa: SLF001
        state=SimpleNamespace(q01=torch.full((7,), -2.0), q99=torch.full((7,), 2.0)),
        actions={
            2: SimpleNamespace(q01=torch.full((7,), -4.0), q99=torch.full((7,), 4.0))
        },
    )
    return config


@pytest.mark.parametrize("condition_on_state", [True, False])
def test_pi05_relative_processor_uses_fixed_shared_pipeline_order(
    monkeypatch: pytest.MonkeyPatch, condition_on_state: bool
):
    tokenizer_module = pytest.importorskip("lerobot.processor.tokenizer_processor")
    monkeypatch.setattr(tokenizer_module, "_transformers_available", True)
    monkeypatch.setattr(
        tokenizer_module,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: object()),
    )
    preprocessor, postprocessor = make_pi05_pre_post_processors(
        config=_relative_7d_config(condition_on_state=condition_on_state),
        dataset_stats=DUMMY_DATASET_STATS,
    )

    expected_pre_steps = [
        AddBatchDimensionProcessorStep,
        RelativeJointProcessorStep,
    ]
    if condition_on_state:
        expected_pre_steps.append(StateNoiseProcessorStep)
    expected_pre_steps.append(NormalizerProcessorStep)
    if not condition_on_state:
        expected_pre_steps.append(ZeroStateProcessorStep)
    expected_pre_steps.extend(
        [Pi05PrepareStateTokenizerProcessorStep, TokenizerProcessorStep, DeviceProcessorStep]
    )
    assert [type(step) for step in preprocessor.steps] == expected_pre_steps
    assert [type(step) for step in postprocessor.steps] == [
        UnnormalizerProcessorStep,
        RelativeJointAbsoluteActionProcessorStep,
        DeviceProcessorStep,
    ]

    normalizer = next(step for step in preprocessor.steps if isinstance(step, NormalizerProcessorStep))
    assert normalizer.norm_map[FeatureType.STATE] is NormalizationMode.QUANTILES
    assert normalizer.norm_map[FeatureType.ACTION] is NormalizationMode.QUANTILES
    assert normalizer.clip_quantiles is True
    torch.testing.assert_close(normalizer._tensor_stats[OBS_STATE]["q01"], torch.full((7,), -2.0))
    torch.testing.assert_close(normalizer._tensor_stats[ACTION]["q99"], torch.full((7,), 4.0))


def test_pi05_absolute_7d_pipeline_includes_state_noise_when_configured(
    monkeypatch: pytest.MonkeyPatch,
):
    tokenizer_module = pytest.importorskip("lerobot.processor.tokenizer_processor")
    monkeypatch.setattr(tokenizer_module, "_transformers_available", True)
    monkeypatch.setattr(tokenizer_module, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: object()))
    config = PI05Config(
        device="cpu",
        joint_representation="absolute",
        joint_gripper_indices=[6],
        state_noise_std_rad=0.003,
        gripper_noise_std_m=0.001,
        chunk_size=2,
        n_action_steps=2,
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    stats = {
        OBS_STATE: {"q01": torch.full((7,), -1.0), "q99": torch.full((7,), 1.0)},
        ACTION: {"q01": torch.full((7,), -1.0), "q99": torch.full((7,), 1.0)},
    }

    preprocessor, _ = make_pi05_pre_post_processors(config=config, dataset_stats=stats)

    noise_step = next(step for step in preprocessor.steps if isinstance(step, StateNoiseProcessorStep))
    assert noise_step.joint_std_rad == 0.003
    assert noise_step.gripper_std_m == 0.001


def test_pi05_absolute_quantiles_pipeline_honors_configured_clipping(
    monkeypatch: pytest.MonkeyPatch,
):
    tokenizer_module = pytest.importorskip("lerobot.processor.tokenizer_processor")
    monkeypatch.setattr(tokenizer_module, "_transformers_available", True)
    monkeypatch.setattr(
        tokenizer_module,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: object()),
    )
    config = create_pi05_config()
    assert config.joint_representation == "absolute"
    assert config.normalization_mapping["STATE"] is NormalizationMode.QUANTILES
    assert config.normalization_mapping["ACTION"] is NormalizationMode.QUANTILES
    assert config.clip_quantiles is True

    preprocessor, _ = make_pi05_pre_post_processors(
        config=config,
        dataset_stats=DUMMY_DATASET_STATS,
    )

    normalizer = next(step for step in preprocessor.steps if isinstance(step, NormalizerProcessorStep))
    assert normalizer.clip_quantiles is True


def test_pi05_image_only_keeps_fixed_state_prompt_with_exact_zero_state():
    transition = batch_to_transition(
        {
            OBS_STATE: torch.tensor([[0.5, -0.5, 0.25, -0.25, 1.0, -1.0, 0.03]]),
            "task": ["grasp_bread"],
        }
    )
    zeroed = ZeroStateProcessorStep()(transition)
    result = Pi05PrepareStateTokenizerProcessorStep()(zeroed)

    assert result[TransitionKey.COMPLEMENTARY_DATA]["task"] == [
        "Task: grasp bread, State: 128 128 128 128 128 128 128;\nAction: "
    ]


def test_pi05_core_processor_test_runs_when_ci_is_true():
    env = os.environ.copy()
    env["CI"] = "true"
    env.pop("GITHUB_ACTIONS", None)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            f"{Path(__file__).resolve()}::test_pi05_image_only_keeps_fixed_state_prompt_with_exact_zero_state",
            "-q",
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout
