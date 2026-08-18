# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""PI052 processor factory with optional recipe rendering and text tokenization.

Without a recipe it delegates to the standard PI0.5 pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from lerobot.configs import FeatureType, NormalizationMode
from lerobot.configs.recipe import TrainingRecipe
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    ActionTokenizerProcessorStep,
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
    StateNoiseProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)

# Import directly to keep optional language dependencies out of ``lerobot.processor``.
from lerobot.processor.render_messages_processor import RenderMessagesStep
from lerobot.utils.constants import (
    ACTION,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from ..pi05.joint_representation import (
    Pi05AbsoluteActionProcessorStep,
    Pi05JointRepresentationProcessorStep,
    can_use_pi05_joint_stats,
    merge_pi05_joint_stats,
)
from ..pi05.configuration_pi05 import DUAL_ARM_20D_STATE_FEATURE_NAMES
from lerobot.processor.relative_joint_processor import (
    RelativeJointAbsoluteActionProcessorStep,
    RelativeJointProcessorStep,
)
from ..pi05.processor_pi05 import make_pi05_pre_post_processors
from .configuration_pi052 import PI052Config
from .text_processor_pi052 import PI052TextTokenizerStep


def make_pi052_pre_post_processors(
    config: PI052Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    dataset_repo_id: str | None = None,
    dataset_root: str | None = None,
    dataset_revision: str | None = None,
    episodes: list[int] | None = None,
    exclude_episodes: list[int] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build PI0.5-v2's pre/post-processor pipelines.

    Falls through to π0.5's stock pipeline when ``recipe_path`` is unset.
    """
    if not config.recipe_path:
        if getattr(config, "enable_fast_action_loss", False):
            raise ValueError("PI052 FAST action loss requires recipe_path to build action supervision.")
        return make_pi05_pre_post_processors(config, dataset_stats=dataset_stats)

    recipe = _load_recipe(config.recipe_path)

    use_legacy_relative_actions = config.use_relative_actions and config.joint_representation == "absolute"
    relative_step = RelativeActionsProcessorStep(
        enabled=use_legacy_relative_actions,
        exclude_joints=getattr(config, "relative_exclude_joints", []),
        action_names=getattr(config, "action_feature_names", None),
    )
    stats = merge_pi05_joint_stats(config, dataset_stats)
    state_feature = config.input_features.get(OBS_STATE)
    action_feature = config.output_features.get(ACTION)
    has_mixed_stats = getattr(config, "absolute_state_relative_action_stats", None) is not None
    use_mapped_relative_step = (
        state_feature is not None
        and action_feature is not None
        and state_feature.shape == (20,)
        and action_feature.shape == (14,)
        and (config.joint_representation == "relative" or use_legacy_relative_actions)
        and (getattr(config, "relative_joint_stats", None) is not None or has_mixed_stats)
    )
    if use_mapped_relative_step:
        state_names = getattr(config, "state_feature_names", None)
        action_names = getattr(config, "action_feature_names", None)
        if state_names is None and state_feature.shape == (20,):
            state_names = list(DUAL_ARM_20D_STATE_FEATURE_NAMES)
        if action_names is None and action_feature.shape == (14,) and state_names is not None:
            action_names = [state_names[index] for index in [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 19]]
        if state_names is None or action_names is None:
            raise ValueError("PI052 mapped relative processing requires state and action feature names")
        state_absolute_indices = list(getattr(config, "state_absolute_indices", []))
        if config.joint_representation == "absolute" and use_legacy_relative_actions:
            state_absolute_indices = list(range(len(state_names)))
        action_state_indices = [state_names.index(name) for name in action_names]
        joint_step = RelativeJointProcessorStep(
            enabled=True,
            condition_on_state=config.condition_on_state,
            execution_horizon=config.n_action_steps,
            joint_names=action_names,
            gripper_indices=config.joint_gripper_indices,
            state_feature_names=state_names,
            action_feature_names=action_names,
            state_gripper_indices=getattr(config, "state_gripper_indices", None),
            state_absolute_indices=state_absolute_indices,
            action_gripper_indices=config.joint_gripper_indices,
            action_state_indices=action_state_indices,
        )
    else:
        joint_step = Pi05JointRepresentationProcessorStep(
            joint_representation=config.joint_representation,
            gripper_indices=config.joint_gripper_indices,
            action_names=getattr(config, "action_feature_names", None),
            joint_limit_profile=config.joint_limit_profile,
            precomputed_relative_chunk=config.precomputed_relative_chunk,
            condition_on_state=config.condition_on_state,
            execution_horizon=config.n_action_steps,
        )
    norm_map = dict(config.normalization_mapping)
    if (
        getattr(config, "relative_joint_stats", None) is not None
        or getattr(config, "absolute_state_relative_action_stats", None) is not None
    ):
        norm_map[FeatureType.STATE] = NormalizationMode.QUANTILES
        norm_map[FeatureType.ACTION] = NormalizationMode.QUANTILES
    elif can_use_pi05_joint_stats(config):
        norm_map[FeatureType.STATE] = NormalizationMode.MIN_MAX
        norm_map[FeatureType.ACTION] = NormalizationMode.MIN_MAX

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        joint_step,
    ]
    if config.input_features.get(OBS_STATE) is not None and config.condition_on_state:
        input_steps.append(
            StateNoiseProcessorStep(
                joint_std_rad=config.state_noise_std_rad,
                gripper_std_m=config.gripper_noise_std_m,
                gripper_indices=getattr(config, "state_gripper_indices", None) or config.joint_gripper_indices,
            )
        )
    if use_legacy_relative_actions and not use_mapped_relative_step:
        input_steps.append(relative_step)
    input_steps.extend(
        [
            NormalizerProcessorStep(
                features={**config.input_features, **config.output_features},
                norm_map=norm_map,
                stats=stats,
                clip_quantiles=config.clip_quantiles,
            ),
            RenderMessagesStep(recipe=recipe),
            PI052TextTokenizerStep(
                tokenizer_name=config.tokenizer_name,
                max_length=config.tokenizer_max_length,
                plan_dropout_prob=getattr(config, "plan_dropout_prob", 0.0),
                memory_dropout_prob=getattr(config, "memory_dropout_prob", 0.0),
                subtask_dropout_prob=getattr(config, "subtask_dropout_prob", 0.0),
            ),
        ]
    )

    # Add FAST action-token supervision only when explicitly enabled.
    if getattr(config, "enable_fast_action_loss", False):
        from .fit_fast_tokenizer import resolve_fast_tokenizer  # noqa: PLC0415

        input_steps.append(
            ActionTokenizerProcessorStep(
                action_tokenizer_name=resolve_fast_tokenizer(
                    config,
                    dataset_repo_id,
                    dataset_root,
                    stats,
                    dataset_revision,
                    episodes,
                    exclude_episodes,
                ),
                max_action_tokens=config.max_action_tokens,
                fast_skip_tokens=config.fast_skip_tokens,
                paligemma_tokenizer_name=config.tokenizer_name,
                allow_truncation=False,
            )
        )

    input_steps.append(DeviceProcessorStep(device=config.device))

    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=norm_map,
            stats=stats,
        ),
    ]
    if use_mapped_relative_step:
        output_steps.append(RelativeJointAbsoluteActionProcessorStep(relative_step=joint_step))
        if config.apply_action_limits:
            output_steps.append(
                Pi05AbsoluteActionProcessorStep(
                    joint_representation="absolute",
                    gripper_indices=config.joint_gripper_indices,
                    joint_limit_profile=config.joint_limit_profile,
                )
            )
    elif config.apply_action_limits:
        output_steps.extend(
            [
                AbsoluteActionsProcessorStep(
                    enabled=use_legacy_relative_actions,
                    relative_step=relative_step,
                ),
                Pi05AbsoluteActionProcessorStep(
                    joint_representation=config.joint_representation,
                    gripper_indices=config.joint_gripper_indices,
                    joint_limit_profile=config.joint_limit_profile,
                    relative_step=joint_step,
                ),
            ]
        )
    elif use_legacy_relative_actions:
        output_steps.append(AbsoluteActionsProcessorStep(enabled=True, relative_step=relative_step))
    output_steps.append(DeviceProcessorStep(device="cpu"))
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


def _load_recipe(path_str: str) -> TrainingRecipe:
    """Resolve ``path_str`` to a ``TrainingRecipe``.

    Accepts an absolute path or a path relative to
    ``src/lerobot/configs/``.
    """
    p = Path(path_str)
    if not p.is_absolute() and not p.exists():
        from lerobot.configs import recipe as _recipe_module  # noqa: PLC0415

        configs_dir = Path(_recipe_module.__file__).resolve().parent
        candidate = configs_dir / path_str
        if candidate.exists():
            p = candidate
    return TrainingRecipe.from_yaml(p)
