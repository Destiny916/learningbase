#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
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
from typing import Any

import torch

from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    PoseQuantileNormalizerProcessorStep,
    PoseQuantileUnnormalizerProcessorStep,
    RelativePoseAbsoluteActionProcessorStep,
    RelativePoseProcessorStep,
    RelativeJointAbsoluteActionProcessorStep,
    RelativeJointProcessorStep,
    RenameObservationsProcessorStep,
    StateNoiseProcessorStep,
    UnnormalizerProcessorStep,
    ZeroStateProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import (
    ACTION,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_act import ACTConfig


def make_act_pre_post_processors(
    config: ACTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Creates the pre- and post-processing pipelines for the ACT policy.

    The pre-processing pipeline handles normalization, batching, and device placement for the model inputs.
    The post-processing pipeline handles unnormalization and moves the model outputs back to the CPU.

    Args:
        config (ACTConfig): The ACT policy configuration object.
        dataset_stats (dict[str, dict[str, torch.Tensor]] | None): A dictionary containing dataset
            statistics (e.g., mean and std) used for normalization. Defaults to None.

    Returns:
        tuple[PolicyProcessorPipeline[dict[str, Any], dict[str, Any]], PolicyProcessorPipeline[PolicyAction, PolicyAction]]: A tuple containing the
        pre-processor pipeline and the post-processor pipeline.
    """

    if config.end_effector_pose_representation == "relative":
        from lerobot.datasets.end_effector_pose_stats import load_relative_pose_stats_paths

        state_feature = config.input_features.get(OBS_STATE)
        action_feature = config.output_features.get(ACTION)
        if state_feature is None or action_feature is None or state_feature.shape != (10,) or action_feature.shape != (10,):
            raise ValueError("relative end-effector pose representation requires 10D state and action features")
        if not config.pose_state_stats_path or not config.pose_action_stats_path:
            raise ValueError("relative end-effector pose representation requires state and action stats paths")
        pose_stats = load_relative_pose_stats_paths(
            config.pose_state_stats_path, config.pose_action_stats_path, expected_horizon=config.chunk_size
        )
        stats = {
            OBS_STATE: {"q01": pose_stats.state.q01, "q99": pose_stats.state.q99},
            ACTION: {"q01": pose_stats.actions[config.chunk_size].q01, "q99": pose_stats.actions[config.chunk_size].q99},
        }
        relative_step = RelativePoseProcessorStep(execution_horizon=config.n_action_steps)
        input_steps = [
            AddBatchDimensionProcessorStep(),
            relative_step,
            PoseQuantileNormalizerProcessorStep(stats=stats, clip_quantiles=config.clip_quantiles),
        ]
        if not config.condition_on_state:
            input_steps.append(ZeroStateProcessorStep())
        input_steps.append(DeviceProcessorStep(device=config.device))
        output_steps = [
            PoseQuantileUnnormalizerProcessorStep(stats=stats),
            RelativePoseAbsoluteActionProcessorStep(relative_step=relative_step),
            DeviceProcessorStep(device="cpu"),
        ]
    elif config.joint_representation == "relative":
        if config.relative_joint_stats is None:
            config.validate_features()
        relative_stats = config.relative_joint_stats
        if relative_stats is None:
            raise ValueError("relative joint representation requires loaded relative joint statistics")

        stats = {key: dict(value) for key, value in (dataset_stats or {}).items()}
        stats[OBS_STATE] = {
            "q01": relative_stats.state.q01.copy(),
            "q99": relative_stats.state.q99.copy(),
        }
        action_stats = relative_stats.actions[config.chunk_size]
        stats[ACTION] = {
            "q01": action_stats.q01.copy(),
            "q99": action_stats.q99.copy(),
        }
        relative_step = RelativeJointProcessorStep(
            # ACT always computes the real relative state; image-only conditioning is expressed by
            # ZeroStateProcessorStep after normalization so action labels retain the real current anchor.
            condition_on_state=True,
            execution_horizon=config.n_action_steps,
            joint_names=config.action_feature_names or [],
            gripper_indices=config.gripper_indices,
            state_feature_names=config.state_feature_names,
            action_feature_names=config.action_feature_names,
            state_gripper_indices=config.state_gripper_indices,
            state_absolute_indices=config.state_absolute_indices,
            action_absolute_indices=config.action_absolute_indices,
            action_gripper_indices=config.gripper_indices,
        )
        input_steps = [
            AddBatchDimensionProcessorStep(),
            relative_step,
        ]
        if config.condition_on_state:
            input_steps.append(
                StateNoiseProcessorStep(
                    joint_std_rad=config.state_noise_std_rad,
                    gripper_std_m=config.gripper_noise_std_m,
                    gripper_indices=config.state_gripper_indices or config.gripper_indices,
                    position_std_m=config.state_position_noise_std_m,
                    position_indices=config.state_position_indices,
                )
            )
        input_steps.append(
            NormalizerProcessorStep(
                features={**config.input_features, **config.output_features},
                norm_map=config.normalization_mapping,
                stats=stats,
                device=config.device,
                clip_quantiles=config.clip_quantiles,
            )
        )
        if not config.condition_on_state:
            input_steps.append(ZeroStateProcessorStep())
        input_steps.append(DeviceProcessorStep(device=config.device))
        output_steps = [
            UnnormalizerProcessorStep(
                features=config.output_features,
                norm_map=config.normalization_mapping,
                stats=stats,
            ),
            RelativeJointAbsoluteActionProcessorStep(relative_step=relative_step),
            DeviceProcessorStep(device="cpu"),
        ]
    else:
        input_steps = [
            RenameObservationsProcessorStep(rename_map={}),
            AddBatchDimensionProcessorStep(),
            DeviceProcessorStep(device=config.device),
            NormalizerProcessorStep(
                features={**config.input_features, **config.output_features},
                norm_map=config.normalization_mapping,
                stats=dataset_stats,
                device=config.device,
            ),
        ]
        output_steps = [
            UnnormalizerProcessorStep(
                features=config.output_features,
                norm_map=config.normalization_mapping,
                stats=dataset_stats,
            ),
            DeviceProcessorStep(device="cpu"),
        ]

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
