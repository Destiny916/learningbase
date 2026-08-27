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

"""Shared relative-joint processor steps with explicit absolute gripper dimensions."""

from dataclasses import dataclass, field
from typing import Any

import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import OBS_STATE

from .pipeline import ProcessorStep, ProcessorStepRegistry


JOINT_NAMES = ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"]
def _validate_joint_names(joint_names: list[str], gripper_indices: list[int]) -> None:
    if not joint_names or any(not isinstance(name, str) or not name for name in joint_names):
        raise ValueError("joint_names must be a nonempty list of strings")
    if len(set(joint_names)) != len(joint_names):
        raise ValueError("joint_names must be unique")
    if not gripper_indices or len(set(gripper_indices)) != len(gripper_indices):
        raise ValueError("gripper_indices must be a nonempty list of unique indices")
    if any(not isinstance(index, int) or isinstance(index, bool) for index in gripper_indices):
        raise ValueError("gripper_indices must contain integers")
    if any(index < 0 or index >= len(joint_names) for index in gripper_indices):
        raise ValueError("gripper_indices must refer to joint_names dimensions")


def _validate_last_dimension(tensor: torch.Tensor, name: str, dimension: int) -> None:
    if tensor.shape[-1] != dimension:
        raise ValueError(f"{name} must have {dimension} dimensions, got {tensor.shape[-1]}")


def _has_action(transition: EnvTransition) -> bool:
    action = transition.get(TransitionKey.ACTION)
    return isinstance(action, torch.Tensor) and action.numel() > 0


@ProcessorStepRegistry.register("relative_joint_processor")
@dataclass
class RelativeJointProcessorStep(ProcessorStep):
    """Convert absolute joint state/action data into relative arm values.

    During offline training, the state must contain paired previous/current absolute
    positions in ``[..., 2, 7]``. During online inference, an action is absent and
    the step tracks the preceding absolute state locally.
    """

    enabled: bool = True
    condition_on_state: bool = True
    execution_horizon: int = 1
    joint_names: list[str] = field(default_factory=lambda: JOINT_NAMES.copy())
    gripper_indices: list[int] = field(default_factory=lambda: [6])
    state_feature_names: list[str] | None = None
    action_feature_names: list[str] | None = None
    state_gripper_indices: list[int] | None = None
    state_absolute_indices: list[int] = field(default_factory=list)
    action_absolute_indices: list[int] = field(default_factory=list)
    action_gripper_indices: list[int] | None = None
    action_state_indices: list[int] | None = None
    _last_observation_state: torch.Tensor | None = field(default=None, init=False, repr=False)
    _action_anchor_state: torch.Tensor | None = field(default=None, init=False, repr=False)
    _action_steps_processed: int = field(default=0, init=False, repr=False)
    _state_names: list[str] = field(default_factory=list, init=False, repr=False)
    _action_names: list[str] = field(default_factory=list, init=False, repr=False)
    _state_grippers: list[int] = field(default_factory=list, init=False, repr=False)
    _state_absolute: list[int] = field(default_factory=list, init=False, repr=False)
    _action_grippers: list[int] = field(default_factory=list, init=False, repr=False)
    _action_absolute: list[int] = field(default_factory=list, init=False, repr=False)
    _action_to_state: list[int] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            self.state_feature_names is None
            and self.action_feature_names is None
            and len(self.joint_names) <= len(JOINT_NAMES)
            and self.joint_names != JOINT_NAMES
        ):
            raise ValueError("joint_names must contain joint_0, joint_1, joint_2, joint_3, joint_4, joint_5, gripper")
        state_names = list(self.state_feature_names or self.joint_names)
        action_names = list(self.action_feature_names or self.joint_names)
        state_grippers = list(self.state_gripper_indices if self.state_gripper_indices is not None else self.gripper_indices)
        state_absolute = list(self.state_absolute_indices)
        action_absolute = list(self.action_absolute_indices)
        action_grippers = list(self.action_gripper_indices if self.action_gripper_indices is not None else self.gripper_indices)
        _validate_joint_names(state_names, state_grippers)
        _validate_joint_names(action_names, action_grippers)
        if len(set(state_absolute)) != len(state_absolute) or any(
            not isinstance(index, int) or isinstance(index, bool) or index < 0 or index >= len(state_names)
            for index in state_absolute
        ):
            raise ValueError("state_absolute_indices must contain unique valid state indices")
        if len(set(action_absolute)) != len(action_absolute) or any(
            not isinstance(index, int) or isinstance(index, bool) or index < 0 or index >= len(action_names)
            for index in action_absolute
        ):
            raise ValueError("action_absolute_indices must contain unique valid action indices")
        state_name_to_index = {name: index for index, name in enumerate(state_names)}
        if self.action_state_indices is None:
            try:
                action_to_state = [state_name_to_index[name] for name in action_names]
            except KeyError as error:
                raise ValueError(f"action feature name {error.args[0]!r} is missing from state feature names") from error
        else:
            action_to_state = list(self.action_state_indices)
            if len(action_to_state) != len(action_names) or len(set(action_to_state)) != len(action_to_state):
                raise ValueError("action_state_indices must contain unique indices for every action feature")
            if any(not isinstance(index, int) or isinstance(index, bool) or index < 0 or index >= len(state_names) for index in action_to_state):
                raise ValueError("action_state_indices must contain valid state indices")
            if [state_names[index] for index in action_to_state] != action_names:
                raise ValueError("action feature names do not match action_state_indices")
        self._state_names = state_names
        self._action_names = action_names
        self._state_grippers = state_grippers
        self._state_absolute = state_absolute
        self._action_absolute = action_absolute
        self._action_grippers = action_grippers
        self._action_to_state = action_to_state
        if isinstance(self.execution_horizon, bool) or not isinstance(self.execution_horizon, int):
            raise ValueError("execution_horizon must be a positive integer")
        if self.execution_horizon <= 0:
            raise ValueError("execution_horizon must be a positive integer")

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition

        observation = transition.get(TransitionKey.OBSERVATION)
        if observation is None or OBS_STATE not in observation:
            return transition

        state = observation[OBS_STATE]
        if not isinstance(state, torch.Tensor):
            raise ValueError("observation.state must be a torch.Tensor")

        if _has_action(transition):
            return self._offline(transition, state)
        return self._online(transition, state)

    def _offline(self, transition: EnvTransition, paired_state: torch.Tensor) -> EnvTransition:
        _validate_last_dimension(paired_state, "offline observation.state", len(self._state_names))

        action = transition[TransitionKey.ACTION]
        assert isinstance(action, torch.Tensor)
        _validate_last_dimension(action, "action", len(self._action_names))

        # Absolute-state/relative-action training has no previous state frame.
        # If every state dimension is explicitly absolute, the current state is
        # the action-relative anchor.
        if paired_state.ndim < 3 or paired_state.shape[-2] != 2:
            state_arm_mask = self._state_arm_mask(paired_state.device)
            if state_arm_mask.any():
                raise ValueError(
                    "offline relative state conversion requires paired previous/current frames; "
                    "use state_absolute_indices for absolute-state/relative-action training"
                )
            relative_action = action.clone()
            action_arm_mask = self._action_arm_mask(action.device)
            action_state_reference = paired_state[..., self._action_to_state]
            action_reference = action_state_reference[..., action_arm_mask].to(
                device=action.device, dtype=action.dtype
            )
            if action.ndim == paired_state.ndim + 1:
                action_reference = action_reference.unsqueeze(-2)
            relative_action[..., action_arm_mask] -= action_reference
            new_transition = transition.copy()
            observation = transition[TransitionKey.OBSERVATION]
            assert observation is not None
            new_observation = observation.copy()
            new_observation[OBS_STATE] = paired_state.clone()
            new_transition[TransitionKey.OBSERVATION] = new_observation
            new_transition[TransitionKey.ACTION] = relative_action
            return new_transition

        previous, current = paired_state.unbind(dim=-2)
        state_arm_mask = self._state_arm_mask(current.device)
        relative_state = current.clone()
        if self.condition_on_state:
            relative_state[..., state_arm_mask] -= previous[..., state_arm_mask]

        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        state_is_pad = complementary_data.get("observation.state_is_pad")
        if state_is_pad is not None:
            if not isinstance(state_is_pad, torch.Tensor):
                raise ValueError("observation.state_is_pad must be a torch.Tensor")
            if state_is_pad.shape != paired_state.shape[:-1]:
                raise ValueError(
                    "observation.state_is_pad must match offline observation.state batch dimensions and time axis"
                )
            previous_is_pad = state_is_pad[..., 0].to(device=relative_state.device, dtype=torch.bool)
            relative_state[..., state_arm_mask] = torch.where(
                previous_is_pad.unsqueeze(-1),
                torch.zeros_like(relative_state[..., state_arm_mask]),
                relative_state[..., state_arm_mask],
            )

        relative_action = action.clone()
        action_arm_mask = self._action_arm_mask(action.device)
        action_state_reference = current[..., self._action_to_state]
        action_reference = action_state_reference[..., action_arm_mask]
        if action.ndim == current.ndim + 1:
            action_reference = action_reference.unsqueeze(-2)
        relative_action[..., action_arm_mask] -= action_reference

        new_transition = transition.copy()
        observation = transition[TransitionKey.OBSERVATION]
        assert observation is not None
        new_observation = observation.copy()
        new_observation[OBS_STATE] = relative_state
        new_transition[TransitionKey.OBSERVATION] = new_observation
        new_transition[TransitionKey.ACTION] = relative_action
        return new_transition

    def _online(self, transition: EnvTransition, current: torch.Tensor) -> EnvTransition:
        _validate_last_dimension(current, "observation.state", len(self._state_names))

        state_arm_mask = self._state_arm_mask(current.device)
        relative_state = current.clone()
        if self._last_observation_state is None:
            relative_state[..., state_arm_mask] = 0.0
        else:
            if self._last_observation_state.shape != current.shape:
                raise ValueError(
                    "cached absolute observation.state shape does not match the current observation.state"
                )
            previous = self._last_observation_state.to(device=current.device, dtype=current.dtype)
            relative_state[..., state_arm_mask] -= previous[..., state_arm_mask]

        self._last_observation_state = current.detach().clone()
        if self._action_anchor_state is None:
            self._action_anchor_state = current.detach().clone()
            self._action_steps_processed = 0
        new_transition = transition.copy()
        observation = transition[TransitionKey.OBSERVATION]
        assert observation is not None
        new_observation = observation.copy()
        new_observation[OBS_STATE] = relative_state
        new_transition[TransitionKey.OBSERVATION] = new_observation
        return new_transition

    def get_cached_absolute_state(self) -> torch.Tensor | None:
        """Return the absolute state anchoring the action chunk for paired postprocessing."""
        return self._action_anchor_state

    def _state_arm_mask(self, device: torch.device) -> torch.Tensor:
        mask = torch.ones(len(self._state_names), device=device, dtype=torch.bool)
        mask[self._state_grippers] = False
        if self._state_absolute:
            mask[self._state_absolute] = False
        return mask

    def _action_arm_mask(self, device: torch.device) -> torch.Tensor:
        mask = torch.ones(len(self._action_names), device=device, dtype=torch.bool)
        mask[self._action_grippers] = False
        if self._action_absolute:
            mask[self._action_absolute] = False
        return mask

    def _arm_mask(self, device: torch.device) -> torch.Tensor:
        """Backward-compatible alias for the action mask."""
        return self._action_arm_mask(device)

    def consume_action_steps(self, action_steps: int) -> None:
        """Release the current action anchor after its configured horizon is consumed."""
        if action_steps <= 0:
            raise ValueError("action_steps must be positive")
        self._action_steps_processed += action_steps
        if self._action_steps_processed >= self.execution_horizon:
            self._action_anchor_state = None
            self._action_steps_processed = 0

    def reset(self) -> None:
        self._last_observation_state = None
        self._action_anchor_state = None
        self._action_steps_processed = 0

    def get_config(self) -> dict[str, Any]:
        config = {
            "enabled": self.enabled,
            "condition_on_state": self.condition_on_state,
            "execution_horizon": self.execution_horizon,
            "joint_names": self.joint_names,
        }
        if self.state_feature_names is not None or self.action_feature_names is not None:
            config.update(
                {
                    "gripper_indices": self.gripper_indices,
                    "state_feature_names": self._state_names,
                    "action_feature_names": self._action_names,
                    "state_gripper_indices": self._state_grippers,
                    "state_absolute_indices": self._state_absolute,
                    "action_absolute_indices": self._action_absolute,
                    "action_gripper_indices": self._action_grippers,
                    "action_state_indices": self._action_to_state,
                }
            )
        return config

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("state_noise_processor")
@dataclass
class StateNoiseProcessorStep(ProcessorStep):
    """Add train-only sensor noise before normalization."""

    joint_std_rad: float = 0.0
    gripper_std_m: float = 0.0
    gripper_indices: list[int] = field(default_factory=lambda: [6])
    position_std_m: float = 0.0
    position_indices: list[int] = field(default_factory=list)
    enabled: bool = True
    _training: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.joint_std_rad < 0 or self.gripper_std_m < 0 or self.position_std_m < 0:
            raise ValueError("state noise standard deviations must be non-negative")
        if not self.gripper_indices or len(set(self.gripper_indices)) != len(self.gripper_indices):
            raise ValueError("gripper_indices must be a nonempty list of unique indices")

    def train(self) -> None:
        self._training = True

    def eval(self) -> None:
        self._training = False

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled or not self._training or (
            self.joint_std_rad == 0 and self.gripper_std_m == 0 and self.position_std_m == 0
        ):
            return transition
        observation = transition.get(TransitionKey.OBSERVATION)
        if observation is None or OBS_STATE not in observation:
            return transition
        state = observation[OBS_STATE]
        if not isinstance(state, torch.Tensor):
            raise ValueError("observation.state must be a torch.Tensor")
        if any(index < 0 or index >= state.shape[-1] for index in self.gripper_indices):
            raise ValueError("gripper_indices must refer to observation.state dimensions")
        if any(index < 0 or index >= state.shape[-1] for index in self.position_indices):
            raise ValueError("position_indices must refer to observation.state dimensions")
        scales = state.new_full((state.shape[-1],), self.joint_std_rad)
        scales[self.gripper_indices] = self.gripper_std_m
        if self.position_indices:
            scales[self.position_indices] = self.position_std_m
        new_transition = transition.copy()
        new_observation = observation.copy()
        new_observation[OBS_STATE] = state + torch.randn_like(state) * scales
        new_transition[TransitionKey.OBSERVATION] = new_observation
        return new_transition

    def get_config(self) -> dict[str, Any]:
        config = {
            "joint_std_rad": self.joint_std_rad,
            "gripper_std_m": self.gripper_std_m,
            "gripper_indices": self.gripper_indices,
            "enabled": self.enabled,
        }
        if self.position_std_m != 0 or self.position_indices:
            config.update({"position_std_m": self.position_std_m, "position_indices": self.position_indices})
        return config

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("relative_joint_absolute_action_processor")
@dataclass
class RelativeJointAbsoluteActionProcessorStep(ProcessorStep):
    """Reconstruct absolute arm actions from a paired relative-joint step cache."""

    relative_step: RelativeJointProcessorStep | None = field(default=None, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition
        if not isinstance(action, torch.Tensor):
            raise ValueError("action must be a torch.Tensor")
        if self.relative_step is None:
            raise RuntimeError(
                "RelativeJointAbsoluteActionProcessorStep requires a paired RelativeJointProcessorStep"
            )
        _validate_last_dimension(action, "action", len(self.relative_step._action_names))
        current = self.relative_step.get_cached_absolute_state()
        if current is None:
            raise RuntimeError(
                "RelativeJointAbsoluteActionProcessorStep requires a paired RelativeJointProcessorStep, "
                "but no absolute state has been cached"
            )
        if action.shape[:-1] != current.shape[:-1] and action.shape[:-2] != current.shape[:-1]:
            raise ValueError("cached absolute observation.state shape does not match action batch dimensions")

        absolute_action = action.clone()
        # The cached absolute anchor is captured before the device processor,
        # while the policy action is normally on the policy device.  Build the
        # boolean mask on the anchor device before indexing it, then move the
        # selected reference to the action device.
        arm_mask = self.relative_step._action_arm_mask(current.device)
        mapped_current = current[..., self.relative_step._action_to_state]
        reference = mapped_current[..., arm_mask].to(device=action.device, dtype=action.dtype)
        if action.ndim == current.ndim + 1:
            reference = reference.unsqueeze(-2)
        absolute_action[..., arm_mask] += reference
        action_steps = action.shape[-2] if action.ndim == current.ndim + 1 else 1
        self.relative_step.consume_action_steps(action_steps)

        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = absolute_action
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("zero_state_processor")
@dataclass
class ZeroStateProcessorStep(ProcessorStep):
    """Replace an existing observation state with an equal-shaped zero tensor."""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION)
        if observation is None or OBS_STATE not in observation:
            return transition

        state = observation[OBS_STATE]
        if not isinstance(state, torch.Tensor):
            raise ValueError("observation.state must be a torch.Tensor")

        new_transition = transition.copy()
        new_observation = observation.copy()
        new_observation[OBS_STATE] = state.clone().zero_()
        new_transition[TransitionKey.OBSERVATION] = new_observation
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
