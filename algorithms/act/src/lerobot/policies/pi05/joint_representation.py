#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import ProcessorStep, ProcessorStepRegistry, RelativeJointProcessorStep
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import ACTION, OBS_STATE

PIPER_PIKA_14D_ABSOLUTE_MIN = torch.tensor(
    [
        -2.618,
        0.0,
        -2.967,
        -1.745,
        -1.22,
        -2.0944,
        0.0,
        -2.618,
        0.0,
        -2.967,
        -1.745,
        -1.22,
        -2.0944,
        0.0,
    ],
    dtype=torch.float32,
)
PIPER_PIKA_14D_ABSOLUTE_MAX = torch.tensor(
    [
        2.618,
        3.14,
        0.0,
        1.745,
        1.22,
        2.0944,
        0.10,
        2.618,
        3.14,
        0.0,
        1.745,
        1.22,
        2.0944,
        0.10,
    ],
    dtype=torch.float32,
)


def build_arm_mask(
    action_dim: int,
    *,
    gripper_indices: list[int] | tuple[int, ...],
    action_names: list[str] | None = None,
) -> torch.Tensor:
    """Return True for arm dimensions and False for absolute gripper dimensions."""
    del action_names  # Grippers are index-defined for this dataset: joint_6 is a gripper.
    mask = torch.ones(action_dim, dtype=torch.bool)
    for index in gripper_indices:
        if 0 <= index < action_dim:
            mask[index] = False
    return mask


def _profile_limits(profile: str) -> tuple[torch.Tensor, torch.Tensor]:
    if profile != "piper_pika_14d":
        raise ValueError(f"Unsupported pi05 joint_limit_profile: {profile}")
    return PIPER_PIKA_14D_ABSOLUTE_MIN.clone(), PIPER_PIKA_14D_ABSOLUTE_MAX.clone()


def make_pi05_joint_stats(
    joint_representation: str,
    *,
    gripper_indices: list[int] | tuple[int, ...],
    joint_limit_profile: str = "piper_pika_14d",
) -> dict[str, dict[str, torch.Tensor]]:
    """Build hardware-limit stats for pi05 14D Piper/Pika state and action normalization."""
    absolute_min, absolute_max = _profile_limits(joint_limit_profile)
    if joint_representation not in {"absolute", "relative"}:
        raise ValueError(f"Unsupported joint_representation: {joint_representation}")

    if joint_representation == "absolute":
        min_val = absolute_min
        max_val = absolute_max
    else:
        arm_mask = build_arm_mask(len(absolute_min), gripper_indices=gripper_indices)
        joint_range = absolute_max - absolute_min
        min_val = -joint_range
        max_val = joint_range
        min_val[~arm_mask] = absolute_min[~arm_mask]
        max_val[~arm_mask] = absolute_max[~arm_mask]

    stats = {
        "mean": (min_val + max_val) / 2.0,
        "std": torch.clamp((max_val - min_val) / 2.0, min=1e-8),
        "min": min_val.clone(),
        "max": max_val.clone(),
        "q01": min_val.clone(),
        "q99": max_val.clone(),
    }
    return {OBS_STATE: {k: v.clone() for k, v in stats.items()}, ACTION: {k: v.clone() for k, v in stats.items()}}


def can_use_pi05_joint_stats(config: Any) -> bool:
    state_feature = getattr(config, "input_features", {}).get(OBS_STATE)
    action_feature = getattr(config, "output_features", {}).get(ACTION)
    if state_feature is None or action_feature is None:
        return False
    return tuple(state_feature.shape) == (14,) and action_feature.shape[-1] == 14


def merge_pi05_joint_stats(
    config: Any, dataset_stats: dict[str, dict[str, Any]] | None
) -> dict[str, dict[str, Any]] | None:
    mixed_stats = getattr(config, "absolute_state_relative_action_stats", None)
    if mixed_stats is not None:
        absolute_bundle, relative_bundle = mixed_stats
        action_stats = relative_bundle.actions[config.chunk_size]
        return {
            OBS_STATE: {
                "q01": torch.tensor(absolute_bundle.state.q01, dtype=torch.float32),
                "q99": torch.tensor(absolute_bundle.state.q99, dtype=torch.float32),
            },
            ACTION: {
                "q01": torch.tensor(action_stats.q01, dtype=torch.float32),
                "q99": torch.tensor(action_stats.q99, dtype=torch.float32),
            },
        }

    absolute_stats = getattr(config, "absolute_action_stats", None)
    if absolute_stats is not None:
        return {
            OBS_STATE: {
                "q01": torch.tensor(absolute_stats.state.q01, dtype=torch.float32),
                "q99": torch.tensor(absolute_stats.state.q99, dtype=torch.float32),
            },
            ACTION: {
                "q01": torch.tensor(absolute_stats.actions[config.chunk_size].q01, dtype=torch.float32),
                "q99": torch.tensor(absolute_stats.actions[config.chunk_size].q99, dtype=torch.float32),
            },
        }

    relative_stats = getattr(config, "relative_joint_stats", None)
    if relative_stats is not None:

        def copied_tensor(values: Any) -> torch.Tensor:
            if isinstance(values, torch.Tensor):
                return values.detach().clone()
            return torch.tensor(values)

        action_stats = relative_stats.actions[config.chunk_size]
        merged = dict(dataset_stats or {})
        merged[OBS_STATE] = {
            "q01": copied_tensor(relative_stats.state.q01),
            "q99": copied_tensor(relative_stats.state.q99),
        }
        merged[ACTION] = {
            "q01": copied_tensor(action_stats.q01),
            "q99": copied_tensor(action_stats.q99),
        }
        return merged

    if not can_use_pi05_joint_stats(config):
        return dataset_stats

    merged = dict(dataset_stats or {})
    merged.update(
        make_pi05_joint_stats(
            getattr(config, "joint_representation", "absolute"),
            gripper_indices=getattr(config, "joint_gripper_indices", [6, 13]),
            joint_limit_profile=getattr(config, "joint_limit_profile", "piper_pika_14d"),
        )
    )
    return merged


def _current_state_from_observation(state: torch.Tensor) -> torch.Tensor:
    if state.ndim >= 3:
        return state[..., -1, :]
    return state


def _previous_state_from_observation(state: torch.Tensor) -> torch.Tensor | None:
    if state.ndim >= 3 and state.shape[-2] >= 2:
        return state[..., -2, :]
    return None


def _previous_is_pad(transition: EnvTransition) -> torch.Tensor | bool:
    complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
    pad = complementary_data.get(f"{OBS_STATE}_is_pad")
    if pad is None:
        return False
    pad_t = torch.as_tensor(pad)
    if pad_t.ndim == 0:
        return bool(pad_t.item())
    if pad_t.shape[-1] >= 2:
        return pad_t[..., -2]
    return pad_t[..., 0]


def _reshape_dim_vector(values: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    shape = [1] * target.ndim
    shape[-1] = values.shape[0]
    return values.to(device=target.device, dtype=target.dtype).reshape(shape)


@ProcessorStepRegistry.register(name="pi05_joint_representation_processor")
@dataclass
class Pi05JointRepresentationProcessorStep(ProcessorStep):
    joint_representation: str = "absolute"
    gripper_indices: list[int] = field(default_factory=lambda: [6, 13])
    action_names: list[str] | None = None
    joint_limit_profile: str = "piper_pika_14d"
    precomputed_relative_chunk: bool = False
    condition_on_state: bool = True
    execution_horizon: int = 1
    _last_observation_state: torch.Tensor | None = field(default=None, init=False, repr=False)
    _action_anchor_state: torch.Tensor | None = field(default=None, init=False, repr=False)
    _action_steps_processed: int = field(default=0, init=False, repr=False)
    _shared_step: RelativeJointProcessorStep = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.joint_representation not in {"absolute", "relative"}:
            raise ValueError(f"Unsupported joint_representation: {self.joint_representation}")
        self._shared_step = RelativeJointProcessorStep(
            enabled=self.joint_representation == "relative",
            condition_on_state=self.condition_on_state,
            execution_horizon=self.execution_horizon,
        )

    def _arm_mask(self, dim: int, *, device: torch.device, dtype: torch.dtype = torch.bool) -> torch.Tensor:
        return build_arm_mask(dim, gripper_indices=self.gripper_indices, action_names=self.action_names).to(
            device=device, dtype=dtype
        )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION) or {}
        state = observation.get(OBS_STATE)
        if state is None:
            return transition

        state = torch.as_tensor(state)
        action = transition.get(TransitionKey.ACTION)
        # Dataset rows in this mode already carry the desired relative state and
        # anchored action chunk. An action identifies the training data path;
        # online inference supplies only the current absolute robot state.
        if self.precomputed_relative_chunk and action is not None:
            return transition
        if self.joint_representation == "relative" and state.shape[-1] == 7:
            return self._shared_step(transition)

        is_offline = action is not None
        cached_previous_state = None if is_offline else self._last_observation_state
        current_state = _current_state_from_observation(state)

        new_transition = transition.copy()
        new_observation = dict(observation)

        if self.joint_representation == "relative":
            previous_state = _previous_state_from_observation(state)
            if previous_state is None and cached_previous_state is not None:
                previous_state = cached_previous_state.to(device=current_state.device, dtype=current_state.dtype)
            relative_state = current_state.clone()
            arm_mask = self._arm_mask(relative_state.shape[-1], device=relative_state.device)

            if previous_state is None:
                relative_state[..., arm_mask] = 0.0
            else:
                relative_state[..., arm_mask] = current_state[..., arm_mask] - previous_state[..., arm_mask]
                previous_pad = _previous_is_pad(transition)
                if isinstance(previous_pad, torch.Tensor):
                    previous_pad = previous_pad.to(device=relative_state.device, dtype=torch.bool)
                    if previous_pad.any():
                        relative_state[..., arm_mask] = torch.where(
                            previous_pad.unsqueeze(-1),
                            torch.zeros_like(relative_state[..., arm_mask]),
                            relative_state[..., arm_mask],
                        )
                elif previous_pad:
                    relative_state[..., arm_mask] = 0.0

            new_observation[OBS_STATE] = relative_state
        else:
            new_observation[OBS_STATE] = current_state

        new_transition[TransitionKey.OBSERVATION] = new_observation

        action = new_transition.get(TransitionKey.ACTION)
        if action is not None and self.joint_representation == "relative":
            action = torch.as_tensor(action)
            relative_action = action.clone()
            arm_mask = self._arm_mask(relative_action.shape[-1], device=relative_action.device)
            current_for_action = current_state.to(device=relative_action.device, dtype=relative_action.dtype)
            relative_action[..., arm_mask] = (
                relative_action[..., arm_mask] - current_for_action[..., None, arm_mask]
                if relative_action.ndim == current_for_action.ndim + 1
                else relative_action[..., arm_mask] - current_for_action[..., arm_mask]
            )
            new_transition[TransitionKey.ACTION] = relative_action

        if not is_offline:
            self._last_observation_state = current_state.detach().clone()
            if self.joint_representation == "relative" and self._action_anchor_state is None:
                self._action_anchor_state = current_state.detach().clone()
                self._action_steps_processed = 0
        return new_transition

    def get_cached_absolute_state(self) -> torch.Tensor | None:
        shared_state = self._shared_step.get_cached_absolute_state()
        if shared_state is not None:
            return shared_state
        return self._action_anchor_state

    def consume_action_steps(self, action_steps: int) -> None:
        if self._shared_step.get_cached_absolute_state() is not None:
            self._shared_step.consume_action_steps(action_steps)
            return
        if self._action_anchor_state is None:
            return
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
        self._shared_step.reset()

    def get_config(self) -> dict[str, Any]:
        return {
            "joint_representation": self.joint_representation,
            "gripper_indices": self.gripper_indices,
            "action_names": self.action_names,
            "joint_limit_profile": self.joint_limit_profile,
            "precomputed_relative_chunk": self.precomputed_relative_chunk,
            "condition_on_state": self.condition_on_state,
            "execution_horizon": self.execution_horizon,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="pi05_absolute_action_processor")
@dataclass
class Pi05AbsoluteActionProcessorStep(ProcessorStep):
    joint_representation: str = "absolute"
    gripper_indices: list[int] = field(default_factory=lambda: [6, 13])
    joint_limit_profile: str = "piper_pika_14d"
    relative_step: Pi05JointRepresentationProcessorStep | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.joint_representation not in {"absolute", "relative"}:
            raise ValueError(f"Unsupported joint_representation: {self.joint_representation}")

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition

        action = torch.as_tensor(action)
        absolute_action = action.clone()

        if self.joint_representation == "relative":
            if self.relative_step is None:
                raise RuntimeError("Pi05AbsoluteActionProcessorStep requires relative_step in relative mode")
            cached_state = self.relative_step.get_cached_absolute_state()
            if cached_state is None:
                raise RuntimeError("Pi05AbsoluteActionProcessorStep needs a cached absolute state")
            cached_state = cached_state.to(device=absolute_action.device, dtype=absolute_action.dtype)
            arm_mask = build_arm_mask(
                absolute_action.shape[-1],
                gripper_indices=self.gripper_indices,
            ).to(device=absolute_action.device)
            absolute_action[..., arm_mask] = (
                absolute_action[..., arm_mask] + cached_state[..., None, arm_mask]
                if absolute_action.ndim == cached_state.ndim + 1
                else absolute_action[..., arm_mask] + cached_state[..., arm_mask]
            )

        min_val, max_val = _profile_limits(self.joint_limit_profile)
        min_val = _reshape_dim_vector(min_val[: absolute_action.shape[-1]], absolute_action)
        max_val = _reshape_dim_vector(max_val[: absolute_action.shape[-1]], absolute_action)
        absolute_action = torch.clamp(absolute_action, min=min_val, max=max_val)

        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = absolute_action
        if self.joint_representation == "relative":
            assert self.relative_step is not None
            action_steps = action.shape[-2] if action.ndim == cached_state.ndim + 1 else 1
            self.relative_step.consume_action_steps(action_steps)
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {
            "joint_representation": self.joint_representation,
            "gripper_indices": self.gripper_indices,
            "joint_limit_profile": self.joint_limit_profile,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
