"""Relative SE(3) processing for the isolated right end-effector pose workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import ACTION, OBS_STATE

from .pipeline import ProcessorStep, ProcessorStepRegistry


POSE_DIM = 10
ROT6D_SLICE = slice(3, 9)
GRIPPER_INDEX = 9
SCALED_INDICES = (0, 1, 2, 9)


def _json_safe_config(value: Any) -> Any:
    """Convert tensor and NumPy-backed statistics into JSON-compatible values."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _json_safe_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_config(item) for item in value]
    tolist = getattr(value, "tolist", None)
    return _json_safe_config(tolist()) if callable(tolist) else value


def _validate_pose(tensor: torch.Tensor, name: str) -> None:
    if tensor.shape[-1] != POSE_DIM:
        raise ValueError(f"{name} must have {POSE_DIM} dimensions, got {tensor.shape[-1]}")


def _normalize(vector: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    norm = vector.norm(dim=-1, keepdim=True)
    return torch.where(norm > 1e-8, vector / norm.clamp_min(1e-8), fallback.expand_as(vector))


def rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """Project continuous 6D orientation vectors onto SO(3)."""
    if rot6d.shape[-1] != 6:
        raise ValueError(f"rot6d must have six dimensions, got {rot6d.shape[-1]}")
    first_fallback = torch.zeros_like(rot6d[..., :3])
    first_fallback[..., 0] = 1.0
    first = _normalize(rot6d[..., :3], first_fallback)
    second = rot6d[..., 3:] - (first * rot6d[..., 3:]).sum(dim=-1, keepdim=True) * first
    basis = torch.eye(3, device=rot6d.device, dtype=rot6d.dtype)
    fallback_index = first.abs().argmin(dim=-1)
    fallback_second = basis[fallback_index]
    fallback_second = fallback_second - (first * fallback_second).sum(dim=-1, keepdim=True) * first
    second = _normalize(second, _normalize(fallback_second, first_fallback))
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def matrix_to_rot6d(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"rotation matrix must end in (3, 3), got {tuple(matrix.shape)}")
    return torch.cat((matrix[..., :, 0], matrix[..., :, 1]), dim=-1)


def _axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    if axis_angle.shape[-1] != 3:
        raise ValueError(f"axis_angle must have three dimensions, got {axis_angle.shape[-1]}")
    x, y, z = axis_angle.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    skew = torch.stack(
        (
            zeros,
            -z,
            y,
            z,
            zeros,
            -x,
            -y,
            x,
            zeros,
        ),
        dim=-1,
    ).reshape(*axis_angle.shape[:-1], 3, 3)
    angle_squared = (axis_angle * axis_angle).sum(dim=-1, keepdim=True)
    angle = angle_squared.sqrt()
    safe_angle = angle.clamp_min(1e-8)
    first = (torch.sin(angle) / safe_angle).unsqueeze(-1)
    second = ((1.0 - torch.cos(angle)) / angle_squared.clamp_min(1e-8)).unsqueeze(-1)
    first = torch.where(angle.unsqueeze(-1) > 1e-4, first, torch.ones_like(first))
    second = torch.where(angle.unsqueeze(-1) > 1e-4, second, torch.full_like(second, 0.5))
    identity = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype).expand_as(skew)
    return identity + first * skew + second * (skew @ skew)


def relative_pose10d(anchor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return ``inv(T_anchor) @ T_target`` and retain target's absolute gripper."""
    _validate_pose(anchor, "anchor pose")
    _validate_pose(target, "target pose")
    while anchor.ndim < target.ndim:
        anchor = anchor.unsqueeze(-2)
    anchor_rotation = rot6d_to_matrix(anchor[..., ROT6D_SLICE])
    target_rotation = rot6d_to_matrix(target[..., ROT6D_SLICE])
    anchor_rotation_transpose = anchor_rotation.transpose(-1, -2)
    translation = (anchor_rotation_transpose @ (target[..., :3] - anchor[..., :3]).unsqueeze(-1)).squeeze(-1)
    rotation = anchor_rotation_transpose @ target_rotation
    return torch.cat((translation, matrix_to_rot6d(rotation), target[..., GRIPPER_INDEX : GRIPPER_INDEX + 1]), dim=-1)


def absolute_pose10d(anchor: torch.Tensor, relative: torch.Tensor) -> torch.Tensor:
    """Compose a relative SE(3) target with its absolute pose anchor."""
    _validate_pose(anchor, "anchor pose")
    _validate_pose(relative, "relative pose")
    while anchor.ndim < relative.ndim:
        anchor = anchor.unsqueeze(-2)
    anchor_rotation = rot6d_to_matrix(anchor[..., ROT6D_SLICE])
    relative_rotation = rot6d_to_matrix(relative[..., ROT6D_SLICE])
    translation = anchor[..., :3] + (anchor_rotation @ relative[..., :3].unsqueeze(-1)).squeeze(-1)
    rotation = anchor_rotation @ relative_rotation
    return torch.cat(
        (translation, matrix_to_rot6d(rotation), relative[..., GRIPPER_INDEX : GRIPPER_INDEX + 1]), dim=-1
    )


def _identity_relative_pose(current: torch.Tensor) -> torch.Tensor:
    relative = torch.zeros_like(current)
    relative[..., 3] = 1.0
    relative[..., 7] = 1.0
    relative[..., GRIPPER_INDEX] = current[..., GRIPPER_INDEX]
    return relative


def _has_action(transition: EnvTransition) -> bool:
    action = transition.get(TransitionKey.ACTION)
    return isinstance(action, torch.Tensor) and action.numel() > 0


@ProcessorStepRegistry.register("pose_state_noise_processor")
@dataclass
class PoseStateNoiseProcessorStep(ProcessorStep):
    """Add train-only geometric noise to a pose10d observation state."""

    position_std_m: float = 0.0
    rotation_std_rad: float = 0.0
    gripper_std_m: float = 0.0
    enabled: bool = True
    _training: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if min(self.position_std_m, self.rotation_std_rad, self.gripper_std_m) < 0:
            raise ValueError("pose state noise standard deviations must be non-negative")

    def train(self) -> None:
        self._training = True

    def eval(self) -> None:
        self._training = False

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled or not self._training:
            return transition
        if self.position_std_m == 0 and self.rotation_std_rad == 0 and self.gripper_std_m == 0:
            return transition
        observation = transition.get(TransitionKey.OBSERVATION)
        if observation is None or OBS_STATE not in observation:
            return transition
        state = observation[OBS_STATE]
        if not isinstance(state, torch.Tensor):
            raise ValueError("observation.state must be a torch.Tensor")
        _validate_pose(state, "observation.state")

        noisy_state = state.clone()
        if self.position_std_m > 0:
            noisy_state[..., :3] += torch.randn_like(noisy_state[..., :3]) * self.position_std_m
        if self.rotation_std_rad > 0:
            rotation_noise = torch.randn_like(noisy_state[..., :3]) * self.rotation_std_rad
            noisy_rotation = _axis_angle_to_matrix(rotation_noise) @ rot6d_to_matrix(noisy_state[..., ROT6D_SLICE])
            noisy_state[..., ROT6D_SLICE] = matrix_to_rot6d(noisy_rotation)
        if self.gripper_std_m > 0:
            noisy_state[..., GRIPPER_INDEX] += torch.randn_like(noisy_state[..., GRIPPER_INDEX]) * self.gripper_std_m

        new_transition = transition.copy()
        new_observation = observation.copy()
        new_observation[OBS_STATE] = noisy_state
        new_transition[TransitionKey.OBSERVATION] = new_observation
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {
            "position_std_m": self.position_std_m,
            "rotation_std_rad": self.rotation_std_rad,
            "gripper_std_m": self.gripper_std_m,
            "enabled": self.enabled,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("relative_end_effector_pose_processor")
@dataclass
class RelativePoseProcessorStep(ProcessorStep):
    """Create relative pose10d state and action chunks from absolute end-effector poses."""

    execution_horizon: int = 1
    _last_observation_state: torch.Tensor | None = field(default=None, init=False, repr=False)
    _action_anchor_state: torch.Tensor | None = field(default=None, init=False, repr=False)
    _action_steps_processed: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.execution_horizon, bool) or not isinstance(self.execution_horizon, int):
            raise ValueError("execution_horizon must be a positive integer")
        if self.execution_horizon <= 0:
            raise ValueError("execution_horizon must be a positive integer")

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION)
        if observation is None or OBS_STATE not in observation:
            return transition
        state = observation[OBS_STATE]
        if not isinstance(state, torch.Tensor):
            raise ValueError("observation.state must be a torch.Tensor")
        return self._offline(transition, state) if _has_action(transition) else self._online(transition, state)

    def _offline(self, transition: EnvTransition, paired_state: torch.Tensor) -> EnvTransition:
        if paired_state.ndim < 2 or paired_state.shape[-2] != 2:
            raise ValueError("offline observation.state must contain previous and current frames in [..., 2, 10]")
        _validate_pose(paired_state, "offline observation.state")
        action = transition[TransitionKey.ACTION]
        assert isinstance(action, torch.Tensor)
        _validate_pose(action, "action")
        previous, current = paired_state.unbind(dim=-2)
        relative_state = relative_pose10d(previous, current)
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        state_is_pad = complementary_data.get(f"{OBS_STATE}_is_pad")
        if state_is_pad is not None:
            if not isinstance(state_is_pad, torch.Tensor) or state_is_pad.shape != paired_state.shape[:-1]:
                raise ValueError("observation.state_is_pad must match paired observation.state dimensions")
            previous_is_pad = state_is_pad[..., 0].to(device=current.device, dtype=torch.bool)
            if previous_is_pad.any():
                identity = _identity_relative_pose(current)
                relative_state = torch.where(previous_is_pad.unsqueeze(-1), identity, relative_state)
        new_transition = transition.copy()
        new_observation = observation = transition[TransitionKey.OBSERVATION]
        assert observation is not None
        new_observation = observation.copy()
        new_observation[OBS_STATE] = relative_state
        new_transition[TransitionKey.OBSERVATION] = new_observation
        new_transition[TransitionKey.ACTION] = relative_pose10d(current, action)
        return new_transition

    def _online(self, transition: EnvTransition, current: torch.Tensor) -> EnvTransition:
        _validate_pose(current, "observation.state")
        if self._last_observation_state is None:
            relative_state = _identity_relative_pose(current)
        else:
            if self._last_observation_state.shape != current.shape:
                raise ValueError("cached absolute observation.state shape does not match the current observation.state")
            previous = self._last_observation_state.to(device=current.device, dtype=current.dtype)
            relative_state = relative_pose10d(previous, current)
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
        return self._action_anchor_state

    def consume_action_steps(self, action_steps: int) -> None:
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
        return {"execution_horizon": self.execution_horizon}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("relative_end_effector_pose_absolute_action_processor")
@dataclass
class RelativePoseAbsoluteActionProcessorStep(ProcessorStep):
    """Restore absolute pose10d targets from a paired relative pose processor cache."""

    relative_step: RelativePoseProcessorStep | None = field(default=None, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition
        if not isinstance(action, torch.Tensor):
            raise ValueError("action must be a torch.Tensor")
        _validate_pose(action, "action")
        if self.relative_step is None:
            raise RuntimeError("RelativePoseAbsoluteActionProcessorStep requires a paired RelativePoseProcessorStep")
        anchor = self.relative_step.get_cached_absolute_state()
        if anchor is None:
            raise RuntimeError("RelativePoseAbsoluteActionProcessorStep requires a cached absolute pose anchor")
        absolute_action = absolute_pose10d(anchor.to(device=action.device, dtype=action.dtype), action)
        action_steps = action.shape[-2] if action.ndim == anchor.ndim + 1 else 1
        self.relative_step.consume_action_steps(action_steps)
        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = absolute_action
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def _pose_quantile_transform(
    tensor: torch.Tensor, stats: dict[str, Any], *, inverse: bool, clip_quantiles: bool
) -> torch.Tensor:
    _validate_pose(tensor, "pose tensor")
    if not isinstance(stats, dict) or "q01" not in stats or "q99" not in stats:
        raise ValueError("pose quantile stats must contain q01 and q99")
    q01 = torch.as_tensor(stats["q01"], device=tensor.device, dtype=tensor.dtype)
    q99 = torch.as_tensor(stats["q99"], device=tensor.device, dtype=tensor.dtype)
    if q01.shape != (POSE_DIM,) or q99.shape != (POSE_DIM,):
        raise ValueError("pose quantile q01/q99 must each have shape (10,)")
    indices = torch.as_tensor(SCALED_INDICES, device=tensor.device)
    output = tensor.clone()
    q01_scaled = q01[indices]
    q99_scaled = q99[indices]
    if inverse:
        output[..., indices] = 0.5 * (output[..., indices] + 1.0) * (q99_scaled - q01_scaled) + q01_scaled
    else:
        normalized = 2.0 * (output[..., indices] - q01_scaled) / (q99_scaled - q01_scaled).clamp_min(1e-8) - 1.0
        output[..., indices] = normalized.clamp(-1.0, 1.0) if clip_quantiles else normalized
    return output


@ProcessorStepRegistry.register("end_effector_pose_quantile_normalizer")
@dataclass
class PoseQuantileNormalizerProcessorStep(ProcessorStep):
    """Quantile-normalize only pose translation and absolute gripper dimensions."""

    stats: dict[str, dict[str, Any]]
    clip_quantiles: bool = True

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        observation = transition.get(TransitionKey.OBSERVATION)
        if observation is not None and OBS_STATE in observation:
            state = observation[OBS_STATE]
            if not isinstance(state, torch.Tensor):
                raise ValueError("observation.state must be a torch.Tensor")
            new_observation = observation.copy()
            new_observation[OBS_STATE] = _pose_quantile_transform(
                state, self.stats.get(OBS_STATE, {}), inverse=False, clip_quantiles=self.clip_quantiles
            )
            new_transition[TransitionKey.OBSERVATION] = new_observation
        action = transition.get(TransitionKey.ACTION)
        if action is not None:
            if not isinstance(action, torch.Tensor):
                raise ValueError("action must be a torch.Tensor")
            new_transition[TransitionKey.ACTION] = _pose_quantile_transform(
                action, self.stats.get(ACTION, {}), inverse=False, clip_quantiles=self.clip_quantiles
            )
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"stats": _json_safe_config(self.stats), "clip_quantiles": self.clip_quantiles}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("end_effector_pose_quantile_unnormalizer")
@dataclass
class PoseQuantileUnnormalizerProcessorStep(ProcessorStep):
    """Undo pose translation/gripper quantile scaling while preserving rot6d."""

    stats: dict[str, dict[str, Any]]

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        action = transition.get(TransitionKey.ACTION)
        if action is not None:
            if not isinstance(action, torch.Tensor):
                raise ValueError("action must be a torch.Tensor")
            new_transition[TransitionKey.ACTION] = _pose_quantile_transform(
                action, self.stats.get(ACTION, {}), inverse=True, clip_quantiles=False
            )
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"stats": _json_safe_config(self.stats)}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
