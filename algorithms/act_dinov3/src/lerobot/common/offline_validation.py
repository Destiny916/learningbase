#!/usr/bin/env python

"""Deterministic, side-effect-free offline policy validation."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import torch
from torch import Tensor

from lerobot.configs import FeatureType, NormalizationMode
from lerobot.processor.converters import create_transition
from lerobot.processor.normalize_processor import UnnormalizerProcessorStep
from lerobot.processor.end_effector_pose_processor import (
    PoseQuantileUnnormalizerProcessorStep,
    absolute_pose10d,
)
from lerobot.types import TransitionKey
from lerobot.utils.constants import ACTION, OBS_STATE

_BASE_METRIC_KEYS = (
    "valid/loss",
    "valid/gripper_loss",
    "valid/action_mse",
    "valid/gripper_mse",
)


class SampleMetricAccumulator:
    """Accumulate per-sample metrics equally across training micro-batches and ranks."""

    def __init__(self, metric_name: str):
        self.metric_name = metric_name
        self.numerator: Tensor | None = None
        self.denominator: Tensor | None = None

    def update(self, details: dict[str, Any]) -> None:
        per_sample = details[f"{self.metric_name}_per_sample"].detach().double()
        valid_samples = details[f"{self.metric_name}_count_per_sample"].detach() > 0
        numerator = per_sample[valid_samples].sum()
        denominator = valid_samples.sum().to(dtype=torch.float64)
        if self.numerator is None:
            self.numerator = numerator
            self.denominator = denominator
        else:
            self.numerator = self.numerator + numerator
            self.denominator = self.denominator + denominator

    @property
    def has_data(self) -> bool:
        return self.numerator is not None

    def compute_global(self, accelerator) -> float:
        if self.numerator is None or self.denominator is None:
            raise ValueError(f"no {self.metric_name} metric parts were accumulated")
        parts = torch.stack((self.numerator, self.denominator))
        global_parts = accelerator.reduce(parts, reduction="sum")
        return (global_parts[0] / global_parts[1].clamp_min(1)).item()


def stable_validation_seed(base_seed: int, episode: int, frame: int, purpose: str) -> int:
    """Return a process-stable seed for one validation sample and random purpose."""
    payload = f"{base_seed}:{episode}:{frame}:{purpose}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)


def make_inference_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """Remove action labels before invoking a policy's prediction path."""
    return {key: value for key, value in batch.items() if key not in {ACTION, "action_is_pad"}}


def make_pi05_validation_randomness(
    policy,
    sample_ids: Sequence[tuple[int, int]],
    *,
    seed: int,
    device: torch.device | str,
) -> tuple[Tensor, Tensor, Tensor]:
    """Create PI0.5 validation randomness keyed only by stable sample identity."""
    shape = (policy.config.chunk_size, policy.config.max_action_dim)
    flow_noise: list[Tensor] = []
    flow_time: list[Tensor] = []
    initial_noise: list[Tensor] = []

    for episode, frame in sample_ids:
        episode = int(episode)
        frame = int(frame)
        flow_generator = torch.Generator(device=device).manual_seed(
            stable_validation_seed(seed, episode, frame, "flow_noise")
        )
        initial_generator = torch.Generator(device=device).manual_seed(
            stable_validation_seed(seed, episode, frame, "initial_noise")
        )
        flow_noise.append(torch.randn(shape, generator=flow_generator, device=device))
        initial_noise.append(torch.randn(shape, generator=initial_generator, device=device))

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(stable_validation_seed(seed, episode, frame, "flow_time"))
            flow_time.append(policy.model.sample_time(1, device))

    return torch.stack(flow_noise), torch.cat(flow_time), torch.stack(initial_noise)


def physical_action_mse_parts(
    predicted: Tensor,
    target: Tensor,
    action_is_pad: Tensor,
    *,
    gripper_indices: Sequence[int],
) -> dict[str, Tensor]:
    """Return per-sample physical-space squared-error numerators and counts."""
    if predicted.shape != target.shape:
        raise ValueError(
            f"predicted and target action shapes must match, got {tuple(predicted.shape)} and "
            f"{tuple(target.shape)}"
        )
    if predicted.ndim != 3:
        raise ValueError(f"actions must have shape (batch, horizon, dim), got {tuple(predicted.shape)}")
    if action_is_pad.shape != predicted.shape[:2]:
        raise ValueError(
            f"action_is_pad must have shape {tuple(predicted.shape[:2])}, got {tuple(action_is_pad.shape)}"
        )
    action_dim = predicted.shape[-1]
    if (
        not gripper_indices
        or len(set(gripper_indices)) != len(gripper_indices)
        or any(type(index) is not int or index < 0 or index >= action_dim for index in gripper_indices)
    ):
        raise ValueError(
            f"gripper indices must be unique integers within [0, {action_dim}), got {gripper_indices}"
        )

    valid_steps = ~action_is_pad.to(device=predicted.device, dtype=torch.bool)
    squared_error = (predicted - target).square()
    valid_actions = valid_steps.unsqueeze(-1).expand_as(squared_error)
    gripper_error = squared_error[:, :, list(gripper_indices)]
    valid_grippers = valid_steps.unsqueeze(-1).expand_as(gripper_error)
    return {
        "action_mse_sum_per_sample": (squared_error * valid_actions).sum(dim=(1, 2)),
        "action_mse_count_per_sample": valid_actions.sum(dim=(1, 2)),
        "gripper_mse_sum_per_sample": (gripper_error * valid_grippers).sum(dim=(1, 2)),
        "gripper_mse_count_per_sample": valid_grippers.sum(dim=(1, 2)),
        "dimension_mse_sum_per_sample": (squared_error * valid_actions).sum(dim=1),
        "dimension_mse_count_per_sample": valid_actions.sum(dim=1),
    }


def restore_absolute_joint_actions(
    relative_actions: Tensor,
    absolute_state: Tensor,
    *,
    gripper_indices: Sequence[int],
) -> Tensor:
    """Restore absolute arm targets from chunk-relative actions and the current absolute state."""
    if relative_actions.ndim != 3:
        raise ValueError(
            f"relative actions must have shape (batch, horizon, dim), got {tuple(relative_actions.shape)}"
        )
    if absolute_state.ndim == 3 and absolute_state.shape[-2] == 2:
        current_state = absolute_state[:, -1]
    elif absolute_state.ndim == 2:
        current_state = absolute_state
    else:
        raise ValueError(
            "absolute observation.state must have shape (batch, dim) or paired shape (batch, 2, dim), "
            f"got {tuple(absolute_state.shape)}"
        )
    if current_state.shape != (relative_actions.shape[0], relative_actions.shape[-1]):
        raise ValueError(
            "current absolute state must match action batch and dimension, got "
            f"{tuple(current_state.shape)} for actions {tuple(relative_actions.shape)}"
        )

    action_dim = relative_actions.shape[-1]
    if (
        not gripper_indices
        or len(set(gripper_indices)) != len(gripper_indices)
        or any(type(index) is not int or index < 0 or index >= action_dim for index in gripper_indices)
    ):
        raise ValueError(
            f"gripper indices must be unique integers within [0, {action_dim}), got {gripper_indices}"
        )

    arm_mask = torch.ones(action_dim, dtype=torch.bool, device=relative_actions.device)
    arm_mask[list(gripper_indices)] = False
    absolute_actions = relative_actions.clone()
    anchor = current_state.to(device=relative_actions.device, dtype=relative_actions.dtype)
    absolute_actions[..., arm_mask] += anchor[:, None, arm_mask]
    return absolute_actions


def restore_absolute_pose_actions(relative_actions: Tensor, absolute_state: Tensor) -> Tensor:
    """Restore absolute pose10d chunk targets through SE(3) composition."""
    if relative_actions.ndim != 3 or relative_actions.shape[-1] != 10:
        raise ValueError("relative pose actions must have shape (batch, horizon, 10)")
    if absolute_state.ndim == 3 and absolute_state.shape[-2:] == (2, 10):
        anchor = absolute_state[:, -1]
    elif absolute_state.ndim == 2 and absolute_state.shape[-1] == 10:
        anchor = absolute_state
    else:
        raise ValueError("absolute pose state must have shape (batch, 10) or (batch, 2, 10)")
    return absolute_pose10d(anchor.to(device=relative_actions.device, dtype=relative_actions.dtype), relative_actions)


def reduce_metric_parts(
    accelerator,
    parts: Iterable[tuple[Tensor, Tensor]],
) -> float:
    """Gather per-sample metric parts and compute one strict global ratio."""
    numerator = torch.zeros((), dtype=torch.float64, device=accelerator.device) if hasattr(
        accelerator, "device"
    ) else torch.zeros((), dtype=torch.float64)
    denominator = torch.zeros_like(numerator)
    found = False
    for per_sample_numerator, per_sample_denominator in parts:
        found = True
        gathered_numerator = accelerator.gather_for_metrics(per_sample_numerator)
        gathered_denominator = accelerator.gather_for_metrics(per_sample_denominator)
        numerator = numerator + gathered_numerator.double().sum().to(numerator.device)
        denominator = denominator + gathered_denominator.double().sum().to(denominator.device)
    if not found:
        raise ValueError("validation dataloader produced no metric parts")
    return (numerator / denominator.clamp_min(1)).item()


def _gather_metric_parts(accelerator, numerator: Tensor, denominator: Tensor) -> tuple[Tensor, Tensor]:
    """Gather one dataloader batch while Accelerate can still trim repeated tail samples."""
    gathered_numerator = accelerator.gather_for_metrics(numerator)
    gathered_denominator = accelerator.gather_for_metrics(denominator)
    return gathered_numerator.double().sum(), gathered_denominator.double().sum()


def make_action_unnormalizer(postprocessor) -> Callable[[Tensor], Tensor]:
    """Extract only q01/q99 action unnormalization from an inference postprocessor."""
    steps = [
        step
        for step in postprocessor.steps
        if isinstance(step, (UnnormalizerProcessorStep, PoseQuantileUnnormalizerProcessorStep))
    ]
    if len(steps) != 1:
        raise ValueError(f"expected exactly one action unnormalizer step, found {len(steps)}")
    step = steps[0]
    if isinstance(step, PoseQuantileUnnormalizerProcessorStep):
        def unnormalize_pose(action: Tensor) -> Tensor:
            result = step(create_transition(action=action))[TransitionKey.ACTION]
            if not isinstance(result, Tensor):
                raise TypeError(f"action unnormalizer returned {type(result)}, expected torch.Tensor")
            return result
        return unnormalize_pose
    action_stats = (step.stats or {}).get(ACTION, {})
    if (
        step.norm_map.get(FeatureType.ACTION) is not NormalizationMode.QUANTILES
        or "q01" not in action_stats
        or "q99" not in action_stats
    ):
        raise ValueError("physical action MSE requires q01/q99 QUANTILES action unnormalization")

    def unnormalize(action: Tensor) -> Tensor:
        transition = step(create_transition(action=action))
        result = transition[TransitionKey.ACTION]
        if not isinstance(result, Tensor):
            raise TypeError(f"action unnormalizer returned {type(result)}, expected torch.Tensor")
        return result

    return unnormalize


def _sample_ids(batch: dict[str, Any]) -> list[tuple[int, int]]:
    try:
        episodes = batch["episode_index"]
        frames = batch["frame_index"]
    except KeyError as error:
        raise KeyError("validation batches require episode_index and frame_index") from error
    if isinstance(episodes, Tensor):
        episodes = episodes.detach().cpu().tolist()
    if isinstance(frames, Tensor):
        frames = frames.detach().cpu().tolist()
    return [(int(episode), int(frame)) for episode, frame in zip(episodes, frames, strict=True)]


def _policy_type(policy) -> str:
    policy_type = getattr(policy.config, "type", None)
    if policy_type not in {"pi05", "act"}:
        raise ValueError(f"offline validation supports only pi05 and act policies, got {policy_type!r}")
    return policy_type


def _gripper_indices(policy) -> list[int]:
    config = policy.config
    if hasattr(config, "joint_gripper_indices"):
        return list(config.joint_gripper_indices)
    return list(config.gripper_indices)


def _is_pose10d_policy(policy) -> bool:
    """Identify pose10d by the configured action feature, not by its representation."""
    output_features = getattr(policy.config, "output_features", {})
    action_feature = output_features.get(ACTION) if isinstance(output_features, dict) else None
    if action_feature is not None and tuple(action_feature.shape) == (10,):
        return True
    return getattr(policy.config, "end_effector_pose_representation", "absolute") == "relative"


def _joint_metric_specs(
    policy,
    action_dim: int,
    *,
    gripper_indices: Sequence[int],
) -> list[tuple[int, str, str, str]]:
    """Return physical MSE and RMSE metric names for every non-gripper action dimension."""
    feature_names = getattr(policy.config, "action_feature_names", None)
    if not isinstance(feature_names, list) or len(feature_names) != action_dim:
        feature_names = [f"joint_{index}" for index in range(action_dim)]

    gripper_set = set(gripper_indices)
    return [
        (
            index,
            f"valid/{feature_name}_mse_rad2",
            f"valid/{feature_name}_rmse_rad",
            f"valid/{feature_name}_rmse_deg",
        )
        for index, feature_name in enumerate(feature_names)
        if index not in gripper_set
    ]


def _gripper_metric_specs(
    policy,
    action_dim: int,
    *,
    gripper_indices: Sequence[int],
) -> list[tuple[int, str, str, str]]:
    """Return physical MSE and RMSE metric names for each configured gripper."""
    feature_names = getattr(policy.config, "action_feature_names", None)
    if not isinstance(feature_names, list) or len(feature_names) != action_dim:
        feature_names = [f"joint_{index}" for index in range(action_dim)]
        for index in gripper_indices:
            feature_names[index] = "gripper" if len(gripper_indices) == 1 else f"gripper_{index}"

    return [
        (
            index,
            f"valid/{feature_names[index]}_mse",
            f"valid/{feature_names[index]}_rmse_m",
            f"valid/{feature_names[index]}_rmse_mm",
        )
        for index in gripper_indices
    ]


def evaluate_offline(
    policy,
    eval_dataloader,
    preprocessor,
    accelerator,
    *,
    action_unnormalizer: Callable[[Tensor], Tensor],
    seed: int,
    camera_keys: Sequence[str] = (),
) -> dict[str, float]:
    """Evaluate normalized loss and physical action MSE without changing training state."""
    unwrapped_policy = accelerator.unwrap_model(policy)
    policy_type = _policy_type(unwrapped_policy)
    pose_mode = _is_pose10d_policy(unwrapped_policy)
    relative_pose_mode = pose_mode and getattr(
        unwrapped_policy.config, "end_effector_pose_representation", "absolute"
    ) == "relative"
    relative_joint_mode = not pose_mode and getattr(
        unwrapped_policy.config, "joint_representation", "relative"
    ) == "relative"
    was_training = policy.training
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    metric_keys = ("valid/loss", "valid/gripper_loss", "valid/gripper_mse") if pose_mode else _BASE_METRIC_KEYS
    metric_totals: dict[str, tuple[Tensor, Tensor] | None] = {key: None for key in metric_keys}
    joint_metric_specs: list[tuple[int, str, str, str]] | None = None
    gripper_metric_specs: list[tuple[int, str, str, str]] | None = None

    policy.eval()
    try:
        with torch.no_grad():
            for raw_batch in eval_dataloader:
                sample_ids = _sample_ids(raw_batch)
                absolute_state = raw_batch.get(OBS_STATE)
                if not isinstance(absolute_state, Tensor):
                    raise KeyError(
                        "validation requires raw absolute observation.state to restore relative arm actions"
                    )
                batch = dict(raw_batch)
                for camera_key in camera_keys:
                    image = batch.get(camera_key)
                    if isinstance(image, Tensor) and image.dtype == torch.uint8:
                        batch[camera_key] = image.to(dtype=torch.float32) / 255.0
                processed_batch = preprocessor(batch)
                target_normalized = processed_batch[ACTION]

                with accelerator.autocast():
                    if policy_type == "pi05":
                        flow_noise, flow_time, initial_noise = make_pi05_validation_randomness(
                            unwrapped_policy,
                            sample_ids,
                            seed=seed,
                            device=target_normalized.device,
                        )
                        _, details = policy(
                            processed_batch,
                            reduction="none",
                            noise=flow_noise,
                            time=flow_time,
                        )
                        predicted_normalized = policy(
                            make_inference_batch(processed_batch),
                            noise=initial_noise,
                            return_action_chunk=True,
                        )
                    else:
                        _, details = policy(processed_batch, reduction="none")
                        predicted_normalized = policy(
                            make_inference_batch(processed_batch),
                            return_action_chunk=True,
                        )

                gripper_indices = [9] if pose_mode else _gripper_indices(unwrapped_policy)
                predicted_actions = action_unnormalizer(predicted_normalized.float())
                target_actions = action_unnormalizer(target_normalized.float())
                if relative_pose_mode:
                    predicted_physical = restore_absolute_pose_actions(predicted_actions, absolute_state)
                    target_physical = restore_absolute_pose_actions(target_actions, absolute_state)
                elif relative_joint_mode:
                    predicted_physical = restore_absolute_joint_actions(
                        predicted_actions, absolute_state, gripper_indices=gripper_indices
                    )
                    target_physical = restore_absolute_joint_actions(
                        target_actions, absolute_state, gripper_indices=gripper_indices
                    )
                else:
                    predicted_physical = predicted_actions
                    target_physical = target_actions
                mse_parts = physical_action_mse_parts(
                    predicted_physical,
                    target_physical,
                    processed_batch["action_is_pad"],
                    gripper_indices=gripper_indices,
                )
                if joint_metric_specs is None and not pose_mode:
                    joint_metric_specs = _joint_metric_specs(
                        unwrapped_policy,
                        predicted_physical.shape[-1],
                        gripper_indices=gripper_indices,
                    )
                    metric_totals.update({mse_key: None for _, mse_key, _, _ in joint_metric_specs})
                if gripper_metric_specs is None and not pose_mode:
                    gripper_metric_specs = _gripper_metric_specs(
                        unwrapped_policy,
                        predicted_physical.shape[-1],
                        gripper_indices=gripper_indices,
                    )
                    metric_totals.update({mse_key: None for _, mse_key, _, _ in gripper_metric_specs})
                batch_metric_parts = {
                    "valid/loss": (
                        details["loss_sum_per_sample"],
                        details["loss_count_per_sample"],
                    ),
                    "valid/gripper_loss": (
                        details["gripper_loss_sum_per_sample"],
                        details["gripper_loss_count_per_sample"],
                    ),
                    "valid/gripper_mse": (
                        mse_parts["gripper_mse_sum_per_sample"],
                        mse_parts["gripper_mse_count_per_sample"],
                    ),
                }
                if not pose_mode:
                    batch_metric_parts["valid/action_mse"] = (
                        mse_parts["action_mse_sum_per_sample"], mse_parts["action_mse_count_per_sample"]
                    )
                for index, mse_key, _, _ in joint_metric_specs or []:
                    batch_metric_parts[mse_key] = (
                        mse_parts["dimension_mse_sum_per_sample"][:, index],
                        mse_parts["dimension_mse_count_per_sample"][:, index],
                    )
                for index, mse_key, _, _ in gripper_metric_specs or []:
                    batch_metric_parts[mse_key] = (
                        mse_parts["dimension_mse_sum_per_sample"][:, index],
                        mse_parts["dimension_mse_count_per_sample"][:, index],
                    )
                for metric_name, (numerator, denominator) in batch_metric_parts.items():
                    gathered = _gather_metric_parts(accelerator, numerator, denominator)
                    totals = metric_totals[metric_name]
                    if totals is None:
                        metric_totals[metric_name] = gathered
                    else:
                        metric_totals[metric_name] = (
                            totals[0] + gathered[0].to(totals[0].device),
                            totals[1] + gathered[1].to(totals[1].device),
                        )

        metrics: dict[str, float] = {}
        for metric_name, totals in metric_totals.items():
            if totals is None:
                raise ValueError("validation dataloader produced no batches")
            numerator, denominator = totals
            metrics[metric_name] = (numerator / denominator.clamp_min(1)).item()
        metrics["valid/gripper_rmse_m"] = math.sqrt(metrics["valid/gripper_mse"])
        metrics["valid/gripper_rmse_mm"] = metrics["valid/gripper_rmse_m"] * 1000.0
        if joint_metric_specs is None and not pose_mode:
            raise ValueError("validation dataloader produced no batches")
        for _, mse_key, rmse_rad_key, rmse_deg_key in joint_metric_specs or []:
            rmse_rad = math.sqrt(metrics[mse_key])
            metrics[rmse_rad_key] = rmse_rad
            metrics[rmse_deg_key] = math.degrees(rmse_rad)
        for _, mse_key, rmse_m_key, rmse_mm_key in gripper_metric_specs or []:
            rmse_m = math.sqrt(metrics[mse_key])
            metrics[rmse_m_key] = rmse_m
            metrics[rmse_mm_key] = rmse_m * 1000.0
        return metrics
    finally:
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        policy.train(was_training)
