from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor.pipeline import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION

from w1_simulation.robot.joints import ACT_STATE_JOINTS


@dataclass(frozen=True)
class ActCheckpointContract:
    image_shapes: dict[str, tuple[int, int, int]]
    state_dim: int
    action_dim: int
    prediction_horizon: int
    execution_horizon: int


def _contract_from_config(config: object) -> ActCheckpointContract:
    if getattr(config, "type", None) != "act":
        raise ValueError(f"Expected an ACT checkpoint, got {getattr(config, 'type', None)}")
    image_shapes = {
        key: (int(feature.shape[1]), int(feature.shape[2]), int(feature.shape[0]))
        for key, feature in config.image_features.items()
    }
    if not image_shapes:
        raise ValueError("ACT checkpoint must declare at least one visual input")
    expected_inputs = {"observation.state", *image_shapes}
    if set(config.input_features) != expected_inputs:
        raise ValueError(f"ACT input feature mismatch: {set(config.input_features)}")
    if tuple(config.input_features["observation.state"].shape) != (len(ACT_STATE_JOINTS),):
        raise ValueError("ACT state must be 19D")
    for key, hwc_shape in image_shapes.items():
        if hwc_shape[2] != 3:
            raise ValueError(
                f"ACT simulation requires three-channel images for {key}: {config.input_features[key].shape}"
            )
    if tuple(config.output_features[ACTION].shape) != (len(ACT_STATE_JOINTS),):
        raise ValueError("ACT output must be 19D")
    prediction_horizon = int(config.chunk_size)
    execution_horizon = int(config.n_action_steps)
    if prediction_horizon <= 0:
        raise ValueError(f"ACT prediction horizon must be positive, got {prediction_horizon}")
    if not 1 <= execution_horizon <= prediction_horizon:
        raise ValueError(
            "ACT execution horizon must be within the prediction horizon: "
            f"chunk={prediction_horizon}, n_action_steps={execution_horizon}"
        )
    return ActCheckpointContract(
        image_shapes=image_shapes,
        state_dim=len(ACT_STATE_JOINTS),
        action_dim=len(ACT_STATE_JOINTS),
        prediction_horizon=prediction_horizon,
        execution_horizon=execution_horizon,
    )


def inspect_checkpoint_contract(checkpoint: Path) -> ActCheckpointContract:
    train_config = TrainPipelineConfig.from_pretrained(Path(checkpoint).resolve(), local_files_only=True)
    return _contract_from_config(train_config.policy)


def inspect_checkpoint_image_shapes(checkpoint: Path) -> dict[str, tuple[int, int, int]]:
    return inspect_checkpoint_contract(checkpoint).image_shapes


class ActPolicyRuntime:
    def __init__(
        self,
        checkpoint: Path,
        device: str = "cuda:0",
        execution_horizon: int | None = None,
    ) -> None:
        self.checkpoint = checkpoint.resolve()
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("ACT simulation inference requires CUDA and does not fall back to CPU")
        if not (self.checkpoint / "model.safetensors").is_file():
            raise FileNotFoundError(self.checkpoint / "model.safetensors")
        train_config = TrainPipelineConfig.from_pretrained(self.checkpoint, local_files_only=True)
        self.config = train_config.policy
        self.config.device = str(self.device)
        self.contract = _contract_from_config(self.config)
        self.execution_horizon = (
            self.contract.execution_horizon if execution_horizon is None else int(execution_horizon)
        )
        if isinstance(execution_horizon, bool) or not (
            1 <= self.execution_horizon <= self.contract.prediction_horizon
        ):
            raise ValueError(
                "ACT runtime execution horizon must be within the prediction horizon: "
                f"prediction={self.contract.prediction_horizon}, execution={execution_horizon}"
            )
        self.image_shapes = self.contract.image_shapes
        self.image_keys = tuple(self.image_shapes)
        self.policy = ACTPolicy.from_pretrained(
            self.checkpoint,
            config=self.config,
            local_files_only=True,
            strict=True,
        )
        self.policy.to(self.device).eval()
        self.preprocessor = PolicyProcessorPipeline.from_pretrained(
            self.checkpoint,
            config_filename="policy_preprocessor.json",
            local_files_only=True,
        )
        self.postprocessor = PolicyProcessorPipeline.from_pretrained(
            self.checkpoint,
            config_filename="policy_postprocessor.json",
            local_files_only=True,
        )
        self._set_processor_devices()
        self._lock = threading.Lock()
        self.last_latency_ms = 0.0
        self.last_model_ms = 0.0

    def _set_processor_devices(self) -> None:
        for step in getattr(self.preprocessor, "steps", []):
            if step.__class__.__name__ == "DeviceProcessorStep":
                step.device = str(self.device)
            elif step.__class__.__name__ == "NormalizerProcessorStep" and hasattr(step, "to"):
                step.to(device=self.device)
        for step in getattr(self.postprocessor, "steps", []):
            if step.__class__.__name__ == "DeviceProcessorStep":
                step.device = "cpu"
            elif step.__class__.__name__ == "UnnormalizerProcessorStep" and hasattr(step, "to"):
                step.to(device="cpu")

    def predict_chunk(self, state: np.ndarray, images: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
        state_array = np.asarray(state, dtype=np.float32)
        if state_array.shape != (len(ACT_STATE_JOINTS),) or not np.isfinite(state_array).all():
            raise ValueError(f"Expected finite 19D ACT state, got {state_array.shape}")
        if set(images) != set(self.image_keys):
            raise ValueError(f"ACT image keys mismatch: {set(images)}")
        observation: dict[str, np.ndarray] = {"observation.state": state_array.copy()}
        for key in self.image_keys:
            image = np.asarray(images[key])
            if image.shape != self.image_shapes[key] or image.dtype != np.uint8:
                raise ValueError(
                    f"Expected uint8 HWC image {self.image_shapes[key]} for {key}, "
                    f"got {image.shape} {image.dtype}"
                )
            observation[key] = image.copy()
        with self._lock:
            total_started = time.perf_counter()
            batch = self.preprocessor(prepare_observation_for_inference(observation, self.device))
            torch.cuda.synchronize(self.device)
            model_started = time.perf_counter()
            with torch.no_grad():
                full_normalized = self.policy.predict_action_chunk(batch)
            torch.cuda.synchronize(self.device)
            model_ms = (time.perf_counter() - model_started) * 1000.0
            expected_full_shape = (
                1,
                self.contract.prediction_horizon,
                self.contract.action_dim,
            )
            if tuple(full_normalized.shape) != expected_full_shape:
                raise ValueError(
                    "ACT prediction chunk shape mismatch: "
                    f"{tuple(full_normalized.shape)}, expected {expected_full_shape}"
                )
            normalized = full_normalized[:, : self.execution_horizon]
            output = self.postprocessor({ACTION: normalized})[ACTION]
            torch.cuda.synchronize(self.device)
            latency_ms = (time.perf_counter() - total_started) * 1000.0
        chunk = output.detach().cpu().numpy()
        expected_shape = (1, self.execution_horizon, self.contract.action_dim)
        if chunk.shape != expected_shape:
            raise ValueError(f"ACT action chunk shape mismatch: {chunk.shape}")
        chunk = chunk[0].astype(np.float32, copy=False)
        if not np.isfinite(chunk).all():
            raise ValueError("ACT action chunk contains non-finite values")
        self.last_latency_ms = latency_ms
        self.last_model_ms = model_ms
        return chunk, latency_ms
