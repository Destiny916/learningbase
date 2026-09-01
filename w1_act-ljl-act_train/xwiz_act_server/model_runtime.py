"""Strict LeRobot ACT checkpoint runtime used by the XWiz adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import types
from typing import Any, Literal, get_args, get_origin, get_type_hints

import numpy as np
from PIL import Image

from .checkpoint_converter import resample_action_chunk


REQUIRED_CHECKPOINT_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
)
EXPECTED_ACTION_SHAPE = (16, 19)
NORMALIZATION_EPS = 1e-8


class CheckpointError(RuntimeError):
    """Raised when the checkpoint or its output violates the runtime contract."""


def validate_checkpoint(policy_path: str | Path) -> Path:
    path = Path(policy_path).expanduser().resolve()
    if not path.is_dir():
        raise CheckpointError(f"checkpoint directory does not exist: {path}")
    if not all((path / name).is_file() for name in REQUIRED_CHECKPOINT_FILES):
        nested = path / "pretrained_model"
        if nested.is_dir():
            path = nested
    missing = [name for name in REQUIRED_CHECKPOINT_FILES if not (path / name).is_file()]
    if missing:
        raise CheckpointError(f"checkpoint missing required files: {', '.join(missing)}")
    return path


def validate_action_chunk(actions: Any, expected_horizon: int = EXPECTED_ACTION_SHAPE[0]) -> np.ndarray:
    array = np.asarray(actions, dtype=np.float32)
    expected_shape = (expected_horizon, EXPECTED_ACTION_SHAPE[1])
    if array.shape == (1, *expected_shape):
        array = array[0]
    if array.shape != expected_shape:
        raise CheckpointError(
            f"model action chunk must have shape {expected_shape}, got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise CheckpointError("model action chunk must contain finite values")
    return array


def resample_runtime_action_chunk(
    actions: Any, *, source_horizon: int, target_horizon: int = EXPECTED_ACTION_SHAPE[0]
) -> np.ndarray:
    try:
        converted = resample_action_chunk(actions, target_horizon)
    except ValueError as exc:
        raise CheckpointError(str(exc)) from exc
    if converted.shape[0] != target_horizon or converted.shape[1] != EXPECTED_ACTION_SHAPE[1]:
        raise CheckpointError(f"converted action chunk has invalid shape: {converted.shape}")
    if int(source_horizon) != np.asarray(actions).shape[0]:
        raise CheckpointError(
            f"checkpoint declared source horizon {source_horizon}, got {np.asarray(actions).shape[0]}"
        )
    return converted


def adapt_observation_to_policy(
    observation: dict[str, np.ndarray], input_features: dict[str, Any]
) -> dict[str, np.ndarray]:
    adapted = dict(observation)
    # The wire contract names the physical left head camera; some checkpoints
    # call the same feature cam_high_right. Preserve the checkpoint's key.
    if (
        "observation.images.cam_high_left" in adapted
        and "observation.images.cam_high_left" not in input_features
        and "observation.images.cam_high_right" in input_features
    ):
        adapted["observation.images.cam_high_right"] = adapted.pop(
            "observation.images.cam_high_left"
        )
    return adapted


def preprocess_observation_image(
    image: np.ndarray, key: str, feature_shape: tuple[int, int, int] | list[int]
) -> np.ndarray:
    """Apply the deterministic Popcorn image conversion used during training."""
    channels, height, width = (int(value) for value in feature_shape)
    if channels != 3:
        raise CheckpointError(f"image feature {key} must have 3 channels")
    array = np.asarray(image, dtype=np.uint8)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise CheckpointError(f"image feature {key} must be HWC RGB")
    pil = Image.fromarray(array, mode="RGB")
    if key == "observation.images.cam_high_right":
        side = max(pil.width, pil.height)
        canvas = Image.new("RGB", (side, side), (0, 0, 0))
        canvas.paste(pil, ((side - pil.width) // 2, (side - pil.height) // 2))
        pil = canvas
    elif key in {"observation.images.cam_hand_left", "observation.images.cam_hand_right"}:
        # The deployed wrist cameras have different native aspect ratios:
        # physical left wrist is 640x360 and physical right wrist is 640x480.
        # Training-time conversion stretches each image to a square using its
        # corresponding source side, then resizes to the model feature size.
        side = max(pil.width, pil.height)
        pil = pil.resize((side, side), Image.Resampling.LANCZOS)
    pil = pil.resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(pil, dtype=np.uint8)


def _draccus_decoder(config_class: Any, raw: dict[str, Any]) -> Any:
    import draccus

    for annotation in get_type_hints(config_class).values():
        if get_origin(annotation) is not Literal:
            continue
        allowed = get_args(annotation)

        def decode_literal(value: Any, path=(), allowed=allowed):
            if value not in allowed:
                raise ValueError(f"{value!r} is not one of {allowed!r}")
            return value

        draccus.decode.register(annotation, decode_literal)
    return draccus.decode(config_class, raw)


def load_policy_config(
    policy_class: Any,
    policy_path: Path,
    device: str,
    *,
    decoder: Any = None,
) -> Any:
    with (policy_path / "config.json").open() as stream:
        raw = json.load(stream)
    config = (decoder or _draccus_decoder)(policy_class.config_class, raw)
    config.device = device
    return config


def normalize_observation(
    observation: dict[str, np.ndarray],
    stats: dict[str, dict[str, np.ndarray]],
    feature_shapes: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    batch: dict[str, np.ndarray] = {}
    for key, value in observation.items():
        if key not in stats:
            raise CheckpointError(f"normalization stats missing feature: {key}")
        array = np.asarray(value)
        if key.startswith("observation.images."):
            if array.ndim != 3 or array.shape[-1] != 3:
                raise CheckpointError(f"image feature {key} must be HWC RGB")
            if feature_shapes and key in feature_shapes:
                array = preprocess_observation_image(array, key, feature_shapes[key]["shape"])
            array = array.astype(np.float32) / 255.0
            array = np.transpose(array, (2, 0, 1))
        else:
            array = array.astype(np.float32)
        mean = np.asarray(stats[key]["mean"], dtype=np.float32)
        std = np.asarray(stats[key]["std"], dtype=np.float32)
        batch[key] = ((array - mean) / (std + NORMALIZATION_EPS))[None, ...]
    return batch


def unnormalize_action_chunk(
    actions: Any,
    stats: dict[str, dict[str, np.ndarray]],
    expected_horizon: int = EXPECTED_ACTION_SHAPE[0],
) -> np.ndarray:
    array = np.asarray(actions, dtype=np.float32)
    expected_shape = (expected_horizon, EXPECTED_ACTION_SHAPE[1])
    if array.shape == (1, *expected_shape):
        array = array[0]
    action_stats = stats.get("action")
    if action_stats is None:
        raise CheckpointError("normalization stats missing feature: action")
    mean = np.asarray(action_stats["mean"], dtype=np.float32)
    std = np.asarray(action_stats["std"], dtype=np.float32)
    return validate_action_chunk(array * std + mean, expected_horizon=expected_horizon)


def load_normalization_stats(path: str | Path) -> dict[str, dict[str, np.ndarray]]:
    from safetensors import safe_open

    grouped: dict[str, dict[str, np.ndarray]] = {}
    with safe_open(str(path), framework="np") as tensors:
        for key in tensors.keys():
            feature, separator, statistic = key.rpartition(".")
            if separator and statistic in {"mean", "std"}:
                grouped.setdefault(feature, {})[statistic] = tensors.get_tensor(key)
    missing = [
        feature
        for feature, values in grouped.items()
        if "mean" not in values or "std" not in values
    ]
    if missing:
        raise CheckpointError(f"incomplete normalization stats: {', '.join(missing)}")
    return grouped


def install_lerobot_inference_import_shims() -> None:
    """Avoid importing unrelated policy, training, teleop and dataset stacks."""
    import lerobot

    lerobot_root = Path(lerobot.__file__).resolve().parent
    policies = types.ModuleType("lerobot.policies")
    policies.__path__ = [str(lerobot_root / "policies")]
    policies.__package__ = "lerobot.policies"
    sys.modules["lerobot.policies"] = policies

    train = types.ModuleType("lerobot.configs.train")
    train.TrainPipelineConfig = type("TrainPipelineConfig", (), {})
    sys.modules["lerobot.configs.train"] = train

    policy_utils = types.ModuleType("lerobot.policies.utils")
    policy_utils.log_model_loading_keys = lambda _missing, _unexpected: None
    sys.modules["lerobot.policies.utils"] = policy_utils


class LeRobotActRuntime:
    def __init__(self, policy_path: str | Path, device: str = "cuda"):
        self.policy_path = validate_checkpoint(policy_path)

        import torch
        install_lerobot_inference_import_shims()
        from lerobot.policies.act.modeling_act import ACTPolicy

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise CheckpointError(f"CUDA device requested but unavailable: {device}")
        self.torch = torch
        self.device = torch.device(device)

        policy_config = load_policy_config(
            ACTPolicy, self.policy_path, str(self.device)
        )
        self.policy = ACTPolicy.from_pretrained(
            self.policy_path,
            config=policy_config,
            local_files_only=True,
            strict=True,
        )
        self.policy.to(self.device).eval()
        self.source_horizon = int(getattr(policy_config, "chunk_size", 0))
        if self.source_horizon < 1:
            raise CheckpointError("checkpoint must declare a positive chunk_size")
        with (self.policy_path / "config.json").open() as stream:
            self.input_features = json.load(stream).get("input_features", {})
        self.input_stats = load_normalization_stats(
            self.policy_path / "policy_preprocessor_step_3_normalizer_processor.safetensors"
        )
        self.output_stats = load_normalization_stats(
            self.policy_path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
        )
        self.reset()

    def reset(self) -> None:
        self.policy.reset()

    def predict(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        observation = adapt_observation_to_policy(observation, self.input_features)
        batch_np = normalize_observation(observation, self.input_stats, self.input_features)
        batch = {
            key: self.torch.from_numpy(value).to(self.device)
            for key, value in batch_np.items()
        }
        with self.torch.inference_mode():
            normalized = self.policy.predict_action_chunk(batch)
        actions = normalized.detach().cpu().numpy()
        source_actions = unnormalize_action_chunk(
            actions, self.output_stats, expected_horizon=self.source_horizon
        )
        if self.source_horizon != EXPECTED_ACTION_SHAPE[0]:
            raise CheckpointError(
                f"checkpoint chunk_size must be {EXPECTED_ACTION_SHAPE[0]} for direct execution, "
                f"got {self.source_horizon}"
            )
        return validate_action_chunk(source_actions, expected_horizon=self.source_horizon)


def _synthetic_observation() -> dict[str, np.ndarray]:
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    return {
        "observation.state": np.zeros(19, dtype=np.float32),
        "observation.images.cam_high_left": image.copy(),
        "observation.images.cam_hand_left": image.copy(),
        "observation.images.cam_hand_right": image.copy(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    runtime = LeRobotActRuntime(args.policy_path, args.device)
    print(f"CHECKPOINT_LOADED path={runtime.policy_path} device={runtime.device}", flush=True)
    if args.smoke_test:
        actions = runtime.predict(_synthetic_observation())
        print(
            f"SMOKE_INFERENCE_OK shape={actions.shape} finite={np.isfinite(actions).all()} "
            f"min={actions.min():.6f} max={actions.max():.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
