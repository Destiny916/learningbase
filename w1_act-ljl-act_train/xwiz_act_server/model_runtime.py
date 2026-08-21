"""Strict LeRobot ACT checkpoint runtime used by the XWiz adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal, get_args, get_origin, get_type_hints

import numpy as np


REQUIRED_CHECKPOINT_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
)
EXPECTED_ACTION_SHAPE = (100, 19)


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


def validate_action_chunk(actions: Any) -> np.ndarray:
    array = np.asarray(actions, dtype=np.float32)
    if array.shape == (1, *EXPECTED_ACTION_SHAPE):
        array = array[0]
    if array.shape != EXPECTED_ACTION_SHAPE:
        raise CheckpointError(
            f"model action chunk must have shape {EXPECTED_ACTION_SHAPE}, got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise CheckpointError("model action chunk must contain finite values")
    return array


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


class LeRobotActRuntime:
    def __init__(self, policy_path: str | Path, device: str = "cuda"):
        self.policy_path = validate_checkpoint(policy_path)

        import torch
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.processor import PolicyProcessorPipeline

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

        self.preprocessor = PolicyProcessorPipeline.from_pretrained(
            self.policy_path,
            config_filename="policy_preprocessor.json",
            local_files_only=True,
        )
        self.postprocessor = PolicyProcessorPipeline.from_pretrained(
            self.policy_path,
            config_filename="policy_postprocessor.json",
            local_files_only=True,
        )
        self._place_processors()
        self.reset()

    def _place_processors(self) -> None:
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

    def reset(self) -> None:
        self.policy.reset()

    def predict(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        from lerobot.policies.utils import prepare_observation_for_inference
        from lerobot.utils.constants import ACTION

        observation_t = prepare_observation_for_inference(observation, self.device)
        batch = self.preprocessor(observation_t)
        with self.torch.inference_mode():
            normalized = self.policy.predict_action_chunk(batch)
            processed = self.postprocessor({ACTION: normalized})
        action = processed[ACTION] if isinstance(processed, dict) else processed
        if hasattr(action, "detach"):
            action = action.detach().cpu().numpy()
        return validate_action_chunk(action)


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
