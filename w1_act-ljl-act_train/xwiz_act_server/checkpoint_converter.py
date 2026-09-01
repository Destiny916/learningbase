"""Convert a short-horizon LeRobot ACT checkpoint for the XWiz wire contract.

The model weights remain unchanged.  The conversion records the source horizon
and lets the runtime resample the model's output to the required horizon.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "policy_preprocessor_step_3_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)

EXPECTED_CAMERA_KEYS = {
    "observation.images.cam_high_right",
    "observation.images.cam_hand_left",
    "observation.images.cam_hand_right",
}
EXPECTED_JOINT_ORDER = [
    "WAIST", "LEFT_J1", "LEFT_J2", "LEFT_J3", "LEFT_J4", "LEFT_J5", "LEFT_J6", "LEFT_J7",
    "NECK1", "NECK2", "RIGHT_J1", "RIGHT_J2", "RIGHT_J3", "RIGHT_J4", "RIGHT_J5", "RIGHT_J6", "RIGHT_J7",
    "LEFT_GRIPPER", "RIGHT_GRIPPER",
]
HAND_JOINT_ORDER = [
    "T_MCP", "T_CMC_YAW", "IF_MCP_PITCH", "MF_MCP_PITCH", "RF_MCP_PITCH", "LF_MCP_PITCH",
]
HAND_CONTRACT = {
    "joint_order": HAND_JOINT_ORDER,
    "unit": "percent_0_100",
    "default_pose": [0.0, 70.0, 0.0, 0.0, 0.0, 0.0],
    "left_scalar_0": [0.0, 100.0, 35.0, 45.0, 47.0, 37.0],
    "left_scalar_100": [0.0, 70.0, 0.0, 0.0, 0.0, 0.0],
    "right_scalar_0": [65.0, 100.0, 70.0, 75.0, 100.0, 100.0],
    "right_scalar_100": [0.0, 70.0, 0.0, 0.0, 0.0, 0.0],
    "scalar_semantics": "openness_0_closed_100_open",
}


class CheckpointConversionError(ValueError):
    pass


def resample_action_chunk(actions: Any, target_horizon: int) -> np.ndarray:
    array = np.asarray(actions, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 19 or array.shape[0] < 1:
        raise CheckpointConversionError(
            f"source action chunk must have shape (N,19), got {array.shape}"
        )
    if target_horizon < 1:
        raise CheckpointConversionError("target_horizon must be positive")
    if not np.isfinite(array).all():
        raise CheckpointConversionError("source action chunk must be finite")
    if array.shape[0] == target_horizon:
        return array.copy()
    source_x = np.linspace(0.0, 1.0, array.shape[0], dtype=np.float32)
    target_x = np.linspace(0.0, 1.0, target_horizon, dtype=np.float32)
    return np.stack(
        [np.interp(target_x, source_x, array[:, index]) for index in range(19)], axis=1
    ).astype(np.float32, copy=False)


def _read_config(source: Path) -> dict[str, Any]:
    try:
        payload = json.loads((source / "config.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointConversionError(f"invalid checkpoint config: {source}") from exc
    if not isinstance(payload, dict):
        raise CheckpointConversionError("checkpoint config must be an object")
    return payload


def _resolve_checkpoint_root(source: Path) -> Path:
    if (source / "config.json").is_file():
        return source
    nested = source / "pretrained_model"
    if nested.is_dir() and (nested / "config.json").is_file():
        return nested
    return source


def convert_checkpoint(
    source: str | Path, target: str | Path, *, target_horizon: int = 16
) -> Path:
    source_path = _resolve_checkpoint_root(Path(source).expanduser().resolve())
    target_path = Path(target).expanduser().resolve()
    if target_horizon < 1:
        raise CheckpointConversionError("target_horizon must be positive")
    if not source_path.is_dir():
        raise CheckpointConversionError(f"checkpoint directory does not exist: {source_path}")
    config = _read_config(source_path)
    model_type = config.get("type")
    if model_type != "act":
        raise CheckpointConversionError(
            f"XWiz converter supports type='act' only, got {model_type!r}"
        )
    source_horizon = int(config.get("chunk_size", 0))
    if source_horizon < 1 or int(config.get("n_action_steps", 0)) < 1:
        raise CheckpointConversionError("checkpoint must declare positive chunk_size and n_action_steps")
    state_shape = config.get("input_features", {}).get("observation.state", {}).get("shape")
    action_shape = config.get("output_features", {}).get("action", {}).get("shape")
    if state_shape != [19] or action_shape != [19]:
        raise CheckpointConversionError("checkpoint must use 19D state and action features")
    visual_keys = {
        key for key in config.get("input_features", {}) if key.startswith("observation.images.")
    }
    if visual_keys != EXPECTED_CAMERA_KEYS:
        raise CheckpointConversionError(
            f"checkpoint camera keys must be {sorted(EXPECTED_CAMERA_KEYS)}, got {sorted(visual_keys)}"
        )
    processor = json.loads((source_path / "policy_preprocessor.json").read_text())
    norm_steps = [step for step in processor.get("steps", []) if step.get("registry_name") == "normalizer_processor"]
    if len(norm_steps) != 1:
        raise CheckpointConversionError("checkpoint must contain exactly one normalizer_processor")
    norm_map = norm_steps[0].get("config", {}).get("norm_map", {})
    if norm_map.get("VISUAL") != "MEAN_STD" or norm_map.get("STATE") != "MEAN_STD" or norm_map.get("ACTION") != "MEAN_STD":
        raise CheckpointConversionError(
            "outputs/500000 converter requires the checkpoint's MEAN_STD normalization contract"
        )
    if any(step.get("registry_name") == "relative_joint_processor" for step in processor.get("steps", [])):
        raise CheckpointConversionError("relative-joint checkpoints need a dedicated absolute-action conversion")
    missing = [name for name in REQUIRED_FILES if not (source_path / name).is_file()]
    if missing:
        raise CheckpointConversionError("checkpoint missing: " + ", ".join(missing))
    if target_path.exists():
        if any(target_path.iterdir()):
            raise CheckpointConversionError(f"target directory is not empty: {target_path}")
    else:
        target_path.mkdir(parents=True)
    for item in source_path.iterdir():
        if item.name == "xwiz_conversion.json":
            continue
        destination = target_path / item.name
        if item.is_file():
            shutil.copy2(item, destination)
    # PC2's deployed ACTConfig selects ACTPolicy explicitly and rejects these
    # metadata-only fields during draccus decoding.
    pc2_config = dict(config)
    pc2_config.pop("type", None)
    pc2_config.pop("camera_keys", None)
    (target_path / "config.json").write_text(json.dumps(pc2_config, indent=2) + "\n")
    manifest = {
        "format": "popcorn-xwiz-runtime-conversion-v1",
        "model_type": model_type,
        "source_horizon": source_horizon,
        "target_horizon": int(target_horizon),
        "action_semantics": "absolute",
        "joint_order": EXPECTED_JOINT_ORDER,
        "hand_contract": HAND_CONTRACT,
        "normalization": {
            "mode": "MEAN_STD",
            "stats_file": "policy_preprocessor_step_3_normalizer_processor.safetensors",
            "q01_q99_present_but_unused": True,
        },
        "wire_camera_to_model": {
            "cam_high_r": "observation.images.cam_high_right",
            "cam_left_wrist": "observation.images.cam_hand_left",
            "cam_right_wrist": "observation.images.cam_hand_right",
        },
        "source_checkpoint": str(source_path),
    }
    (target_path / "xwiz_conversion.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return target_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("target")
    parser.add_argument("--target-horizon", type=int, default=16)
    args = parser.parse_args()
    print(convert_checkpoint(args.source, args.target, target_horizon=args.target_horizon))


if __name__ == "__main__":
    main()
