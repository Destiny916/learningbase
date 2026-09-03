"""Create an isolated full-100-step ACT deployment directory."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


FILES = (
    "config.json", "model.safetensors", "policy_preprocessor.json",
    "policy_postprocessor.json",
    "policy_preprocessor_step_3_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    "train_config.json",
)


def convert(source: Path, target: Path) -> Path:
    source = source.expanduser().resolve()
    if (source / "pretrained_model").is_dir():
        source = source / "pretrained_model"
    target = target.expanduser().resolve()
    if target.exists() and any(target.iterdir()):
        raise RuntimeError(f"target is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    config = json.loads((source / "config.json").read_text())
    if config.get("chunk_size") != 100:
        raise RuntimeError(f"expected chunk_size=100, got {config.get('chunk_size')}")
    config["n_action_steps"] = 100
    # ACTPolicy.config_class is decoded directly by the PC2 runtime and
    # rejects deployment metadata fields that are not dataclass members.
    # Keep those fields in the source checkpoint, but omit them from the
    # isolated PC2 policy copy.
    config.pop("type", None)
    config.pop("camera_keys", None)
    for name in FILES:
        src = source / name
        if not src.is_file():
            raise RuntimeError(f"missing checkpoint file: {src}")
        shutil.copy2(src, target / name)
    (target / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (target / "xwiz_conversion.json").write_text(json.dumps({
        "format": "popcorn-act-full-chunk-v1",
        "source_checkpoint": str(source),
        "chunk_size": 100,
        "n_action_steps": 100,
        "state_action_dim": 19,
        "action_semantics": "absolute",
        "normalization": "MEAN_STD",
        "hand_contract": {
            "joint_order": ["T_MCP", "T_CMC_YAW", "IF_MCP_PITCH", "MF_MCP_PITCH", "RF_MCP_PITCH", "LF_MCP_PITCH"],
            "left_scalar_0": [0, 100, 35, 45, 47, 37],
            "right_scalar_0": [65, 100, 70, 75, 100, 100],
            "scalar_100": [0, 70, 0, 0, 0, 0],
        },
    }, indent=2) + "\n")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    print(convert(args.source, args.target))
