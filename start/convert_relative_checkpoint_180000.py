"""Convert the 180000 ACT-DINOv3 training checkpoint to the PC2 runtime layout."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


POLICY_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "policy_preprocessor_step_3_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)
STATS_FILES = ("relative_state_q01_q99.json", "relative_action_chunk16_q01_q99.json")


def convert(source: Path, target: Path, stats_root: Path) -> Path:
    source = source.expanduser().resolve()
    source = source / "pretrained_model" if (source / "pretrained_model").is_dir() else source
    target = target.expanduser().resolve()
    stats_root = stats_root.expanduser().resolve()
    if target.exists() and any(target.iterdir()):
        raise RuntimeError(f"target is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    for name in POLICY_FILES:
        src = source / name
        if not src.is_file():
            raise RuntimeError(f"missing checkpoint file: {src}")
        shutil.copy2(src, target / name)
    stats_dest = target / "relative_stats"
    stats_dest.mkdir()
    for name in STATS_FILES:
        src = stats_root / name
        if not src.is_file():
            raise RuntimeError(f"missing relative stats file: {src}")
        shutil.copy2(src, stats_dest / name)

    config = json.loads((target / "config.json").read_text())
    if config.get("type") != "act_dinov3":
        raise RuntimeError(f"expected act_dinov3 checkpoint, got {config.get('type')!r}")
    if config.get("chunk_size") != 16 or config.get("n_action_steps") != 16:
        raise RuntimeError("checkpoint horizon must be 16")
    if config.get("joint_representation") != "relative":
        raise RuntimeError("checkpoint must use relative joint representation")
    config["relative_state_stats_path"] = str(stats_dest / STATS_FILES[0])
    config["relative_action_stats_path"] = str(stats_dest / STATS_FILES[1])
    (target / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    manifest = {
        "format": "popcorn-pc2-relative-act-dinov3-v1",
        "source_checkpoint": str(source),
        "action_horizon": 16,
        "state_action_dim": 19,
        "joint_representation": "relative_arm_joints_absolute_waist_neck_grippers",
        "normalization": "q01_q99",
        "camera_contract": {
            "cam_high_right": "head padded 960x960 then resize 224x224",
            "cam_hand_left": "/camera_r 640x360 stretch to 360x360 then resize 224x224",
            "cam_hand_right": "/camera_l 640x480 stretch to 480x480 then resize 224x224",
        },
        "hand_contract": {
            "scalar_semantics": "openness_0_closed_100_open",
            "left_closed": [0, 100, 35, 45, 47, 37],
            "right_closed": [65, 100, 70, 75, 100, 100],
            "open": [0, 70, 0, 0, 0, 0],
        },
    }
    (target / "xwiz_conversion.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--stats-root", type=Path, required=True)
    args = parser.parse_args()
    print(convert(args.source, args.target, args.stats_root))


if __name__ == "__main__":
    main()
