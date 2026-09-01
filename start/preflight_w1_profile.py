#!/usr/bin/env python3
"""Read-only validation of a W1 profile and local checkpoint contracts.

This tool never sources ROS, opens a control socket, publishes commands, or
changes robot state. It validates the portable contract before deployment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MODEL_FILES = {
    "220000": ("config.json", "model.safetensors", "relative_stats/relative_state_q01_q99.json", "relative_stats/relative_action_chunk16_q01_q99.json"),
    "180000": ("config.json", "model.safetensors", "relative_stats/relative_state_q01_q99.json", "relative_stats/relative_action_chunk16_q01_q99.json"),
    "500000": ("config.json", "model.safetensors"),
}


def load_profile(path: Path) -> dict:
    profile = json.loads(path.read_text(encoding="utf-8"))
    parent_name = profile.get("inherit_from")
    if parent_name:
        parent_path = path.parent / f"{parent_name}.json"
        if not parent_path.is_file():
            fail(f"profile parent not found: {parent_path}")
        base = load_profile(parent_path)
        base.update({key: value for key, value in profile.items() if key not in {"inherit_from", "overrides", "notes"}})
        overrides = profile.get("overrides", {})
        if "pc1_host" in overrides:
            base.setdefault("pc1", {})["host"] = overrides["pc1_host"]
        if "pc2_host" in overrides:
            base.setdefault("pc2", {})["host"] = overrides["pc2_host"]
        if "pc1_w1_act_root" in overrides:
            base.setdefault("pc1", {})["w1_act_root"] = overrides["pc1_w1_act_root"]
        if "pc2_workspace_root" in overrides:
            base.setdefault("pc2", {})["workspace_root"] = overrides["pc2_workspace_root"]
        if "left_wrist_serial" in overrides:
            base.setdefault("cameras", {})["left_wrist_serial"] = overrides["left_wrist_serial"]
        if "right_wrist_serial" in overrides:
            base.setdefault("cameras", {})["right_wrist_serial"] = overrides["right_wrist_serial"]
        return base
    return profile


def fail(message: str) -> None:
    raise SystemExit(f"PREFLIGHT_FAIL: {message}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--model", choices=tuple(MODEL_FILES))
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()

    profile = load_profile(args.profile)
    for section in ("pc1", "pc2", "ros", "ports", "models", "topics", "robot_contract", "cameras"):
        if section not in profile:
            fail(f"profile missing section {section}")
    if profile["ros"]["domain_id"] != 20 or profile["ros"]["rmw"] != "rmw_cyclonedds_cpp":
        fail("ROS baseline must be domain_id=20 and rmw_cyclonedds_cpp, or update the runtime explicitly")
    contract = profile["robot_contract"]
    if len(contract["robot_order"]) != 20 or len(contract["model_order"]) != 19:
        fail("robot/model joint order length must be 20/19")
    if contract["act_default_hand"] != [0, 70, 0, 0, 0, 0]:
        fail("ACT default hand must preserve thumb-root yaw 70")
    cameras = profile["cameras"]
    if cameras["left_wrist_serial"].startswith("REPLACE_") or cameras["right_wrist_serial"].startswith("REPLACE_"):
        fail("camera serials are placeholders; perform a hardware audit first")
    if args.model:
        if args.checkpoint is None:
            fail("--checkpoint is required with --model")
        if not args.checkpoint.is_dir():
            fail(f"checkpoint directory not found: {args.checkpoint}")
        missing = [name for name in MODEL_FILES[args.model] if not (args.checkpoint / name).exists()]
        if missing:
            fail(f"checkpoint missing: {', '.join(missing)}")
        config = json.loads((args.checkpoint / "config.json").read_text(encoding="utf-8"))
        expected_relative = args.model in {"180000", "220000"}
        if expected_relative and config.get("joint_representation") != "relative":
            fail(f"{args.model} checkpoint is not relative")
        if not expected_relative and config.get("normalization_mapping", {}).get("STATE") not in {"MEAN_STD", None}:
            fail("500000 legacy checkpoint normalization is not MEAN_STD")
        print(f"checkpoint={args.checkpoint} contract=PASS chunk={config.get('chunk_size')} normalization={config.get('normalization_mapping')}")
    print(f"profile={args.profile} contract=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
