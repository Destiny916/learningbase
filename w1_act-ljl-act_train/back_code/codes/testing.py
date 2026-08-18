#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
testing.py — Minimal, strict 11-D ACT forward tester (no ROS).

- Forces the state dimension to be exactly 11 (10 body + 1 hand scalar).
- Builds a batch with the same default keys as your inference node:
    observation.images.cam_high_left
    observation.images.cam_high_right
    observation.state
- Uses zero images by default (960x540, RGB, CHW, [0,1]).
- Prints detailed input/output stats so you can verify if the model outputs zeros.

Example:
  python3 testing.py \
    --policy_path /home/grabotics/workspace/act_inference/pick_doll/160000/pretrained_model \
    --joints -0.015303855204140342,-0.07998271701834976,-0.0045779739138462104,-0.14019146294286117,1.2418772900424802,-0.13309680179889008,1.2152842949774938,0.33386852163832875,0.0176887159603063,-0.12879446505786707,0.0

Tips:
- If your model used different keys, override with --image_left_key/--image_right_key/--state_key.
- Use --image_noise or --image_ones to sanity-check image sensitivity.
"""

import argparse
import sys
import numpy as np

try:
    import torch
except Exception as e:
    print(f"[ERROR] torch import failed: {e}", file=sys.stderr)
    sys.exit(1)

try:
    from act.modeling_act import ACTPolicy
except Exception as e:
    print(
        "[ERROR] Failed to import ACTPolicy from act.modeling_act. "
        "Ensure 'act' (with modeling_act.py) is on PYTHONPATH.\n"
        f"Import error: {e}",
        file=sys.stderr)
    sys.exit(1)


def parse_args():
    ap = argparse.ArgumentParser(description="Strict 11-D ACT forward test")
    ap.add_argument("--policy_path", required=True, type=str)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    # Keys (default match your inference node)
    ap.add_argument("--image_left_key",
                    default="observation.images.cam_high_left")
    ap.add_argument("--image_right_key",
                    default="observation.images.cam_high_right")
    ap.add_argument("--state_key", default="observation.state")

    # Images
    ap.add_argument("--target_width", type=int, default=960)
    ap.add_argument("--target_height", type=int, default=540)
    ap.add_argument("--image_noise", action="store_true")
    ap.add_argument("--image_ones", action="store_true")

    # Joints input (must be exactly 11 numbers)
    ap.add_argument("--joints", type=str, default=None, help="CSV of 11 floats")
    ap.add_argument("--joints_file",
                    type=str,
                    default=None,
                    help="File with 11 floats (comma/space/newline separated)")
    ap.add_argument("--print_first_n", type=int, default=11)
    return ap.parse_args()


def parse_11d(args) -> np.ndarray:
    vals = None
    if args.joints is not None:
        try:
            vals = [
                float(x.strip())
                for x in args.joints.split(",")
                if x.strip() != ""
            ]
        except Exception as e:
            raise RuntimeError(f"Failed to parse --joints: {e}")
    elif args.joints_file is not None:
        import os
        if not os.path.isfile(args.joints_file):
            raise RuntimeError(f"joints_file not found: {args.joints_file}")
        txt = open(args.joints_file, "r", encoding="utf-8").read()
        tokens = [t for t in txt.replace(",", " ").split() if t.strip() != ""]
        try:
            vals = [float(t) for t in tokens]
        except Exception as e:
            raise RuntimeError(f"Failed to parse joints_file: {e}")
    else:
        raise RuntimeError("Provide exactly one of --joints or --joints_file")

    if vals is None or len(vals) != 11:
        raise RuntimeError(
            f"Expected exactly 11 joint values, got {0 if vals is None else len(vals)}"
        )
    arr = np.asarray(vals, dtype=np.float32)
    return arr


def make_images(H: int, W: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    C = 3
    if mode == "noise":
        L = np.random.uniform(0.0, 1.0, (C, H, W)).astype(np.float32)
        R = np.random.uniform(0.0, 1.0, (C, H, W)).astype(np.float32)
    elif mode == "ones":
        L = np.ones((C, H, W), dtype=np.float32)
        R = np.ones((C, H, W), dtype=np.float32)
    else:
        L = np.zeros((C, H, W), dtype=np.float32)
        R = np.zeros((C, H, W), dtype=np.float32)
    return L, R


def main():
    args = parse_args()
    device = torch.device(args.device if (
        args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    print(f"[LOAD] policy_path={args.policy_path}\n[DEVICE] {device}")

    # Load policy
    policy = ACTPolicy.from_pretrained(args.policy_path)
    policy.to(device)
    policy.eval()

    # Build 11-D state
    joints = parse_11d(args)
    H, W = int(args.target_height), int(args.target_width)
    mode = "noise" if args.image_noise else (
        "ones" if args.image_ones else "zeros")
    left, right = make_images(H, W, mode)

    batch_np = {
        args.image_left_key: left[None, ...],  # (1,3,H,W)
        args.image_right_key: right[None, ...],  # (1,3,H,W)
        args.state_key: joints[None, ...],  # (1,11)
    }
    batch = {k: torch.from_numpy(v).to(device) for k, v in batch_np.items()}

    print(f"[BATCH] keys={list(batch.keys())}")
    print(
        f"[INPUT] joints (11) = {np.array2string(joints, precision=6, separator=', ')}"
    )
    print(
        f"[IMG]   mode={mode}, L min/max={left.min():.3f}/{left.max():.3f}, R min/max={right.min():.3f}/{right.max():.3f}"
    )

    with torch.no_grad():
        out = policy.select_action(batch)

    if isinstance(out, torch.Tensor):
        act = out.detach().to("cpu").numpy()
        if act.ndim == 2 and act.shape[0] == 1:
            act = act[0]
    else:
        act = np.asarray(out, dtype=np.float32)

    # Prints
    print("------------------------------------------------------------")
    print(f"[OUTPUT] action shape={act.shape}")
    print(
        f"[OUTPUT] min/max/mean = {act.min():.6f}/{act.max():.6f}/{act.mean():.6f}"
    )
    n = min(args.print_first_n, act.shape[-1] if act.ndim > 0 else 0)
    if n > 0:
        print(
            f"[OUTPUT] first {n} = {np.array2string(act[:n], precision=6, separator=', ')}"
        )
    else:
        print("[OUTPUT] (no elements)")

    # Extra guards
    if np.isnan(act).any():
        print("[WARN] NaN detected in action.")
    if act.shape[-1] != 11:
        print(
            f"[WARN] Action dim != 11 (got {act.shape[-1]}). Check model head / keys / state dim."
        )


if __name__ == "__main__":
    main()
