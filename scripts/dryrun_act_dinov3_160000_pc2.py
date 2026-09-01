#!/usr/bin/env python3
"""Offline PC2 dry-run for the Popcorn W1 ACT-DINOv3 160000 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import typing

import numpy as np
from PIL import Image
import torch
import typing_extensions


# PC2's Jetson Torch is built for Python 3.10 while this LeRobot source uses
# two typing-only names introduced in newer Python releases.
if not hasattr(typing, "Self"):
    typing.Self = typing_extensions.Self
if not hasattr(typing, "Unpack"):
    typing.Unpack = typing_extensions.Unpack

from lerobot.configs import PreTrainedConfig  # noqa: E402
from lerobot.policies import get_policy_class, make_pre_post_processors  # noqa: E402
from lerobot.policies.utils import prepare_observation_for_inference  # noqa: E402


ARM_INDICES = np.asarray((*range(1, 8), *range(10, 17)), dtype=np.int64)
ABSOLUTE_INDICES = np.asarray((0, 8, 9, 17, 18), dtype=np.int64)
LEFT_CLOSED = np.asarray((0, 100, 35, 45, 47, 37), dtype=np.float32)
RIGHT_CLOSED = np.asarray((65, 100, 70, 75, 100, 100), dtype=np.float32)
HAND_OPEN = np.asarray((0, 70, 0, 0, 0, 0), dtype=np.float32)
BODY_NAMES = (
    "WAIST",
    "LEFT_J1", "LEFT_J2", "LEFT_J3", "LEFT_J4", "LEFT_J5", "LEFT_J6", "LEFT_J7",
    "NECK1", "NECK2",
    "RIGHT_J1", "RIGHT_J2", "RIGHT_J3", "RIGHT_J4", "RIGHT_J5", "RIGHT_J6", "RIGHT_J7",
)
BODY_LIMITS = np.asarray(
    (
        (-2.9670597284, 2.9670597284),
        (-2.9670597284, 2.9670597284), (-2.0943951024, 1.5707963268),
        (-2.9670597284, 2.9670597284), (-2.3561944902, 1.5707963268),
        (-2.9670597284, 2.9670597284), (-0.7853981634, 0.7853981634),
        (-1.5707963268, 1.0471975512),
        (-1.5707963268, 1.5707963268), (-0.7853981634, 0.4363323130),
        (-2.9670597284, 2.9670597284), (-1.5707963268, 2.0943951024),
        (-2.9670597284, 2.9670597284), (-1.5707963268, 2.3561944902),
        (-2.9670597284, 2.9670597284), (-0.7853981634, 0.7853981634),
        (-1.0471975512, 1.5707963268),
    ),
    dtype=np.float32,
)


def hand_scalar(entries: list[list[object]], closed: np.ndarray) -> float:
    by_name = {str(name): float(value) for name, value in entries}
    values = np.asarray(
        [by_name[name] for name in ("T_MCP", "T_CMC_YAW", "IF_MCP_PITCH", "MF_MCP_PITCH", "RF_MCP_PITCH", "LF_MCP_PITCH")],
        dtype=np.float32,
    )
    direction = HAND_OPEN - closed
    fraction = float(np.dot(values - closed, direction) / np.dot(direction, direction))
    return float(np.clip(fraction, 0.0, 1.0) * 100.0)


def build_state(capture: dict[str, object]) -> tuple[np.ndarray, dict[str, object]]:
    robot = json.loads(str(capture["state"]))
    positions = np.asarray(robot["joint_position"], dtype=np.float32)
    if positions.shape != (20,):
        raise ValueError(f"expected 20 robot joints, got {positions.shape}")
    left = hand_scalar(capture["ee_left"], LEFT_CLOSED)
    right = hand_scalar(capture["ee_right"], RIGHT_CLOSED)
    state = np.concatenate(
        (positions[3:4], positions[6:13], positions[4:6], positions[13:20], [left, right])
    ).astype(np.float32)
    return state, robot


def convert_images(snapshot: Path, output: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    output.mkdir(parents=True, exist_ok=True)
    head = Image.open(snapshot / "live_high_right.png").convert("RGB")
    physical_left = Image.open(snapshot / "live_hand_right.png").convert("RGB")
    physical_right = Image.open(snapshot / "live_hand_left.png").convert("RGB")
    if head.size != (960, 540) or physical_left.size != (640, 360) or physical_right.size != (640, 480):
        raise ValueError(
            f"unexpected image sizes: head={head.size}, left={physical_left.size}, right={physical_right.size}"
        )

    head_square = Image.new("RGB", (960, 960), (0, 0, 0))
    head_square.paste(head, (0, (960 - 540) // 2))
    left_square = physical_left.resize((360, 360), Image.Resampling.BILINEAR)
    right_square = physical_right.resize((480, 480), Image.Resampling.BILINEAR)
    converted = {
        "observation.images.cam_high_right": head_square.resize((224, 224), Image.Resampling.BILINEAR),
        "observation.images.cam_hand_left": left_square.resize((224, 224), Image.Resampling.BILINEAR),
        "observation.images.cam_hand_right": right_square.resize((224, 224), Image.Resampling.BILINEAR),
    }
    for key, image in converted.items():
        image.save(output / f"{key.rsplit('.', 1)[-1]}.png")
    details = {
        "cam_high_right": "live_high_right.png: 960x540 -> centered black-pad 960x960 -> 224x224",
        "cam_hand_left": "physical left live_hand_right.png: 640x360 -> stretch 360x360 -> 224x224",
        "cam_hand_right": "physical right live_hand_left.png: 640x480 -> stretch 480x480 -> 224x224",
    }
    return {key: np.asarray(image).copy() for key, image in converted.items()}, details


def load_quantiles(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text())
    return np.asarray(payload["q01"], dtype=np.float32), np.asarray(payload["q99"], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    capture = json.loads((args.snapshot / "capture.json").read_text())
    state, robot = build_state(capture)
    images, image_contract = convert_images(args.snapshot, args.output)
    observation = {"observation.state": state, **images}

    config = PreTrainedConfig.from_pretrained(args.policy, local_files_only=True)
    policy_class = get_policy_class(config.type)
    policy = policy_class.from_pretrained(
        args.policy, config=config, local_files_only=True, strict=True
    ).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        config, pretrained_path=str(args.policy)
    )
    prepared = prepare_observation_for_inference(observation, torch.device(config.device))
    batch = preprocessor(prepared)
    with torch.inference_mode():
        raw = policy.predict_action_chunk(batch)
        absolute = postprocessor(raw)

    raw_np = raw.detach().cpu().numpy()
    absolute_np = absolute.detach().cpu().numpy()
    if raw_np.shape != (1, 16, 19) or absolute_np.shape != (1, 16, 19):
        raise ValueError(f"unexpected action shapes raw={raw_np.shape}, absolute={absolute_np.shape}")
    if not np.isfinite(raw_np).all() or not np.isfinite(absolute_np).all():
        raise ValueError("model actions contain non-finite values")

    state_q01, state_q99 = load_quantiles(args.policy / "relative_stats/relative_state_q01_q99.json")
    action_q01, action_q99 = load_quantiles(args.policy / "relative_stats/relative_action_chunk16_q01_q99.json")
    first_online_relative_state = state.copy()
    first_online_relative_state[ARM_INDICES] = 0.0
    expected_normalized_state = np.clip(
        2.0 * (first_online_relative_state - state_q01) / (state_q99 - state_q01) - 1.0,
        -1.0,
        1.0,
    )
    actual_normalized_state = batch["observation.state"].detach().cpu().numpy()[0]
    physical_relative_action = (raw_np + 1.0) * (action_q99 - action_q01) / 2.0 + action_q01
    expected_absolute = physical_relative_action.copy()
    expected_absolute[..., ARM_INDICES] += state[ARM_INDICES]

    body = absolute_np[0, :, :17]
    below = body < BODY_LIMITS[:, 0]
    above = body > BODY_LIMITS[:, 1]
    violations = [
        {"step": int(step), "joint": BODY_NAMES[int(joint)], "value": float(body[step, joint])}
        for step, joint in np.argwhere(below | above)
    ]
    clipped_grippers = np.where(absolute_np[0, :, 17:19] < 95.0, 0.0, np.clip(absolute_np[0, :, 17:19], 0.0, 100.0))
    result = {
        "capture_timestamp": robot["timestamp"],
        "robot_status": robot["status"],
        "model_type": config.type,
        "strict_parameter_count": sum(parameter.numel() for parameter in policy.parameters()),
        "image_contract": image_contract,
        "input_state_absolute_19d": state.tolist(),
        "input_grippers": {"left": float(state[17]), "right": float(state[18])},
        "preprocessor": {
            "first_online_arm_state_is_zero": bool(np.allclose(first_online_relative_state[ARM_INDICES], 0.0)),
            "absolute_indices_unchanged_before_q_normalization": bool(
                np.array_equal(first_online_relative_state[ABSOLUTE_INDICES], state[ABSOLUTE_INDICES])
            ),
            "q01_q99_clip_matches": bool(np.allclose(actual_normalized_state, expected_normalized_state, atol=1e-5)),
            "normalized_state_19d": actual_normalized_state.tolist(),
        },
        "raw_model_output": {
            "shape": list(raw_np.shape[1:]),
            "finite": bool(np.isfinite(raw_np).all()),
            "min": float(raw_np.min()),
            "max": float(raw_np.max()),
        },
        "postprocessor": {
            "shape": list(absolute_np.shape[1:]),
            "finite": bool(np.isfinite(absolute_np).all()),
            "q01_q99_inverse_and_absolute_reconstruction_match": bool(
                np.allclose(absolute_np, expected_absolute, atol=1e-4)
            ),
            "first_frame_absolute_19d": absolute_np[0, 0].tolist(),
            "last_frame_absolute_19d": absolute_np[0, -1].tolist(),
        },
        "runtime_gate": {
            "body_limit_violation_count_before_runtime_clip": len(violations),
            "body_limit_violations": violations,
            "gripper_raw_16x2": absolute_np[0, :, 17:19].tolist(),
            "gripper_after_less_than_95_to_zero_16x2": clipped_grippers.tolist(),
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
