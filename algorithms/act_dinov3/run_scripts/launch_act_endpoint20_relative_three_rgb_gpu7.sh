#!/usr/bin/env bash
set -euo pipefail

# 20D state: 12 joint deltas + 2 absolute grippers + 6 absolute endpoint xyz values.
DATA_ROOT="${DATA_ROOT:-/data/wengyikun/datasets/joint_songling/730_subtask_doubletop_rgbd_endpoint20}"
STATS_ROOT="${STATS_ROOT:-$DATA_ROOT/normalization_relative_state20_action14_endpoint_absolute_chunk16}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/wengyikun/outputs/act_730_subtask_endpoint20_relative_three_rgb_chunk16_b32_100k_gpu7/train_out}"
STEPS="${STEPS:-100000}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
BATCH_SIZE="${BATCH_SIZE:-32}"

STATE_NAMES='["left_joint_0","left_joint_1","left_joint_2","left_joint_3","left_joint_4","left_joint_5","left_gripper","right_joint_0","right_joint_1","right_joint_2","right_joint_3","right_joint_4","right_joint_5","right_gripper","right_endpoint_x","right_endpoint_y","right_endpoint_z","left_endpoint_x","left_endpoint_y","left_endpoint_z"]'

python3 - "$DATA_ROOT" "$STATS_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root, stats_root = map(Path, sys.argv[1:])
info = json.loads((root / "meta/info.json").read_text())
features = info["features"]
expected_images = {
    "observation.images.top_right": (405, 720, 3),
    "observation.images.gripper_left": (480, 640, 3),
    "observation.images.gripper_right": (480, 640, 3),
}
for key, shape in expected_images.items():
    if tuple(features.get(key, {}).get("shape", ())) != shape:
        raise SystemExit(f"{key} has unexpected shape: {features.get(key, {}).get('shape')}")
if tuple(features["observation.state"]["shape"]) != (20,) or tuple(features["action"]["shape"]) != (14,):
    raise SystemExit("expected 20D state and 14D action")
manifest = json.loads((stats_root / "relative_stats_manifest.json").read_text())
if manifest["state_gripper_indices"] != [6, 13, 14, 15, 16, 17, 18, 19]:
    raise SystemExit("stats do not preserve grippers and endpoint xyz as absolute state values")
print("dataset preflight: 20D relative-joint state, absolute grippers/endpoint xyz, 14D relative-joint action")
PY

exec accelerate launch --num_processes=1 --mixed_precision=bf16 \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/730_subtask_doubletop_rgbd_endpoint20 \
  --dataset.root="$DATA_ROOT" \
  --dataset.image_transforms.enable=false \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --policy.input_features='{"observation.state":{"type":"STATE","shape":[20]},"observation.images.top_right":{"type":"VISUAL","shape":[3,405,720]},"observation.images.gripper_left":{"type":"VISUAL","shape":[3,480,640]},"observation.images.gripper_right":{"type":"VISUAL","shape":[3,480,640]}}' \
  --policy.chunk_size=16 --policy.n_action_steps=16 \
  --policy.joint_representation=relative \
  --policy.gripper_indices='[6,13]' \
  --policy.state_feature_names="$STATE_NAMES" \
  --policy.state_gripper_indices='[6,13,14,15,16,17,18,19]' \
  --policy.condition_on_state=true \
  --policy.relative_state_stats_path="$STATS_ROOT/relative_state_q01_q99.json" \
  --policy.relative_action_stats_path="$STATS_ROOT/relative_action_chunk16_q01_q99.json" \
  --policy.clip_quantiles=true --policy.state_noise_std_rad=0 --policy.gripper_noise_std_m=0 \
  --batch_size="$BATCH_SIZE" --gradient_accumulation_steps=1 --num_workers=8 \
  --steps="$STEPS" --save_checkpoint=true --save_freq="$SAVE_FREQ" --log_freq=10 \
  --eval_steps=0 --wandb.enable=false \
  --output_dir="$OUTPUT_DIR" \
  --job_name="${JOB_NAME:-act_730_subtask_endpoint20_relative_three_rgb_chunk16_b32_100k_gpu7}"
