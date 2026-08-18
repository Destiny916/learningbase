#!/usr/bin/env bash
set -euo pipefail

# Independent six-camera ACT run for the endpoint20 dataset.
# State: 12 joint deltas + absolute grippers + absolute endpoint xyz.
DATA_ROOT="${DATA_ROOT:-/data/wengyikun/datasets/joint_songling/730_subtask_doubletop_rgbd_endpoint20}"
STATS_ROOT="${STATS_ROOT:-$DATA_ROOT/normalization_relative_state20_action14_xyz_absolute_chunk16_v2}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/wengyikun/outputs/act_730_endpoint20_relative_stereo_rgbd_chunk16_b32_100k_gpu7/train_out}"
JOB_NAME="${JOB_NAME:-act_730_endpoint20_relative_stereo_rgbd_chunk16_b32_100k_gpu7}"
STEPS="${STEPS:-100000}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GRAD_ACC="${GRAD_ACC:-1}"

export TORCH_HOME="${TORCH_HOME:-/data/wengyikun/models/torch}"
export DINO_V2_REPO="${DINO_V2_REPO:-/data/wengyikun/models/dinov2}"
export PYTHONPATH="${PYTHONPATH:-/workspace/lerobot/src}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/data/wengyikun/.cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/data/wengyikun/.cache/torchinductor}"
export HOME="${HOME:-/home/wengyikun}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/home/wengyikun/.cache}"
export HF_HOME="${HF_HOME:-/home/wengyikun/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/wengyikun/.cache/huggingface/datasets}"

python3 - "$DATA_ROOT" "$STATS_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root, stats_root = map(Path, sys.argv[1:])
info = json.loads((root / "meta/info.json").read_text())
features = info["features"]
expected = {
    "observation.images.top_left": (405, 720, 3),
    "observation.images.top_right": (405, 720, 3),
    "observation.images.gripper_left": (480, 640, 3),
    "observation.images.gripper_right": (480, 640, 3),
    "observation.images.gripper_left_depth": (1, 480, 640),
    "observation.images.gripper_right_depth": (1, 480, 640),
}
for key, shape in expected.items():
    actual = tuple(features.get(key, {}).get("shape", ()))
    if actual != shape:
        raise SystemExit(f"{key}: expected {shape}, got {actual}")
if tuple(features["observation.state"]["shape"]) != (20,):
    raise SystemExit("observation.state must be 20D")
if tuple(features["action"]["shape"]) != (14,):
    raise SystemExit("action must be 14D")
if info.get("fps") != 25:
    raise SystemExit(f"dataset fps must be 25, got {info.get('fps')}")
manifest = json.loads((stats_root / "relative_stats_manifest.json").read_text())
if manifest.get("format_version") != 3:
    raise SystemExit("stats must use relative_joint_v3")
if manifest.get("state_gripper_indices") != [6, 13]:
    raise SystemExit("stats gripper indices must be [6, 13]")
if manifest.get("state_absolute_indices") != [14, 15, 16, 17, 18, 19]:
    raise SystemExit("stats must keep endpoint xyz indices [14..19] absolute")
print("dataset preflight: v3, fps=25, state=20D, action=14D, images=6, stats=relative_joint_v3")
PY

exec accelerate launch --num_processes=1 --mixed_precision=bf16 \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/730_subtask_doubletop_rgbd_endpoint20 \
  --dataset.root="$DATA_ROOT" \
  --dataset.depth_output_unit=m \
  --dataset.image_transforms.enable=false \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.stereo_visual_mode=stereo_top_rgbd \
  --policy.input_features='{"observation.state":{"type":"STATE","shape":[20]},"observation.images.top_left":{"type":"VISUAL","shape":[3,405,720]},"observation.images.top_right":{"type":"VISUAL","shape":[3,405,720]},"observation.images.gripper_left":{"type":"VISUAL","shape":[3,480,640]},"observation.images.gripper_right":{"type":"VISUAL","shape":[3,480,640]},"observation.images.gripper_left_depth":{"type":"VISUAL","shape":[1,480,640]},"observation.images.gripper_right_depth":{"type":"VISUAL","shape":[1,480,640]}}' \
  --policy.chunk_size=16 \
  --policy.n_action_steps=16 \
  --policy.joint_representation=relative \
  --policy.gripper_indices='[6,13]' \
  --policy.state_feature_names='["left_joint_0","left_joint_1","left_joint_2","left_joint_3","left_joint_4","left_joint_5","left_gripper","right_joint_0","right_joint_1","right_joint_2","right_joint_3","right_joint_4","right_joint_5","right_gripper","right_endpoint_x","right_endpoint_y","right_endpoint_z","left_endpoint_x","left_endpoint_y","left_endpoint_z"]' \
  --policy.action_feature_names='["left_joint_0","left_joint_1","left_joint_2","left_joint_3","left_joint_4","left_joint_5","left_gripper","right_joint_0","right_joint_1","right_joint_2","right_joint_3","right_joint_4","right_joint_5","right_gripper"]' \
  --policy.state_gripper_indices='[6,13]' \
  --policy.state_absolute_indices='[14,15,16,17,18,19]' \
  --policy.state_position_indices='[14,15,16,17,18,19]' \
  --policy.condition_on_state=true \
  --policy.relative_state_stats_path="$STATS_ROOT/relative_state_q01_q99.json" \
  --policy.relative_action_stats_path="$STATS_ROOT/relative_action_chunk16_q01_q99.json" \
  --policy.clip_quantiles=true \
  --policy.state_noise_std_rad=0.003 \
  --policy.state_position_noise_std_m=0.003 \
  --policy.gripper_noise_std_m=0.001 \
  --batch_size="$BATCH_SIZE" \
  --gradient_accumulation_steps="$GRAD_ACC" \
  --num_workers=8 \
  --steps="$STEPS" \
  --save_checkpoint=true \
  --save_freq="$SAVE_FREQ" \
  --log_freq=10 \
  --eval_steps=0 \
  --wandb.enable=false \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME"
