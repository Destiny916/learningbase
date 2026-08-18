#!/usr/bin/env bash
set -euo pipefail

# Independent ACT run for the 17D state / 14D action LeRobot v3 dataset.
DATA_ROOT="${DATA_ROOT:-/data/wengyikun/datasets/joint_songling/0729_rgbd_xyz17d_v30}"
STATS_ROOT="${STATS_ROOT:-$DATA_ROOT/normalization_relative_state17_action14_chunk16}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/wengyikun/outputs/act_0729_rgbd_xyz17d_stereo_top_rgbd_relative_chunk16_b8_100k_gpu6/train_out}"
STEPS="${STEPS:-100000}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
JOB_NAME="${JOB_NAME:-act_0729_rgbd_xyz17d_stereo_top_rgbd_relative_chunk16_b8_100k_gpu6}"

# The Docker wrapper exposes host GPU6 as cuda:0 inside the container.
export TORCH_HOME="${TORCH_HOME:-/data/wengyikun/models/torch}"
export DINO_V2_REPO="${DINO_V2_REPO:-/data/wengyikun/models/dinov2}"

python3 - "$DATA_ROOT" "$STATS_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
stats_root = Path(sys.argv[2])
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
        raise SystemExit(f"{key}: expected shape {shape}, got {actual}")
if tuple(features["observation.state"]["shape"]) != (17,):
    raise SystemExit("observation.state must be 17D")
if tuple(features["action"]["shape"]) != (14,):
    raise SystemExit("action must be 14D")
if info.get("fps") != 25:
    raise SystemExit(f"dataset fps must be 25, got {info.get('fps')}")
for filename in ("relative_state_q01_q99.json", "relative_action_chunk16_q01_q99.json", "relative_stats_manifest.json"):
    if not (stats_root / filename).is_file():
        raise SystemExit(f"missing independent normalization file: {stats_root / filename}")
print("dataset preflight: v3.0, 25Hz, state=17D, action=14D, images=6, depth=2")
print(f"normalization preflight: {stats_root}")
PY

exec accelerate launch --num_processes=1 --mixed_precision=bf16 \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/0729_rgbd_xyz17d_v30 \
  --dataset.root="$DATA_ROOT" --dataset.depth_output_unit=m \
  --dataset.image_transforms.enable=false \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --policy.stereo_visual_mode=stereo_top_rgbd \
  --policy.chunk_size=16 --policy.n_action_steps=16 \
  --policy.joint_representation=relative --policy.gripper_indices='[6,13]' \
  --policy.state_gripper_indices='[6,13]' \
  --policy.state_position_indices='[14,15,16]' \
  --policy.condition_on_state=true \
  --policy.relative_state_stats_path="$STATS_ROOT/relative_state_q01_q99.json" \
  --policy.relative_action_stats_path="$STATS_ROOT/relative_action_chunk16_q01_q99.json" \
  --policy.clip_quantiles=true \
  --policy.state_noise_std_rad=0.003 \
  --policy.state_position_noise_std_m=0.003 \
  --policy.gripper_noise_std_m=0.001 \
  --batch_size="$BATCH_SIZE" --gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS" --num_workers=8 \
  --steps="$STEPS" --save_checkpoint=true --save_freq="$SAVE_FREQ" --log_freq=10 \
  --eval_steps=0 --wandb.enable=false \
  --output_dir="$OUTPUT_DIR" --job_name="$JOB_NAME"
