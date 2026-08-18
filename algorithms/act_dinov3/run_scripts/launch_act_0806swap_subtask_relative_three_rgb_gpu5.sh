#!/usr/bin/env bash
set -euo pipefail

# Required per-model paths are supplied by the Docker launch command.
for required_var in DATA_ROOT STATS_ROOT OUTPUT_DIR JOB_NAME; do
  if [[ -z "${!required_var:-}" ]]; then
    echo "missing required environment variable: ${required_var}" >&2
    exit 2
  fi
done

# 20D state: only the 12 arm joints are relative; endpoint xyz and grippers stay absolute.
STEPS="${STEPS:-500000}"
SAVE_FREQ="${SAVE_FREQ:-50000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WARMUP_STEPS="${WARMUP_STEPS:-25000}"
DECAY_STEPS="${DECAY_STEPS:-500000}"

STATE_NAMES='["left_joint_0","left_joint_1","left_joint_2","left_joint_3","left_joint_4","left_joint_5","left_endpoint_x","left_endpoint_y","left_endpoint_z","left_gripper","right_joint_0","right_joint_1","right_joint_2","right_joint_3","right_joint_4","right_joint_5","right_endpoint_x","right_endpoint_y","right_endpoint_z","right_gripper"]'
STATE_ABSOLUTE_INDICES='[6,7,8,9,16,17,18,19]'

python3 - "$DATA_ROOT" "$STATS_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root, stats_root = map(Path, sys.argv[1:])
info = json.loads((root / "meta/info.json").read_text())
features = info["features"]
expected_images = {
    "observation.images.top": (405, 720, 3),
    "observation.images.wrist_left": (480, 640, 3),
    "observation.images.wrist_right": (480, 640, 3),
}
for key, shape in expected_images.items():
    if tuple(features.get(key, {}).get("shape", ())) != shape or features[key].get("names", "missing") is not None:
        raise SystemExit(f"{key} metadata is not an expected RGB v3 feature")
if tuple(features["observation.state"]["shape"]) != (20,) or tuple(features["action"]["shape"]) != (14,):
    raise SystemExit("expected 20D state and 14D action")
manifest = json.loads((stats_root / "relative_stats_manifest.json").read_text())
if manifest["horizons"] != [16] or manifest["state_absolute_indices"] != [6, 7, 8, 9, 16, 17, 18, 19]:
    raise SystemExit("relative state stats do not preserve xyz and grippers as absolute")
if manifest["state_gripper_indices"] != [9, 19] or manifest["gripper_indices"] != [6, 13]:
    raise SystemExit("relative stats have incorrect gripper indices")
print("dataset preflight: 12 relative joints, absolute xyz/grippers, three RGB cameras")
PY

exec accelerate launch --num_processes=1 --mixed_precision=bf16 \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="local/${JOB_NAME}" --dataset.root="$DATA_ROOT" --dataset.image_transforms.enable=false \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --policy.input_features='{"observation.state":{"type":"STATE","shape":[20]},"observation.images.top":{"type":"VISUAL","shape":[3,405,720]},"observation.images.wrist_left":{"type":"VISUAL","shape":[3,480,640]},"observation.images.wrist_right":{"type":"VISUAL","shape":[3,480,640]}}' \
  --policy.chunk_size=16 --policy.n_action_steps=16 --policy.joint_representation=relative \
  --policy.gripper_indices='[6,13]' --policy.state_feature_names="$STATE_NAMES" \
  --policy.state_gripper_indices='[9,19]' --policy.state_absolute_indices="$STATE_ABSOLUTE_INDICES" \
  --policy.condition_on_state=true --policy.relative_state_stats_path="$STATS_ROOT/relative_state_q01_q99.json" \
  --policy.relative_action_stats_path="$STATS_ROOT/relative_action_chunk16_q01_q99.json" \
  --policy.clip_quantiles=true --policy.state_noise_std_rad=0 --policy.gripper_noise_std_m=0 \
  --batch_size="$BATCH_SIZE" --gradient_accumulation_steps=1 --num_workers=8 \
  --use_policy_training_preset=false \
  --optimizer.type=adamw --optimizer.lr=1e-5 --optimizer.weight_decay=1e-4 --optimizer.grad_clip_norm=10 \
  --scheduler.type=cosine_decay_with_warmup --scheduler.num_warmup_steps="$WARMUP_STEPS" \
  --scheduler.num_decay_steps="$DECAY_STEPS" --scheduler.peak_lr=1e-5 --scheduler.decay_lr=1e-6 \
  --steps="$STEPS" --save_checkpoint=true --save_freq="$SAVE_FREQ" --log_freq=10 \
  --eval_steps=0 --wandb.enable=false --output_dir="$OUTPUT_DIR" --job_name="$JOB_NAME"
