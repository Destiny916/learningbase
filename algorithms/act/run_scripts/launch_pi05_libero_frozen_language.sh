#!/usr/bin/env bash
set -euo pipefail

DATASET_REPO_ID="HuggingFaceVLA/libero"
PRETRAINED_PATH="lerobot/pi05_libero"
OUTPUT_DIR="${OUTPUT_DIR:-/data/wengyikun/pi05_libero_frozen_language/train_out}"
JOB_NAME="${JOB_NAME:-pi05_libero_frozen_language}"
STEPS="${STEPS:-250000}"
SAVE_FREQ="${SAVE_FREQ:-50000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
EMPTY_CAMERAS="${EMPTY_CAMERAS:-1}"

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite existing output directory: $OUTPUT_DIR" >&2
  exit 2
fi

export MUJOCO_GL=egl

echo "dataset=$DATASET_REPO_ID"
echo "pretrained_path=$PRETRAINED_PATH"
echo "output_dir=$OUTPUT_DIR"
echo "steps=$STEPS save_freq=$SAVE_FREQ batch_size=$BATCH_SIZE empty_cameras=$EMPTY_CAMERAS"
echo "LIBERO uses its dataset-native 7D relative action target without an additional action transform."
echo "freeze_language_model=true freeze_vision_encoder=false train_expert_only=false"
echo "Set BATCH_SIZE=8 and relaunch with a new OUTPUT_DIR if batch size 16 runs out of memory."

exec accelerate launch --num_processes=1 --mixed_precision=bf16 \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --policy.type=pi05 \
  --policy.pretrained_path="$PRETRAINED_PATH" \
  --policy.freeze_language_model=true \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.gradient_checkpointing=true \
  --policy.empty_cameras="$EMPTY_CAMERAS" \
  --policy.apply_action_limits=false \
  --policy.scheduler_decay_steps=250000 \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.joint_gripper_indices='[6]' \
  --policy.push_to_hub=false \
  --env_eval_freq=0 \
  --eval_steps=0 \
  --batch_size="$BATCH_SIZE" \
  --num_workers="$NUM_WORKERS" \
  --steps="$STEPS" \
  --save_checkpoint=true \
  --save_freq="$SAVE_FREQ" \
  --wandb.enable="$WANDB_ENABLE" \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME"
