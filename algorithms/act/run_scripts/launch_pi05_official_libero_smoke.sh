#!/usr/bin/env bash
set -euo pipefail

DATASET_REPO_ID="HuggingFaceVLA/libero"
PRETRAINED_PATH="lerobot/pi05_libero"
OUTPUT_DIR="${OUTPUT_DIR:-/data/wengyikun/pi05_official_libero_smoke/train_out}"
JOB_NAME="${JOB_NAME:-pi05_official_libero_smoke}"
STEPS="${STEPS:-1}"
SAVE_FREQ="${SAVE_FREQ:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
DATASET_EPISODES="${DATASET_EPISODES:-[0]}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite existing output directory: $OUTPUT_DIR" >&2
  exit 2
fi

export MUJOCO_GL=egl

echo "dataset=$DATASET_REPO_ID episodes=$DATASET_EPISODES"
echo "pretrained_path=$PRETRAINED_PATH"
echo "output_dir=$OUTPUT_DIR"
echo "steps=$STEPS batch_size=$BATCH_SIZE"

exec accelerate launch --num_processes=1 --mixed_precision=bf16 \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.episodes="$DATASET_EPISODES" \
  --policy.type=pi05 \
  --policy.pretrained_path="$PRETRAINED_PATH" \
  --policy.freeze_language_model=true \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.gradient_checkpointing=true \
  --policy.chunk_size=50 \
  --policy.n_action_steps=10 \
  --policy.joint_gripper_indices='[6]' \
  --policy.push_to_hub=false \
  --env.type=libero \
  --env.task=libero_spatial \
  --env.task_ids='[0]' \
  --env.control_mode=relative \
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
