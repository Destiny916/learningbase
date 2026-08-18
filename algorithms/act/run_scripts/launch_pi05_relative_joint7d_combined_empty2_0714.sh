#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/wengyikun/datasets/joint_songling/0714_gripper_bread_combined_split_seed42}"
TRAIN_ROOT="${DATA_ROOT}/train"
TEST_ROOT="${DATA_ROOT}/test"
STATS_ROOT="${DATA_ROOT}/normalization"
STATE_STATS="${STATS_ROOT}/relative_state_q01_q99.json"
ACTION_STATS="${STATS_ROOT}/relative_action_chunk50_q01_q99.json"

BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACC="${GRAD_ACC:-1}"
STEPS="${STEPS:-100000}"
SAVE_FREQ="${SAVE_FREQ:-20000}"
SCHEDULER_DECAY_STEPS="${SCHEDULER_DECAY_STEPS:-$STEPS}"
EVAL_STEPS="${EVAL_STEPS:-5000}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
STATE_NOISE_STD_RAD="${STATE_NOISE_STD_RAD:-0.003}"
GRIPPER_NOISE_STD_M="${GRIPPER_NOISE_STD_M:-0.001}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/wengyikun/outputs/pi05_0714_joint7d_combined_relative_empty2/train_out}"
JOB_NAME="${JOB_NAME:-pi05_0714_joint7d_combined_relative_empty2}"

for path in "$TRAIN_ROOT" "$TEST_ROOT" "$STATE_STATS" "$ACTION_STATS"; do
  [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 1; }
done

echo "effective batch=$((BATCH_SIZE * GRAD_ACC)) (batch=$BATCH_SIZE grad_acc=$GRAD_ACC)"
echo "empty cameras=2; both slots are masked after visual encoding"
echo "train root=$TRAIN_ROOT test root=$TEST_ROOT"
echo "steps=$STEPS save=$SAVE_FREQ decay=$SCHEDULER_DECAY_STEPS eval=$EVAL_STEPS output=$OUTPUT_DIR"
sha256sum "$STATE_STATS" "$ACTION_STATS"

exec accelerate launch --num_processes=1 --mixed_precision=bf16 \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/0714_bread_combined_joint7d_train \
  --dataset.root="$TRAIN_ROOT" \
  --dataset.image_transforms.enable=false \
  --validation_dataset.repo_id=local/0714_bread_combined_joint7d_test \
  --validation_dataset.root="$TEST_ROOT" \
  --validation_dataset.image_transforms.enable=false \
  --policy.type=pi05 \
  --policy.pretrained_path=/data/wengyikun/openpi/lerobot_pi05_base \
  --policy.freeze_language_model=true \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.gradient_checkpointing=true \
  --policy.push_to_hub=false \
  --policy.empty_cameras=2 \
  --policy.joint_representation=relative \
  --policy.joint_gripper_indices='[6]' \
  --policy.condition_on_state=true \
  --policy.relative_state_stats_path="$STATE_STATS" \
  --policy.relative_action_stats_path="$ACTION_STATS" \
  --policy.clip_quantiles=true \
  --policy.state_noise_std_rad="$STATE_NOISE_STD_RAD" \
  --policy.gripper_noise_std_m="$GRIPPER_NOISE_STD_M" \
  --policy.scheduler_decay_steps="$SCHEDULER_DECAY_STEPS" \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --eval_steps="$EVAL_STEPS" \
  --max_eval_samples="$MAX_EVAL_SAMPLES" \
  --env_eval_freq=0 \
  --log_freq=10 \
  --batch_size="$BATCH_SIZE" \
  --gradient_accumulation_steps="$GRAD_ACC" \
  --num_workers="$NUM_WORKERS" \
  --steps="$STEPS" \
  --save_checkpoint=true \
  --save_freq="$SAVE_FREQ" \
  --wandb.enable=false \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME"
