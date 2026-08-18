#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/wengyikun/datasets/joint_songling/0714_gripper_bread_single_teleop_normal_differentplace_wrongplace_right_fisheye_endpose_pose10d_combined_v30_split_seed42}"
TRAIN_ROOT="${DATA_ROOT}/train"
TEST_ROOT="${DATA_ROOT}/test"
STATS_ROOT="${DATA_ROOT}/normalization"
STATE_STATS="${STATS_ROOT}/relative_pose_state_q01_q99.json"
ACTION_STATS="${STATS_ROOT}/relative_pose_action_chunk16_q01_q99.json"

BATCH_SIZE="${BATCH_SIZE:-32}"
GRAD_ACC="${GRAD_ACC:-1}"
STEPS="${STEPS:-100000}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/wengyikun/outputs/act_0714_end_effector_pose_relative_chunk16/train_out}"
JOB_NAME="${JOB_NAME:-act_0714_end_effector_pose_relative_chunk16}"

for path in "$TRAIN_ROOT" "$TEST_ROOT" "$STATE_STATS" "$ACTION_STATS"; do
  [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 1; }
done

exec accelerate launch --num_processes=1 --mixed_precision=bf16 \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/0714_bread_endpose_pose10d_train \
  --dataset.root="$TRAIN_ROOT" \
  --validation_dataset.repo_id=local/0714_bread_endpose_pose10d_test \
  --validation_dataset.root="$TEST_ROOT" \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.end_effector_pose_representation=relative \
  --policy.pose_state_stats_path="$STATE_STATS" \
  --policy.pose_action_stats_path="$ACTION_STATS" \
  --policy.condition_on_state=true \
  --policy.clip_quantiles=true \
  --policy.chunk_size=16 \
  --policy.n_action_steps=16 \
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
