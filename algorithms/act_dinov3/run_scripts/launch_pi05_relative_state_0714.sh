#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="/data/wengyikun/datasets/joint_songling/0714_gripper_bread_combined_split_seed42"
TRAIN_ROOT="${DATA_ROOT}/train"
TEST_ROOT="${DATA_ROOT}/test"
STATS_ROOT="${DATA_ROOT}/normalization"
STATE_STATS="${STATS_ROOT}/relative_state_q01_q99.json"
ACTION16_STATS="${STATS_ROOT}/relative_action_chunk16_q01_q99.json"
ACTION50_STATS="${STATS_ROOT}/relative_action_chunk50_q01_q99.json"

NUM_PROCESSES="${NUM_PROCESSES:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACC="${GRAD_ACC:-1}"
STEPS="${STEPS:-10000}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}"
LOG_FREQ="${LOG_FREQ:-10}"
NUM_WORKERS="${NUM_WORKERS:-8}"
STATE_NOISE_STD_RAD="${STATE_NOISE_STD_RAD:-0.003}"
STATE_POSITION_NOISE_STD_M="${STATE_POSITION_NOISE_STD_M:-0.003}"
GRIPPER_NOISE_STD_M="${GRIPPER_NOISE_STD_M:-0.001}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
JOB_NAME="${JOB_NAME:-pi05_0714_relative_state_chunk50}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/wengyikun/outputs/pi05_0714_relative_state_chunk50/train_out}"

for stats_file in "$STATE_STATS" "$ACTION16_STATS" "$ACTION50_STATS"; do
  [[ -f "$stats_file" ]] || { echo "Missing normalization stats: $stats_file" >&2; exit 1; }
done

echo "GPU=${CUDA_VISIBLE_DEVICES:-container-visible-device}"
echo "NUM_PROCESSES=$NUM_PROCESSES"
echo "effective batch=$((BATCH_SIZE * NUM_PROCESSES * GRAD_ACC)) (batch=$BATCH_SIZE grad_acc=$GRAD_ACC)"
echo "train root=$TRAIN_ROOT"
echo "test root=$TEST_ROOT"
echo "stats SHA256:"
sha256sum "$STATE_STATS" "$ACTION16_STATS" "$ACTION50_STATS"
echo "condition_on_state=true chunk=50"
echo "steps=$STEPS save=$SAVE_FREQ eval=$EVAL_STEPS max_eval_samples=$MAX_EVAL_SAMPLES log=$LOG_FREQ"
echo "freeze_language_model=true freeze_vision_encoder=false train_expert_only=false include_projector=true"
echo "output_dir=$OUTPUT_DIR job_name=$JOB_NAME"

ACCELERATE_ARGS=(--num_processes="$NUM_PROCESSES" --mixed_precision=bf16)
if [[ "$NUM_PROCESSES" -gt 1 ]]; then
  ACCELERATE_ARGS+=(--multi_gpu)
fi

exec accelerate launch "${ACCELERATE_ARGS[@]}" \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/0714_bread_train \
  --dataset.root="$TRAIN_ROOT" \
  --validation_dataset.repo_id=local/0714_bread_test \
  --validation_dataset.root="$TEST_ROOT" \
  --policy.type=pi05 \
  --policy.pretrained_path=/data/wengyikun/openpi/lerobot_pi05_base \
  --policy.visual_pretrained_path=/data/wengyikun/models/TeleEmbodied_VISTA/pretrained_model/model.safetensors \
  --policy.visual_pretrained_include_projector=true \
  --policy.freeze_language_model=true \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.gradient_checkpointing=true \
  --policy.push_to_hub=false \
  --policy.joint_representation=relative \
  --policy.joint_gripper_indices='[6]' \
  --policy.condition_on_state=true \
  --policy.relative_state_stats_path="$STATE_STATS" \
  --policy.relative_action_stats_path="$ACTION50_STATS" \
  --policy.clip_quantiles=true \
  --policy.state_noise_std_rad="$STATE_NOISE_STD_RAD" \
  --policy.state_position_noise_std_m="$STATE_POSITION_NOISE_STD_M" \
  --policy.gripper_noise_std_m="$GRIPPER_NOISE_STD_M" \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --eval_steps="$EVAL_STEPS" \
  --max_eval_samples="$MAX_EVAL_SAMPLES" \
  --env_eval_freq=0 \
  --log_freq="$LOG_FREQ" \
  --batch_size="$BATCH_SIZE" \
  --gradient_accumulation_steps="$GRAD_ACC" \
  --num_workers="$NUM_WORKERS" \
  --steps="$STEPS" \
  --save_checkpoint=true \
  --save_freq="$SAVE_FREQ" \
  --wandb.enable="$WANDB_ENABLE" \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME"
