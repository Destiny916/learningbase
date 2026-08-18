#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/wengyikun/datasets/joint_songling/0714_gripper_bread_normal_even_wrongplace_joint7d_v30_split_seed42}"
TRAIN_ROOT="${DATA_ROOT}/train"
TEST_ROOT="${DATA_ROOT}/test"
STATS_ROOT="${DATA_ROOT}/normalization_absolute"
STATE_STATS="${STATS_ROOT}/absolute_state_q01_q99.json"
ACTION_STATS="${STATS_ROOT}/absolute_action_chunk50_q01_q99.json"
STATS_MANIFEST="${STATS_ROOT}/absolute_stats_manifest.json"

NUM_PROCESSES="${NUM_PROCESSES:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACC="${GRAD_ACC:-1}"
STEPS="${STEPS:-32000}"
SAVE_FREQ="${SAVE_FREQ:-4000}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}"
LOG_FREQ="${LOG_FREQ:-10}"
NUM_WORKERS="${NUM_WORKERS:-8}"
STATE_NOISE_STD_RAD="${STATE_NOISE_STD_RAD:-0.003}"
STATE_POSITION_NOISE_STD_M="${STATE_POSITION_NOISE_STD_M:-0.003}"
GRIPPER_NOISE_STD_M="${GRIPPER_NOISE_STD_M:-0.001}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
JOB_NAME="${JOB_NAME:-pi05_0714_joint7d_normal_even_wrongplace_absolute_chunk50}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/wengyikun/outputs/pi05_0714_joint7d_normal_even_wrongplace_absolute_chunk50/train_out}"

for path in "$TRAIN_ROOT" "$TEST_ROOT" "$STATE_STATS" "$ACTION_STATS" "$STATS_MANIFEST"; do
  [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 1; }
done

echo "GPU=${CUDA_VISIBLE_DEVICES:-container-visible-device}"
echo "effective batch=$((BATCH_SIZE * NUM_PROCESSES * GRAD_ACC)) (batch=$BATCH_SIZE processes=$NUM_PROCESSES grad_acc=$GRAD_ACC)"
echo "absolute joint7d state/action quantiles: $STATS_ROOT"
sha256sum "$STATE_STATS" "$ACTION_STATS" "$STATS_MANIFEST"

ACCELERATE_ARGS=(--num_processes="$NUM_PROCESSES" --mixed_precision=bf16)
if [[ "$NUM_PROCESSES" -gt 1 ]]; then
  ACCELERATE_ARGS+=(--multi_gpu)
fi

exec accelerate launch "${ACCELERATE_ARGS[@]}" \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/0714_bread_joint7d_absolute_train \
  --dataset.root="$TRAIN_ROOT" \
  --validation_dataset.repo_id=local/0714_bread_joint7d_absolute_test \
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
  --policy.joint_representation=absolute \
  --policy.joint_gripper_indices='[6]' \
  --policy.absolute_state_stats_path="$STATE_STATS" \
  --policy.absolute_action_stats_path="$ACTION_STATS" \
  --policy.condition_on_state=true \
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
