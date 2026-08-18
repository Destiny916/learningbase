#!/usr/bin/env bash
set -euo pipefail

TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-/data/wengyikun/pi05_official_libero_smoke/train_out}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-000001}"
PRETRAINED_MODEL_DIR="${TRAIN_OUTPUT_DIR}/checkpoints/${CHECKPOINT_STEP}/pretrained_model"
OUTPUT_DIR="${OUTPUT_DIR:-/data/wengyikun/pi05_official_libero_smoke/eval_out}"
JOB_NAME="${JOB_NAME:-pi05_official_libero_smoke_eval}"

[[ -d "$PRETRAINED_MODEL_DIR" ]] || {
  echo "Missing checkpoint model directory: $PRETRAINED_MODEL_DIR" >&2
  exit 2
}
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite existing output directory: $OUTPUT_DIR" >&2
  exit 2
fi

export MUJOCO_GL=egl

echo "policy_path=$PRETRAINED_MODEL_DIR"
echo "output_dir=$OUTPUT_DIR"

exec python -m lerobot.scripts.lerobot_eval \
  --policy.path="$PRETRAINED_MODEL_DIR" \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.n_action_steps=10 \
  --env.type=libero \
  --env.task=libero_spatial \
  --env.task_ids='[0]' \
  --env.control_mode=relative \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME"
