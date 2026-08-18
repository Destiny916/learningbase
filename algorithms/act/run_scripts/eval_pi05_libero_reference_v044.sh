#!/usr/bin/env bash
set -euo pipefail

POLICY_PATH="lerobot/pi05_libero_finetuned_v044"
OUTPUT_DIR="${OUTPUT_DIR:-/data/wengyikun/pi05_libero_reference_v044/eval_out}"
JOB_NAME="${JOB_NAME:-pi05_libero_reference_v044}"

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite existing output directory: $OUTPUT_DIR" >&2
  exit 2
fi

export MUJOCO_GL=egl

echo "policy_path=$POLICY_PATH"
echo "output_dir=$OUTPUT_DIR"

exec python -m lerobot.scripts.lerobot_eval \
  --policy.path="$POLICY_PATH" \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.n_action_steps=10 \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --env.control_mode=relative \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME"
