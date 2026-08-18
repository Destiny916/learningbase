#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/wengyikun/datasets/joint_songling/0729_0727_last10_doublefripper_top_grippebread_25fps_combined}"
STATS_ROOT="$DATA_ROOT/normalization"
OUTPUT_DIR="${OUTPUT_DIR:-/data/wengyikun/outputs/pi05_0729_0727_dualarm14d_relative_base_chunk50_b16_100k_camorder_top_left_right_visionbf16_20260803/train_out}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/0729_0727_last10_doublefripper_top_grippebread_25fps_full}"
JOB_NAME="${JOB_NAME:-pi05_0729_0727_dualarm14d_relative_camorder_top_left_right_visionbf16_b16_100k}"

for path in \
  "$DATA_ROOT" \
  "$STATS_ROOT/relative_state_q01_q99.json" \
  "$STATS_ROOT/relative_action_chunk50_q01_q99.json"; do
  [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 1; }
done

export TORCHDYNAMO_DISABLE=1

exec accelerate launch --num_processes=1 --mixed_precision=bf16 \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.root="$DATA_ROOT" \
  --dataset.image_transforms.enable=false \
  --policy.type=pi05 \
  --policy.pretrained_path=/data/wengyikun/openpi/lerobot_pi05_base \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.compile_model=false \
  --policy.gradient_checkpointing=true \
  --policy.freeze_language_model=true \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.empty_cameras=0 \
  --policy.image_feature_order='["observation.images.top","observation.images.gripper_left","observation.images.gripper_right"]' \
  --policy.joint_representation=relative \
  --policy.joint_gripper_indices='[6,13]' \
  --policy.condition_on_state=true \
  --policy.relative_state_stats_path="$STATS_ROOT/relative_state_q01_q99.json" \
  --policy.relative_action_stats_path="$STATS_ROOT/relative_action_chunk50_q01_q99.json" \
  --policy.clip_quantiles=true \
  --policy.state_noise_std_rad=0.003 \
  --policy.gripper_noise_std_m=0.001 \
  --policy.scheduler_warmup_steps=5000 \
  --policy.scheduler_decay_steps=100000 \
  --batch_size=16 \
  --gradient_accumulation_steps=1 \
  --num_workers=8 \
  --steps=100000 \
  --save_checkpoint=true \
  --save_freq=10000 \
  --log_freq=10 \
  --eval_steps=0 \
  --wandb.enable=false \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME"
