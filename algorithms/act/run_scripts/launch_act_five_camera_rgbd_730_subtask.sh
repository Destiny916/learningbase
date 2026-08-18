#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/wengyikun/datasets/joint_songling/730_subtask_with_depth}"
STATS_ROOT="${STATS_ROOT:-$DATA_ROOT/normalization}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/wengyikun/outputs/act_730_subtask_with_depth_five_camera_rgbd_dualarm14d_relative_chunk16_b32_500k/train_out}"
STEPS="${STEPS:-500000}"
SAVE_FREQ="${SAVE_FREQ:-50000}"
BATCH_SIZE="${BATCH_SIZE:-32}"

export TORCH_HOME="${TORCH_HOME:-/data/wengyikun/.cache/torch}"

exec accelerate launch --num_processes=1 --mixed_precision=bf16 \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/730_subtask_with_depth_five_camera_rgbd \
  --dataset.root="$DATA_ROOT" --dataset.depth_output_unit=m \
  --dataset.image_transforms.enable=false \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --policy.stereo_visual_mode=five_camera_rgbd \
  --policy.chunk_size=16 --policy.n_action_steps=16 \
  --policy.joint_representation=relative --policy.gripper_indices='[6,13]' \
  --policy.condition_on_state=true \
  --policy.relative_state_stats_path="$STATS_ROOT/relative_state_q01_q99.json" \
  --policy.relative_action_stats_path="$STATS_ROOT/relative_action_chunk16_q01_q99.json" \
  --policy.clip_quantiles=true --policy.state_noise_std_rad=0.003 --policy.gripper_noise_std_m=0.001 \
  --batch_size="$BATCH_SIZE" --gradient_accumulation_steps=1 --num_workers=8 \
  --steps="$STEPS" --save_checkpoint=true --save_freq="$SAVE_FREQ" --log_freq=10 \
  --eval_steps=0 --wandb.enable=false \
  --output_dir="$OUTPUT_DIR" --job_name="${JOB_NAME:-act_730_subtask_with_depth_five_camera_rgbd_dualarm14d_relative_chunk16_b32_500k}"
