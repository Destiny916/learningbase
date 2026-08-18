#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/wengyikun/datasets/joint_songling/0724_0727_doublefripper_top_grippebread_combined_split_seed42_80train_19test_top_only}"
STATS_ROOT="${STATS_ROOT:-/data/wengyikun/datasets/joint_songling/0724_0727_doublefripper_top_grippebread_combined_split_seed42_80train_19test/normalization}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/wengyikun/outputs/act_0724_0727_dualarm14d_relative_top_only_chunk16_b32_100k/train_out}"
STEPS="${STEPS:-100000}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
EVAL_STEPS="${EVAL_STEPS:-5000}"

exec accelerate launch --num_processes=1 --mixed_precision=bf16 \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/0724_0727_doublefripper_top_grippebread_combined_train_top_only \
  --dataset.root="$DATA_ROOT/train" \
  --dataset.image_transforms.enable=false \
  --validation_dataset.repo_id=local/0724_0727_doublefripper_top_grippebread_combined_test_top_only \
  --validation_dataset.root="$DATA_ROOT/test" \
  --validation_dataset.image_transforms.enable=false \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --policy.chunk_size=16 --policy.n_action_steps=16 \
  --policy.joint_representation=relative --policy.gripper_indices='[6,13]' \
  --policy.condition_on_state=true \
  --policy.relative_state_stats_path="$STATS_ROOT/relative_state_q01_q99.json" \
  --policy.relative_action_stats_path="$STATS_ROOT/relative_action_chunk16_q01_q99.json" \
  --policy.clip_quantiles=true --policy.state_noise_std_rad=0.003 --policy.gripper_noise_std_m=0.001 \
  --batch_size="${BATCH_SIZE:-32}" --gradient_accumulation_steps=1 --num_workers=8 \
  --steps="$STEPS" --save_checkpoint=true --save_freq="$SAVE_FREQ" --log_freq=10 \
  --eval_steps="$EVAL_STEPS" --max_eval_samples=0 --wandb.enable=false \
  --output_dir="$OUTPUT_DIR" --job_name="${JOB_NAME:-act_0724_0727_dualarm14d_relative_top_only_chunk16_b32_100k}"
