#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/wengyikun/datasets/joint_songling/0724_doublefripper_top_grippebread_v30_split_seed42}"
STATS_ROOT="$DATA_ROOT/normalization"
OUTPUT_DIR="${OUTPUT_DIR:-/data/wengyikun/outputs/pi05_0724_dualarm14d_relative_base_chunk50_b16_20k/train_out}"

# DeepSpeed's imported Muon optimizer uses torch.compile decorators even when PI05 compile_model is false.
export TORCHDYNAMO_DISABLE=1

exec accelerate launch --num_processes=1 --mixed_precision=bf16 \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/0724_doublefripper_top_grippebread_train \
  --dataset.root="$DATA_ROOT/train" \
  --dataset.image_transforms.enable=false \
  --validation_dataset.repo_id=local/0724_doublefripper_top_grippebread_test \
  --validation_dataset.root="$DATA_ROOT/test" \
  --validation_dataset.image_transforms.enable=false \
  --policy.type=pi05 --policy.pretrained_path=/data/wengyikun/openpi/lerobot_pi05_base \
  --policy.dtype=bfloat16 --policy.device=cuda --policy.push_to_hub=false --policy.compile_model=false \
  --policy.chunk_size=50 --policy.n_action_steps=50 --policy.empty_cameras=0 \
  --policy.joint_representation=relative --policy.joint_gripper_indices='[6,13]' \
  --policy.condition_on_state=true \
  --policy.relative_state_stats_path="$STATS_ROOT/relative_state_q01_q99.json" \
  --policy.relative_action_stats_path="$STATS_ROOT/relative_action_chunk50_q01_q99.json" \
  --policy.clip_quantiles=true --policy.state_noise_std_rad=0.003 --policy.gripper_noise_std_m=0.001 \
  --policy.freeze_language_model=true --policy.freeze_vision_encoder=false --policy.train_expert_only=false \
  --policy.gradient_checkpointing=true --policy.scheduler_warmup_steps=2000 --policy.scheduler_decay_steps=20000 \
  --batch_size="${BATCH_SIZE:-16}" --gradient_accumulation_steps=1 --num_workers=8 \
  --steps=20000 --save_checkpoint=true --save_freq=5000 --log_freq=10 \
  --eval_steps=5000 --max_eval_samples=0 --wandb.enable=false \
  --output_dir="$OUTPUT_DIR" --job_name="${JOB_NAME:-pi05_0724_dualarm14d_relative_base_chunk50_b16_20k}"
