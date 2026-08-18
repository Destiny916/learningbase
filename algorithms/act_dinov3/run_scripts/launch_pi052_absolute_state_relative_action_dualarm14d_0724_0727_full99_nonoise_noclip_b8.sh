#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/wengyikun/datasets/joint_songling/0724_0727_doublefripper_top_grippebread_combined_full_99episodes_task_en}"
REL_STATS_ROOT="$DATA_ROOT/normalization"
ABS_STATS_ROOT="$DATA_ROOT/normalization_absolute"
PI052_BASE="${PI052_BASE:-/data/wengyikun/openpi/lerobot_pi052_base}"
PALIGEMMA_TOKENIZER="${PALIGEMMA_TOKENIZER:-/data/jianan/weight/paligemma-3b-pt-224}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/wengyikun/outputs/pi052_0724_0727_full99_absolute_state_relative_action_task_en_base_chunk50_b8_100k_gpu7_20260805/train_out}"
JOB_NAME="${JOB_NAME:-pi052_0724_0727_full99_absolute_state_relative_action_task_en_base_chunk50_b8_100k_gpu7}"

for path in \
  "$DATA_ROOT" \
  "$REL_STATS_ROOT/relative_stats_manifest.json" \
  "$REL_STATS_ROOT/relative_state_q01_q99.json" \
  "$REL_STATS_ROOT/relative_action_chunk50_q01_q99.json" \
  "$ABS_STATS_ROOT/absolute_stats_manifest.json" \
  "$ABS_STATS_ROOT/absolute_state_q01_q99.json" \
  "$ABS_STATS_ROOT/absolute_action_chunk50_q01_q99.json" \
  "$PI052_BASE/config.json" \
  "$PI052_BASE/model.safetensors" \
  "$PI052_BASE/action_tokenizer"; do
  [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 1; }
done
[[ -d "$PALIGEMMA_TOKENIZER" ]] || { echo "Missing required path: $PALIGEMMA_TOKENIZER" >&2; exit 1; }
[[ ! -e "$OUTPUT_DIR" ]] || { echo "Refusing to overwrite output directory: $OUTPUT_DIR" >&2; exit 1; }

export TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec accelerate launch --num_processes=1 --mixed_precision=bf16 \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/0724_0727_doublefripper_top_grippebread_combined_full_99episodes_task_en \
  --dataset.root="$DATA_ROOT" \
  --dataset.image_transforms.enable=false \
  --policy.type=pi052 \
  --policy.pretrained_path="$PI052_BASE" \
  --policy.action_tokenizer_name="$PI052_BASE/action_tokenizer" \
  --policy.tokenizer_name="$PALIGEMMA_TOKENIZER" \
  --policy.auto_fit_fast_tokenizer=false \
  --policy.recipe_path=recipes/subtask_mem.yaml \
  --policy.enable_fast_action_loss=true \
  --policy.flow_loss_weight=10 \
  --policy.fast_action_loss_weight=1 \
  --policy.text_loss_weight=1 \
  --policy.knowledge_insulation=true \
  --policy.flow_num_repeats=5 \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.compile_model=false \
  --policy.gradient_checkpointing=true \
  --policy.freeze_language_model=true \
  --policy.unfreeze_lm_head=false \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.empty_cameras=0 \
  --policy.image_feature_order='["observation.images.top","observation.images.gripper_left","observation.images.gripper_right"]' \
  --policy.joint_representation=absolute \
  --policy.use_relative_actions=true \
  --policy.relative_exclude_joints='["gripper"]' \
  --policy.joint_gripper_indices='[6,13]' \
  --policy.condition_on_state=true \
  --policy.absolute_state_stats_path="$ABS_STATS_ROOT/absolute_state_q01_q99.json" \
  --policy.relative_action_stats_path="$REL_STATS_ROOT/relative_action_chunk50_q01_q99.json" \
  --policy.apply_action_limits=true \
  --policy.clip_quantiles=false \
  --policy.state_noise_std_rad=0 \
  --policy.gripper_noise_std_m=0 \
  --policy.scheduler_warmup_steps=5000 \
  --policy.scheduler_decay_steps=100000 \
  --batch_size="$BATCH_SIZE" \
  --gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS" \
  --num_workers=8 \
  --steps=100000 \
  --save_checkpoint=true \
  --save_freq=10000 \
  --log_freq=10 \
  --eval_steps=0 \
  --wandb.enable=false \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME"
