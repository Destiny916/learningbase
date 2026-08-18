#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/wengyikun/datasets/joint_songling/0806swap_gripper_fixed_pi052_task_en_v2}"
MIXED_ROOT="$DATA_ROOT/normalization_absolute_state_relative_action"
PI052_BASE="${PI052_BASE:-/data/wengyikun/openpi/lerobot_pi052_base}"
PALIGEMMA_TOKENIZER="${PALIGEMMA_TOKENIZER:-/data/jianan/weight/paligemma-3b-pt-224}"
BATCH_SIZE="${BATCH_SIZE:-4}"
STEPS="${STEPS:-400000}"
SAVE_FREQ="${SAVE_FREQ:-40000}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/wengyikun/outputs/pi052_0806_absolute_state_relative_action_task_en_chunk50_b4_400k_gpu1/train_out}"

for path in \
  "$DATA_ROOT/meta/info.json" \
  "$MIXED_ROOT/relative_stats_manifest.json" \
  "$MIXED_ROOT/absolute_state_q01_q99.json" \
  "$MIXED_ROOT/relative_state_q01_q99.json" \
  "$MIXED_ROOT/relative_action_chunk50_q01_q99.json" \
  "$PI052_BASE/config.json" \
  "$PI052_BASE/model.safetensors" \
  "$PI052_BASE/action_tokenizer"; do
  [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 1; }
done
[[ -d "$PALIGEMMA_TOKENIZER" ]] || { echo "Missing required path: $PALIGEMMA_TOKENIZER" >&2; exit 1; }
[[ ! -e "$OUTPUT_DIR" ]] || { echo "Refusing to overwrite output directory: $OUTPUT_DIR" >&2; exit 1; }

export TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="/workspace/lerobot/src${PYTHONPATH:+:$PYTHONPATH}"

exec accelerate launch --num_processes=1 --mixed_precision=bf16 \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/0806swap_gripper_fixed_pi052_task_en_v2 \
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
  --policy.state_gripper_indices='[9,19]' \
  --policy.condition_on_state=true \
  --policy.absolute_state_stats_path="$MIXED_ROOT/absolute_state_q01_q99.json" \
  --policy.relative_action_stats_path="$MIXED_ROOT/relative_action_chunk50_q01_q99.json" \
  --policy.apply_action_limits=true \
  --policy.clip_quantiles=false \
  --policy.state_noise_std_rad=0 \
  --policy.gripper_noise_std_m=0 \
  --policy.scheduler_warmup_steps=20000 \
  --policy.scheduler_decay_steps=400000 \
  --batch_size="$BATCH_SIZE" \
  --gradient_accumulation_steps=1 \
  --num_workers=8 \
  --steps="$STEPS" \
  --save_checkpoint=true \
  --save_freq="$SAVE_FREQ" \
  --log_freq=10 \
  --eval_steps=0 \
  --wandb.enable=false \
  --output_dir="$OUTPUT_DIR" \
  --job_name=pi052_0806_absolute_state_relative_action_task_en_chunk50_b4_400k_gpu1
