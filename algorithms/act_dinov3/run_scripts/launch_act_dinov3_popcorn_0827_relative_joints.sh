#!/usr/bin/env bash
set -euo pipefail

# Popcorn W1: 19D state/action. Joint dimensions are relative; grippers stay absolute.
DATA_ROOT="${DATA_ROOT:-/data/wengyikun/datasets/popcorn/0827_lerobot_v30_action_nextstate}"
STATS_ROOT="${STATS_ROOT:-/data/wengyikun/act_stats/popcorn_0827_19d_relative_arm_joints_absolute_waist_neck_grippers_chunk16}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/wengyikun/outputs/act_dinov3_popcorn_0827_19d_relative_arm_joints_nextstate_chunk16_b64_500k_gpu1_2/train_out}"
STEPS="${STEPS:-500000}"
SAVE_FREQ="${SAVE_FREQ:-50000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-16}"
WARMUP_STEPS="${WARMUP_STEPS:-25000}"
DECAY_STEPS="${DECAY_STEPS:-500000}"
JOB_NAME="${JOB_NAME:-act_dinov3_popcorn_0827_19d_relative_arm_joints_nextstate_chunk16_b64_500k_gpu1_2}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"

STATE_NAMES='["WAIST","LEFT_J1","LEFT_J2","LEFT_J3","LEFT_J4","LEFT_J5","LEFT_J6","LEFT_J7","NECK1","NECK2","RIGHT_J1","RIGHT_J2","RIGHT_J3","RIGHT_J4","RIGHT_J5","RIGHT_J6","RIGHT_J7","LEFT_GRIPPER","RIGHT_GRIPPER"]'

[[ -d "$DATA_ROOT" ]] || { echo "missing dataset: $DATA_ROOT" >&2; exit 2; }
[[ ! -e "$OUTPUT_DIR" ]] || { echo "refusing existing output: $OUTPUT_DIR" >&2; exit 2; }

python3 - "$DATA_ROOT" "$STATS_ROOT" <<'PY'
import json, sys
from pathlib import Path
root, stats = map(Path, sys.argv[1:])
info = json.loads((root / "meta/info.json").read_text())
features = info["features"]
assert info.get("codebase_version") == "v3.0", info.get("codebase_version")
assert info.get("total_episodes") == 50, info.get("total_episodes")
assert tuple(features["observation.state"]["shape"]) == (19,)
assert tuple(features["action"]["shape"]) == (19,)
expected = {
    "observation.images.cam_high_right": (3, 224, 224),
    "observation.images.cam_hand_left": (3, 224, 224),
    "observation.images.cam_hand_right": (3, 224, 224),
}
for key, shape in expected.items():
    assert tuple(features[key]["shape"]) == shape, (key, features[key]["shape"])
print(f"dataset preflight OK: episodes={info['total_episodes']} frames={info['total_frames']} fps={info['fps']} state=19 action=19 cameras=3")
PY

if [[ ! -f "$STATS_ROOT/relative_stats_manifest.json" || ! -f "$STATS_ROOT/relative_state_q01_q99.json" || ! -f "$STATS_ROOT/relative_action_chunk16_q01_q99.json" ]]; then
  mkdir -p "$STATS_ROOT"
  python3 -m lerobot.scripts.compute_popcorn_relative_joint_stats \
    --dataset-root="$DATA_ROOT" \
    --output-dir="$STATS_ROOT" \
    --horizon=16
fi

python3 - "$DATA_ROOT" "$STATS_ROOT" <<'PY'
import json, sys
from pathlib import Path
root, stats = map(Path, sys.argv[1:])
manifest = json.loads((stats / "relative_stats_manifest.json").read_text())
assert manifest["format_version"] == 4
assert manifest["horizons"] == [16]
assert manifest["gripper_indices"] == [17, 18]
assert manifest["state_gripper_indices"] == [17, 18]
assert manifest["state_absolute_indices"] == [0, 8, 9, 17, 18]
assert manifest["action_absolute_indices"] == [0, 8, 9, 17, 18]
assert manifest["source_dataset_root"] == str(root.resolve())
for name in ("relative_state_q01_q99.json", "relative_action_chunk16_q01_q99.json"):
    payload = json.loads((stats / name).read_text())
    assert len(payload["q01"]) == 19 and len(payload["q99"]) == 19
    assert all(a <= b for a, b in zip(payload["q01"], payload["q99"]))
print("relative stats preflight OK: state/action independent q01-q99, D=19, joints=0..16 relative, grippers=17..18 absolute")
PY

export PYTHONPATH="${PYTHONPATH:-/data/wengyikun/popcorn/algorithms/act_dinov3/src}"
export TORCHDYNAMO_DISABLE=1

exec accelerate launch --num_processes="$NUM_PROCESSES" --multi_gpu --mixed_precision=bf16 \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/popcorn_0827_w1_v30 \
  --dataset.root="$DATA_ROOT" \
  --dataset.image_transforms.enable=false \
  --policy.type=act_dinov3 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.dinov3_pretrained_path=/data/wengyikun/models/turbovla_joint_songling/dinov3-vitl16-pretrain-lvd1689m \
  --policy.dinov3_gradient_checkpointing=true \
  --policy.dinov3_autocast_dtype=bfloat16 \
  --policy.joint_representation=relative \
  --policy.gripper_indices='[17,18]' \
  --policy.state_gripper_indices='[17,18]' \
  --policy.state_absolute_indices='[0,8,9,17,18]' \
  --policy.action_absolute_indices='[0,8,9,17,18]' \
  --policy.state_feature_names="$STATE_NAMES" \
  --policy.action_feature_names="$STATE_NAMES" \
  --policy.condition_on_state=true \
  --policy.relative_state_stats_path="$STATS_ROOT/relative_state_q01_q99.json" \
  --policy.relative_action_stats_path="$STATS_ROOT/relative_action_chunk16_q01_q99.json" \
  --policy.clip_quantiles=true \
  --policy.state_noise_std_rad=0 \
  --policy.gripper_noise_std_m=0 \
  --policy.chunk_size=16 --policy.n_action_steps=16 \
  --policy.dropout=0.1 \
  --batch_size="$BATCH_SIZE" --gradient_accumulation_steps=1 --num_workers="$NUM_WORKERS" \
  --optimizer.type=adamw --optimizer.lr=1e-5 --optimizer.weight_decay=1e-4 --optimizer.grad_clip_norm=10 \
  --scheduler.type=cosine_decay_with_warmup --scheduler.num_warmup_steps="$WARMUP_STEPS" --scheduler.num_decay_steps="$DECAY_STEPS" --scheduler.peak_lr=1e-5 --scheduler.decay_lr=1e-6 \
  --steps="$STEPS" --save_checkpoint=true --save_freq="$SAVE_FREQ" --log_freq=10 \
  --eval_steps=0 --wandb.enable=false \
  --output_dir="$OUTPUT_DIR" --job_name="$JOB_NAME"
