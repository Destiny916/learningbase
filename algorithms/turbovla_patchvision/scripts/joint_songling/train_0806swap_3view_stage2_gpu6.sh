#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

selected_gpu="${CUDA_VISIBLE_DEVICES:-6}"
if [[ "${IN_TURBOVLA_CONTAINER:-0}" != "1" ]]; then
  : "${JOINT_SONGLING_DATA_ROOT:?Set the source LeRobot v3 dataset root.}"
  : "${TURBOVLA_STAGE1_CKPT:?Set the completed stage-1 non-EMA safetensors checkpoint.}"
  : "${BERT_MODEL_PATH:?Set a local bert-base-uncased directory.}"
  : "${DINOV3_MODEL_PATH:?Set a local DINOv3 ViT-L directory.}"
  : "${RUN_ROOT_DIR:?Set a new writable output directory.}"
  [[ -f "$TURBOVLA_STAGE1_CKPT" ]] || { echo "Missing stage-1 checkpoint: $TURBOVLA_STAGE1_CKPT" >&2; exit 1; }
  [[ ! -e "$RUN_ROOT_DIR" ]] || { echo "Refusing to overwrite RUN_ROOT_DIR: $RUN_ROOT_DIR" >&2; exit 1; }
  image="${TURBOVLA_DOCKER_IMAGE:-turbovla-joint-songling:20260803}"
  exec docker run --rm --gpus "device=${selected_gpu}" --ipc=host \
    -e IN_TURBOVLA_CONTAINER=1 \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e JOINT_SONGLING_DATA_ROOT \
    -e TURBOVLA_STAGE1_CKPT \
    -e BERT_MODEL_PATH \
    -e DINOV3_MODEL_PATH \
    -e RUN_ROOT_DIR \
    -e WANDB_MODE="${WANDB_MODE:-disabled}" \
    -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    -e RUN_ID="${RUN_ID:-turbovla_0806swap_gripper_fixed_3view_gpu6_stage2}" \
    -e JOINT_SONGLING_OVERLAY_ROOT="${JOINT_SONGLING_OVERLAY_ROOT:-}" \
    -v "${repo_root}:${repo_root}" \
    -v /data:/data \
    -w "$repo_root" \
    "$image" bash scripts/joint_songling/train_0806swap_3view_stage2_gpu6.sh "$@"
fi

: "${JOINT_SONGLING_DATA_ROOT:?Set the source LeRobot v3 dataset root.}"
: "${TURBOVLA_STAGE1_CKPT:?Set the completed stage-1 non-EMA safetensors checkpoint.}"
: "${BERT_MODEL_PATH:?Set a local bert-base-uncased directory.}"
: "${DINOV3_MODEL_PATH:?Set a local DINOv3 ViT-L directory.}"
: "${RUN_ROOT_DIR:?Set a writable output directory.}"

[[ -f "$TURBOVLA_STAGE1_CKPT" ]] || { echo "Missing stage-1 checkpoint: $TURBOVLA_STAGE1_CKPT" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export JOINT_SONGLING_OVERLAY_ROOT="${JOINT_SONGLING_OVERLAY_ROOT:-${RUN_ROOT_DIR}/joint_songling_overlay}"
export PYTHONPATH="${repo_root}:${repo_root}/third_party/starvla_runtime:${PYTHONPATH:-}"

python3 - "$JOINT_SONGLING_DATA_ROOT" "$JOINT_SONGLING_OVERLAY_ROOT" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

source, overlay = map(Path, sys.argv[1:])
task = "Pick up the bread with the right gripper, transfer it to the left gripper, and place it in the bowl."
info = json.loads((source / "meta" / "info.json").read_text())
features = info["features"]
if tuple(features["observation.state"]["shape"]) != (20,) or tuple(features["action"]["shape"]) != (14,):
    raise SystemExit("expected state=20D and action=14D")
for key, shape in {
    "observation.images.top": (405, 720, 3),
    "observation.images.gripper_left": (480, 640, 3),
    "observation.images.gripper_right": (480, 640, 3),
}.items():
    if tuple(features.get(key, {}).get("shape", ())) != shape:
        raise SystemExit(f"unexpected image feature {key}: {features.get(key)}")
if info.get("splits") != {"train": "0:73"}:
    raise SystemExit(f"expected all 73 episodes in train, got {info.get('splits')}")
overlay.mkdir(parents=True, exist_ok=False)
for name in ("data", "videos"):
    (overlay / name).symlink_to(source / name, target_is_directory=True)
meta = overlay / "meta"
meta.mkdir()
for name in ("info.json", "episodes", "stats.json"):
    (meta / name).symlink_to(source / "meta" / name, target_is_directory=(source / "meta" / name).is_dir())
tasks = pd.read_parquet(source / "meta" / "tasks.parquet")
tasks = tasks[["task_index"]].copy()
tasks.index = pd.Index([task] * len(tasks), name="task")
tasks.to_parquet(meta / "tasks.parquet")
modality = {
    "state": {
        "left_joints": {"start": 0, "end": 6, "original_key": "observation.state"},
        "left_endpoint": {"start": 6, "end": 9, "original_key": "observation.state"},
        "left_gripper": {"start": 9, "end": 10, "original_key": "observation.state"},
        "right_joints": {"start": 10, "end": 16, "original_key": "observation.state"},
        "right_endpoint": {"start": 16, "end": 19, "original_key": "observation.state"},
        "right_gripper": {"start": 19, "end": 20, "original_key": "observation.state"},
    },
    "action": {
        "left_joints": {"start": 0, "end": 6, "original_key": "action"},
        "left_gripper": {"start": 6, "end": 7, "original_key": "action"},
        "right_joints": {"start": 7, "end": 13, "original_key": "action"},
        "right_gripper": {"start": 13, "end": 14, "original_key": "action"},
    },
    "video": {
        "top": {"original_key": "observation.images.top"},
        "gripper_left": {"original_key": "observation.images.gripper_left"},
        "gripper_right": {"original_key": "observation.images.gripper_right"},
    },
    "annotation": {"human.action.task_description": {"original_key": "task_index"}},
}
(meta / "modality.json").write_text(json.dumps(modality, indent=2) + "\n")
print("[INFO] state=left_joints, left_xyz, left_gripper, right_joints, right_xyz, right_gripper")
print("[INFO] relative=left_joints,right_joints; absolute=endpoint_xyz,grippers")
print("[INFO] camera_order=top,gripper_left,gripper_right")
print(f"[INFO] task={task}")
PY

exec python3 -m accelerate.commands.launch --num_processes 1 \
  third_party/starvla_runtime/starVLA/training/train_robotwin_clean_act_pi05_recipe.py \
  --config_yaml experiments/joint_songling/configs/0806swap_3view_stage2.yaml \
  --run_root_dir "$RUN_ROOT_DIR" \
  --run_id "${RUN_ID:-turbovla_0806swap_gripper_fixed_3view_gpu6_stage2}" \
  "$@"
