#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

selected_gpu="${CUDA_VISIBLE_DEVICES:-2}"
if [[ "${IN_TURBOVLA_CONTAINER:-0}" != "1" ]]; then
  : "${JOINT_SONGLING_DATA_ROOT:?Set the source LeRobot v3 dataset root.}"
  : "${TURBOVLA_RELEASE_CKPT:?Set the released RoboTwin safetensors checkpoint.}"
  : "${BERT_MODEL_PATH:?Set a local bert-base-uncased directory.}"
  : "${DINOV3_MODEL_PATH:?Set a local DINOv3 ViT-L directory.}"
  : "${RUN_ROOT_DIR:?Set a writable output directory.}"
  image="${TURBOVLA_DOCKER_IMAGE:-turbovla-joint-songling:20260803}"
  exec docker run --rm --gpus "device=${selected_gpu}" --ipc=host \
    -e IN_TURBOVLA_CONTAINER=1 \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e JOINT_SONGLING_DATA_ROOT \
    -e TURBOVLA_RELEASE_CKPT \
    -e BERT_MODEL_PATH \
    -e DINOV3_MODEL_PATH \
    -e RUN_ROOT_DIR \
    -e RUN_ID="${RUN_ID:-turbovla_joint_songling_relative_3view}" \
    -e JOINT_SONGLING_OVERLAY_ROOT="${JOINT_SONGLING_OVERLAY_ROOT:-}" \
    -v "${repo_root}:${repo_root}" \
    -v /data:/data \
    -w "$repo_root" \
    "$image" bash scripts/joint_songling/train_relative_3view.sh "$@"
fi

: "${JOINT_SONGLING_DATA_ROOT:?Set the source LeRobot v3 dataset root.}"
: "${TURBOVLA_RELEASE_CKPT:?Set the released RoboTwin safetensors checkpoint.}"
: "${BERT_MODEL_PATH:?Set a local bert-base-uncased directory.}"
: "${DINOV3_MODEL_PATH:?Set a local DINOv3 ViT-L directory.}"
: "${RUN_ROOT_DIR:?Set a writable output directory.}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export JOINT_SONGLING_OVERLAY_ROOT="${JOINT_SONGLING_OVERLAY_ROOT:-${RUN_ROOT_DIR}/joint_songling_overlay}"
export PYTHONPATH="${repo_root}:${repo_root}/third_party/starvla_runtime:${PYTHONPATH:-}"

python3 - "$JOINT_SONGLING_DATA_ROOT" "$JOINT_SONGLING_OVERLAY_ROOT" <<'PY'
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

source, overlay = map(Path, sys.argv[1:])
if not (source / "meta/info.json").is_file():
    raise SystemExit(f"Not a LeRobot dataset: {source}")
overlay.mkdir(parents=True, exist_ok=True)
for name in ("data", "videos"):
    target = overlay / name
    if not target.exists():
        target.symlink_to(source / name, target_is_directory=True)
meta = overlay / "meta"
meta.mkdir(exist_ok=True)
for name in ("info.json", "episodes", "stats.json"):
    target = meta / name
    if not target.exists():
        target.symlink_to(source / "meta" / name, target_is_directory=(source / "meta" / name).is_dir())
task = "Pick up the bread with the right arm, transfer it to the left arm, then place it in the bowl with the left arm."
tasks = pd.read_parquet(source / "meta" / "tasks.parquet").reset_index()
tasks = tasks[["task_index"]].copy()
tasks.index = pd.Index([task] * len(tasks), name="task")
tasks.to_parquet(meta / "tasks.parquet")
modality = {
    "state": {
        "left_joints": {"start": 0, "end": 6, "original_key": "observation.state"},
        "left_gripper": {"start": 6, "end": 7, "original_key": "observation.state"},
        "right_joints": {"start": 7, "end": 13, "original_key": "observation.state"},
        "right_gripper": {"start": 13, "end": 14, "original_key": "observation.state"},
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
print(f"[INFO] overlay={overlay}")
print("[INFO] camera_order=top,gripper_left,gripper_right")
PY

exec python3 -m accelerate.commands.launch --num_processes 1 \
  third_party/starvla_runtime/starVLA/training/train_robotwin_clean_act_pi05_recipe.py \
  --config_yaml experiments/joint_songling/configs/relative_3view.yaml \
  --run_root_dir "$RUN_ROOT_DIR" \
  --run_id "${RUN_ID:-turbovla_joint_songling_relative_3view}" \
  "$@"
