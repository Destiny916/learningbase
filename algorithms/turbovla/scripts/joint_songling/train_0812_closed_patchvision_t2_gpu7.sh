#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
selected_gpu="${CUDA_VISIBLE_DEVICES:-7}"
export JOINT_SONGLING_DATA_ROOT="${JOINT_SONGLING_DATA_ROOT:-/data/wengyikun/datasets/joint_songling/0812_closed_gripper_zero_without_ep173_174}"
export RUN_ID="${RUN_ID:-turbovla_0812_closed_patchvision_t2_gpu7}"
export JOINT_SONGLING_OVERLAY_ROOT="${JOINT_SONGLING_OVERLAY_ROOT:-/data/wengyikun/outputs/${RUN_ID}_overlay}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

if [[ "${IN_TURBOVLA_CONTAINER:-0}" != "1" ]]; then
  : "${BERT_MODEL_PATH:?Set BERT_MODEL_PATH}"
  : "${DINOV3_MODEL_PATH:?Set DINOV3_MODEL_PATH}"
  : "${RUN_ROOT_DIR:?Set a new RUN_ROOT_DIR}"
  [[ ! -e "$RUN_ROOT_DIR" ]] || { echo "Refusing to overwrite RUN_ROOT_DIR: $RUN_ROOT_DIR" >&2; exit 1; }
  image="${TURBOVLA_DOCKER_IMAGE:-turbovla-joint-songling:20260803}"
  exec docker run --rm --name "${TURBOVLA_CONTAINER_NAME:-${RUN_ID}}" --ipc=host \
    --gpus "device=${selected_gpu}" -e IN_TURBOVLA_CONTAINER=1 -e CUDA_VISIBLE_DEVICES=0 \
    -e JOINT_SONGLING_DATA_ROOT -e JOINT_SONGLING_OVERLAY_ROOT -e BERT_MODEL_PATH -e DINOV3_MODEL_PATH \
    -e RUN_ROOT_DIR -e RUN_ID \
    -e CONFIG_YAML="${CONFIG_YAML:-experiments/joint_songling/configs/0812_closed_patchvision_t2_gpu7.yaml}" \
    -e SMOKE_TEST="${SMOKE_TEST:-0}" -e WANDB_MODE="${WANDB_MODE:-disabled}" \
    -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    -e OMP_NUM_THREADS -e MKL_NUM_THREADS \
    -v "${repo_root}:${repo_root}" -v /data:/data -w "$repo_root" "$image" \
    bash scripts/joint_songling/train_0812_closed_patchvision_t2_gpu7.sh "$@"
fi

: "${BERT_MODEL_PATH:?Set BERT_MODEL_PATH}"
: "${DINOV3_MODEL_PATH:?Set DINOV3_MODEL_PATH}"
: "${RUN_ROOT_DIR:?Set RUN_ROOT_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${repo_root}:${repo_root}/third_party/starvla_runtime:${PYTHONPATH:-}"

python3 - "$JOINT_SONGLING_DATA_ROOT" "$JOINT_SONGLING_OVERLAY_ROOT" <<'PY'
import json, shutil, sys
from pathlib import Path
import pandas as pd

source, overlay = map(Path, sys.argv[1:])
task = "Pick up the bread with the right gripper, transfer it to the left gripper, and place it in the bowl."
if not source.is_dir():
    raise SystemExit(f"dataset root does not exist: {source}")
info = json.loads((source / "meta/info.json").read_text())
features = info["features"]
expected = {
    "observation.images.top": (405, 720, 3),
    "observation.images.gripper_left": (480, 640, 3),
    "observation.images.gripper_right": (480, 640, 3),
}
if info.get("total_episodes") != 202 or info.get("total_frames") != 125558:
    raise SystemExit("0812 dataset totals do not match the audited 202/125558 contract")
if tuple(features["observation.state"]["shape"]) != (20,) or tuple(features["action"]["shape"]) != (14,):
    raise SystemExit("expected state=20D and action=14D")
for key, shape in expected.items():
    if tuple(features[key]["shape"]) != shape:
        raise SystemExit(f"unexpected {key} shape: {features[key]}")
if overlay.exists():
    contract = json.loads((overlay / "overlay_contract.json").read_text())
    if contract.get("source") != str(source.resolve()) or contract.get("task") != task:
        raise SystemExit(f"overlay contract mismatch: {overlay}")
else:
    (overlay / "data/chunk-000").mkdir(parents=True)
    (overlay / "meta/episodes/chunk-000").mkdir(parents=True)
    (overlay / "videos").symlink_to(source.resolve() / "videos", target_is_directory=True)
    data = pd.concat([pd.read_parquet(p) for p in sorted((source / "data").glob("*/*.parquet"))], ignore_index=True)
    episodes = pd.concat([pd.read_parquet(p) for p in sorted((source / "meta/episodes").glob("*/*.parquet"))], ignore_index=True)
    data["task_index"] = 0
    episodes["tasks"] = [[task] for _ in range(len(episodes))]
    data.to_parquet(overlay / "data/chunk-000/file-000.parquet", index=False)
    episodes.to_parquet(overlay / "meta/episodes/chunk-000/file-000.parquet", index=False)
    filtered = dict(info); filtered["total_tasks"] = 1; filtered["splits"] = {"train": "0:202"}
    (overlay / "meta/info.json").write_text(json.dumps(filtered, indent=2) + "\n")
    pd.DataFrame({"task_index": [0]}, index=pd.Index([task], name="task")).to_parquet(overlay / "meta/tasks.parquet")
    modality = {
      "state": {"left_joints": {"start": 0, "end": 6, "original_key": "observation.state"}, "left_endpoint": {"start": 6, "end": 9, "original_key": "observation.state"}, "left_gripper": {"start": 9, "end": 10, "original_key": "observation.state"}, "right_joints": {"start": 10, "end": 16, "original_key": "observation.state"}, "right_endpoint": {"start": 16, "end": 19, "original_key": "observation.state"}, "right_gripper": {"start": 19, "end": 20, "original_key": "observation.state"}},
      "action": {"left_joints": {"start": 0, "end": 6, "original_key": "action"}, "left_gripper": {"start": 6, "end": 7, "original_key": "action"}, "right_joints": {"start": 7, "end": 13, "original_key": "action"}, "right_gripper": {"start": 13, "end": 14, "original_key": "action"}},
      "video": {"top": {"original_key": "observation.images.top"}, "gripper_left": {"original_key": "observation.images.gripper_left"}, "gripper_right": {"original_key": "observation.images.gripper_right"}},
      "annotation": {"human.action.task_description": {"original_key": "task_index"}},
    }
    (overlay / "meta/modality.json").write_text(json.dumps(modality, indent=2) + "\n")
    (overlay / "overlay_contract.json").write_text(json.dumps({"source": str(source.resolve()), "total_episodes": 202, "total_frames": 125558, "task": task, "camera_order": ["top", "gripper_left", "gripper_right"], "image_layout": "joint_songling", "temporal_window": [-1, 0], "relative_action_anchor": "current", "gripper_mode": "binary_absolute_closed_zero"}, indent=2) + "\n")
print("[INFO] 0812 PatchVision T2 overlay verified")
PY

config_yaml="${CONFIG_YAML:-experiments/joint_songling/configs/0812_closed_patchvision_t2_gpu7.yaml}"
if [[ "${SMOKE_TEST:-0}" == "1" ]]; then
  python3 - "$config_yaml" /tmp/0812_patchvision_t2_smoke.yaml <<'PY'
import sys, yaml
from pathlib import Path
cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
cfg["run_id"] = cfg["run_id"] + "_smoke"
for key, value in {"max_train_steps": 1, "num_warmup_steps": 0, "save_interval": 1000, "eval_interval": 1000, "logging_frequency": 1}.items(): cfg["trainer"][key] = value
Path(sys.argv[2]).write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
  config_yaml=/tmp/0812_patchvision_t2_smoke.yaml
fi
exec python3 -m accelerate.commands.launch --num_processes 1 \
  third_party/starvla_runtime/starVLA/training/train_robotwin_clean_act_pi05_recipe.py \
  --config_yaml "$config_yaml" --run_root_dir "$RUN_ROOT_DIR" --run_id "$RUN_ID" "$@"
