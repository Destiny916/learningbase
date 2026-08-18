#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

selected_gpu="${CUDA_VISIBLE_DEVICES:-0}"
training_variant="${TRAINING_VARIANT:-warm}"
export JOINT_SONGLING_DATA_ROOT="${JOINT_SONGLING_DATA_ROOT:-/data/wengyikun/datasets/joint_songling/0812_closed_gripper_zero_without_ep173_174}"
case "$training_variant" in
  warm)
    config_yaml="experiments/joint_songling/configs/0812_closed_gripper_top_padded_warm.yaml"
    : "${TURBOVLA_INITIAL_CKPT:?Set retry8 steps_200000_model.safetensors for warm training.}"
    ;;
  fresh)
    config_yaml="experiments/joint_songling/configs/0812_closed_gripper_top_padded_fresh.yaml"
    ;;
  *)
    echo "TRAINING_VARIANT must be warm or fresh, got: $training_variant" >&2
    exit 1
    ;;
esac

if [[ "${IN_TURBOVLA_CONTAINER:-0}" != "1" ]]; then
  : "${JOINT_SONGLING_DATA_ROOT:?Set the source LeRobot v3 dataset root.}"
  : "${BERT_MODEL_PATH:?Set a local bert-base-uncased directory.}"
  : "${DINOV3_MODEL_PATH:?Set a local DINOv3 ViT-L directory.}"
  : "${RUN_ROOT_DIR:?Set a new writable output directory.}"
  [[ ! -e "$RUN_ROOT_DIR" ]] || {
    echo "Refusing to overwrite RUN_ROOT_DIR: $RUN_ROOT_DIR" >&2
    exit 1
  }
  image="${TURBOVLA_DOCKER_IMAGE:-turbovla-joint-songling:20260803}"
  container_name="${TURBOVLA_CONTAINER_NAME:-turbovla_0812_closed_gripper_top_padded_${training_variant}_gpu${selected_gpu}}"
  docker_args=(--rm --name "$container_name" --ipc=host)
  if [[ "${PREPARE_ONLY:-0}" != "1" ]]; then
    docker_args+=(--gpus "device=${selected_gpu}")
  fi
  exec docker run "${docker_args[@]}" \
    -e IN_TURBOVLA_CONTAINER=1 \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e TRAINING_VARIANT="$training_variant" \
    -e JOINT_SONGLING_DATA_ROOT \
    -e TURBOVLA_INITIAL_CKPT="${TURBOVLA_INITIAL_CKPT:-}" \
    -e BERT_MODEL_PATH \
    -e DINOV3_MODEL_PATH \
    -e RUN_ROOT_DIR \
    -e PREPARE_ONLY="${PREPARE_ONLY:-0}" \
    -e SMOKE_TEST="${SMOKE_TEST:-0}" \
    -e WANDB_MODE="${WANDB_MODE:-disabled}" \
    -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    -e RUN_ID="${RUN_ID:-turbovla_0812_closed_gripper_zero_top_padded_${training_variant}_gpu${selected_gpu}}" \
    -e JOINT_SONGLING_OVERLAY_ROOT="${JOINT_SONGLING_OVERLAY_ROOT:-}" \
    -v "${repo_root}:${repo_root}" \
    -v /data:/data \
    -w "$repo_root" \
    "$image" bash scripts/joint_songling/train_0812_closed_gripper_top_padded.sh "$@"
fi

: "${JOINT_SONGLING_DATA_ROOT:?Set the source LeRobot v3 dataset root.}"
: "${BERT_MODEL_PATH:?Set a local bert-base-uncased directory.}"
: "${DINOV3_MODEL_PATH:?Set a local DINOv3 ViT-L directory.}"
: "${RUN_ROOT_DIR:?Set a writable output directory.}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export JOINT_SONGLING_OVERLAY_ROOT="${JOINT_SONGLING_OVERLAY_ROOT:-${JOINT_SONGLING_DATA_ROOT}_turbovla_overlay}"
export PYTHONPATH="${repo_root}:${repo_root}/third_party/starvla_runtime:${PYTHONPATH:-}"

python3 - "$JOINT_SONGLING_DATA_ROOT" "$JOINT_SONGLING_OVERLAY_ROOT" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

source, overlay = map(Path, sys.argv[1:])
task = (
    "first pick up the bread with the right hand, then hand it to the left hand "
    "at the middle point, then place the bread in the bowl with the left hand."
)
if not source.is_dir():
    raise SystemExit(f"dataset root does not exist: {source}")

info = json.loads((source / "meta/info.json").read_text())
features = info["features"]
expected_cameras = {
    "observation.images.top": (405, 720, 3),
    "observation.images.gripper_left": (480, 640, 3),
    "observation.images.gripper_right": (480, 640, 3),
}
if info.get("total_episodes") != 202 or info.get("total_frames") != 125558:
    raise SystemExit("expected source totals 202 episodes and 125558 frames")
if tuple(features["observation.state"]["shape"]) != (20,):
    raise SystemExit("expected observation.state=20D")
if tuple(features["action"]["shape"]) != (14,):
    raise SystemExit("expected action=14D")
for key, shape in expected_cameras.items():
    if tuple(features.get(key, {}).get("shape", ())) != shape:
        raise SystemExit(f"unexpected image feature {key}: {features.get(key)}")

data_paths = sorted((source / "data").glob("*/*.parquet"))
episode_paths = sorted((source / "meta/episodes").glob("*/*.parquet"))
if not data_paths or not episode_paths:
    raise SystemExit("source data or episode metadata is missing")
data = pd.concat([pd.read_parquet(path) for path in data_paths], ignore_index=True)
episodes = pd.concat([pd.read_parquet(path) for path in episode_paths], ignore_index=True)
if set(map(int, episodes["episode_index"])) != set(range(202)):
    raise SystemExit("expected contiguous source episode IDs 0..201")
if len(data) != 125558 or data["episode_index"].nunique() != 202:
    raise SystemExit("data totals do not match source metadata")
states = np.stack(data["observation.state"].to_numpy())
actions = np.stack(data["action"].to_numpy())
if states.shape != (125558, 20) or actions.shape != (125558, 14):
    raise SystemExit(f"unexpected state/action shapes: {states.shape}/{actions.shape}")
if not np.isfinite(states).all() or not np.isfinite(actions).all():
    raise SystemExit("state/action contains non-finite values")
for name, values in {
    "state.left_gripper": states[:, 9],
    "state.right_gripper": states[:, 19],
    "action.left_gripper": actions[:, 6],
    "action.right_gripper": actions[:, 13],
}.items():
    if values.min() < -1e-6 or values.max() > 0.101:
        raise SystemExit(f"{name} is outside expected continuous absolute range")

contract = {
    "source": str(source.resolve()),
    "total_episodes": 202,
    "total_frames": 125558,
    "task": task,
    "camera_order": ["top", "gripper_left", "gripper_right"],
    "image_layout": "joint_songling_top_padded",
    "relative_action_anchor": "current",
    "gripper_mode": "binary_absolute_closed_zero",
}
contract_path = overlay / "overlay_contract.json"
if overlay.exists():
    if not contract_path.is_file() or json.loads(contract_path.read_text()) != contract:
        raise SystemExit(f"existing overlay contract differs: {overlay}")
    overlay_data = pd.read_parquet(overlay / "data/chunk-000/file-000.parquet")
    overlay_episodes = pd.read_parquet(overlay / "meta/episodes/chunk-000/file-000.parquet")
    if len(overlay_data) != 125558 or len(overlay_episodes) != 202:
        raise SystemExit("existing overlay totals are invalid")
    if set(overlay_data["task_index"]) != {0}:
        raise SystemExit("existing overlay does not have a uniform task index")
    print(f"[INFO] reusing verified overlay={overlay}")
else:
    (overlay / "data/chunk-000").mkdir(parents=True)
    (overlay / "meta/episodes/chunk-000").mkdir(parents=True)
    (overlay / "videos").symlink_to(source.resolve() / "videos", target_is_directory=True)
    data["task_index"] = 0
    episodes["tasks"] = [[task] for _ in range(len(episodes))]
    data.to_parquet(overlay / "data/chunk-000/file-000.parquet", index=False)
    episodes.to_parquet(overlay / "meta/episodes/chunk-000/file-000.parquet", index=False)
    filtered_info = dict(info)
    filtered_info["total_tasks"] = 1
    filtered_info["splits"] = {"train": "0:202"}
    (overlay / "meta/info.json").write_text(json.dumps(filtered_info, indent=2) + "\n")
    pd.DataFrame({"task_index": [0]}, index=pd.Index([task], name="task")).to_parquet(
        overlay / "meta/tasks.parquet"
    )
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
    (overlay / "meta/modality.json").write_text(json.dumps(modality, indent=2) + "\n")
    contract_path.write_text(json.dumps(contract, indent=2) + "\n")
    print(f"[INFO] created uniform-task overlay={overlay}")

print("[INFO] total_episodes=202 total_frames=125558 all=train")
print("[INFO] state=20D action=14D horizon=50")
print("[INFO] relative=left_joints,right_joints; absolute=endpoint_xyz,continuous_grippers")
print("[INFO] camera_order=top,gripper_left,gripper_right")
print("[INFO] preprocessing=top 405x720 pad(157,158)->720x720->224x224; wrists 480x640->224x224")
print(f"[INFO] task={task}")
PY

if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
  echo "[INFO] PREPARE_ONLY=1; formal training was not started."
  exit 0
fi

if [[ "${SMOKE_TEST:-0}" == "1" ]]; then
  smoke_config="/tmp/turbovla_0812_closed_${training_variant}_smoke.yaml"
  python3 - "$config_yaml" "$smoke_config" <<'PY'
import sys
from pathlib import Path

import yaml

source, destination = map(Path, sys.argv[1:])
config = yaml.safe_load(source.read_text())
config["run_id"] = f"{config['run_id']}_smoke"
trainer = config["trainer"]
trainer["max_train_steps"] = 1
trainer["num_warmup_steps"] = 0
trainer["save_interval"] = 1000
trainer["eval_interval"] = 1000
trainer["logging_frequency"] = 1
destination.write_text(yaml.safe_dump(config, sort_keys=False))
PY
  config_yaml="$smoke_config"
  echo "[INFO] SMOKE_TEST=1; using file-level max_train_steps=1 config."
fi

exec python3 -m accelerate.commands.launch --num_processes 1 \
  third_party/starvla_runtime/starVLA/training/train_robotwin_clean_act_pi05_recipe.py \
  --config_yaml "$config_yaml" \
  --run_root_dir "$RUN_ROOT_DIR" \
  --run_id "${RUN_ID:-turbovla_0812_closed_gripper_zero_top_padded_${training_variant}}" \
  "$@"
