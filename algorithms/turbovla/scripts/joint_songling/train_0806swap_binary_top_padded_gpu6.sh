#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

selected_gpu="${CUDA_VISIBLE_DEVICES:-6}"
if [[ "${IN_TURBOVLA_CONTAINER:-0}" != "1" ]]; then
  : "${JOINT_SONGLING_DATA_ROOT:?Set the source LeRobot v3 dataset root.}"
  : "${TURBOVLA_INITIAL_CKPT:?Set retry8 steps_200000_model.safetensors.}"
  : "${BERT_MODEL_PATH:?Set a local bert-base-uncased directory.}"
  : "${DINOV3_MODEL_PATH:?Set a local DINOv3 ViT-L directory.}"
  : "${RUN_ROOT_DIR:?Set a new writable output directory.}"
  [[ ! -e "$RUN_ROOT_DIR" ]] || {
    echo "Refusing to overwrite RUN_ROOT_DIR: $RUN_ROOT_DIR" >&2
    exit 1
  }
  image="${TURBOVLA_DOCKER_IMAGE:-turbovla-joint-songling:20260803}"
  container_name="${TURBOVLA_CONTAINER_NAME:-turbovla_0806swap_binary_top_padded_gpu6}"
  docker_args=(--rm --name "$container_name" --ipc=host)
  if [[ "${PREPARE_ONLY:-0}" != "1" ]]; then
    docker_args+=(--gpus "device=${selected_gpu}")
  fi
  exec docker run "${docker_args[@]}" \
    -e IN_TURBOVLA_CONTAINER=1 \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e JOINT_SONGLING_DATA_ROOT \
    -e TURBOVLA_INITIAL_CKPT \
    -e BERT_MODEL_PATH \
    -e DINOV3_MODEL_PATH \
    -e RUN_ROOT_DIR \
    -e PREPARE_ONLY="${PREPARE_ONLY:-0}" \
    -e SMOKE_TEST="${SMOKE_TEST:-0}" \
    -e WANDB_MODE="${WANDB_MODE:-disabled}" \
    -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    -e RUN_ID="${RUN_ID:-turbovla_0806swap_binary_gripper_top_padded_3view_gpu6}" \
    -e JOINT_SONGLING_OVERLAY_ROOT="${JOINT_SONGLING_OVERLAY_ROOT:-}" \
    -v "${repo_root}:${repo_root}" \
    -v /data:/data \
    -w "$repo_root" \
    "$image" bash scripts/joint_songling/train_0806swap_binary_top_padded_gpu6.sh "$@"
fi

: "${JOINT_SONGLING_DATA_ROOT:?Set the source LeRobot v3 dataset root.}"
: "${TURBOVLA_INITIAL_CKPT:?Set retry8 steps_200000_model.safetensors.}"
: "${BERT_MODEL_PATH:?Set a local bert-base-uncased directory.}"
: "${DINOV3_MODEL_PATH:?Set a local DINOv3 ViT-L directory.}"
: "${RUN_ROOT_DIR:?Set a writable output directory.}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export JOINT_SONGLING_OVERLAY_ROOT="${JOINT_SONGLING_OVERLAY_ROOT:-${RUN_ROOT_DIR}_overlay}"
export PYTHONPATH="${repo_root}:${repo_root}/third_party/starvla_runtime:${PYTHONPATH:-}"

python3 - "$JOINT_SONGLING_DATA_ROOT" "$JOINT_SONGLING_OVERLAY_ROOT" "$TURBOVLA_INITIAL_CKPT" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

source, overlay, checkpoint = map(Path, sys.argv[1:])
EXCLUDED_EPISODES = {16, 17}
task = "Pick up the bread with the right gripper, transfer it to the left gripper, and place it in the bowl."
expected_checkpoint_name = "steps_200000_model.safetensors"

if not source.is_dir():
    raise SystemExit(f"dataset root does not exist: {source}")
if not checkpoint.is_file() or checkpoint.name != expected_checkpoint_name:
    raise SystemExit(
        f"expected non-EMA {expected_checkpoint_name}, got {checkpoint}"
    )

info = json.loads((source / "meta" / "info.json").read_text())
features = info["features"]
if info.get("total_episodes") != 73 or info.get("total_frames") != 50360:
    raise SystemExit(
        f"expected source totals 73/50360, got {info.get('total_episodes')}/{info.get('total_frames')}"
    )
if tuple(features["observation.state"]["shape"]) != (20,):
    raise SystemExit("expected observation.state=20D")
if tuple(features["action"]["shape"]) != (14,):
    raise SystemExit("expected action=14D")
for key, shape in {
    "observation.images.top": (405, 720, 3),
    "observation.images.wrist_left": (480, 640, 3),
    "observation.images.wrist_right": (480, 640, 3),
}.items():
    if tuple(features.get(key, {}).get("shape", ())) != shape:
        raise SystemExit(f"unexpected image feature {key}: {features.get(key)}")

episode_files = sorted((source / "meta" / "episodes").glob("*/*.parquet"))
data_files = sorted((source / "data").glob("*/*.parquet"))
if not episode_files or not data_files:
    raise SystemExit("source episode/data parquet files are missing")
episodes = pd.concat([pd.read_parquet(path) for path in episode_files], ignore_index=True)
data = pd.concat([pd.read_parquet(path) for path in data_files], ignore_index=True)
source_episode_ids = set(int(value) for value in episodes["episode_index"])
if source_episode_ids != set(range(73)):
    raise SystemExit(f"expected source episode IDs 0..72, got {sorted(source_episode_ids)}")

allowed_episode_ids = sorted(source_episode_ids - EXCLUDED_EPISODES)
episodes = episodes[episodes["episode_index"].isin(allowed_episode_ids)].copy()
data = data[data["episode_index"].isin(allowed_episode_ids)].copy()
if allowed_episode_ids != [*range(16), *range(18, 73)]:
    raise SystemExit(f"unexpected allowed episode IDs: {allowed_episode_ids}")
if len(episodes) != 71 or len(data) != 49430:
    raise SystemExit(f"expected filtered totals 71/49430, got {len(episodes)}/{len(data)}")
if set(data["episode_index"]) != set(allowed_episode_ids):
    raise SystemExit("filtered data and episode metadata IDs differ")
row_counts = data.groupby("episode_index", sort=False).size().to_dict()
for row in episodes.itertuples(index=False):
    episode_id = int(row[episodes.columns.get_loc("episode_index")])
    length = int(row[episodes.columns.get_loc("length")])
    if row_counts.get(episode_id) != length:
        raise SystemExit(f"episode {episode_id} length mismatch")

states = np.stack(data["observation.state"].to_numpy())
actions = np.stack(data["action"].to_numpy())
if states.shape != (49430, 20) or actions.shape != (49430, 14):
    raise SystemExit(f"unexpected filtered state/action shapes: {states.shape}/{actions.shape}")
if not np.isfinite(states).all() or not np.isfinite(actions).all():
    raise SystemExit("state/action contains non-finite values")
for name, values in {
    "state.left_gripper": states[:, 9],
    "state.right_gripper": states[:, 19],
    "action.left_gripper": actions[:, 6],
    "action.right_gripper": actions[:, 13],
}.items():
    if not np.isclose(values[:, None], np.array([0.0, 0.1]), atol=1e-6).any(axis=1).all():
        raise SystemExit(f"{name} is not binary in physical units 0.0/0.1")

contract = {
    "source": str(source.resolve()),
    "excluded_episode_ids": sorted(EXCLUDED_EPISODES),
    "allowed_episode_ids": allowed_episode_ids,
    "total_episodes": 71,
    "total_frames": 49430,
    "task": task,
    "camera_order": ["top", "gripper_left", "gripper_right"],
    "image_layout": "joint_songling_top_padded",
    "relative_action_anchor": "current",
}
contract_path = overlay / "overlay_contract.json"
if overlay.exists():
    if not contract_path.is_file() or json.loads(contract_path.read_text()) != contract:
        raise SystemExit(f"existing overlay contract differs: {overlay}")
    overlay_data = pd.read_parquet(overlay / "data" / "chunk-000" / "file-000.parquet")
    overlay_episodes = pd.read_parquet(
        overlay / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    )
    if len(overlay_data) != 49430 or len(overlay_episodes) != 71:
        raise SystemExit("existing overlay totals are invalid")
    if EXCLUDED_EPISODES & set(overlay_data["episode_index"]):
        raise SystemExit("existing overlay contains an excluded episode")
    print(f"[INFO] reusing verified overlay={overlay}")
else:
    (overlay / "data" / "chunk-000").mkdir(parents=True)
    (overlay / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (overlay / "videos").symlink_to(source.resolve() / "videos", target_is_directory=True)
    data.to_parquet(overlay / "data" / "chunk-000" / "file-000.parquet", index=False)
    episodes.to_parquet(
        overlay / "meta" / "episodes" / "chunk-000" / "file-000.parquet", index=False
    )
    filtered_info = dict(info)
    filtered_info["total_episodes"] = 71
    filtered_info["total_frames"] = 49430
    filtered_info["splits"] = {"train": "0:71"}
    (overlay / "meta" / "info.json").write_text(json.dumps(filtered_info, indent=2) + "\n")
    tasks = pd.DataFrame(
        {"task_index": [0]},
        index=pd.Index([task], name="task"),
    )
    tasks.to_parquet(overlay / "meta" / "tasks.parquet")
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
            "gripper_left": {"original_key": "observation.images.wrist_left"},
            "gripper_right": {"original_key": "observation.images.wrist_right"},
        },
        "annotation": {
            "human.action.task_description": {"original_key": "task_index"}
        },
    }
    (overlay / "meta" / "modality.json").write_text(json.dumps(modality, indent=2) + "\n")
    contract_path.write_text(json.dumps(contract, indent=2) + "\n")
    print(f"[INFO] created filtered overlay={overlay}")

print("[INFO] episodes=71 frames=49430 excluded=16,17")
print("[INFO] state=20D; action=14D horizon=50")
print("[INFO] relative=left_joints,right_joints; absolute=endpoint_xyz,grippers")
print("[INFO] camera_order=top,gripper_left,gripper_right")
print("[INFO] preprocessing=top 405x720 pad(157,158)->720x720->224x224; wrists 480x640->224x224")
print(f"[INFO] task={task}")
print(f"[INFO] initialization={checkpoint} (non-EMA weights; optimizer/scheduler restart)")
PY

if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
  echo "[INFO] PREPARE_ONLY=1; formal training was not started."
  exit 0
fi

config_yaml="experiments/joint_songling/configs/0806swap_binary_top_padded_3view.yaml"
if [[ "${SMOKE_TEST:-0}" == "1" ]]; then
  smoke_config="/tmp/turbovla_binary_top_padded_smoke_config.yaml"
  python3 - "$config_yaml" "$smoke_config" <<'PY'
import sys
from pathlib import Path

import yaml

source, destination = map(Path, sys.argv[1:])
config = yaml.safe_load(source.read_text())
config["run_id"] = "turbovla_binary_top_padded_smoke"
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
  --run_id "${RUN_ID:-turbovla_0806swap_binary_gripper_top_padded_3view_gpu6}" \
  "$@"
