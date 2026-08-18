#!/usr/bin/env bash
set -euo pipefail

: "${LEROBOT_GPUS:?Set LEROBOT_GPUS to device=<gpu-id> before launching the container}"

if [[ ! "$LEROBOT_GPUS" =~ ^device=[0-9]+$ ]]; then
  echo "LEROBOT_GPUS must use exactly one GPU as device=<gpu-id>" >&2
  exit 2
fi

RUNTIME_HOME=/data/wengyikun/pi05_official_libero_smoke/home
HF_HOME=/data/wengyikun/.cache/huggingface
HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$RUNTIME_HOME" "$HF_HOME" "$HF_DATASETS_CACHE" /data/wengyikun/.cache/triton /data/wengyikun/.cache/torchinductor

exec docker run --rm \
  --gpus "${LEROBOT_GPUS}" \
  --user "$(id -u):$(id -g)" \
  --ipc=host \
  --shm-size=16g \
  -e HOME="$RUNTIME_HOME" \
  -e USER=wengyikun \
  -e LOGNAME=wengyikun \
  -e HF_ENDPOINT=https://hf-mirror.com \
  -e HF_HOME="$HF_HOME" \
  -e HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
  -e MUJOCO_GL=egl \
  -e PYTHONPATH=/data/wengyikun/lerobot_py311/site-packages:/workspace/lerobot/src \
  -e TRITON_CACHE_DIR=/data/wengyikun/.cache/triton \
  -e TORCHINDUCTOR_CACHE_DIR=/data/wengyikun/.cache/torchinductor \
  -v /home/wengyikun/lerobot:/workspace/lerobot:ro \
  -v /home/wengyikun/.cache:/home/wengyikun/.cache \
  -v /data/wengyikun:/data/wengyikun \
  -w /workspace/lerobot \
  lerobot-pi05-libero-smoke:20260717-libero-rootfix-assets "$@"
