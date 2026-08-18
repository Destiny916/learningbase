#!/usr/bin/env bash
set -euo pipefail

: "${LEROBOT_GPUS:?Set LEROBOT_GPUS to device=<gpu-id> before launching the container}"

if [[ ! "$LEROBOT_GPUS" =~ ^device=[0-9]+(,[0-9]+)*$ ]]; then
  echo "LEROBOT_GPUS must use device=<gpu-id> or device=<gpu-id>,<gpu-id>" >&2
  exit 2
fi

mkdir -p /data/wengyikun/.cache/triton /data/wengyikun/.cache/torchinductor

exec docker run --rm \
  --gpus "${LEROBOT_GPUS}" \
  --user "$(id -u):$(id -g)" \
  --ipc=host \
  --shm-size=16g \
  -e HOME=/home/wengyikun \
  -e USER=wengyikun \
  -e LOGNAME=wengyikun \
  -e PYTHONPATH=/data/wengyikun/lerobot_py311/site-packages:/workspace/lerobot/src \
  -e TRITON_CACHE_DIR=/data/wengyikun/.cache/triton \
  -e TORCHINDUCTOR_CACHE_DIR=/data/wengyikun/.cache/torchinductor \
  -v /home/wengyikun/lerobot:/workspace/lerobot \
  -v /home/wengyikun/.cache:/home/wengyikun/.cache \
  -v /data/wengyikun:/data/wengyikun \
  -w /workspace/lerobot \
  lerobot-pi05-train:20260706 "$@"
