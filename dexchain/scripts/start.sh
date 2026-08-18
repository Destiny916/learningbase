#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"
[[ -f "${ENV_FILE}" ]] || ENV_FILE="${ROOT_DIR}/.env.example"
HOST_DISPLAY="${DISPLAY:-}"

set -a
source "${ENV_FILE}"
set +a

if [[ -n "${HOST_DISPLAY}" ]]; then
    export DISPLAY="${HOST_DISPLAY}"
fi

BASE_IMAGE="${BASE_IMAGE:-192.168.3.13:5000/dexsdk:ubuntu22.04-cuda12.8.0-h5ffmpeg-v3}"
NVIDIA_CDI_DEVICE="${NVIDIA_CDI_DEVICE:-nvidia.com/gpu=0}"

docker compose version >/dev/null
docker image inspect "${BASE_IMAGE}" >/dev/null 2>&1 || {
    printf 'base image is missing; run scripts/pull_image.sh first\n' >&2
    exit 1
}

[[ -d /tmp/.X11-unix ]] || {
    printf '/tmp/.X11-unix is missing; X11 applications will not work\n' >&2
    exit 1
}

printf 'checking CDI GPU device %s\n' "${NVIDIA_CDI_DEVICE}"
docker run --rm \
    --device "${NVIDIA_CDI_DEVICE}" \
    --entrypoint nvidia-smi \
    "${BASE_IMAGE}" \
    --query-gpu=index,name --format=csv,noheader

if command -v xhost >/dev/null && [[ -n "${DISPLAY:-}" ]]; then
    if ! xhost +si:localuser:root >/dev/null 2>&1; then
        printf 'X11 authorization skipped: cannot access display %s\n' "${DISPLAY}" >&2
    fi
fi

cd "${ROOT_DIR}"
docker compose up -d --build
docker compose ps
