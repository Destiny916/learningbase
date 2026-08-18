#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

require_file() {
    local path="$1"
    [[ -f "${ROOT_DIR}/${path}" ]] || {
        printf 'missing required file: %s\n' "${path}" >&2
        return 1
    }
}

require_text() {
    local path="$1"
    local text="$2"
    rg -F --quiet -- "${text}" "${ROOT_DIR}/${path}" || {
        printf 'missing text in %s: %s\n' "${path}" "${text}" >&2
        return 1
    }
}

for path in Dockerfile compose.yaml .env.example .dockerignore \
    scripts/fix_registry_route.sh scripts/pull_image.sh scripts/start.sh scripts/verify.sh README.md; do
    require_file "${path}"
done

require_text .env.example '192.168.3.13:5000/dexsdk:ubuntu22.04-cuda12.8.0-h5ffmpeg-v3'
require_text .env.example 'DEXECHAIN_VERSION=0.1.6'
require_text .env.example 'EMBODICHAIN_VERSION=0.2.4'
require_text .env.example 'NVIDIA_CDI_DEVICE=nvidia.com/gpu=0'
require_text Dockerfile 'dexechain==${DEXECHAIN_VERSION}'
require_text Dockerfile 'embodichain==${EMBODICHAIN_VERSION}'
require_text compose.yaml '${NVIDIA_CDI_DEVICE:-nvidia.com/gpu=0}'
require_text compose.yaml '/tmp/.X11-unix:/tmp/.X11-unix:rw'
require_text compose.yaml '/usr/share/vulkan:/usr/share/vulkan:ro'
require_text scripts/verify.sh 'import dexechain'
require_text scripts/verify.sh 'import embodichain'
require_text scripts/verify.sh 'nvidia-smi'
require_text scripts/pull_image.sh '--retry 3'
require_text scripts/pull_image.sh 'registry preflight failed; docker pull will provide the final result'
require_text scripts/pull_image.sh 'fix_registry_route.sh'
require_text scripts/fix_registry_route.sh 'to 192.168.3.0/24 lookup main'
require_text scripts/start.sh 'HOST_DISPLAY="${DISPLAY:-}"'
require_text scripts/start.sh 'X11 authorization skipped'

printf 'docker configuration contract: PASS\n'
