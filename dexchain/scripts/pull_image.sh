#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"
[[ -f "${ENV_FILE}" ]] || ENV_FILE="${ROOT_DIR}/.env.example"

set -a
source "${ENV_FILE}"
set +a

BASE_IMAGE="${BASE_IMAGE:-192.168.3.13:5000/dexsdk:ubuntu22.04-cuda12.8.0-h5ffmpeg-v3}"
REGISTRY="${BASE_IMAGE%%/*}"
MIN_FREE_KB=$((45 * 1024 * 1024))

command -v docker >/dev/null || {
    printf 'docker is not installed\n' >&2
    exit 1
}
docker info >/dev/null

"${ROOT_DIR}/scripts/fix_registry_route.sh"

FREE_KB="$(df -Pk /var/lib/docker | awk 'NR == 2 {print $4}')"
if [[ -z "${FREE_KB}" || "${FREE_KB}" -lt "${MIN_FREE_KB}" ]]; then
    printf 'insufficient free space for the DexSDK image: %s KB available, %s KB required\n' \
        "${FREE_KB:-unknown}" "${MIN_FREE_KB}" >&2
    exit 1
fi

if command -v curl >/dev/null; then
    if ! curl --fail --silent --show-error \
        --connect-timeout 5 --max-time 10 \
        --retry 3 --retry-delay 2 --retry-all-errors \
        "http://${REGISTRY}/v2/" >/dev/null; then
        printf 'registry preflight failed; docker pull will provide the final result\n' >&2
    fi
fi

printf 'pulling %s\n' "${BASE_IMAGE}"
docker pull "${BASE_IMAGE}"

IMAGE_ID="$(docker image inspect "${BASE_IMAGE}" --format '{{.Id}}')"
printf 'base image ready: %s (%s)\n' "${BASE_IMAGE}" "${IMAGE_ID}"
