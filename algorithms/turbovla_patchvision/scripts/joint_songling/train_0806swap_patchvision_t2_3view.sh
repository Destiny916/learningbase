#!/usr/bin/env bash
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to one idle host GPU.}"

export TURBOVLA_CONFIG_YAML=experiments/joint_songling/configs/0806swap_patchvision_t2_3view.yaml
export RUN_ID="${RUN_ID:-turbovla_0806swap_patchvision_t2_3view}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "${repo_root}/scripts/joint_songling/train_0806swap_3view_gpu6.sh" "$@"
