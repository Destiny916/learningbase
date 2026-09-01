#!/usr/bin/env bash
set -euo pipefail

W1_ACT_ROOT="${W1_ACT_ROOT:-/home/dexforce/w1/w1_act}"
POLICY_PATH="${POLICY_PATH:-/home/dexforce/workspace/outputs/220000_pc2}"
ACT_DINOV3_SRC="${ACT_DINOV3_SRC:-/home/dexforce/workspace/act_dinov3_c8c674b/src}"
RUNTIME_DEPS="${RUNTIME_DEPS:-/home/dexforce/workspace/act_dinov3_runtime_deps}"
PC2_PORT="${PC2_PORT:-8889}"

for required in \
  "${POLICY_PATH}/config.json" \
  "${POLICY_PATH}/model.safetensors" \
  "${POLICY_PATH}/relative_stats/relative_state_q01_q99.json" \
  "${POLICY_PATH}/relative_stats/relative_action_chunk16_q01_q99.json" \
  "${ACT_DINOV3_SRC}/lerobot" \
  "${W1_ACT_ROOT}/xwiz_act_server/server_160000.py"
do
  test -e "${required}" || { echo "missing 220000 runtime path: ${required}" >&2; exit 1; }
done

export PYTHONPATH="${RUNTIME_DEPS}:${ACT_DINOV3_SRC}:${W1_ACT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec python3 -m xwiz_act_server.server_160000 \
  --host 0.0.0.0 \
  --port "${PC2_PORT}" \
  --policy-path "${POLICY_PATH}" \
  --device cuda
