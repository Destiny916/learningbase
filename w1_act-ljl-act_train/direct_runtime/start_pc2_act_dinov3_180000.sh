#!/usr/bin/env bash
set -euo pipefail

W1_ACT_ROOT=/home/dexforce/w1/w1_act
POLICY_PATH=/home/dexforce/workspace/outputs/act_dinov3_popcorn_0827_19d_relative_arm_joints_nextstate_chunk16_b64_500k_gpu1_2_180000_pc2
ACT_DINOV3_SRC=/home/dexforce/workspace/act_dinov3_c8c674b/src
RUNTIME_DEPS=/home/dexforce/workspace/act_dinov3_runtime_deps

for required in \
  "${POLICY_PATH}/config.json" \
  "${POLICY_PATH}/model.safetensors" \
  "${POLICY_PATH}/relative_stats/relative_state_q01_q99.json" \
  "${POLICY_PATH}/relative_stats/relative_action_chunk16_q01_q99.json" \
  "${ACT_DINOV3_SRC}/lerobot" \
  "${W1_ACT_ROOT}/xwiz_act_server/server_160000.py" \
  "${W1_ACT_ROOT}/direct_runtime/run_server_180000.py"
do
  test -e "${required}" || { echo "missing 180000 runtime path: ${required}" >&2; exit 1; }
done

export PYTHONPATH="${RUNTIME_DEPS}:${ACT_DINOV3_SRC}:${W1_ACT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec python3 "${W1_ACT_ROOT}/direct_runtime/run_server_180000.py" \
  --host 0.0.0.0 \
  --port 8892 \
  --policy-path "${POLICY_PATH}" \
  --device cuda
