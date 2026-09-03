#!/usr/bin/env bash
# 类型：PC2 / ACT 500000 chunk100 policy server（默认端口 8894）。
# 逻辑：仅当 checkpoint config.json 确认 chunk100 时使用；PC2 只推理不控机。
set -euo pipefail
W1_ACT_ROOT="${W1_ACT_ROOT:-/home/dexforce/w1/w1_act}"
RUNTIME_DEPS="${RUNTIME_DEPS:-/home/dexforce/.local/share/xwiz-act-server/runtime-deps}"
POLICY_PATH="${POLICY_PATH:-/home/dexforce/workspace/outputs/act_0827_action_nextstate_v024_eefk_3cam_chunk100_gpu4_500k_500000_full100_pc2}"
PC2_PORT="${PC2_PORT:-8894}"
test -f "${POLICY_PATH}/config.json"
test -f "${POLICY_PATH}/model.safetensors"
export XWIZ_ACTION_HORIZON=100
export PYTHONPATH="${RUNTIME_DEPS}:${W1_ACT_ROOT}/w1_lerobot/src:${W1_ACT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec python3 -m xwiz_act_server.server --host 0.0.0.0 --port "${PC2_PORT}" --policy-path "${POLICY_PATH}" --device cuda
