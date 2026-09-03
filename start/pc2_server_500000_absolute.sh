#!/usr/bin/env bash
# 类型：PC2 / 500000 旧 ACT absolute policy server（默认端口 8889）。
# 逻辑：按旧 checkpoint 自带 processor 处理绝对 state/action，不套用 relative q01/q99。
set -euo pipefail

W1_ACT_ROOT="${W1_ACT_ROOT:-/home/dexforce/w1/w1_act}"
RUNTIME_DEPS="${RUNTIME_DEPS:-/home/dexforce/.local/share/xwiz-act-server/runtime-deps}"
POLICY_PATH="${POLICY_PATH:-${XWIZ_ACT_POLICY_PATH:-/home/dexforce/workspace/outputs/500000_pc2}}"
PC2_PORT="${PC2_PORT:-8889}"

export PYTHONPATH="${RUNTIME_DEPS}:${W1_ACT_ROOT}/w1_lerobot/src:${W1_ACT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec python3 -m xwiz_act_server.server \
  --host 0.0.0.0 \
  --port "${PC2_PORT}" \
  --policy-path "${POLICY_PATH}" \
  --device cuda
