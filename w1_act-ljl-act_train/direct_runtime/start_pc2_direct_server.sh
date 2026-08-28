#!/usr/bin/env bash
set -euo pipefail

W1_ACT_ROOT=/home/dexforce/w1/w1_act
RUNTIME_DEPS=/home/dexforce/.local/share/xwiz-act-server/runtime-deps
POLICY_PATH=/home/dexforce/workspace/outputs/act_popcorn_45w_xwiz

export PYTHONPATH="${RUNTIME_DEPS}:${W1_ACT_ROOT}/w1_lerobot/src:${W1_ACT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec python3 -m xwiz_act_server.server \
  --host 0.0.0.0 \
  --port 8889 \
  --policy-path "${POLICY_PATH}" \
  --device cuda
