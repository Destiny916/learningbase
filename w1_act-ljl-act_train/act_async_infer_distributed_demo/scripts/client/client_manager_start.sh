#!/usr/bin/env bash
set -euo pipefail

W1_ACT_HOME="${W1_ACT_HOME:-/home/dexforce/w1/w1_act}"
W1_CLIENT_CONFIG="${W1_CLIENT_CONFIG:-${W1_ACT_HOME}/act_async_infer_distributed_demo/scripts/client/client_config.json}"

python -m act_async_infer_distributed_demo.scripts.client.run_robot_client \
    --config "${W1_CLIENT_CONFIG}" \
    --service
