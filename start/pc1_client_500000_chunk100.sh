#!/usr/bin/env bash
set -euo pipefail
W1_ACT_ROOT="${W1_ACT_ROOT:-/home/dexforce/w1/w1_act}"
CONFIG_PATH="${CONFIG_PATH:-${W1_ACT_ROOT}/direct_runtime/client_runtime_500000_chunk100.json}"
set +u
source /opt/ros/humble/setup.bash
source /home/dexforce/w1/install/setup.bash
set -u
export ROS_DOMAIN_ID=20 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp XWIZ_ACTION_HORIZON=100
unset XWIZ_ASYNC_REPLAN XWIZ_ASYNC_SAMPLE_FACTOR XWIZ_ASYNC_BLEND_POINTS
export PYTHONPATH="${W1_ACT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec python3 -m xwiz_real_runtime.client_service --config "${CONFIG_PATH}"
