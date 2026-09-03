#!/usr/bin/env bash
set -euo pipefail
W1_ACT_ROOT="${W1_ACT_ROOT:-/home/dexforce/w1/w1_act}"
CONFIG_PATH="${CONFIG_PATH:-${W1_ACT_ROOT}/direct_runtime/client_runtime_200000_nointerp_async.json}"
set +u
source /opt/ros/humble/setup.bash
source /home/dexforce/w1/install/setup.bash
set -u
export ROS_DOMAIN_ID=20 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export XWIZ_ACTION_HORIZON=100 XWIZ_ASYNC_REPLAN=1
export XWIZ_ASYNC_SAMPLE_FACTOR=1 XWIZ_ASYNC_BLEND_POINTS=15
export PYTHONPATH="${W1_ACT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec python3 -m xwiz_real_runtime.client_service --config "${CONFIG_PATH}"
