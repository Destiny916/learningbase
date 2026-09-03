#!/usr/bin/env bash
# 类型：PC1 / ACT 200000 async100（无插值）客户端。
# 逻辑：100 策略点直接作为 100 控制点；剩 15 点请求下一块，身体最多 15 点 LIPO，手部 scalar 直接取新块。
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
