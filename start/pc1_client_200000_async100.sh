#!/usr/bin/env bash
# 类型：PC1 / ACT 200000 async100（插值）客户端。
# 逻辑：100 策略点×2=200 控制点；剩 30 控制点异步请求，身体做最多 30 点 LIPO，左右手 scalar 不融合。
set -euo pipefail
W1_ACT_ROOT="${W1_ACT_ROOT:-/home/dexforce/w1/w1_act}"
CONFIG_PATH="${CONFIG_PATH:-${W1_ACT_ROOT}/direct_runtime/client_runtime_200000_async100.json}"
set +u
source /opt/ros/humble/setup.bash
source /home/dexforce/w1/install/setup.bash
set -u
export ROS_DOMAIN_ID=20 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp XWIZ_ACTION_HORIZON=100 XWIZ_ASYNC_REPLAN=1
export PYTHONPATH="${W1_ACT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec python3 -m xwiz_real_runtime.client_service --config "${CONFIG_PATH}"
