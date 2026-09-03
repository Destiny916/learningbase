#!/usr/bin/env bash
# 类型：PC1 / 旧 ACT absolute 客户端。
# 逻辑：state/action 均按绝对值解释；不做 relative delta 还原，手部仍由 scalar 展开为六关节。
set -euo pipefail
W1_ACT_ROOT="${W1_ACT_ROOT:-/home/dexforce/w1/w1_act}"
RUNTIME_ROOT="${RUNTIME_ROOT:-${W1_ACT_ROOT}/xwiz_real_runtime}"
set +u
source /opt/ros/humble/setup.bash
source /home/dexforce/w1/install/setup.bash
set -u
export ROS_DOMAIN_ID=20
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export PYTHONPATH=${W1_ACT_ROOT}:${PYTHONPATH:-}
exec python3 -m xwiz_real_runtime.client_service --config ${RUNTIME_ROOT}/client_runtime_direct_pc2.json
