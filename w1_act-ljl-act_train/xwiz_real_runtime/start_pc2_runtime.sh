#!/usr/bin/env bash
set -euo pipefail

W1_ACT_ROOT=/home/dexforce/w1/w1_act
RUNTIME_ROOT="${W1_ACT_ROOT}/xwiz_real_runtime"

export ROS_DOMAIN_ID=20
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
set +u
source /opt/ros/humble/setup.bash
source /home/dexforce/w1/install/setup.bash
set -u
export PYTHONPATH="${W1_ACT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

exec python3 -m xwiz_real_runtime.client_service \
  --config "${RUNTIME_ROOT}/client_runtime.json"
