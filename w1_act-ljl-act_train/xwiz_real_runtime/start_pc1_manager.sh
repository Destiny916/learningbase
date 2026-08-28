#!/usr/bin/env bash
set -euo pipefail

W1_ACT_ROOT=/home/dexforce/w1/w1_act

export ROS_DOMAIN_ID=20
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
set +u
source /opt/ros/humble/setup.bash
source /home/dexforce/w1/install/setup.bash
set -u
export PYTHONPATH="${W1_ACT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

exec python3 -m xwiz_real_runtime.manager_service
