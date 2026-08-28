#!/usr/bin/env bash
set -euo pipefail
set +u
source /opt/ros/humble/setup.bash
source /home/dexforce/w1/install/setup.bash
set -u
export ROS_DOMAIN_ID=20
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export PYTHONPATH=/home/dexforce/w1/w1_act:${PYTHONPATH:-}
exec python3 -m xwiz_real_runtime.black_wrist_images
