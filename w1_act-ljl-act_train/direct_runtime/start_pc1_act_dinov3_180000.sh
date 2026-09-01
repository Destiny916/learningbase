#!/usr/bin/env bash
set -euo pipefail

W1_ACT_ROOT=/home/dexforce/w1/w1_act
CONFIG_PATH="${W1_ACT_ROOT}/direct_runtime/client_runtime_180000.json"

set +u
source /opt/ros/humble/setup.bash
source /home/dexforce/w1/install/setup.bash
set -u
export ROS_DOMAIN_ID=20
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

test -f "${CONFIG_PATH}" || { echo "missing 180000 client config: ${CONFIG_PATH}" >&2; exit 1; }
test -f "${W1_ACT_ROOT}/xwiz_real_runtime/client_service_160000.py" || {
  echo "missing protocol-v2 client service" >&2
  exit 1
}

export PYTHONPATH="${W1_ACT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec python3 -m xwiz_real_runtime.client_service_160000 --config "${CONFIG_PATH}"
