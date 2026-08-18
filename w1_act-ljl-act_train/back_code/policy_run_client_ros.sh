#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${1:-${SCRIPT_DIR}/../test.json}"
export PYTHONPATH=/root/w1_act:$PYTHONPATH
# （可选）如果本次不使用手的标量输入，这一行可以删掉或保留：
python3 inference_codes/hand_scalar.py &

# 启动 ACT 推理节点，只通过 config json 传参
python3 inference_codes/smooth_policy_new_act_remote_client_ros_from_json.py --config "${CONFIG_PATH}"
