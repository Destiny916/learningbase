#!/usr/bin/env bash
set -euo pipefail

# 1) 相机中继（后台常驻）
setsid ./camera_hub.sh >/tmp/camera_hub.log 2>&1 &
hub_pid=$!
echo "camera_hub pid=$hub_pid"

# 2) A / B 后台跑（它们从 shmsrc 读帧，不再占用 /dev/video99）
setsid ./policy_run_policya.sh & pidA=$!; pgidA=$(ps -o pgid= "$pidA" | tr -d ' ')
setsid ./policy_run_policyb.sh & pidB=$!; pgidB=$(ps -o pgid= "$pidB" | tr -d ' ')
echo "A pid=$pidA pgid=$pgidA   B pid=$pidB pgid=$pgidB"

cleanup() {
  echo "Stopping mux / A / B / camera_hub ..."
  for g in "$pgidA" "$pgidB"; do
    kill -INT "-$g" 2>/dev/null || true; sleep 0.2
    kill -TERM "-$g" 2>/dev/null || true; sleep 0.2
    kill -KILL "-$g" 2>/dev/null || true
  done
  kill -INT "$hub_pid" 2>/dev/null || true; sleep 0.2
  kill -TERM "$hub_pid" 2>/dev/null || true; sleep 0.2
  kill -KILL "$hub_pid" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

# 3) 运行 mux（按回车切换；'a'/'b' 强制；'q' 退出）
python3 policy_mux.py --ros-args \
  -p source_a:=/policy_mux/A \
  -p source_b:=/policy_mux/B \
  -p output:=/control/joint_position \
  -p blend_ms:=400 \
  -p loop_hz:=100.0
