#!/usr/bin/env bash
set -euo pipefail

A="./policy_run_policya.sh"   # 政策A启动脚本
B="./policy_run_policyb.sh"   # 政策B启动脚本

child_pid=""
pgid=""
active="A"

start_child() {
  local script="$1"
  echo ""
  echo "▶ Starting ${script} ..."
  # 用 setsid 启动形成独立进程组，后面好“一把梭”杀干净
  setsid bash "$script" &
  child_pid=$!
  # 取进程组ID
  pgid="$(ps -o pgid= "$child_pid" | tr -d ' ')"
  echo "   pid=${child_pid}, pgid=${pgid}, active=${active}"
}

stop_child() {
  if [[ -n "${pgid:-}" ]]; then
    echo "⛔ Stopping active policy (pgid=${pgid}) ..."
    # 先温柔打断，再升级，确保ROS节点能触发 KeyboardInterrupt 并清理
    kill -INT -"${pgid}" 2>/dev/null || true
    sleep 0.6
    kill -TERM -"${pgid}" 2>/dev/null || true
    sleep 0.6
    kill -KILL -"${pgid}" 2>/dev/null || true
    # 回收
    wait "${child_pid}" 2>/dev/null || true
    child_pid=""
    pgid=""
  fi
}

cleanup() {
  echo ""
  stop_child
  echo "Bye."
}
trap cleanup INT TERM EXIT

# 启动A
active="A"
start_child "$A"

# 交互循环：回车切换，q 回车退出
while true; do
  echo ""
  echo "Press Enter to switch policy (A ↔ B), or type 'q' then Enter to quit."
  IFS= read -r line || true
  if [[ "${line:-}" == "q" || "${line:-}" == "Q" ]]; then
    break
  fi
  stop_child
  if [[ "$active" == "A" ]]; then
    active="B"
    start_child "$B"
  else
    active="A"
    start_child "$A"
  fi
done
