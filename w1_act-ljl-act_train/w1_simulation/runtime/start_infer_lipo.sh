#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
PROFILE="${W1_SIMULATION_PROFILE:-$PROJECT_ROOT/configs/w1_popcorn_v1.json}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BRIDGE_MODE="async"
CHECKPOINT="${CHECKPOINT:-$($PYTHON_BIN -c 'import sys; from pathlib import Path; from w1_simulation.w1_profile import W1Profile; print(W1Profile.load(Path(sys.argv[1])).checkpoint)' "$PROFILE")}"
SERVER_PORT="${SERVER_PORT:-$($PYTHON_BIN -c 'import sys; from pathlib import Path; from w1_simulation.w1_profile import W1Profile; print(W1Profile.load(Path(sys.argv[1])).runtime["server_port"])' "$PROFILE")}"

BRIDGE_MODE_ARGS=()
case "$BRIDGE_MODE" in
  sync)
    BRIDGE_MODE_ARGS=(-p sample_factor:=1)
    ;;
  async)
    ;;
  *)
    echo "错误：BRIDGE_MODE只能是sync或async，当前值为${BRIDGE_MODE}"
    exit 2
    ;;
esac

MODEL_PID=""
BRIDGE_PID=""

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM

  echo "检测到退出信号，正在关闭推理进程..."

  if [[ -n "${BRIDGE_PID}" ]] && kill -0 "${BRIDGE_PID}" 2>/dev/null; then
    kill -TERM "${BRIDGE_PID}" 2>/dev/null || true
  fi

  if [[ -n "${MODEL_PID}" ]] && kill -0 "${MODEL_PID}" 2>/dev/null; then
    kill -TERM "${MODEL_PID}" 2>/dev/null || true
  fi

  wait "${BRIDGE_PID}" 2>/dev/null || true
  wait "${MODEL_PID}" 2>/dev/null || true
  exit "${exit_code}"
}

trap cleanup EXIT INT TERM

echo "正在启动ACT本地模型服务器..."

cd "$WORKSPACE_ROOT"
taskset -c 3 "$PYTHON_BIN" -u -m w1_simulation.runtime.policy_infer_act --config <(cat <<EOF
{
  "port": $SERVER_PORT,
  "device": "cuda:0",
  "models": {
    "0": "$CHECKPOINT"
  }
}
EOF
) &

MODEL_PID=$!

echo "模型服务器正在加载，等待监听端口 $SERVER_PORT..."

wait_for_model_server() {
  local deadline=$((SECONDS + 90))
  while ((SECONDS < deadline)); do
    if ! kill -0 "${MODEL_PID}" 2>/dev/null; then
      set +e
      wait "${MODEL_PID}"
      local status=$?
      set -e
      echo "错误：模型服务器在监听端口前退出，status=${status}"
      exit "${status}"
    fi
    if "$PYTHON_BIN" -c 'import sys; from multiprocessing.connection import Client; c=Client(("127.0.0.1", int(sys.argv[1])), authkey=b"w1_simulation_secret"); c.close()' "$SERVER_PORT" 2>/dev/null; then
      echo "模型服务器已就绪，启动${BRIDGE_MODE} Policy Bridge..."
      return
    fi
    sleep 0.5
  done
  echo "错误：模型服务器在 90s 内未监听端口 $SERVER_PORT"
  exit 1
}

wait_for_model_server

W1_SIMULATION_BRIDGE_MODE="$BRIDGE_MODE" taskset -c 5 "$PYTHON_BIN" -m w1_simulation.runtime.bridge --ros-args \
  -p server_port:="$SERVER_PORT" \
  -p shadow_mode:=false \
  -p tolerance_ms:=250.0 \
  -p head_target_width:=640 \
  -p head_target_height:=360 \
  -p hand_target_width:=640 \
  -p hand_target_height:=360 \
  "${BRIDGE_MODE_ARGS[@]}" &

BRIDGE_PID=$!

echo "ACT模型服务器 PID=${MODEL_PID}，CPU=3"
echo "Policy Bridge PID=${BRIDGE_PID}，CPU=5，模式=${BRIDGE_MODE}"
echo "W1关节顺序、三路控制话题及运行参数来自：$PROFILE"
echo "Shadow Mode=false"

set +e
wait -n "${MODEL_PID}" "${BRIDGE_PID}"
CHILD_STATUS=$?
set -e

if [[ "${CHILD_STATUS}" -eq 0 ]]; then
  echo "错误：模型服务器或Policy Bridge意外退出。"
  exit 1
fi

exit "${CHILD_STATUS}"
