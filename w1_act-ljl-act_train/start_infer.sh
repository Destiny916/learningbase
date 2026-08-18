#!/usr/bin/env bash
set -euo pipefail

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

taskset -c 3 python3 -u policy_infer_act.py --config <(cat <<EOF
{
  "port": 8888,
  "device": "cuda:0",
  "models": {
    "0": "/root/w1_act/checkpoints/1950000/pretrained_model"
  }
}
EOF
) &

MODEL_PID=$!

echo "模型服务器正在加载，等待监听端口 8888..."

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
    if python3 -c 'from multiprocessing.connection import Client; c=Client(("127.0.0.1", 8888), authkey=b"w1_act_secret"); c.close()' 2>/dev/null; then
      echo "模型服务器已就绪，启动LIPO Policy Bridge..."
      return
    fi
    sleep 0.5
  done
  echo "错误：模型服务器在 90s 内未监听端口 8888"
  exit 1
}

wait_for_model_server

taskset -c 5 python3 policy_bridge_act_lipo.py --ros-args \
  -p server_port:=8888 \
  -p joint_topic:=/feedback/robot_server_state \
  -p publish_topic:=/feedback/body_act \
  -p set_left_hand_qpos6_topic:=/feedback/hand/left_act \
  -p set_right_hand_qpos6_topic:=/feedback/hand/right_act \
  -p policy_hz:=20.0 \
  -p sample_factor:=2 \
  -p lipo_trigger_points:=15 \
  -p lipo_blend_points:=6 \
  -p shadow_mode:=false \
  -p tolerance_ms:=250.0 \
  -p head_target_width:=640 \
  -p head_target_height:=360 \
  -p hand_target_width:=640 \
  -p hand_target_height:=360 \
  -p cam_high_left_topic:=/camera/left_eye_resize \
  -p cam_hand_left_topic:=/camera_l/color/image_rect_raw \
  -p cam_hand_right_topic:=/camera_r/color/image_rect_raw \
  -p image_keys:='["observation.images.cam_high_left","observation.images.cam_hand_left","observation.images.cam_hand_right"]' \
  -p urdf_path:=/root/w1_act/assets/DexforceW1_v02_1/DexforceW1_v02_1.urdf \
  -p selected_body_names:='["WAIST","LEFT_J1","LEFT_J2","LEFT_J3","LEFT_J4","LEFT_J5","LEFT_J6","LEFT_J7","NECK1","NECK2","RIGHT_J1","RIGHT_J2","RIGHT_J3","RIGHT_J4","RIGHT_J5","RIGHT_J6","RIGHT_J7"]' \
  -p hand_input_mode:=scalar \
  -p hand_sides:='["left","right"]' \
  -p gripper_invert_left:=true \
  -p gripper_invert_right:=true \
  -p left_hand_start:='[0.0,70.0,0.0,0.0,0.0,0.0]' \
  -p left_hand_end:='[0.0,100.0,35.0,45.0,47.0,37.0]' \
  -p right_hand_start:='[0.0,70.0,0.0,0.0,0.0,0.0]' \
  -p right_hand_end:='[65.0,100.0,70.0,75.0,100.0,100.0]' &

BRIDGE_PID=$!

echo "ACT模型服务器 PID=${MODEL_PID}，CPU=3"
echo "LIPO Policy Bridge PID=${BRIDGE_PID}，CPU=5"

set +e
wait -n "${MODEL_PID}" "${BRIDGE_PID}"
CHILD_STATUS=$?
set -e

if [[ "${CHILD_STATUS}" -eq 0 ]]; then
  echo "错误：模型服务器或LIPO Policy Bridge意外退出。"
  exit 1
fi

exit "${CHILD_STATUS}"
