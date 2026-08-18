#!/usr/bin/env bash
set -euo pipefail
W1_ACT_CONTAINER_HOME="${W1_ACT_CONTAINER_HOME:-/root/w1_act}"
W1_HELLO_POLICY_PATH="${W1_HELLO_POLICY_PATH:-${W1_ACT_CONTAINER_HOME}/checkpoints/w1_hello_1118/1060000/pretrained_model}"

# （可选）如果本次不使用手的标量输入，这一行可以删掉或保留：
#python3 inference_codes/hand_scalar.py &

# 启动 10 维身体输入的 ACT 推理节点
python3 inference_codes/smooth_policy_new.py --ros-args \
  -p policy_path:="${W1_HELLO_POLICY_PATH}" \
  -p joint_topic:=/feedback/robot_server_state \
  -p publish_topic:=/control/joint_position \
  -p device:=cuda -p policy_hz:=60.0 -p tolerance_ms:=50.0 \
  -p head_target_width:=640 -p head_target_height:=360 \
  -p hand_target_width:=640 -p hand_target_height:=360 \
  -p image_keys:='["observation.images.cam_high_left","observation.images.cam_high_right"]' \
  -p hand_input_mode:=none \
  -p selected_body_names:='["WAIST","LEFT_J1","LEFT_J2","LEFT_J3","LEFT_J4","LEFT_J5","LEFT_J6","LEFT_J7","NECK1","NECK2",
  "RIGHT_J1","RIGHT_J2","RIGHT_J3","RIGHT_J4","RIGHT_J5","RIGHT_J6","RIGHT_J7"]'

  # -p hand_sides:='[]'\

# joint_topic:=/feedback/robot_server_state
# joint_topic:=/feedback/joint

# ["ANKLE","KNEE","BUTTOCK","WAIST",
# "LEFT_J1","LEFT_J2","LEFT_J3","LEFT_J4","LEFT_J5","LEFT_J6","LEFT_J7", "LEFT_GRIPPER",
# "NECK1","NECK2",
# "RIGHT_J1","RIGHT_J2","RIGHT_J3","RIGHT_J4","RIGHT_J5","RIGHT_J6","RIGHT_J7", "RIGHT_GRIPPER"
# "LEFT_HAND_THUMB1","LEFT_HAND_THUMB2","LEFT_HAND_INDEX","LEFT_HAND_MIDDLE","LEFT_HAND_RING","LEFT_HAND_PINKY",
# "RIGHT_HAND_THUMB1","RIGHT_HAND_THUMB2","RIGHT_HAND_INDEX","RIGHT_HAND_MIDDLE","RIGHT_HAND_RING","RIGHT_HAND_PINKY"]
