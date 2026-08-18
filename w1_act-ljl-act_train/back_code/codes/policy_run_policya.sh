#!/usr/bin/env bash
set -euo pipefail

# （可选）如果本次不使用手的标量输入，这一行可以删掉或保留：
#python3 hand_scalar.py &

# 启动 10 维身体输入的 ACT 推理节点
python3 smooth_policy_v2.py --ros-args \
  -p policy_path:=/home/grabotics/workspace/act_inference/put_doll_dex/280000/pretrained_model \
  -p gst_pipeline:='shmsrc socket-path=/tmp/cam99.sock do-timestamp=true is-live=true ! video/x-raw,format=BGR,width=3840,height=1080,framerate=30/1 ! videoconvert ! appsink max-buffers=1 drop=True' \
  -p joint_topic:=/feedback/joint \
  -p publish_topic:=/policy_mux/A \
  -p device:=cuda -p policy_hz:=20.0 -p tolerance_ms:=200.0 \
  -p target_width:=960 -p target_height:=540 \
  -p image_left_key:=observation.images.cam_high_left \
  -p image_right_key:=observation.images.cam_high_right \
  -p state_key:=observation.state \
  -p hand_input_mode:=qpos6 \
  -p hand_sides:='["right"]'\
  -p selected_body_names:='["WAIST","NECK1","NECK2","RIGHT_J1","...EX","RIGHT_HAND_MIDDLE","RIGHT_HAND_RING","RIGHT_HAND_PINKY"]' \
  -p drop_joint_names:='["ANKLE","KNEE","BUTTOCK"]'
