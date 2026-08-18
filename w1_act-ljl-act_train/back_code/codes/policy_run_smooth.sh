#!/usr/bin/env bash

# 启动双手 HandScalar 节点
python3 hand_scalar.py &

# 启动 policy 节点
python3 smooth_policy_v1.py --ros-args \
  -p policy_path:=/home/grabotics/workspace/act_inference/280000/pretrained_model \
  -p gst_pipeline:='v4l2src device=/dev/video99 ! image/jpeg,width=3840,height=1080,framerate=30/1 ! jpegdec ! videoconvert ! appsink max-buffers=1 drop=True' \
  -p joint_topic:=/feedback/joint \
  -p publish_topic:=/control/joint_position \
  -p device:=cuda -p policy_hz:=30.0 -p tolerance_ms:=200.0 \
  -p arm_tau_ms:=200.0 \
  -p target_width:=960 -p target_height:=540 \
  -p image_left_key:=observation.images.cam_high_left \
  -p image_right_key:=observation.images.cam_high_right \
  -p state_key:=observation.state \
  -p drop_joint_names:="['ANKLE','KNEE','BUTTOCK']"
