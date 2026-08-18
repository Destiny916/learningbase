python3 w1_act_policy_inference_node.py --ros-args \
  -p policy_path:=/home/grabotics/workspace/act_inference/120000/pretrained_model \
  -p gst_pipeline:='v4l2src device=/dev/video99 ! image/jpeg,width=3840,height=1080,framerate=30/1 ! jpegdec ! videoconvert ! appsink max-buffers=1 drop=True' \
  -p joint_topic:=/feedback/joint \
  -p publish_topic:=/w1/policy/desired_joint_positions \
  -p device:=cuda -p policy_hz:=20.0 -p tolerance_ms:=30.0 \
  -p target_width:=960 -p target_height:=540 \
  -p image_left_key:=observation.images.cam_high_left \
  -p image_right_key:=observation.images.cam_high_right \
  -p state_key:=observation.state
