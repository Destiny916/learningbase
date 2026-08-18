#!/bin/bash
W1_ACT_HOST_HOME="${W1_ACT_HOST_HOME:-${W1_ACT_HOME:-/home/dexforce/w1/w1_act}}"
W1_LEROBOT_HOST_HOME="${W1_LEROBOT_HOST_HOME:-${W1_ACT_HOST_HOME}/w1_lerobot}"
W1_ACT_CONTAINER_HOME="${W1_ACT_CONTAINER_HOME:-/root/w1_act}"
W1_LEROBOT_CONTAINER_HOME="${W1_LEROBOT_CONTAINER_HOME:-/root/lerobot}"
W1_DOCKER_IMAGE="${W1_DOCKER_IMAGE:-jetson_lerobot:service}"
W1_DOCKER_NAME="${W1_DOCKER_NAME:-act_ros2}"
W1_VIDEO_DEVICE="${W1_VIDEO_DEVICE:-/dev/video99}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-20}"
W1_LEGACY_ACT_HOME="${W1_LEGACY_ACT_HOME:-/home/dexforce/w1/w1_act}"
W1_CHECKPOINTS_HOST_HOME="${W1_CHECKPOINTS_HOST_HOME:-}"

if [[ -z "${W1_CHECKPOINTS_HOST_HOME}" && ! -e "${W1_ACT_HOST_HOME}/checkpoints" && -d "${W1_LEGACY_ACT_HOME}/checkpoints" ]]; then
  W1_CHECKPOINTS_HOST_HOME="${W1_LEGACY_ACT_HOME}/checkpoints"
fi

CHECKPOINT_ARGS=()
if [[ -n "${W1_CHECKPOINTS_HOST_HOME}" ]]; then
  if [[ ! -d "${W1_CHECKPOINTS_HOST_HOME}" ]]; then
    echo "W1_CHECKPOINTS_HOST_HOME does not exist or is not a directory: ${W1_CHECKPOINTS_HOST_HOME}" >&2
    exit 1
  fi
  CHECKPOINT_ARGS=(-v "${W1_CHECKPOINTS_HOST_HOME}:${W1_ACT_CONTAINER_HOME}/checkpoints:ro")
fi

docker run -dit \
  --name "${W1_DOCKER_NAME}" \
  --runtime nvidia --gpus all \
  --network host \
  -v "${W1_ACT_HOST_HOME}:${W1_ACT_CONTAINER_HOME}" \
  -v "${W1_LEROBOT_HOST_HOME}:${W1_LEROBOT_CONTAINER_HOME}" \
  "${CHECKPOINT_ARGS[@]}" \
  -v /usr/local/cuda:/usr/local/cuda \
  -v /usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu \
  -e LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH \
  -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID}" \
  -e ROS_LOCALHOST_ONLY=0 \
  --device "${W1_VIDEO_DEVICE}:${W1_VIDEO_DEVICE}" \
  "${W1_DOCKER_IMAGE}" bash

  #-v /home/grabotics/workspace/act_inference:/root/w1_act \
