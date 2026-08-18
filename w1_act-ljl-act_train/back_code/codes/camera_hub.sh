#!/usr/bin/env bash
set -euo pipefail

# 把物理相机写到共享内存 /tmp/cam99.sock
gst-launch-1.0 -e \
  v4l2src device=/dev/video99 ! image/jpeg,width=3840,height=1080,framerate=30/1 ! \
  jpegdec ! videoconvert ! video/x-raw,format=BGR ! queue max-size-buffers=1 leaky=downstream ! \
  shmsink socket-path=/tmp/cam99.sock wait-for-connection=false shm-size=134217728 sync=false
