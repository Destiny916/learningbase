#!/usr/bin/env bash
set -euo pipefail

run_dir=/home/kw/runs/turbovla_patchvision_t2_180000_dryrun
exec /home/kw/miniforge3/envs/lerobot/bin/python "$run_dir/dual_turbovla_patchvision_t2_real.py" \
  --server-uri ws://127.0.0.1:18067 \
  --stats-path "$run_dir/dataset_statistics.json" \
  --task "Pick up the bread with the right gripper, transfer it to the left gripper, and place it in the bowl." \
  --fps 30 \
  --left-can can1 \
  --right-can can0 \
  --top-device /dev/video26 \
  --left-pika-port /dev/serial/by-path/pci-0000:c4:00.3-usb-0:3.4:1.0-port0 \
  --right-pika-port /dev/serial/by-path/pci-0000:c6:00.4-usb-0:1.4:1.0-port0 \
  --gripper-max-m 0.10 \
  --enable-arms \
  --enable-grippers \
  --execute-robot-actions
