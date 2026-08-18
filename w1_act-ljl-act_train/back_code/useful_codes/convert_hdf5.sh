#!/bin/bash
# 批量转换 recorded_data 下所有子文件夹里的数据为 hdf5

python3 convert_to_hdf5_batch.py \
  --root_dir . \
  --qpos_pattern 'pose_record*.json' \
  --out_root 'hdf5_files'
