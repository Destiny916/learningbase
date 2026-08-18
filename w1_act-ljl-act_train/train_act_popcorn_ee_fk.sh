#!/bin/bash
set -euo pipefail

source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_NO_CUDA_MEMORY_CACHING_ALLOCATOR=1

RUN_ID=$(date +%Y%m%d_%H%M%S)
RUN_BASE=/home/ubuntu/lerobot/runs/popcorn_${RUN_ID}
OUTPUT_DIR=$RUN_BASE/checkpoints
TB_LOGDIR=$RUN_BASE/tb_logs
LOG_DIR=$RUN_BASE/logs

mkdir -p "$LOG_DIR" "$TB_LOGDIR"
LOG_FILE=$LOG_DIR/train_${RUN_ID}.log

cleanup() {
    echo "Cleaning up..."
    kill $(jobs -p) 2>/dev/null || true
    echo "Train log: $LOG_FILE"
    echo "TB logs: $TB_LOGDIR"
}
trap cleanup EXIT

cd /home/ubuntu/lerobot

echo "=== Starting TensorBoard (port 6006) ==="
tensorboard --logdir "$TB_LOGDIR" --bind_all --port 6006 &
sleep 2

echo "=== Starting TB monitor ==="
python tb_monitor.py --logdir "$TB_LOGDIR" --follow "$LOG_FILE" --tags loss,grdn,l1,ee,fk,aux &
sleep 1

echo "=== Starting ACT EE/FK FP32 training ==="
echo "=== TensorBoard: http://192.168.15.107:6006 ==="
echo "=== Output dir: $RUN_BASE ==="
echo ""

accelerate launch --mixed_precision=no --num_processes=1 \
  /home/ubuntu/lerobot/src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=/home/ubuntu/lerobot/dataset/lerobotv30/popcorn_lerobotv30 \
  --dataset.video_backend=pyav \
  --policy.chunk_size=100 \
  --policy.n_action_steps=100 \
  --policy.vision_encoder_type="resnet" \
  --policy.vision_normalize_in_processor=true \
  --policy.vision_normalize_in_model=false \
  --policy.vision_freeze=false \
  --policy.ee_pose_loss_weight=0.05 \
  --policy.fk_loss_weight=0.10 \
  --policy.kinematics_urdf_path=/home/ubuntu/lerobot/urdf/w1/v022/w1/robot_with_ee.urdf \
  --policy.kinematics_urdf_sha256=8cf7516b5ba5de7d8091b1db4254ab1879c0747fb506e2420805e4ce99fb714e \
  --policy.ee_reference_link=buttock \
  --policy.ee_left_link=left_ee \
  --policy.ee_right_link=right_ee \
  --policy.ee_position_scale_m=0.1 \
  --policy.ee_rotation_loss_weight=0.25 \
  --policy.push_to_hub=false \
  --policy.repo_id="" \
  --policy.type=act \
  --batch_size=8 \
  --num_workers=28 \
  --steps=5000000 \
  --save_freq=50000 \
  --output_dir="$OUTPUT_DIR" \
  --job_name=popcorn \
  --policy.device=cuda \
  --wandb.enable=false \
  --log_freq=100 2>&1 | tee "$LOG_FILE"
