# TurboVLA 训练说明（Joint Songling 0812）

本文记录当前已经核验过的 TurboVLA 训练流程，适用于远端服务器：

```text
SSH: wengyikun@183.230.224.121:50210
远端仓库: /home/wengyikun/tmp/TurboVLA-joint-songling-binary-top-padded
Docker 镜像: turbovla-joint-songling:20260803
```

数据集：

```text
/data/wengyikun/datasets/joint_songling/0812_binary_gripper_without_ep173_174
```

## 1. 训练契约

数据集为 LeRobot v3.0，共 202 个 episode、125558 帧，全部用于训练。

相机顺序固定为：

```text
top -> gripper_left -> gripper_right
```

任务文本固定为：

```text
first pick up the bread with the right hand, then hand it to the left hand at the middle point, then place the bread in the bowl with the left hand.
```

模型输入为 20D state、14D action，action horizon 为 50，batch size 为 16。

当前 DataLoader 配置为：

```yaml
num_workers: 8
prefetch_factor: 2
persistent_workers: true
```

PyAV 的每个视频解码器固定使用 1 个 FFmpeg 解码线程，避免 `workers × FFmpeg auto threads` 导致 CPU 过度订阅。`prefetch_factor: 2` 表示每个 worker 最多提前准备 2 个 batch，因此单个训练的理论预取队列上限为 32 个 batch。

状态和动作语义：

```text
state 左右机械臂关节：q[t] - q[t-1]
state 左右 endpoint xyz：绝对值
state 左右夹爪：绝对 binary 值（0.0 / 0.1）

action 左右机械臂关节：q[t+n] - q[t]
action 左右夹爪：绝对 binary 值（0.0 / 0.1）
```

state 和 action 使用各自独立的 q01/q99 统计。统计按 episode 独立计算，action chunk 末尾使用当前 episode 最后一帧填充，不会跨 episode。

## 2. 图像预处理

配置使用：

```yaml
image_layout: joint_songling_top_padded
obs_image_size: [224, 224]
num_views: 3
```

处理流程：

```text
top:           405x720 -> 上补157行黑边、下补158行黑边 -> 720x720 -> 224x224
gripper_left:  480x640 -> 直接 resize -> 224x224
gripper_right: 480x640 -> 直接 resize -> 224x224
```

DINOv3 processor 使用 224×224，`do_center_crop=null`、`crop_size=null`，不会再次中心裁剪。

## 3. 环境变量

在远端仓库目录中设置：

```bash
cd /home/wengyikun/tmp/TurboVLA-joint-songling-binary-top-padded

export JOINT_SONGLING_DATA_ROOT=/data/wengyikun/datasets/joint_songling/0812_binary_gripper_without_ep173_174
export BERT_MODEL_PATH=/data/wengyikun/models/turbovla_joint_songling/bert-base-uncased
export DINOV3_MODEL_PATH=/data/wengyikun/models/turbovla_joint_songling/dinov3-vitl16-pretrain-lvd1689m
export TURBOVLA_DOCKER_IMAGE=turbovla-joint-songling:20260803
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

训练脚本会自动创建或复用已检查的 overlay：

```text
/data/wengyikun/outputs/turbovla_0812_binary_gripper_without_ep173_174_top_padded_overlay
```

不要手动覆盖 overlay，也不要把其他数据集的统计文件复制进去。该 overlay 的
`overlay_contract.json` 必须包含 `"gripper_mode": "binary"`。

## 4. 启动前检查

```bash
ssh -p 50210 wengyikun@183.230.224.121 \
  "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits; docker images | grep turbovla"

ssh -p 50210 wengyikun@183.230.224.121 \
  "docker ps --format '{{.Names}}|{{.Image}}|{{.Status}}'"
```

不要使用 `--gpus all`。正式训练应明确指定单卡，例如 GPU2：

```bash
export CUDA_VISIBLE_DEVICES=2
```

## 5. 1-step smoke test

Smoke test 检查数据集、overlay、BERT/DINOv3、TurboVLA，并执行 1 个训练 step。它应使用独立输出目录：

```bash
export CUDA_VISIBLE_DEVICES=2
export TRAINING_VARIANT=warm
export TURBOVLA_INITIAL_CKPT=/data/wengyikun/outputs/turbovla_0806swap_gripper_fixed_3view_gpu6_retry8/turbovla_0806swap_gripper_fixed_3view_gpu6_retry8/checkpoints/steps_200000_model.safetensors
export RUN_ROOT_DIR=/data/wengyikun/outputs/turbovla_0812_binary_gripper_top_padded_smoke_gpu2
export RUN_ID=turbovla_0812_binary_gripper_top_padded_smoke_gpu2
export SMOKE_TEST=1

bash scripts/joint_songling/train_0812_closed_gripper_top_padded.sh
unset SMOKE_TEST
```

正常标准：出现 1 step loss，且没有 `Traceback`、`CUDA out of memory`、`nan` 或 `inf`。

## 6. Warm-start 训练（GPU2）

Warm-start 从 retry8 的非 EMA 200000-step checkpoint 初始化；优化器和 scheduler 从新训练 step 0 开始。

```bash
export CUDA_VISIBLE_DEVICES=2
export TRAINING_VARIANT=warm
export TURBOVLA_INITIAL_CKPT=/data/wengyikun/outputs/turbovla_0806swap_gripper_fixed_3view_gpu6_retry8/turbovla_0806swap_gripper_fixed_3view_gpu6_retry8/checkpoints/steps_200000_model.safetensors
export RUN_ROOT_DIR=/data/wengyikun/outputs/turbovla_0812_binary_gripper_top_padded_warm_gpu2
export RUN_ID=turbovla_0812_binary_gripper_top_padded_warm_gpu2

bash scripts/joint_songling/train_0812_closed_gripper_top_padded.sh
```

主要参数：

```yaml
per_device_batch_size: 16
num_workers: 16
prefetch_factor: 2
max_train_steps: 500000
num_warmup_steps: 25000
save_interval: 20000
eval_interval: 20000
lr_scheduler_type: cosine
learning_rate: 5.0e-5
gradient_accumulation_steps: 1
ema_decay: 0.999
```

## 7. Fresh 训练（GPU1）

Fresh 不加载完整 TurboVLA checkpoint，但仍加载本地 BERT 和 DINOv3 backbone。

```bash
export CUDA_VISIBLE_DEVICES=1
export TRAINING_VARIANT=fresh
unset TURBOVLA_INITIAL_CKPT
export RUN_ROOT_DIR=/data/wengyikun/outputs/turbovla_0812_binary_gripper_top_padded_fresh_gpu1
export RUN_ID=turbovla_0812_binary_gripper_top_padded_fresh_gpu1

bash scripts/joint_songling/train_0812_closed_gripper_top_padded.sh
```

## 8. 监控与保存

```bash
ssh -p 50210 wengyikun@183.230.224.121 \
  "docker ps --filter name=turbovla_0812_binary_gripper_top_padded --format '{{.Names}}|{{.Status}}'; nvidia-smi"

ssh -p 50210 wengyikun@183.230.224.121 \
  "find /data/wengyikun/outputs/turbovla_0812_binary_gripper_top_padded_warm_gpu2 -maxdepth 4 -type f | sort | tail -30"
```

每 20000 step 保存 checkpoint，输出目录为：

```text
/data/wengyikun/outputs/turbovla_0812_binary_gripper_top_padded_warm_gpu2/
/data/wengyikun/outputs/turbovla_0812_binary_gripper_top_padded_fresh_gpu1/
```

典型文件：

```text
config.yaml
config.full.yaml
dataset_statistics.json
checkpoints/steps_<step>_model.safetensors
checkpoints/steps_<step>_ema_model.safetensors
```

## 9. 续训

续训使用已有 checkpoint 和新的输出目录；脚本默认拒绝覆盖已有 `RUN_ROOT_DIR`：

```bash
export CUDA_VISIBLE_DEVICES=2
export TRAINING_VARIANT=warm
export TURBOVLA_INITIAL_CKPT=/data/wengyikun/outputs/<已有run>/<已有run>/checkpoints/steps_200000_model.safetensors
export RUN_ROOT_DIR=/data/wengyikun/outputs/<新run>
export RUN_ID=<新run>
bash scripts/joint_songling/train_0812_closed_gripper_top_padded.sh
```

这属于加载模型权重后重新开始优化器/scheduler 的 warm-start，不是恢复原 optimizer step。只传 `TURBOVLA_INITIAL_CKPT` 不能恢复 optimizer、scheduler 和 EMA 状态。

## 10. 停止与故障检查

先确认精确容器名，再停止单个容器：

```bash
ssh -p 50210 wengyikun@183.230.224.121 \
  "docker ps --filter name=turbovla_0812_binary_gripper_top_padded --format '{{.Names}}'"

ssh -p 50210 wengyikun@183.230.224.121 \
  "docker stop turbovla_0812_binary_gripper_top_padded_warm_gpu2"
```

不要使用 `docker stop $(docker ps -q)`，也不要删除共享 overlay、数据集或其他训练输出。

重点检查 `Traceback`、`CUDA out of memory`、`nan/inf`、checkpoint 保存失败和 q01/q99 cache config mismatch。

## 11. 关键文件

```text
scripts/joint_songling/train_0812_closed_gripper_top_padded.sh
experiments/joint_songling/configs/0812_closed_gripper_top_padded_warm.yaml
experiments/joint_songling/configs/0812_closed_gripper_top_padded_fresh.yaml
third_party/starvla_runtime/starVLA/dataloader/gr00t_lerobot/datasets.py
experiments/joint_songling/data_registry/data_config.py
```

本文不包含 Hugging Face token、SSH 密码或其他凭据。
