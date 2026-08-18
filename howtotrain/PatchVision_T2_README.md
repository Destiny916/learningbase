# TurboVLA + Patch Policy（PatchVision T2）训练说明

本文档对应当前已经验证过的训练方案：使用 Patch Policy 风格的 DINOv3 dense patch-token 视觉编码，保留 TurboVLA 原有的 ACT action head。它不是独立的 Patch-ACT；TurboVLA 的 action head、14D action 输出和 50 步 action horizon 保持不变。

## 1. 训练目标和数据

数据集：

```text
/data/wengyikun/datasets/joint_songling/0812_binary_gripper_without_ep173_174
```

任务文本：

```text
Pick up the bread with the right gripper, transfer it to the left gripper, and place it in the bowl.
```

数据契约：

- 202 episodes，125558 frames，全部用于训练；
- state 20D：左关节 6D、左 xyz 3D、左夹爪 1D、右关节 6D、右 xyz 3D、右夹爪 1D；
- action 14D：左关节 6D、左夹爪 1D、右关节 6D、右夹爪 1D；
- 机械臂关节使用当前状态相对量；
- xyz 和夹爪使用绝对量；夹爪为 binary，闭合为 0.0、张开为 0.1；
- state 与 action 分别独立计算本数据集的 q01/q99，不能复用其它数据集的统计量。

## 2. PatchVision T2 输入

时间窗口固定为 `[-1, 0]`，也就是当前训练样本的“前一帧、当前帧”，不是 `[t, t+1]`。

每个时间点输入三路相机，顺序固定为：

```text
top, gripper_left, gripper_right
```

当前配置使用 `image_layout: joint_songling`：

- top：405×720，先上下补黑边（上 157、下 158）成为 720×720，再 resize 到 224×224；
- 左右腕部：480×640，中心裁剪左右各 80 像素成为 480×480，再 resize 到 224×224。

DINOv3 输出 patch tokens；时间轴、相机轴和 patch 轴在视觉编码阶段保持为：

```text
[batch, time=2, views=3, patches, hidden]
```

之后加入时间/视角/patch 位置编码并送入 TurboVLA 的视觉语言交互模块。DINOv3 backbone 冻结，文本 BERT 也冻结；训练参数主要是视觉投影、交互模块和原 ACT action head。

## 3. 训练配置

配置文件：

```text
TurboVLA/experiments/joint_songling/configs/0812_closed_patchvision_t2_gpu7.yaml
```

关键参数：

```text
GPU                         7
per_device_batch_size      16
gradient_accumulation      1
total batch size            16
max_train_steps            500000
warmup                      25000
lr scheduler                cosine
save_interval               20000
eval_interval               20000
action_dim                  14
state_dim                   20
action horizon              50
temporal_window_size        2
freeze vision encoder       true
freeze text encoder         true
```

当前正式输出目录：

```text
/data/wengyikun/outputs/turbovla_0812_binary_gripper_patchvision_t2_gpu7
```

每个 checkpoint 位于该目录下的 run 子目录 `checkpoints/` 中，保存格式为 safetensors；EMA 模型也会按训练器配置保存。

## 4. 远程环境变量

远程机器为 `183.230.224.121`（主机名 `cloud`）。模型目录：

```bash
export BERT_MODEL_PATH=/data/wengyikun/models/turbovla_joint_songling/bert-base-uncased
export DINOV3_MODEL_PATH=/data/wengyikun/models/turbovla_joint_songling/dinov3-vitl16-pretrain-lvd1689m
export JOINT_SONGLING_DATA_ROOT=/data/wengyikun/datasets/joint_songling/0812_binary_gripper_without_ep173_174
```

不要把 `BERT_MODEL_PATH`、`DINOV3_MODEL_PATH` 写成本地开发机路径；训练实际发生在远程 Docker 容器内。

## 5. 先做 smoke test

smoke test 只运行 1 个 optimizer step，使用独立输出目录：

```bash
ssh 183.230.224.121 '
  CUDA_VISIBLE_DEVICES=7 \
  BERT_MODEL_PATH=/data/wengyikun/models/turbovla_joint_songling/bert-base-uncased \
  DINOV3_MODEL_PATH=/data/wengyikun/models/turbovla_joint_songling/dinov3-vitl16-pretrain-lvd1689m \
  RUN_ROOT_DIR=/data/wengyikun/outputs/turbovla_0812_binary_gripper_patchvision_t2_gpu7_smoke \
  RUN_ID=turbovla_0812_binary_gripper_patchvision_t2_gpu7_smoke \
  JOINT_SONGLING_OVERLAY_ROOT=/data/wengyikun/outputs/turbovla_0812_binary_gripper_patchvision_t2_gpu7_smoke_overlay \
  SMOKE_TEST=1 \
  bash /home/wengyikun/tmp/TurboVLA-0812-patchvision-t2-gpu7/scripts/joint_songling/train_0812_closed_patchvision_t2_gpu7.sh
'
```

smoke test 至少应确认：数据集成功加载、输入是双帧三视角、state/action 维度正确、q01/q99 缓存生成、loss 为有限值、无 OOM/Traceback。

## 6. 启动正式训练

正式训练使用新的 GPU7 和新的输出目录。启动脚本自带 Docker GPU 隔离，不要改成 `--gpus all`：

```bash
ssh 183.230.224.121 '
  CUDA_VISIBLE_DEVICES=7 \
  BERT_MODEL_PATH=/data/wengyikun/models/turbovla_joint_songling/bert-base-uncased \
  DINOV3_MODEL_PATH=/data/wengyikun/models/turbovla_joint_songling/dinov3-vitl16-pretrain-lvd1689m \
  RUN_ROOT_DIR=/data/wengyikun/outputs/turbovla_0812_binary_gripper_patchvision_t2_gpu7 \
  RUN_ID=turbovla_0812_binary_gripper_patchvision_t2_gpu7 \
  JOINT_SONGLING_OVERLAY_ROOT=/data/wengyikun/outputs/turbovla_0812_binary_gripper_patchvision_t2_gpu7_overlay \
  WANDB_MODE=disabled \
  bash /home/wengyikun/tmp/TurboVLA-0812-patchvision-t2-gpu7/scripts/joint_songling/train_0812_closed_patchvision_t2_gpu7.sh \
  > /data/wengyikun/outputs/turbovla_0812_binary_gripper_patchvision_t2_gpu7.log 2>&1 &
'
```

脚本会拒绝覆盖已存在的 `RUN_ROOT_DIR`。若需要重新开一组实验，必须改 `RUN_ID`、`RUN_ROOT_DIR` 和 overlay 路径，不能复用旧目录。

## 7. 监控训练

查看容器和最近日志：

```bash
ssh 183.230.224.121 'docker ps --format "table {{.Names}}\t{{.Status}}" | grep turbovla_0812_binary_gripper_patchvision_t2_gpu7'
ssh 183.230.224.121 'tail -f /data/wengyikun/outputs/turbovla_0812_binary_gripper_patchvision_t2_gpu7.log'
```

查看 GPU7：

```bash
ssh 183.230.224.121 'nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | grep "^7,"'
```

正常训练日志会出现类似：

```text
optimization steps = 500000
Per device batch size = 16
Step N, Loss: {'action_dit_loss': ...}
```

## 8. 断点和恢复

每 20000 步保存一次 checkpoint。恢复前先确认目标 checkpoint 存在，并使用新的实验目录记录恢复运行；不要覆盖原 run。恢复参数应在配置中设置：

```yaml
trainer:
  pretrained_checkpoint: /data/wengyikun/outputs/<old_run>/<run_id>/checkpoints/steps_20000_model.safetensors
```

如果只是继续同一训练进程，优先保留原容器和原输出目录；如果启动新实验，则必须记录这是 warm initialization，优化器/调度器是否从零开始要明确区分。

## 9. 重要注意事项

- GPU1/GPU2 上的旧 0812 训练是独立实验，不要停止或覆盖；
- PatchVision T2 的时间索引是 `[-1, 0]`；
- TurboVLA 原 ACT head 的 horizon 是 50，不是独立 Patch-ACT 的 chunk 16；
- state 与 action 的 q01/q99 必须分别生成；
- top 和腕部相机排版必须在 resize 前完成；
- `image_layout: joint_songling` 与 `joint_songling_top_padded` 含义不同，当前 PatchVision T2 使用前者（腕部中心裁剪）；
- 训练全部为 train split，没有额外验证集；`eval_interval` 仅由训练器配置保留，不代表有独立 validation episode；
- 不要把 HF token、SSH 密钥或其它凭据写入配置和日志。
