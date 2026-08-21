# TurboVLA + PatchVision T2：0812 Binary Gripper GPU7 训练说明

本文档对应当前实际运行的训练方案：把 Patch Policy 的 dense patch-token 与短时间窗口思想接入 TurboVLA 视觉编码部分，同时保留 TurboVLA 原有的视觉语言交互模块和 action head。

这不是直接运行 `patch_policy` 仓库，也不是把 TurboVLA action head 替换成 Patch Policy diffusion policy。当前方案只借用 Patch Policy 的视觉表示方法，因此在本文中称为 **PatchVision T2**。

## 1. 当前训练

远程环境：

```text
SSH: wengyikun@183.230.224.121:50210
远端仓库: /home/wengyikun/tmp/TurboVLA-0812-patchvision-t2-gpu7
Docker 镜像: turbovla-joint-songling:20260803
GPU: 7
```

数据集：

```text
/data/wengyikun/datasets/joint_songling/0812_binary_gripper_without_ep173_174
```

当前 fixed run：

```text
turbovla_0812_binary_gripper_patchvision_t2_gpu7_workers8_fixed
```

输出目录：

```text
/data/wengyikun/outputs/turbovla_0812_binary_gripper_patchvision_t2_gpu7_workers8_fixed
```

overlay：

```text
/data/wengyikun/outputs/turbovla_0812_binary_gripper_patchvision_t2_gpu7_workers8_fixed_overlay
```

任务文本：

```text
Pick up the bread with the right gripper, transfer it to the left gripper, and place it in the bowl.
```

## 2. 数据契约

数据集共有 202 episodes、125558 frames，全部属于训练集。

state 为完整 20D：

```text
left_joints    0:6
left_endpoint  6:9
left_gripper   9
right_joints   10:16
right_endpoint 16:19
right_gripper  19
```

action 为 14D：

```text
left_joints   0:6
left_gripper  6
right_joints  7:13
right_gripper 13
```

数值语义：

- state 左右关节：`q[t] - q[t-1]`；
- state 左右 endpoint xyz：当前时刻绝对值；
- state 左右夹爪：当前时刻绝对二值；
- action 左右关节：`q[t+n] - q[t]`；
- action 左右夹爪：未来时刻绝对二值。

夹爪二值规则：

```text
value <= 0.06 m -> 0.0
value >  0.06 m -> 0.1
```

二值索引：

```text
state:  9, 19
action: 6, 13
```

overlay 合同必须记录：

```json
"gripper_mode": "binary_absolute_closed_zero"
```

state 与 action 各自独立计算 q01/q99。当前统计中四个夹爪维度均满足：

```text
q01 = 0.0
q99 = 0.1
```

不得复用 continuous-gripper 数据集或其他训练的统计文件。

## 3. Patch Policy 方法在这里如何使用

原始 Patch Policy 的核心视觉思路是：不把视觉编码器输出压缩为一个 CLS token，而是保留每个图像 patch 的 dense token；同时使用多帧观察窗口，使策略能够利用局部空间信息和短时运动信息。

当前 TurboVLA 改造保留了这两点：

1. DINOv3 输出 dense patch tokens；
2. `window_size=2`，视觉时间索引为 `[-1, 0]`；
3. 每帧包含 top、左腕、右腕三路相机；
4. 为 patch、相机视角和时间分别加入可学习位置编码；
5. 所有视觉 token 送入 TurboVLA 原视觉语言交互模块；
6. 最终仍由 TurboVLA 原 action head 输出 50 步、14D action chunk。

数据流为：

```text
6 张图像
  = 2 个时间点 × 3 个视角
        |
        v
DINOv3 dense patch tokens
        |
        v
[B, T=2, V=3, P=196, D_dino]
        |
        v
VisionProjection -> hidden_dim=256
        |
        v
+ patch position embedding
+ view embedding
+ time embedding
        |
        v
[B, 2*3*196=1176, 256]
        |
        v
TurboVLA vision-language interaction
        |
        v
TurboVLA action head
        |
        v
[B, horizon=50, action_dim=14]
```

DINOv3 ViT-L/16 在 224×224 输入上产生 `14×14=196` 个 patch token。CLS/register prefix tokens 会被移除，不能把它们误当成图像 patch。

## 4. PatchVision 的代码实现

### 4.1 配置字段

配置文件：

```text
TurboVLA/experiments/joint_songling/configs/0812_closed_patchvision_t2_gpu7_workers8.yaml
```

关键配置：

```yaml
framework:
  vision:
    image_size: 224
    num_views: 3
    temporal_window_size: 2
    freeze_vision_encoder: true
    position_init_std: 0.01
    position_scale_init: 0.01

  action:
    action_dim: 14
    state_dim: 20
    horizon: 50
```

`temporal_window_size`、`learned_patch` 位置编码及其参数定义在：

```text
TurboVLA/turbovla/models/configuration.py
```

### 4.2 双帧数据索引

文件：

```text
TurboVLA/experiments/joint_songling/data_registry/data_config.py
```

`JointSonglingSwapEndpoint20Temporal2DataConfig` 将视觉索引设置为：

```python
observation_indices = [-1, 0]
```

因此输入是 `[t-1, t]`，不是 `[t, t+1]`。episode 第一帧的越界索引由数据加载器在本 episode 内处理，不跨 episode 取图像。

action 使用：

```python
action_indices = list(range(50))
```

因此 action horizon 保持 TurboVLA 的 50，不使用 Patch Policy 配置里的 action window。

### 4.3 图像排版

文件：

```text
TurboVLA/third_party/starvla_runtime/starVLA/dataloader/gr00t_lerobot/datasets.py
```

`preprocess_joint_songling_frame()` 在统一 resize 前处理三路图像：

```text
top:
405×720 -> 上补157、下补158黑边 -> 720×720 -> 224×224

gripper_left / gripper_right:
480×640 -> 水平方向中心裁剪 80:560 -> 480×480 -> 224×224
```

相机顺序固定为：

```text
top, gripper_left, gripper_right
```

### 4.4 DINOv3 dense patch tokens

文件：

```text
TurboVLA/turbovla/models/vision_encoder.py
```

`DINOv3VisionEncoder` 接受：

```text
[B, T, V, 3, H, W]
```

内部先展平时间和视角维度送入 DINOv3，再恢复为：

```text
[B, T, V, P, D]
```

实现不会对 patch tokens 做平均池化。输出中的 CLS/register tokens 会根据 `prefix_tokens` 被删除，仅保留 P 个空间 patch。

DINOv3 backbone 冻结时使用 `torch.no_grad()`；训练的主要视觉参数是 `VisionProjection` 和后续位置编码/交互模块。

### 4.5 patch、view、time 位置编码

文件：

```text
TurboVLA/turbovla/models/turbovla.py
```

模型创建三类可学习参数：

```text
patch_position_embedding [1, V, P, H]
view_embedding           [1, V, 1, H]
time_embedding           [1, T, 1, 1, H]
```

视觉 token 的组合为：

```text
token
+ patch_position_scale * patch_position_embedding
+ view_embedding
+ time_embedding
```

之后把 T、V、P 展平为 token 序列。展平前必须保持 `[B,T,V,P,H]`，否则无法区分不同时间、视角和空间 patch。

### 4.6 StarVLA/TurboVLA 训练接口

文件：

```text
TurboVLA/third_party/starvla_runtime/starVLA/model/framework/VLM4A/TurboVLA.py
```

wrapper 将每个 batch 的图像整理为：

```text
[batch, temporal_window_size, num_views, 3, 224, 224]
```

并在 `_core_config()` 中强制使用：

```python
position_embedding="learned_patch"
```

这里没有替换 TurboVLA action head。视觉和文本条件融合后，仍调用原 `TurboVLAActionHead` 预测 50×14 action chunk，并使用当前配置的 L1 loss。

## 5. 数据加载和 PyAV

配置：

```yaml
video_backend: pyav
num_workers: 8
prefetch_factor: 2
persistent_workers: true
pin_memory: true
```

文件：

```text
TurboVLA/third_party/starvla_runtime/starVLA/dataloader/gr00t_lerobot/video.py
```

所有直接 PyAV 打开操作通过 `_open_pyav_single_thread()`：

```python
container = av.open(video_path)
stream = container.streams.video[0]
stream.codec_context.thread_count = 1
```

含义是每个 DataLoader worker 的每个 PyAV 解码器只使用一个 FFmpeg 线程，避免 `8 workers × FFmpeg 自动线程数` 导致 CPU 过度订阅。

启动时同时限制：

```bash
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
```

## 6. 完整训练参数

```text
GPU                         7
per_device_batch_size       16
gradient_accumulation       1
total batch size            16
num_workers                 8
PyAV threads per worker     1
max_train_steps             500000
warmup_steps                25000
learning rate               5e-5
lr scheduler                cosine
save_interval               20000
eval_interval               20000
action horizon              50
action_dim                   14
state_dim                    20
temporal_window_size         2
num_views                    3
vision encoder frozen        true
text encoder frozen          true
EMA decay                    0.999
```

## 7. 启动前检查

```bash
ssh -p 50210 wengyikun@183.230.224.121 '
  test -d /data/wengyikun/datasets/joint_songling/0812_binary_gripper_without_ep173_174
  grep -n "thread_count = 1" \
    /home/wengyikun/tmp/TurboVLA-0812-patchvision-t2-gpu7/third_party/starvla_runtime/starVLA/dataloader/gr00t_lerobot/video.py
  grep -n "binary_absolute_closed_zero" \
    /home/wengyikun/tmp/TurboVLA-0812-patchvision-t2-gpu7/scripts/joint_songling/train_0812_closed_patchvision_t2_gpu7.sh
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits | grep "^7,"
'
```

不要使用 `--gpus all`，也不要停止 GPU1/GPU2 上的其他训练。

## 8. 1-step smoke test

必须使用全新的 smoke run ID 和输出目录：

```bash
ssh -p 50210 wengyikun@183.230.224.121 '
  cd /home/wengyikun/tmp/TurboVLA-0812-patchvision-t2-gpu7
  CUDA_VISIBLE_DEVICES=7 \
  JOINT_SONGLING_DATA_ROOT=/data/wengyikun/datasets/joint_songling/0812_binary_gripper_without_ep173_174 \
  BERT_MODEL_PATH=/data/wengyikun/models/turbovla_joint_songling/bert-base-uncased \
  DINOV3_MODEL_PATH=/data/wengyikun/models/turbovla_joint_songling/dinov3-vitl16-pretrain-lvd1689m \
  CONFIG_YAML=experiments/joint_songling/configs/0812_closed_patchvision_t2_gpu7_workers8.yaml \
  RUN_ID=turbovla_0812_binary_patchvision_t2_gpu7_smoke \
  RUN_ROOT_DIR=/data/wengyikun/outputs/turbovla_0812_binary_patchvision_t2_gpu7_smoke \
  JOINT_SONGLING_OVERLAY_ROOT=/data/wengyikun/outputs/turbovla_0812_binary_patchvision_t2_gpu7_smoke_overlay \
  SMOKE_TEST=1 \
  WANDB_MODE=disabled \
  bash scripts/joint_songling/train_0812_closed_patchvision_t2_gpu7.sh
'
```

smoke test 必须确认：

- overlay source 指向 binary 数据集；
- `gripper_mode` 为 `binary_absolute_closed_zero`；
- DataLoader 日志显示 `num_workers=8`；
- state/action 分别生成 20D/14D q01/q99；
- state/action 夹爪 q01=0、q99=0.1；
- 出现 Step 1 且 loss 有限；
- 没有 OOM、Traceback 或 RuntimeError。

## 9. 启动正式训练

每次重新训练必须使用新的 `RUN_ID`、`RUN_ROOT_DIR` 和 overlay 目录。以下示例使用新的日期后缀，不能直接覆盖已有 fixed run：

```bash
run_name=turbovla_0812_binary_gripper_patchvision_t2_gpu7_workers8_20260814_v2

ssh -p 50210 wengyikun@183.230.224.121 "
  cd /home/wengyikun/tmp/TurboVLA-0812-patchvision-t2-gpu7
  CUDA_VISIBLE_DEVICES=7 \\
  JOINT_SONGLING_DATA_ROOT=/data/wengyikun/datasets/joint_songling/0812_binary_gripper_without_ep173_174 \\
  BERT_MODEL_PATH=/data/wengyikun/models/turbovla_joint_songling/bert-base-uncased \\
  DINOV3_MODEL_PATH=/data/wengyikun/models/turbovla_joint_songling/dinov3-vitl16-pretrain-lvd1689m \\
  CONFIG_YAML=experiments/joint_songling/configs/0812_closed_patchvision_t2_gpu7_workers8.yaml \\
  RUN_ID=${run_name} \\
  RUN_ROOT_DIR=/data/wengyikun/outputs/${run_name} \\
  JOINT_SONGLING_OVERLAY_ROOT=/data/wengyikun/outputs/${run_name}_overlay \\
  OMP_NUM_THREADS=1 \\
  MKL_NUM_THREADS=1 \\
  WANDB_MODE=disabled \\
  bash scripts/joint_songling/train_0812_closed_patchvision_t2_gpu7.sh \\
  > /data/wengyikun/outputs/${run_name}.log 2>&1 &
"
```

启动脚本会拒绝覆盖已存在的 `RUN_ROOT_DIR`。不要删除旧 run 来复用名称。

## 10. 监控训练

当前 fixed run：

```bash
run_name=turbovla_0812_binary_gripper_patchvision_t2_gpu7_workers8_fixed
```

查看容器：

```bash
ssh -p 50210 wengyikun@183.230.224.121 \
  "docker ps --filter name=${run_name} --format '{{.Names}}|{{.Status}}'"
```

查看日志：

```bash
ssh -p 50210 wengyikun@183.230.224.121 \
  "docker logs --tail 100 ${run_name} 2>&1"
```

查看最新 step：

```bash
ssh -p 50210 wengyikun@183.230.224.121 \
  "docker logs ${run_name} 2>&1 | grep -E 'Step [0-9]+, Loss' | tail"
```

检查真实错误：

```bash
ssh -p 50210 wengyikun@183.230.224.121 \
  "docker logs ${run_name} 2>&1 | grep -E 'Traceback|CUDA out of memory|OutOfMemory|RuntimeError:|Killed process|AssertionError' | tail"
```

查看 GPU7：

```bash
ssh -p 50210 wengyikun@183.230.224.121 \
  "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | grep '^7,'"
```

## 11. 验证 q01/q99 和 binary gripper

```bash
ssh -p 50210 wengyikun@183.230.224.121 '
  run_name=turbovla_0812_binary_gripper_patchvision_t2_gpu7_workers8_fixed
  python3 - /data/wengyikun/outputs/$run_name/$run_name/dataset_statistics.json <<"PY"
import json, sys
stats = json.load(open(sys.argv[1]))["new_embodiment"]
for key, indices in (("state", (9, 19)), ("action", (6, 13))):
    value = stats[key]
    print(key, "dims=", len(value["q01"]))
    print(" q01 grippers=", [value["q01"][i] for i in indices])
    print(" q99 grippers=", [value["q99"][i] for i in indices])
PY
'
```

预期：

```text
state dims=20,  q01=[0,0], q99=[0.1,0.1]
action dims=14, q01=[0,0], q99=[0.1,0.1]
```

## 12. checkpoint 和续训

每 20000 steps 保存一次，典型路径：

```text
/data/wengyikun/outputs/$run_name/$run_name/checkpoints/steps_20000_model.safetensors
/data/wengyikun/outputs/$run_name/$run_name/checkpoints/steps_20000_ema_model.safetensors
```

加载 checkpoint 后重新训练时必须使用新输出目录。仅设置 `pretrained_checkpoint` 通常表示权重初始化，不等同于恢复原 optimizer、scheduler 和 EMA step；文档中应明确区分 warm-start 与完整断点恢复。

## 13. 停止训练

只停止精确容器名：

```bash
ssh -p 50210 wengyikun@183.230.224.121 \
  'docker stop turbovla_0812_binary_gripper_patchvision_t2_gpu7_workers8_fixed'
```

禁止使用：

```bash
docker stop $(docker ps -q)
```

不要删除数据集、overlay、旧 run 输出或 GPU1/GPU2 上其他训练。

## 14. 关键实现文件

```text
TurboVLA/experiments/joint_songling/configs/0812_closed_patchvision_t2_gpu7_workers8.yaml
TurboVLA/experiments/joint_songling/data_registry/data_config.py
TurboVLA/scripts/joint_songling/train_0812_closed_patchvision_t2_gpu7.sh
TurboVLA/turbovla/models/configuration.py
TurboVLA/turbovla/models/vision_encoder.py
TurboVLA/turbovla/models/turbovla.py
TurboVLA/third_party/starvla_runtime/starVLA/model/framework/VLM4A/TurboVLA.py
TurboVLA/third_party/starvla_runtime/starVLA/dataloader/gr00t_lerobot/datasets.py
TurboVLA/third_party/starvla_runtime/starVLA/dataloader/gr00t_lerobot/video.py
```

以上文件共同实现 PatchVision T2；`patch_policy` 仓库本身不参与当前训练进程。

## 15. 正确性检查清单

- [ ] 数据源是 `0812_binary_gripper_without_ep173_174`；
- [ ] overlay 合同 source 与数据源一致；
- [ ] `gripper_mode=binary_absolute_closed_zero`；
- [ ] state=20D，action=14D；
- [ ] 关节相对，endpoint xyz 和夹爪绝对；
- [ ] state/action 使用各自 q01/q99；
- [ ] 夹爪值只有 0 和 0.1；
- [ ] 时间窗口是 `[-1,0]`；
- [ ] 相机顺序为 top、左腕、右腕；
- [ ] top 补黑边，腕部中心裁剪后再 resize；
- [ ] DINOv3 保留 dense patch tokens；
- [ ] patch/view/time 三类位置编码生效；
- [ ] TurboVLA action head 和 horizon=50 保持不变；
- [ ] `num_workers=8`；
- [ ] PyAV 每 worker 单线程；
- [ ] GPU7 单卡隔离；
- [ ] loss 有限且 step 持续增长；
- [ ] 无 OOM、Traceback 和 RuntimeError。

本文不记录 Hugging Face token、SSH 密钥或其他凭据。
