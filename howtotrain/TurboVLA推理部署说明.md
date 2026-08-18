# TurboVLA 双臂真机推理部署说明

本文说明 Joint Songling 双 Piper/Pika 机器人如何部署 TurboVLA checkpoint。部署链路为：机器人采集三路 RGB 图像和 20D 实测状态，经 WebSocket 发送到远端 GPU 推理服务；服务返回一个 50 步、14D 的归一化 action chunk；机器人端反归一化并转换为绝对关节目标后执行完整 chunk，再从实际反馈重新推理下一 chunk。

本文不包含任何密码、SSH 私钥或访问令牌。

## 1. 适用的模型契约

以下 checkpoint 使用同一套双臂关节部署协议：

```text
state:  [20]
action: [50, 14]
views:  top -> gripper_left -> gripper_right
```

20D state 的顺序固定为：

```text
0..5    left_joint_0..5                 (rad)
6..8    left_endpoint_x/y/z             (m, absolute)
9       left_gripper                    (m, absolute)
10..15  right_joint_0..5                (rad)
16..18  right_endpoint_x/y/z            (m, absolute)
19      right_gripper                   (m, absolute)
```

14D action 的顺序固定为：

```text
0..5    left_joint_0..5
6       left_gripper
7..12   right_joint_0..5
13      right_gripper
```

对于 `state_mode: delta` 与 `action_mode: rel`：

- 模型 state 中的左右关节是连续两次**真机实测反馈**的差 `q(t)-q(t-1)`；endpoint xyz 与夹爪仍为绝对值。
- 模型 action 中的左右关节是相对当前 chunk 起始实测关节的目标 `q(t+k)-q(t)`。
- action 的两个夹爪是绝对开口宽度，不做相对累加。
- 每个 action chunk 的 50 个关节 action 都必须锚定到同一个 chunk 起始实测关节 `q(t)`；不能把第 k 步继续累加到第 k-1 步。

state 和 action 各自使用 checkpoint 同目录 `dataset_statistics.json` 中独立的 q01/q99 统计。不可混用别的训练 run 的统计文件。

## 2. 选择 checkpoint

一个 checkpoint 通常会同时存在：

```text
steps_<N>_model.safetensors
steps_<N>_ema_model.safetensors
```

两者不是同一个文件：

- `model`：第 N step 的当前训练参数。
- `ema_model`：训练期间以 `ema_decay`（常用 `0.999`）更新的指数滑动平均参数。

真机推理默认优先使用 `steps_<N>_ema_model.safetensors`。只在需要精确复现实验中原始 step 参数时使用普通 `model` 文件。

推理前必须确认目录中同时存在：

```text
config.yaml
config.full.yaml
dataset_statistics.json
checkpoints/steps_<N>_ema_model.safetensors
```

## 3. 三路相机与图像布局

机器人端相机语义与顺序不可更改：

```text
top             顶部双目相机的右眼
gripper_left    左 D405 的 RGB 图
gripper_right   右 D405 的 RGB 图
```

原始上传尺寸为：

```text
top             405 x 720 x 3 RGB
gripper_left    480 x 640 x 3 RGB
gripper_right   480 x 640 x 3 RGB
```

顶部相机从 `3840x1080` 双目 BGR 帧中取右半边 `x=[1920:3840]`，缩放到 `720x405`，再转换为 RGB。左右 D405 分别读取 `640x480 rgb8`。

训练配置中的 `image_layout` 决定 resize 前的几何处理。必须与 checkpoint 的 `config.full.yaml` 一致：

| `image_layout` | top | 左右腕部 |
|---|---|---|
| `joint_songling` | 上补 157、下补 158 黑边，`405x720 -> 720x720` | 水平中心裁剪 `x=[80:560]`，`480x640 -> 480x480` |
| `joint_songling_top_padded` | 上补 157、下补 158 黑边，`405x720 -> 720x720` | 保持原始 `480x640`，不裁剪 |

随后 DINOv3 image processor 将每张图 resize 到 `224x224`，按 `1/255` 缩放，并使用 ImageNet mean/std 归一化。不要把 ACT 的图像处理规则直接套到 TurboVLA checkpoint；是否裁剪腕部完全由 `image_layout` 决定。

## 4. 真机硬件映射

使用稳定的设备路径，不能依赖会变化的 `/dev/ttyUSB*` 编号：

```text
left_piper       CAN: left_piper
right_piper      CAN: right_piper

left_gripper     pci-0000:c4:00.3-usb-0:3.4:1.0-port0 -> ttyUSB2
right_gripper    pci-0000:c6:00.4-usb-0:1.4:1.0-port0 -> ttyUSB0

top              /dev/video26
left D405        serial 412622273326
right D405       serial 260622271788
```

先只读检查，确认 CAN 都是 `ERROR-ACTIVE`、`1 Mbps`，设备路径存在，且没有其他真机控制客户端占用同一 CAN：

```bash
ip -details link show left_piper
ip -details link show right_piper
ls -l /dev/serial/by-path/pci-0000:c4:00.3-usb-0:3.4:1.0-port0 \
      /dev/serial/by-path/pci-0000:c6:00.4-usb-0:1.4:1.0-port0 \
      /dev/video26
ps -eo pid,user,stat,args | grep -Ei 'dual_turbovla|robot_client|teleop_piper|piper_single_ctrl' | grep -v grep
```

## 5. 推理服务端

远端运行时使用 Docker 镜像：

```text
turbovla-joint-songling:20260803
```

容器环境需要同时设置仓库根目录和 `starVLA` runtime：

```bash
export PYTHONPATH=/workspace/TurboVLA-joint-songling-relative:/workspace/TurboVLA-joint-songling-relative/third_party/starvla_runtime
```

镜像没有预装 WebSocket 服务依赖，启动前安装固定版本：

```bash
python -m pip install websockets==15.0.1
```

示例：将 EMA checkpoint 绑定到单张 GPU 与独立端口 `18065`：

```bash
docker run -d --name turbovla_example_gpu3 \
  --gpus 'device=3' --network host --ipc=host \
  -e PYTHONPATH=/workspace/TurboVLA-joint-songling-relative:/workspace/TurboVLA-joint-songling-relative/third_party/starvla_runtime \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v /home/wengyikun/tmp/TurboVLA-joint-songling-relative:/workspace/TurboVLA-joint-songling-relative:ro \
  -v /data:/data:ro \
  turbovla-joint-songling:20260803 \
  bash -lc 'python -m pip install --quiet websockets==15.0.1 && exec python /workspace/TurboVLA-joint-songling-relative/third_party/starvla_runtime/deployment/model_server/server_policy.py \
    --ckpt_path /data/wengyikun/outputs/<run>/<run>/checkpoints/steps_<N>_ema_model.safetensors \
    --port 18065 --use_bf16 --idle_timeout=-1'
```

严格加载是默认行为。不要传 `--allow_partial_load`；若 checkpoint 键或形状不匹配，应让服务启动失败，而不是跳过权重。

服务端日志必须出现：

```text
Loading checkpoint metadata from ...
server running ...
server listening on 0.0.0.0:<port>
```

## 6. 机器人端协议

机器人客户端使用 WebSocket 请求：

```text
payload.examples[0].image = [top, gripper_left, gripper_right]
payload.examples[0].lang  = 训练任务文本
payload.examples[0].state = 归一化后的 20D state
```

服务端返回：

```text
normalized_actions: [1, 50, 14]
```

机器人端处理顺序必须为：

```text
相邻实测状态
-> 关节 state delta，保留 endpoint/gripper 绝对量
-> state q99 归一化
-> 推理
-> action q99 反归一化
-> 相对关节 action 加到 chunk 起始实测关节
-> 夹爪宽度保持绝对米单位
-> 逐步发送 50 个绝对 14D 目标
-> 重新读取真机反馈，开始下一 chunk
```

Pika 夹爪最大宽度可设为 `0.10 m`，以避免训练统计上限约 `96 mm` 的输出被错误截断到 `90 mm`。

## 7. 启动顺序

推荐采用独立目录保存每次部署脚本、日志、统计与客户端副本，例如：

```text
/home/kw/runs/turbovla_<run>_<step>_ema_gpu<id>/
```

1. 启动远端服务并等待端口监听。
2. 在机器人端建立仅用于该服务的 SSH 隧道：

```bash
ssh -f -N -o ExitOnForwardFailure=yes \
  -L 18065:127.0.0.1:18065 \
  -p 50210 wengyikun@183.230.224.121
```

3. 先运行一次 dry-run。dry-run 可以打开 CAN、相机和 Pika 读取，但不得传入 `--enable-arms`、`--enable-grippers` 或 `--execute-robot-actions`。
4. 核对以下输出后才能真机执行：

```text
images=[训练布局后的三张图尺寸]
normalized_actions shape = [1, 50, 14]
DRY_RUN: action chunk was not sent to hardware
```

5. 真机执行时才同时传入：

```text
--enable-arms --enable-grippers --execute-robot-actions
```

## 8. 停止

停止时只处理本次部署的精确客户端、容器和隧道，不能使用会影响其他实验的广泛 `pkill` 或 `docker stop $(docker ps -q)`。

```bash
# 机器人端：停止本次客户端
kill -TERM <robot_client_pid>

# 远端：停止本次容器
docker stop turbovla_example_gpu3

# 机器人端：关闭对应端口的 SSH 隧道
ss -ltnp | grep ':18065'
kill -TERM <ssh_tunnel_pid>
```

停止后确认：

```bash
pgrep -af dual_turbovla_endpoint20_robot_client.py
ss -ltnp | grep ':18065'
docker ps --filter name=turbovla_example_gpu3
```

## 9. 常见问题

| 现象 | 原因与处理 |
|---|---|
| `ModuleNotFoundError: websockets` | 容器内安装 `websockets==15.0.1`。 |
| `ModuleNotFoundError: turbovla` 或 `deployment` | `PYTHONPATH` 必须同时包含 TurboVLA 仓库根与 `third_party/starvla_runtime`。 |
| checkpoint 加载报键/形状错误 | 不要使用 partial load；核对 checkpoint、同目录 config 和当前 TurboVLA runtime 是否匹配。 |
| 图像行为异常 | 核对 camera 顺序必须是 top、left、right，并根据 `image_layout` 选择 top padding 和腕部裁剪策略。 |
| 夹爪左右相反 | 确认 action index `6` 到 c4 左 Pika，index `13` 到 c6 右 Pika。 |
| 夹爪几乎不动 | 检查 action q01/q99、单位是否为米、是否被 `gripper_max_m` 截断，以及 Pika 是否被其他进程占用。 |
