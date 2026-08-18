# 如何使用 ACT 推理

本文说明 0812 双臂三相机 ACT checkpoint 的远端推理与 Piper/Pika 真机部署流程。

## 1. 模型契约

部署前必须读取目标 checkpoint 的 `pretrained_model/config.json`，不能只根据目录名复用命令。当前这一类 ACT checkpoint 的已验证契约为：

- policy：`act`
- state：20D，顺序为左臂 6 个关节、左端点 XYZ、左夹爪、右臂 6 个关节、右端点 XYZ、右夹爪。
- action：14D，顺序为左臂 6 个关节、左夹爪、右臂 6 个关节、右夹爪。
- `joint_representation="relative"`：12 个机械臂关节 state 使用相邻两帧真实反馈构造相对值；端点 XYZ 和两个夹爪仍为绝对值。
- action chunk：`[1, 16, 14]`。每次完整执行一个 16 步 chunk，再以新的真实反馈重新推理。
- 当前训练输入图像：`top` 为 `(3,405,720)`，`gripper_left` 与 `gripper_right` 均为 `(3,480,640)`。
- checkpoint 的 `clip_quantiles=true` 必须保持不变。state 与 action 的 q01/q99 由 checkpoint 中独立的 processor 文件提供，不能互换。

真机 client 向服务端发送原始绝对 20D state 与原始三路 RGB 图像。服务端的 checkpoint preprocessor 负责相对 state 和归一化；postprocessor 负责 action 反归一化，并将相对关节 action 变为以当前真实关节 state 为锚点的绝对关节目标。

## 2. 已验证的硬件映射

| 项目 | 配置 |
| --- | --- |
| 左臂 CAN | `left_piper`，1 Mbps |
| 右臂 CAN | `right_piper`，1 Mbps |
| 顶部相机 | `/dev/video26`，YUY2 `3840x1080`；client 取右眼并 resize 为 `720x405` RGB |
| 左 D405 RGB | serial `412622273326`，`480x640`，上传为 `gripper_left` |
| 右 D405 RGB | serial `260622271788`，`480x640`，上传为 `gripper_right` |
| 左 Pika | `/dev/serial/by-path/pci-0000:c4:00.3-usb-0:3.4:1.0-port0` |
| 右 Pika | `/dev/serial/by-path/pci-0000:c6:00.4-usb-0:1.4:1.0-port0` |

不要使用会漂移的 `/dev/ttyUSB*` 编号。当前真机 Pika 物理上限为 `0.09 m`；模型输出超过此值时由 client 截断到 `90 mm`，不要通过修改 checkpoint 的归一化配置来处理。

## 3. 部署前检查

GPU 服务端：`wengyikun@183.230.224.121:50210`。

真机端：`kw@192.168.10.82`。先确认两路 CAN 都是 `ERROR-ACTIVE`、1 Mbps，且没有旧 client、隧道或其它控制器占用。

```bash
ip -details link show left_piper
ip -details link show right_piper
ps -eo pid,ppid,user,stat,etime,args | \
  grep -Ei 'robot_client|policy_server|piper|pika|teleop' | grep -v grep
```

检查 checkpoint：

```bash
CKPT=/data/wengyikun/outputs/<run>/train_out/checkpoints/<step>/pretrained_model
python - <<'PY'
import json, os
c = json.load(open(os.environ['CKPT'] + '/config.json'))
print(c['type'], c['chunk_size'], c['n_action_steps'])
print(c['input_features'])
print(c['output_features'])
PY
```

运行前设置 `CKPT` 环境变量，例如：

```bash
export CKPT=/data/wengyikun/outputs/act_0812_closed_gripper_zero_relative_three_rgb_chunk16_b64_500k_workers8_pyav1_gpu0_20260813/train_out/checkpoints/020000/pretrained_model
```

## 4. 启动服务端

服务端使用隔离 runtime，避免改动原始 LeRobot 源码。`GPU` 按实际空闲卡调整。

```bash
RUNTIME=/home/wengyikun/lerobot_runtime_act_0806swap_050000
GPU=0

CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$RUNTIME/src" \
  /data/miniconda3/envs/lerobot/bin/python \
  -m lerobot.async_inference.policy_server \
  --host 0.0.0.0 --port 8080 --fps 30 \
  --inference_latency 0.033 --obs_queue_timeout 1
```

服务端没有 checkpoint 参数。checkpoint 由真机 client 连接时发送。正常日志应包含：

```text
Primed relative state from consecutive actual feedback ... state_dim=20
action shape: torch.Size([1, 16, 14])
```

## 5. 建立真机到服务端的隧道

公网不直接暴露 8080；在真机端建立本地隧道：

```bash
KEY=/home/kw/workspace/lerobot_wyk_dev/runtime/act_0806swap_050000/robot_to_cloud_8080
ssh -N -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
  -i "$KEY" -p 50210 \
  -L 127.0.0.1:18080:127.0.0.1:8080 \
  wengyikun@183.230.224.121
```

验证：

```bash
ss -ltn | grep ':18080'
```

## 6. 真机 client 与 dry-run

使用与 checkpoint 图像键名一致的 client。对于 `top/gripper_left/gripper_right` 模型，使用已验证的隔离 client：

```bash
CLIENT=/home/kw/workspace/lerobot_wyk_dev/runtime/act_0812_full_020000/dual_act_endpoint20_robot_client.py
```

先不带 `--enable-arms`、`--enable-grippers`、`--execute-robot-actions`，只做网络 dry-run：

```bash
/home/kw/miniforge3/envs/lerobot/bin/python "$CLIENT" \
  --server-address 127.0.0.1:18080 \
  --checkpoint "$CKPT" --policy-type act \
  --actions-per-chunk 16 --task 'grasp bread' --fps 30 \
  --left-can left_piper --right-can right_piper \
  --left-d405-serial 412622273326 --right-d405-serial 260622271788 \
  --top-device /dev/video26 --top-codec raw \
  --left-pika-port /dev/serial/by-path/pci-0000:c4:00.3-usb-0:3.4:1.0-port0 \
  --right-pika-port /dev/serial/by-path/pci-0000:c6:00.4-usb-0:1.4:1.0-port0 \
  --gripper-max-m 0.09
```

dry-run 必须看到 `Dry-run action not sent to hardware`，并在服务端看到 20D 相邻真实反馈和 `[1,16,14]` 输出。确认 action 数值、图像键名与相机方向无误后，才允许执行下一节。

## 7. 启动真机执行

在上一节命令末尾增加：

```bash
--enable-arms --enable-grippers --execute-robot-actions
```

完整运行时，每个 chunk 执行完后会采集真机最新的两帧连续反馈；下一次推理使用 `q(t)-q(t-1)` 构造 relative state，而不是使用旧推理请求的状态，也不是累计模型 action。

## 8. 停止

先停止真机 client，再停止隧道，最后停止服务端。使用保存的 PID，避免根据模糊的进程名误杀其它任务：

```bash
kill -TERM <robot_client_pid>
kill -TERM <ssh_tunnel_pid>
kill -TERM <policy_server_pid>
```

验证：

```bash
ss -ltnp | grep ':8080' || echo '8080 closed'
ss -ltnp | grep ':18080' || echo '18080 closed'
```

收到停止信号后 client 会断开 Pika，并禁用已由它使能的 Piper。不要在 policy client 仍运行时再启动 ROS/SDK 的另一套关节控制程序。
