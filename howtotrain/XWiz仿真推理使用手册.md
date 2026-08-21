# XWiz 仿真推理使用手册

## 1. 重启后能否直接开始

当前不能保证整机重启后不检查就直接点击“仿真推理”。

原因是目前只有本机 ACT 模型服务配置了用户级自启动；以下进程仍是手动后台进程，重启 PC1 或 PC2 后不会自动恢复：

- PC1 `run_inference_manager_safe.py` 推理管理器。
- PC1 KFC 头部双目相机节点。
- PC2 `safe_client_service.py` 仿真安全客户端。
- PC2 `black_wrist_images.py` 左右腕部黑图发布器。

此外，PC1 的 `dexe-auto.timer` 当前虽然未运行，但仍处于 enabled。PC1 重启后必须检查 Auto 是否被拉起，防止它与仿真推理同时占用控制链路。

本机 `xwiz-act-server.service` 已 enabled，但本机用户 `Linger=no`，因此它会在用户登录桌面后启动，而不是在无人登录时启动。

结论：

- 只重启 XWiz 图形界面、不重启三台机器：通常可直接重新打开 XWiz 使用。
- 重启本机并重新登录桌面：先确认本机 8889 服务正常，再使用。
- 重启 PC1 或 PC2：需要按本文第 3 节恢复安全推理链路。

## 2. 当前部署结构

```text
本机 192.168.20.164
  XWiz 图形界面
  act_popcorn_45w CUDA 推理服务 :8889
            |
            v
PC1 192.168.20.20
  inference_manager ROS 服务
            |
            v
PC2 192.168.20.21
  safe_client_service :8890
  左右腕部黑图
  仅发布 /mj_sim/control/*
```

XWiz 中使用：

- 模型 ID：`1`
- 任务 ID：`1`
- 推理模式：`仿真`
- 模型目录：`/home/wengyikun/workplace/popcorn/act_popcorn_45w/pretrained_model`
- 模型输入：19 维状态、真实左头部图像、左右腕部黑图。
- 模型输出：`100x19` 绝对关节目标。

## 3. 三台机器重启后的启动顺序

### 3.1 检查网络

在本机执行：

```bash
ping -c 1 192.168.20.20
ping -c 1 192.168.20.21
```

两台机器人 PC 都能访问后再继续。

### 3.2 检查本机模型服务

```bash
systemctl --user status xwiz-act-server.service --no-pager
ss -ltnp | grep ':8889'
```

如果没有运行：

```bash
systemctl --user restart xwiz-act-server.service
journalctl --user -u xwiz-act-server.service -n 50 --no-pager
```

正常日志应包含：

```text
Loading weights from local directory
listening on 0.0.0.0:8889
```

### 3.3 停止 PC1 的真机控制模式

登录 PC1：

```bash
ssh dexforce@192.168.20.20
```

检查状态：

```bash
systemctl is-active dexe-auto.service dexe-auto.timer \
  dexe-tele.service dexe-act.service dexe-map.service
docker ps --filter name=act_ros2
pgrep -af 'policy_infer.py|policy_bridge_origin.py'
```

仿真推理前应停止 Auto 和旧 ACT 容器：

```bash
sudo systemctl stop dexe-auto.service dexe-auto.timer
docker stop act_ros2 2>/dev/null || true
```

不要停止以下机器人底层服务：

```text
dexe-system.service
dexe-basic.service
```

### 3.4 启动 PC1 头部相机

先检查：

```bash
pgrep -af 'kfc_nodes.launch|kfc_publisher'
```

如果没有进程，执行：

```bash
export ROS_DOMAIN_ID=20
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
source /opt/ros/humble/setup.bash
source /home/dexforce/w1/install/setup.bash

nohup ros2 launch dexe_sensors_launch kfc_nodes.launch.py \
  kfc_mode:=resize_compressed \
  > /home/dexforce/kfc_camera.log 2>&1 < /dev/null &
```

检查双目图像：

```bash
ros2 topic hz /camera/left_eye_resize
ros2 topic hz /camera/right_eye_resize
```

### 3.5 启动 PC2 黑图和安全客户端

登录 PC2：

```bash
ssh dexforce@192.168.20.21
```

加载环境：

```bash
export PYTHONPATH=/home/dexforce/w1/w1_act
export ROS_DOMAIN_ID=20
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
source /opt/ros/humble/setup.bash
source /home/dexforce/w1/install/setup.bash
cd /home/dexforce/w1/w1_act
```

先检查，避免重复启动：

```bash
pgrep -af 'black_wrist_images.py|safe_client_service.py'
```

缺少黑图发布器时执行：

```bash
nohup python3 xwiz_safe_runtime/black_wrist_images.py \
  > /home/dexforce/xwiz_black_wrist.log 2>&1 < /dev/null &
```

缺少安全客户端时执行：

```bash
cd /home/dexforce/w1/w1_act/xwiz_safe_runtime
nohup python3 safe_client_service.py --config client_simulation.json \
  >> /home/dexforce/xwiz_safe_client.log 2>&1 < /dev/null &
```

检查：

```bash
ss -ltn | grep ':8890'
ros2 topic hz /camera_l/color/image_rect_raw
ros2 topic hz /camera_r/color/image_rect_raw
```

### 3.6 启动 PC1 推理管理器

回到 PC1，先检查：

```bash
pgrep -af run_inference_manager_safe.py
```

如果没有进程，执行：

```bash
cd /home/dexforce/w1/w1_act
export PYTHONPATH=$PWD
export ROS_DOMAIN_ID=20
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
source /opt/ros/humble/setup.bash
source /home/dexforce/w1/install/setup.bash

nohup python3 run_inference_manager_safe.py \
  > /home/dexforce/inference_manager.log 2>&1 < /dev/null &
```

检查服务和状态：

```bash
ros2 service list | grep /inference
ros2 topic echo --once /inference/status
```

空闲状态应为：

```text
status: 1
message: ok
```

## 4. 在 XWiz 中开始仿真推理

本机启动 XWiz：

```bash
xwiz
```

操作步骤：

1. 打开“模型部署”。
2. 选择模型 ID `1`。
3. 选择任务 ID `1`。
4. 选择“仿真”模式，不要选择真机模式。
5. 点击“开始推理”或“仿真推理”。

成功时：

- XWiz 显示推理服务启动成功。
- 本机日志出现 `inference completed ... action_shape=(100,19)`。
- PC2 日志出现 `Actions received` 和 `Executed Action Step`。
- `/mj_sim/control/joint_position` 有约 10 Hz 数据。

检查命令：

```bash
journalctl --user -u xwiz-act-server.service -f
```

PC2：

```bash
tail -f /home/dexforce/xwiz_safe_client.log
ros2 topic hz /mj_sim/control/joint_position
```

## 5. 停止仿真推理

优先在 XWiz 中点击“停止推理”。也可以在 PC1 调用：

```bash
export ROS_DOMAIN_ID=20
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
source /opt/ros/humble/setup.bash
source /home/dexforce/w1/install/setup.bash
ros2 service call /inference/stop_inference std_srvs/srv/Trigger '{}'
```

停止后检查：

```bash
ros2 topic echo --once /inference/status
```

应返回 `status: 1` 和 `message: ok`。

## 6. 安全检查

仿真客户端只能出现在 `/mj_sim/control/*`：

```bash
ros2 topic info -v /mj_sim/control/joint_position
ros2 topic info -v /control/joint_position
```

预期结果：

- `/mj_sim/control/joint_position` 的发布者为 `optimized_robot_client`。
- 真实 `/control/joint_position` 不能新增 `optimized_robot_client`。
- 真实话题可能存在基础界面的 `fastapi_bridge_node`，它不是本次仿真客户端。

PC2 安全包装会强制：

```text
mode=1
home_position=""
chunk_size_threshold=0.0
```

因此本流程不会执行复位动作，也不会把 ACT 动作发布到真机控制话题。

## 7. 常见故障

### 启动调用推理复位超时

依次检查：

```text
本机 8889 -> PC2 8890 -> PC1 inference_manager -> XWiz ROS 服务
```

### 提示头部相机没有数据

```bash
ros2 topic hz /camera/left_eye_resize
ros2 topic hz /camera/right_eye_resize
tail -100 /home/dexforce/kfc_camera.log
```

### 腕部相机没有数据

```bash
pgrep -af black_wrist_images.py
ros2 topic echo --once /camera_l/color/image_rect_raw
ros2 topic echo --once /camera_r/color/image_rect_raw
```

黑图的正确规格是 `640x360`、`bgr8`、像素全部为 0。

### 本机模型服务失败

```bash
systemctl --user restart xwiz-act-server.service
journalctl --user -u xwiz-act-server.service -n 100 --no-pager
```

不要同时手工运行 `start_local.sh` 和 systemd 服务，否则会发生 8889 端口冲突。
