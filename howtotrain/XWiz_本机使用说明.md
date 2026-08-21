# XWiz 本机使用说明

## 1. 安装位置

XWiz 安装在本机 `wengyikun-host`，不是机器人 PC1/PC2。

- XWiz 版本：`0.1.6+20260617035053+g587688f7`
- 用户运行目录：`~/xwiz`
- ROS 2 推理接口：`~/workspace/install`
- ROS 2 配置：`~/xwiz/config/ros2.env`
- ROS 2 话题配置：`~/xwiz/config/ros2_bridge.yaml`

本机机器人网口地址为 `192.168.20.164`。CycloneDDS 已配置为使用该地址，ROS 域为 `20`。

## 2. 启动图形界面

打开终端执行：

```bash
xwiz
```

XWiz 启动脚本会自动加载 `/opt/ros/jazzy` 和 `~/xwiz/config/ros2.env` 中的配置。

无图形界面或远程服务器环境可使用：

```bash
xwiz --headless
```

停止界面：在 XWiz 窗口中退出，或在终端执行：

```bash
pkill -f '/xwiz/bin/xwiz'
```

## 3. ROS 2 通信配置

当前配置文件为 `~/xwiz/config/ros2.env`，关键参数：

```bash
ROS2_PREFIX=/opt/ros/jazzy
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ROS_DOMAIN_ID=20
CYCLONEDDS_URI=使用本机 192.168.20.164 网卡
```

双目推理图像已在 `~/xwiz/config/ros2_bridge.yaml` 中启用：

```yaml
/camera/left_eye_resize: sensor_msgs/msg/Image
/camera/right_eye_resize: sensor_msgs/msg/Image
```

机器人 PC1 的 ROS 2 通信地址是 `192.168.20.20`。启动前应确认本机可以访问：

```bash
ping -c 1 192.168.20.20
```

## 4. 推理相关话题和服务

主要输入话题：

- `/camera/left_eye_resize`：左目图像
- `/camera/right_eye_resize`：右目图像
- `/inference/prompt`：推理提示词

主要输出/控制话题：

- `/control/joint_position`：机器人关节位置命令
- `/inference/status`：推理状态

推理服务包括：

- `/inference/deploy`：部署模型
- `/inference/start_inference`：开始推理
- `/inference/stop_inference`：停止推理
- `/inference/get_model_info`：读取模型信息

查看当前 ROS 2 话题：

```bash
source /opt/ros/jazzy/setup.bash
source ~/workspace/install/setup.bash
export ROS_DOMAIN_ID=20
ros2 topic list
```

## 5. 常用检查

检查 XWiz 是否正在运行：

```bash
pgrep -af '/xwiz/bin/xwiz'
```

查看 XWiz 日志：

```bash
find ~/xwiz -type f -name '*.log' -o -path '~/.dexforce/XWiz/*'
```

检查双目图像是否发布：

```bash
ros2 topic hz /camera/left_eye_resize
ros2 topic hz /camera/right_eye_resize
```

检查推理状态：

```bash
ros2 topic echo /inference/status --once
```

## 6. 注意事项

1. XWiz 本机的 `ROS_DOMAIN_ID` 必须与机器人 ROS 2 域保持一致，目前为 `20`。
2. 不要同时运行多个会向 `/control/joint_position` 发布命令的控制程序。
3. 运行推理前确认机器人处于安全姿态，并确认 Tele、Auto、ACT 等其他控制源已停止。
4. 如果本机网卡地址改变，需要同步修改 `~/xwiz/config/ros2.env` 中的 CycloneDDS `NetworkInterface address`。
5. 双目图像话题会增加相机带宽和 CPU/GPU 负载；不进行推理时可以在 `ros2_bridge.yaml` 中重新注释这两行。

## 7. `act_popcorn_45w` 仿真推理

当前模型已经作为 XWiz 的模型 ID `1`、任务 ID `1` 配置完成。点击“仿真推理”时的数据链路为：

```text
XWiz(本机) -> PC1 推理管理器 -> PC2 安全客户端
             PC2 -> 本机 192.168.20.164:8889 ACT 服务
             PC2 -> /mj_sim/control/*
```

模型权重实际由本机 RTX GPU 从以下目录严格加载：

```text
/home/wengyikun/workplace/popcorn/act_popcorn_45w/pretrained_model
```

输入契约：

- 头部图像：`/camera/left_eye_resize`，模型键 `observation.images.cam_high_left`，`640x360`。
- 左右腕部图像：PC2 发布固定黑图，模型键分别为 `cam_hand_left`、`cam_hand_right`，均为 `640x360`。
- 状态：腰 1 + 左臂 7 + 头 2 + 右臂 7 + 左右夹爪各 1，共 19 维。
- W1 当前为灵巧手，没有二指夹反馈；仿真客户端仅给模型注入两个 `0.0` 夹爪状态，不写入真机。
- 输出：每次 CUDA 推理产生 `100x19` 绝对关节目标，仅发布到 `/mj_sim/control/*`。

### 7.1 在 XWiz 中运行

1. 确认本机 ACT 服务、PC2 安全客户端和黑图发布器在线。
2. 在 XWiz 模型部署中选择模型 `1` 和任务 `1`。
3. 选择“仿真”模式，点击“开始推理/仿真推理”。
4. 结束时点击“停止推理”。

任务配置位于：

```text
~/.dexforce/XWiz/model_deployments/tasks/1/task_config.json
```

关键值为 `server_host=192.168.20.164`、`server_port=8889`、`mode=1`、`action_horizon=100`、`home_position=""` 和 `data_type=simulation`。

### 7.2 检查运行状态

本机服务：

```bash
systemctl --user status xwiz-act-server.service
journalctl --user -u xwiz-act-server.service -n 100 --no-pager
ss -ltnp | grep ':8889'
```

PC2 日志：

```bash
ssh dexforce@192.168.20.21
tail -f /home/dexforce/xwiz_safe_client.log
```

仿真输出频率：

```bash
ros2 topic hz /mj_sim/control/joint_position
```

正常时本机日志包含 `inference completed ... action_shape=(100,19)`，PC2 日志包含 `Actions received` 和 `Executed Action Step`。

### 7.3 安全边界

- PC2 包装器强制 `mode=1`、清空 `home_position` 并禁用 LiPo，XWiz 下发的真机模式值不会覆盖这些限制。
- 仿真客户端发布者必须属于 `/mj_sim/control/*`，不能属于 `/control/*`。
- Auto、Tele、ACT、Map 和 `act_ros2` 容器不得与本链路同时运行。
- `/control/joint_position` 可能存在 W1 基础界面的 `fastapi_bridge_node` 发布端；它不是本仿真客户端。应使用 `ros2 topic info -v` 核对发布者名称。

### 7.4 后续全部迁移到 PC2

如果以后把模型服务也迁移到 PC2，需要先满足：PC2 有可用 CUDA GPU/驱动、能够严格加载同一 checkpoint、完整安装 LeRobot 和处理器依赖，并完成一次 `100x19` 有限值 smoke test。完成后把任务的 `server_host` 改为 `192.168.20.21`；在验证前不要删除或覆盖当前本机服务配置。
