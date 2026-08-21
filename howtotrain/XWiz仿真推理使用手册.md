# XWiz 仿真与真机推理使用手册

## 1. 当前交互规则

XWiz 原厂“部署”按钮会下发配置并立即启动推理，不需要再点一次“开始推理”。当前用两个任务隔离模式：

| XWiz选择 | 模式 | 输出位置 | 点击“部署”后的行为 |
|---|---|---|---|
| 模型1 + 任务1 | 仿真 | `/mj_sim/control/*` | 立即推理并执行100帧仿真动作 |
| 模型1 + 任务2 | 真机 | `/control/joint_position`、`/control/ee/left`、`/control/ee/right` | 安全检查通过后立即推理并执行一个100帧动作块 |

两个任务都固定为 `action_horizon=100`、`max_steps=100`、`sample_factor=1`、`chunk_size_threshold=0`、`home_position=""`。真机任务不会自动复位；机器人必须事先位于 ACT 默认姿势。

真机动作是 `100×19` 绝对目标：17维身体关节加左右夹爪开合度。夹爪标量 `0=完全闭合`、`100=完全张开`，运行时转换为左右 Linker L6 的6维手指命令。控制频率10 Hz，一个动作块约10秒，第100帧后自动停止。

## 2. 三端结构

```text
本机 192.168.20.164
  XWiz GUI
  act_popcorn_45w CUDA模型服务 :8889
                 |
PC1 192.168.20.20
  xwiz_real_runtime.manager_service
                 |
PC2 192.168.20.21
  xwiz_real_runtime.client_service :8890
  xwiz_real_runtime.black_wrist_images
```

模型目录：`/home/wengyikun/workplace/popcorn/act_popcorn_45w`。

PC1/PC2运行代码：`/home/dexforce/w1/w1_act/xwiz_real_runtime/`。

## 3. 全部重启后的启动顺序

### 3.1 本机

```bash
ping -c 1 192.168.20.20
ping -c 1 192.168.20.21

systemctl --user restart xwiz-act-server.service
systemctl --user status xwiz-act-server.service --no-pager
ss -ltnp | grep ':8889'
journalctl --user -u xwiz-act-server.service -n 50 --no-pager

xwiz
```

### 3.2 PC1：停止冲突控制源并恢复相机

```bash
ssh dexforce@192.168.20.20
sudo systemctl stop dexe-auto.service dexe-auto.timer \
  dexe-tele.service dexe-act.service dexe-map.service
sudo systemctl disable --now dexe-auto.timer
docker stop act_ros2 2>/dev/null || true
```

不要停止 `dexe-system.service` 和 `dexe-basic.service`。

上面的 `disable` 是为了防止 Auto 在重启后自动重新占用控制链路。需要恢复原厂Auto时，再显式执行 `sudo systemctl enable --now dexe-auto.timer`。

确认头部图像；没有发布器时启动KFC相机：

```bash
export ROS_DOMAIN_ID=20
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
source /opt/ros/humble/setup.bash
source /home/dexforce/w1/install/setup.bash

ros2 topic info /camera/left_eye_resize
ros2 topic info /camera/right_eye_resize

nohup ros2 launch dexe_sensors_launch kfc_nodes.launch.py \
  kfc_mode:=resize_compressed \
  > /home/dexforce/kfc_camera.log 2>&1 < /dev/null &
```

### 3.3 PC2：腕部黑图与双模式客户端

```bash
ssh dexforce@192.168.20.21
export ROS_DOMAIN_ID=20
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
source /opt/ros/humble/setup.bash
source /home/dexforce/w1/install/setup.bash
export PYTHONPATH=/home/dexforce/w1/w1_act:${PYTHONPATH}

nohup python3 -m xwiz_real_runtime.black_wrist_images \
  > /home/dexforce/xwiz_black_wrist.log 2>&1 < /dev/null &

nohup python3 -m xwiz_real_runtime.client_service \
  --config /home/dexforce/w1/w1_act/xwiz_real_runtime/client_runtime.json \
  > /home/dexforce/xwiz_real_client.log 2>&1 < /dev/null &
```

检查：

```bash
pgrep -af 'xwiz_real_runtime.(black_wrist_images|client_service)'
ss -ltnp | grep ':8890'
ros2 topic info /camera_l/color/image_rect_raw
ros2 topic info /camera_r/color/image_rect_raw
tail -100 /home/dexforce/xwiz_real_client.log
```

不要重复启动同名进程。

### 3.4 PC1：推理管理器

```bash
ssh dexforce@192.168.20.20
export ROS_DOMAIN_ID=20
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
source /opt/ros/humble/setup.bash
source /home/dexforce/w1/install/setup.bash
export PYTHONPATH=/home/dexforce/w1/w1_act:${PYTHONPATH}

nohup python3 -m xwiz_real_runtime.manager_service \
  > /home/dexforce/xwiz_real_manager.log 2>&1 < /dev/null &
```

检查：

```bash
pgrep -af 'xwiz_real_runtime.manager_service'
ros2 service type /inference/deploy
ros2 service type /inference/stop_inference
tail -100 /home/dexforce/xwiz_real_manager.log
```

## 4. 仿真操作

1. 在“模型部署”中选择模型1、任务1。
2. 确认任务名称为仿真任务。
3. 点击“部署”。这一步会立即启动仿真推理。
4. 结束时点击“停止推理”；单块模式也会在100帧后停止。

验证：

```bash
ros2 topic hz /mj_sim/control/joint_position
tail -f /home/dexforce/xwiz_real_client.log
journalctl --user -u xwiz-act-server.service -f
```

仿真任务不会创建 `optimized_robot_client` 的真机话题发布器。

## 5. 真机操作

点击前必须确认：机器人周围无人和障碍、物理急停可立即触达、Auto/Tele/ACT/Map均已停止、机器人已处于ACT默认姿势且状态为Idle。

1. 在“模型部署”中选择模型1、任务2 `ACT Popcorn 真机100帧`。
2. 再次核对任务ID是2，不是1。
3. 准备好物理急停后，由操作者本人点击“部署”。
4. “部署”会立即执行安全门禁；通过后开始推理和真机动作，不通过则返回失败且不发布动作。
5. 一个100帧块执行完后自动停止；需要提前停止时点击“停止推理”。

真机开始门禁包括：20个电机为OP、错误码为零、状态Idle、当前姿势与ACT默认姿势最大误差不超过0.05 rad、头图/双腕黑图/身体和双手反馈齐全、模型输出严格为有限的 `100×19`。

## 6. 停止与检查

软件停止：

```bash
ros2 service call /inference/stop_inference std_srvs/srv/Trigger '{}'
```

紧急情况优先使用物理急停，软件停止不能替代物理急停。

检查是否仍有控制流：

```bash
timeout 3 ros2 topic echo --once /control/joint_position
timeout 3 ros2 topic echo --once /control/ee/left
timeout 3 ros2 topic echo --once /control/ee/right
```

空闲时三个命令都应超时且无消息。话题可能仍显示基础节点或XWiz的发布端点；存在端点不等于正在发送控制命令。

## 7. 配置位置与日志

本机XWiz任务：

```text
~/.dexforce/XWiz/model_deployments/tasks/1/task_config.json  # 仿真
~/.dexforce/XWiz/model_deployments/tasks/2/task_config.json  # 真机100帧
```

PC1任务副本：

```text
/home/dexforce/workspace/.dexforce/XWiz/model_deployments/tasks/1/task_config.json
/home/dexforce/workspace/.dexforce/XWiz/model_deployments/tasks/2/task_config.json
```

日志：

```text
本机：journalctl --user -u xwiz-act-server.service
PC1：/home/dexforce/xwiz_real_manager.log
PC2：/home/dexforce/xwiz_real_client.log
PC2：/home/dexforce/xwiz_black_wrist.log
```

如果XWiz未显示新任务，退出“模型部署”页面后重新进入；仍未刷新时重启XWiz图形界面，但不要重复启动后台三端服务。
