# W1：PC2 部署 ACT 并直连 PC1 使用说明

> 适用范围：DexForce W1/W1 Pro，采用“PC2 运行模型、PC1 连接机器人 ROS 2 并执行动作”的部署方式。
>
> 本方案不依赖 XWiz GUI。文中的 `PC1`、`PC2` 只是角色名；迁移到其他 W1 时，必须重新确认 IP、ROS 域、话题、关节顺序、相机、末端执行器和 checkpoint 合同。
>
> 本文按 2026-08-24 实际代码和部署状态整理。代码与本文冲突时，以部署代码和 checkpoint 元数据为准。

## 1. 结论与运行结构

当前 Popcorn W1 的直接推理链路为：

```text
                         启动/状态/停止控制
操作者所在主机（可断开） ───────────────→ PC1:8890
                                          │
W1 状态、头部相机、腕部相机、双手反馈 ──→ │ PC1 观测与安全执行器
                                          │
                                          ├── TCP 观测请求 ──→ PC2:8889
                                          │                    ACT checkpoint
                                          │                    CUDA 推理
                                          │←── 100×19 动作块 ──┘
                                          │
                                          └── ROS 2 控制话题 ──→ W1 真机
```

当前实例参数：

| 角色 | 地址 | 作用 |
| --- | --- | --- |
| PC1 | `192.168.20.20` | 订阅机器人观测、执行安全门禁、向 PC2 请求推理、发布真机动作 |
| PC2 | `192.168.20.21` | 加载 ACT checkpoint，在 GPU 上推理，返回动作块 |
| 操作者主机 | 当前为 `192.168.20.164` | 只发送 `start/status/stop`；推理开始后可以断开，不参与闭环 |

关键语义：

- PC2 不订阅机器人 ROS 2 话题，也不直接控制电机。
- PC1 不加载 ACT 权重，只负责观测、动作适配、安全检查和发布。
- 操作者主机不是持续推理链路的一部分。`start` 成功后，即使操作者主机退出或更换，PC1 与 PC2 仍可继续运行。
- 停止 PC1 客户端、PC2 模型服务、网络断开或安全门禁失败都会中断闭环。
- 软件服务启动不等于机器人运动。只有向 PC1 `8890` 发送真机 `SETUP_CONFIG/start` 后才会开始推理和动作发布。

## 2. 与 XWiz 的关系

本方案不使用：

- XWiz GUI；
- XWiz 模型部署面板；
- `xwiz-inference-manager.service`；
- PC2 原有的 `xwiz-real-client.service`；
- PC2 原有的 `xwiz-act-server.service`。

当前直接运行服务为：

```text
PC2: w1-act-server-direct.service
PC1: w1-act-client-direct.service
```

代码目录仍保留 `xwiz_act_server`、`xwiz_real_runtime` 名称，是因为复用了已经验证的网络协议和 W1 动作适配代码；运行过程本身不依赖 XWiz GUI。

## 3. 三层通信接口

### 3.1 操作者到 PC1：控制通道 `8890/TCP`

控制端连接 PC1 的 `manager_port=8890`，发送长度前缀 JSON。用途只有：

- `SETUP_CONFIG`：提交 PC1 客户端配置和 PC2 服务配置，并启动推理；
- `STATUS`：读取 PC1 与 PC2 当前状态；
- `STOP`：停止推理、清空后续动作并通知 PC2 回到 `idle`。

这不是动作流。操作者主机不会逐帧发送关节目标。

当前控制程序：

```text
/home/wengyikun/workplace/popcorn/w1_act-ljl-act_train/direct_runtime/control_direct.py
```

### 3.2 PC1 到 PC2：模型通道 `8889/TCP`

PC1 主动连接 PC2 `8889`。协议为：

```text
4 字节大端长度 + pickle payload
```

支持的请求类型：

- `SETUP_CONFIG`
- `STATUS`
- `observation`
- `get_actions`
- `STOP`
- `SHUTDOWN`

主要数据流：

```text
PC1 observation(start_infer=true)
  → PC2 解码状态和图像
  → checkpoint 生成 100×19
  → PC1 轮询 get_actions
  → PC2 返回并删除已消费动作块
```

该协议使用 pickle，只应部署在可信的机器人私有网络，不能直接暴露到互联网。

### 3.3 PC1 到 W1：ROS 2/DDS

当前环境：

```bash
export ROS_DOMAIN_ID=20
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
source /opt/ros/humble/setup.bash
source /home/dexforce/w1/install/setup.bash
```

PC1 通过 ROS 2 订阅状态和图像，并发布机身和双手命令。PC2 不需要加入这个 ROS 2 域。

PC1 直接推理客户端的 ROS 接口汇总：

| 方向 | 话题 | 消息类型 | 作用 |
| --- | --- | --- | --- |
| 订阅 | `/feedback/robot_server_state` | `std_msgs/msg/String` | JSON 字符串，包含状态、20D 关节、电机状态和错误码 |
| 订阅 | `/camera/left_eye_resize` | `sensor_msgs/msg/Image` | 左头目图像 |
| 订阅 | `/camera/right_eye_resize` | `sensor_msgs/msg/Image` | 右头目图像和就绪检查 |
| 订阅 | `/camera_l/color/image_rect_raw` | `sensor_msgs/msg/Image` | 左腕真实彩色图 |
| 订阅 | `/camera_r/color/image_rect_raw` | `sensor_msgs/msg/Image` | 右腕真实彩色图 |
| 订阅 | `/feedback/ee/left` | `end_effector_interfaces/msg/EEFeedback` | 左 Linker L6 六关节反馈 |
| 订阅 | `/feedback/ee/right` | `end_effector_interfaces/msg/EEFeedback` | 右 Linker L6 六关节反馈 |
| 发布 | `/control/joint_position` | `joint_interfaces/msg/JointPositionControl` | 17 个机身关节绝对位置命令 |
| 发布 | `/control/ee/left` | `end_effector_interfaces/msg/EEJointControl` | 左 Linker L6 POSITION 命令 |
| 发布 | `/control/ee/right` | `end_effector_interfaces/msg/EEJointControl` | 右 Linker L6 POSITION 命令 |

## 4. 当前 checkpoint 合同

当前模型：

```text
/home/dexforce/workspace/outputs/act_popcorn_45w_xwiz
```

其 `config.json` 定义：

```text
n_obs_steps=1
chunk_size=100
n_action_steps=100
observation.state:                 19
observation.images.cam_high_left:  3×360×640
observation.images.cam_hand_left:  3×360×640
observation.images.cam_hand_right: 3×360×640
action:                            19
```

checkpoint 必须包含：

```text
config.json
model.safetensors
policy_preprocessor.json
policy_preprocessor_step_3_normalizer_processor.safetensors
policy_postprocessor.json
policy_postprocessor_step_0_unnormalizer_processor.safetensors
```

`train_config.json` 和 `commit_id.json` 建议保留，用于追踪训练配置与版本。

不得绕过 checkpoint 自带的 preprocessor/postprocessor，也不得混用其他数据集生成的归一化统计。

PC1 到 PC2 的观测 payload 使用以下分组名：

```text
states.waistqpos          1
states.left_armqpos       7
states.headqpos           2
states.right_armqpos      7
states.left_eefgripper    1
states.right_eefgripper   1

cam_high                  左头目 BGR bytes
cam_left_wrist            左腕 BGR bytes
cam_right_wrist           右腕 BGR bytes
```

PC2 将它们转换为 LeRobot 特征 key。动作返回时再按同样的六个状态分组拆分 `100×19`，PC1 最终恢复为逐帧 19D 动作。

## 5. 19D 状态输入

PC1 从 `/feedback/robot_server_state` 读取 20 个机身关节反馈，但当前模型只使用其中 17 个，再追加左右手开合度，组成 19D：

| 索引 | 字段 |
| ---: | --- |
| 0 | `WAIST` |
| 1..7 | `LEFT_J1..LEFT_J7` |
| 8..9 | `NECK1, NECK2` |
| 10..16 | `RIGHT_J1..RIGHT_J7` |
| 17 | 左手开合度 |
| 18 | 右手开合度 |

不输入模型的机身关节：

```text
ANKLE, KNEE, BUTTOCK
```

状态和动作都是绝对关节位置，不是相对上一帧的增量。

### 5.1 Linker L6 反馈映射

PC1 订阅：

```text
/feedback/ee/left
/feedback/ee/right
```

每只手按名称重排为：

```text
T_CMC_YAW
T_MCP
IF_MCP_PITCH
MF_MCP_PITCH
RF_MCP_PITCH
LF_MCP_PITCH
```

不同 W1 固件可能返回 `0..1` 或 `0..100`。当前适配器同时兼容：

- 全部值在 `0..1`：先乘 100；
- 存在大于 1 的值：按 `0..100` 百分比处理；
- 小于 0、大于 100、缺少关节或非有限值：拒绝观测。

六指反馈再投影到 checkpoint 使用的单一开合度标量 `0..100`。

## 6. 图像输入

### 6.1 PC1 订阅的话题

| 用途 | ROS 2 话题 | 当前原始格式 | PC1 内部处理 |
| --- | --- | --- | --- |
| 左头目 | `/camera/left_eye_resize` | `960×540 bgr8` | 转 `bgr8`，缩放为 `640×360` |
| 右头目 | `/camera/right_eye_resize` | `960×540 bgr8` | 转 `bgr8`，缩放为 `640×360` |
| 左腕 | `/camera_l/color/image_rect_raw` | `640×360 rgb8` | `cv_bridge` 转 `bgr8` |
| 右腕 | `/camera_r/color/image_rect_raw` | `640×360 rgb8` | `cv_bridge` 转 `bgr8` |

PC1 的观测门禁要求左右头目、左右腕、机器人状态和左右手反馈都已经进入缓冲区。

### 6.2 真正进入当前模型的图像

当前 checkpoint 只定义三路视觉特征：

```text
左头目 → observation.images.cam_high_left
左腕   → observation.images.cam_hand_left
右腕   → observation.images.cam_hand_right
```

右头目当前仍被 PC1 订阅和用于观测就绪检查，但不会传入 `act_popcorn_45w` 模型。若新模型需要右头目，必须同时修改 checkpoint 特征合同和 PC2 `IMAGE_MAPPING`，仅修改 ROS 话题配置不够。

### 6.3 颜色和尺寸转换

数据经过：

```text
ROS Image
  → PC1 cv_bridge desired_encoding=bgr8
  → resize 640×360
  → BGR bytes 发送至 PC2
  → PC2 BGR 转 RGB
  → checkpoint processor
  → 3×360×640 视觉特征
```

因此腕部相机发布 `rgb8` 是允许的，PC1 会显式转换为 `bgr8`。

### 6.4 不再使用黑腕图

当前左右腕部已有真实相机，后续不再需要：

```text
w1-black-wrist-direct.service
xwiz_real_runtime.black_wrist_images
```

确保它处于停止状态：

```bash
ssh dexforce@192.168.20.20
systemctl --user stop w1-black-wrist-direct.service
systemctl --user disable w1-black-wrist-direct.service
```

左右腕部话题必须各只有一个真实相机发布者。两个发布者会导致客户端随机混入黑图或真实图。

### 6.5 图像检查命令

在 PC1：

```bash
export ROS_DOMAIN_ID=20
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
source /opt/ros/humble/setup.bash
source /home/dexforce/w1/install/setup.bash

for topic in \
  /camera/left_eye_resize \
  /camera/right_eye_resize \
  /camera_l/color/image_rect_raw \
  /camera_r/color/image_rect_raw
do
  echo "=== ${topic} ==="
  ros2 topic info -v "${topic}"
  timeout 5 ros2 topic hz "${topic}" --window 10
done
```

验收要求：

- 四路都有数据；
- 每路只有预期的真实相机发布者；
- PC1 `optimized_robot_client` 是订阅者；
- 头部输入可稳定接近 30 Hz；
- 腕部输入尺寸为 `640×360`；
- 像素不是全黑、全常数或损坏数据。

## 7. 动作输出与真机话题

PC2 每次返回严格的：

```text
100×19 float action chunk
```

每帧顺序与状态 19D 相同。PC1 在发布前检查：

- shape 必须是 `(100, 19)`；
- 所有值必须有限，禁止 NaN/Inf；
- 机身 17D 按 W1 关节范围裁剪；
- 左右开合度裁剪为 `0..100` 后映射为 Linker L6 六关节命令。

发布话题：

| 动作 | ROS 2 话题 | 消息语义 |
| --- | --- | --- |
| 17D 机身 | `/control/joint_position` | 绝对关节位置，单位 rad |
| 左 Linker L6 | `/control/ee/left` | 六关节 POSITION 命令 |
| 右 Linker L6 | `/control/ee/right` | 六关节 POSITION 命令 |

当前 checkpoint 中：

```text
开合度 0   = 任务定义的闭合端
开合度 100 = 张开端
```

左右手的闭合端并不相同，具体六关节端点定义在：

```text
w1_act-ljl-act_train/xwiz_real_runtime/runtime.py
LEFT_CLOSED / LEFT_OPEN / RIGHT_CLOSED / RIGHT_OPEN
```

更换末端执行器、修改开合方向或换成 6D 灵巧手输出时，必须修改适配器并重新验证，不能只替换 checkpoint。

## 8. 连续推理语义

配置：

```json
"execution_mode": "continuous",
"action_horizon": 100,
"control_frequency": 10,
"collect_frequency": 10,
"chunk_size_threshold": 0.0,
"sample_factor": 1.0
```

连续模式不是重复同一个动作块：

```text
生成 chunk N
  → PC1 以 10 Hz 完整执行第 1..100 帧
  → 确认动作队列为空
  → 等待 chunk 完成后的新机器人反馈
  → 使用最新状态和最新图像生成 chunk N+1
  → 重复，直到 STOP 或安全门禁触发
```

每个 chunk 在 10 Hz 下约 10 秒。模型的 `100` 是单次输出长度，不是整个任务只能运行 100 帧。

连续模式内部使用极大有限 `max_steps` 兼容原厂线程；实际停止边界由人工 `STOP` 或安全门禁决定。

## 9. PC2 模型服务

### 9.1 当前路径

```text
代码根目录：/home/dexforce/w1/w1_act
模型目录：  /home/dexforce/workspace/outputs/act_popcorn_45w_xwiz
启动脚本：  /home/dexforce/w1/w1_act/start_pc2_direct_server.sh
systemd：   /home/dexforce/.config/systemd/user/w1-act-server-direct.service
监听地址：  0.0.0.0:8889
设备：      cuda
```

### 9.2 启动器组成

当前 `start_pc2_direct_server.sh` 主要设置：

```bash
W1_ACT_ROOT=/home/dexforce/w1/w1_act
RUNTIME_DEPS=/home/dexforce/.local/share/xwiz-act-server/runtime-deps
POLICY_PATH=/home/dexforce/workspace/outputs/act_popcorn_45w_xwiz

export PYTHONPATH="${RUNTIME_DEPS}:${W1_ACT_ROOT}/w1_lerobot/src:${W1_ACT_ROOT}:${PYTHONPATH:-}"

python3 -m xwiz_act_server.server \
  --host 0.0.0.0 \
  --port 8889 \
  --policy-path "${POLICY_PATH}" \
  --device cuda
```

systemd 只负责启动该脚本并在异常退出后重启：

```ini
[Service]
ExecStart=/home/dexforce/w1/w1_act/start_pc2_direct_server.sh
Restart=on-failure
RestartSec=2
Environment=PYTHONUNBUFFERED=1
```

### 9.3 服务命令

```bash
ssh dexforce@192.168.20.21

systemctl --user start w1-act-server-direct.service
systemctl --user stop w1-act-server-direct.service
systemctl --user restart w1-act-server-direct.service
systemctl --user is-active w1-act-server-direct.service
systemctl --user status w1-act-server-direct.service --no-pager -l
ss -ltnp | grep ':8889'
```

如果希望 PC2 登录后自动加载模型，但仍不自动启动真机推理：

```bash
systemctl --user enable w1-act-server-direct.service
```

这只启动模型服务，不会让机器人运动。

## 10. PC1 观测与执行服务

### 10.1 当前路径

```text
代码根目录：/home/dexforce/w1/w1_act
配置：      /home/dexforce/w1/w1_act/xwiz_real_runtime/client_runtime_direct_pc2.json
启动脚本：  /home/dexforce/w1/w1_act/start_pc1_direct.sh
systemd：   /home/dexforce/.config/systemd/user/w1-act-client-direct.service
监听地址：  0.0.0.0:8890
模型服务：  192.168.20.21:8889
```

启动脚本负责加载 ROS 2 Humble、W1 overlay、Domain 20 和 CycloneDDS，再运行：

```bash
python3 -m xwiz_real_runtime.client_service \
  --config /home/dexforce/w1/w1_act/xwiz_real_runtime/client_runtime_direct_pc2.json
```

### 10.2 服务命令

```bash
ssh dexforce@192.168.20.20

systemctl --user start w1-act-client-direct.service
systemctl --user stop w1-act-client-direct.service
systemctl --user restart w1-act-client-direct.service
systemctl --user is-active w1-act-client-direct.service
systemctl --user status w1-act-client-direct.service --no-pager -l
ss -ltnp | grep ':8890'
```

PC1 客户端启动后默认为 `idle`，只缓存观测，不请求模型、不保存训练数据、不发布动作。

### 10.3 客户端配置字段

当前配置文件：

```text
/home/dexforce/w1/w1_act/xwiz_real_runtime/client_runtime_direct_pc2.json
```

主要字段：

| 字段 | 当前值 | 含义 |
| --- | --- | --- |
| `server_host` | `192.168.20.21` | PC2 模型服务地址 |
| `server_port` | `8889` | PC2 模型服务端口 |
| `manager_port` | `8890` | PC1 控制监听端口 |
| `mode` | 配置文件为 `1`，`start` 时强制为 `2` | `1` 仿真，`2` 真机 |
| `execution_mode` | `continuous` | 完成每个 chunk 后请求下一块 |
| `action_horizon` | `100` | 单个动作块长度 |
| `control_frequency` | `10` | 动作发布频率 Hz |
| `collect_frequency` | `10` | 观测收集频率 Hz |
| `head_target_size` | `[640,360]` | 发送给 PC2 的头图尺寸 |
| `hand_target_size` | `[640,360]` | 发送给 PC2 的腕图尺寸 |
| `use_hand_camera` | `true` | 要求左右腕图像 |
| `end_effector_type` | `gripper` | 使用左右开合度标量合同 |
| `end_effector_position_limit` | `[0,100]` | 开合度数值范围 |
| `joint_topic` | `/feedback/robot_server_state` | 机器人状态反馈 |
| `joint_control_topic` | `/control/joint_position` | 机身控制输出 |
| `cam_*_topic` | 见图像章节 | 四路相机订阅 |
| `*_hand_qpos6_topic` | `/feedback/ee/left/right` | 双手六关节反馈 |
| `set_*_hand_qpos6_topic` | `/control/ee/left/right` | 双手六关节控制 |
| `save_actionchunks` | `false` | 运行时不保存动作块 |
| `service` | `true` | 启用 PC1 `8890` 管理服务 |

`prepare_client_config()` 会在启动时强制：

```text
action_horizon=100
sample_factor=1.0
chunk_size_threshold=0.0
home_position=""
continuous 真机模式使用极大有限 max_steps
```

因此直接修改配置中的 `max_steps=100` 不会把连续推理限制为总共 100 帧。

## 11. 全部重启后的启动顺序

### 11.1 启动机器人基础 ROS 和相机

先确认 PC1 机器人基础节点正常，左右头目和左右腕部真实相机有数据。不要启动 Tele、原厂 ACT、回放、拖拽或其他自定义控制器。

### 11.2 启动 PC2 模型服务

```bash
ssh dexforce@192.168.20.21 \
  'systemctl --user start w1-act-server-direct.service'

ssh dexforce@192.168.20.21 \
  'systemctl --user is-active w1-act-server-direct.service; ss -ltnp | grep :8889'
```

### 11.3 启动 PC1 客户端

```bash
ssh dexforce@192.168.20.20 \
  'systemctl --user start w1-act-client-direct.service'

ssh dexforce@192.168.20.20 \
  'systemctl --user is-active w1-act-client-direct.service; ss -ltnp | grep :8890'
```

不要启动 `w1-black-wrist-direct.service`。

### 11.4 查询 idle 状态

```bash
cd /home/wengyikun/workplace/popcorn
PYTHONPATH=w1_act-ljl-act_train python3 \
  w1_act-ljl-act_train/direct_runtime/control_direct.py status
```

期望：

```json
{
  "state": "idle",
  "server": "192.168.20.21:8889"
}
```

`server_state` 在 PC1 尚未主动连接 PC2 时可能显示 `unknown`，这不等于端口不可达；仍应单独检查 PC2 `8889`。

## 12. 真机启动前检查

真机将运动。必须同时满足：

- 现场无人和障碍；机器人支撑稳定；物理急停可立即触达；
- 只有一个控制源；Auto、Tele、原厂 ACT、Map、drag、replay、自定义控制均未运行；
- PC2 checkpoint 已完成离线 shape/有限值检查；
- PC1 状态为 `idle`；
- 机器人为 `Idle`；
- 20/20 电机均为 `OP`；
- 电机错误码全为 `0`；
- server error 全为 `None`；
- 机器人在 ACT 默认姿态，20D 最大误差不超过 `0.05 rad`；
- 左右头目、左右腕真实图、机器人状态和左右手反馈均已就绪；
- 三个真机控制话题没有其他持续发布者。

检查竞争进程示例：

```bash
ssh dexforce@192.168.20.20 \
  'ps -ef | grep -E "tele|act_ros2|policy|replay|drag" | grep -v grep || true'
```

PC1 会在收到 `start` 时再次执行姿态、健康和观测门禁，不满足时拒绝启动。

## 13. 恢复 ACT 默认姿态

目标文件：

```text
/home/dexforce/w1/dexe_mobile_application/script/act_default_pose_runtime.json
```

先停止推理和其他控制源，再在 PC1 执行：

```bash
export ROS_DOMAIN_ID=20
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
source /opt/ros/humble/setup.bash
source /home/dexforce/w1/install/setup.bash
cd /home/dexforce/w1/dexe_mobile_application/script
python3 slowly_move_to.py act_default_pose_runtime.json 1000
```

约 20 秒完成。完成后仍需检查机器人 `Idle`、20/20 `OP`、错误码为零和姿态误差。

## 14. 启动持续真机推理

当前配置已经设置：

```json
"execution_mode": "continuous"
```

从操作者主机执行：

```bash
cd /home/wengyikun/workplace/popcorn
PYTHONPATH=w1_act-ljl-act_train python3 \
  w1_act-ljl-act_train/direct_runtime/control_direct.py start
```

`control_direct.py` 会把运行模式强制设为真机 `mode=2`，再向 PC1 `192.168.20.20:8890` 发送配置。PC1 门禁通过后连接 PC2 `192.168.20.21:8889` 并开始闭环。

启动命令返回成功后，操作者主机可以退出；PC1 和 PC2 会继续运行。

查询：

```bash
PYTHONPATH=w1_act-ljl-act_train python3 \
  w1_act-ljl-act_train/direct_runtime/control_direct.py status
```

正常运行：

```json
{
  "state": "running",
  "server_state": "running"
}
```

## 15. 停止

软件停止无需等待确认：

```bash
cd /home/wengyikun/workplace/popcorn
PYTHONPATH=w1_act-ljl-act_train python3 \
  w1_act-ljl-act_train/direct_runtime/control_direct.py stop
```

停止顺序：

```text
PC1 通知 PC2 STOP
  → PC2 删除未消费动作并回到 idle
  → PC1 停止推理/执行线程并回到 idle
  → 不再发布新动作
```

软件停止后机器人保持最后一帧姿态，不自动复位。必须询问操作者返回：

1. ACT/VLA 默认姿态；
2. 站立/Standby 默认姿态；
3. 蹲姿/Rest 默认姿态。

紧急情况优先使用物理急停，软件 `stop` 不能替代急停。

## 16. 固定姿态文件

当前部署：

| 姿态 | 文件 |
| --- | --- |
| ACT 默认 | `/home/dexforce/w1/dexe_mobile_application/script/act_default_pose_runtime.json` |
| Standby | `/home/dexforce/w1/dexe_mobile_application/script/records.d/default/standby.json` |
| Rest | `/home/dexforce/w1/dexe_mobile_application/script/records.d/default/rest.json` |

姿态恢复属于新的真机运动，必须重新检查现场、控制源和机器人健康。

若插值期间进入 `Init/MOTOR_FAULT`，程序会停止。不要绕过状态继续发布。先检查错误；当前原厂清错接口为：

```bash
ros2 service call /control/set_property \
  robot_server_interfaces/srv/SetProperty \
  "{property_type: ClearError, value: ''}"
```

只有恢复 `Idle/ENABLED`、20/20 `OP`、错误码清零后才能重新插值。

## 17. 切换同合同 ACT 模型

这里的“同合同”必须同时满足：

- `observation.state` 为 19D，顺序完全一致；
- 输入视觉 key 与当前三路一致；
- 图像均为 `3×360×640`；
- 输出 `action` 为 19D；
- `chunk_size=100`、`n_action_steps=100`；
- 状态和动作都是相同单位的绝对值；
- 左右末端仍是 `0..100` 开合度且方向一致；
- checkpoint 自带正确的 processor 和归一化统计。

### 17.1 放置新模型

不要覆盖旧模型，使用独立目录：

```bash
ssh dexforce@192.168.20.21
mkdir -p /home/dexforce/workspace/outputs/<new_model_name>
```

将 checkpoint 文件复制或解压到该目录。

### 17.2 验证文件与合同

```bash
test -f /home/dexforce/workspace/outputs/<new_model_name>/config.json
test -f /home/dexforce/workspace/outputs/<new_model_name>/model.safetensors
python3 -m json.tool \
  /home/dexforce/workspace/outputs/<new_model_name>/config.json >/dev/null
```

检查 `input_features`、`output_features`、`chunk_size` 和 `n_action_steps`，不能只看模型目录能否加载。

### 17.3 修改 PC2 启动器

当前启动器的模型路径是固定值。停止推理和 PC2 服务后，编辑：

```text
/home/dexforce/w1/w1_act/start_pc2_direct_server.sh
```

修改：

```bash
POLICY_PATH=/home/dexforce/workspace/outputs/<new_model_name>
```

然后：

```bash
systemctl --user restart w1-act-server-direct.service
systemctl --user status w1-act-server-direct.service --no-pager -l
ss -ltnp | grep ':8889'
```

先做离线/dry-run 合同验证，再恢复 ACT 默认姿态并启动真机。不要在旧模型仍运行时直接覆盖权重文件。

### 17.4 更通用的启动器写法

新部署建议把启动脚本改为可由环境变量覆盖：

```bash
POLICY_PATH="${W1_ACT_POLICY_PATH:-/home/dexforce/workspace/outputs/default_model}"
ACT_PORT="${W1_ACT_PORT:-8889}"
ACT_DEVICE="${W1_ACT_DEVICE:-cuda}"
```

systemd 可通过 drop-in 选择模型：

```bash
systemctl --user edit w1-act-server-direct.service
```

示例：

```ini
[Service]
Environment=W1_ACT_POLICY_PATH=/home/dexforce/workspace/outputs/<new_model_name>
Environment=W1_ACT_PORT=8889
Environment=W1_ACT_DEVICE=cuda
```

注意：这是推荐的通用模板；当前 PC2 脚本仍使用固定 `POLICY_PATH`，在未实际改造前不要假设环境变量已经生效。

## 18. 切换不同合同模型

以下任一变化都不是“只换模型目录”：

- 状态维数或顺序变化；
- 新增右头目、删除腕部或更改相机 key；
- 图像尺寸变化；
- action 维数、单位、绝对/相对语义变化；
- chunk 长度变化；
- 末端由开合度变为六关节、二指夹或其他类型；
- 新增底盘、腿部或其他关节输出。

必须同步修改并测试：

```text
PC2 xwiz_act_server/contract.py
PC2 xwiz_act_server/model_runtime.py
PC1 xwiz_real_runtime/runtime.py
PC1 xwiz_real_runtime/client_service.py
PC1 client_runtime_direct_pc2.json
控制频率、动作话题和安全门禁
```

还必须重新采集/验证数据合同和 normalization statistics。不得通过裁剪、补零或重排“猜测”兼容真机模型。

## 19. 修改 IP、端口或迁移 PC

### 19.1 PC2 地址或模型端口变化

修改 PC1：

```json
"server_host": "<PC2_IP>",
"server_port": 8889
```

文件：

```text
/home/dexforce/w1/w1_act/xwiz_real_runtime/client_runtime_direct_pc2.json
```

随后重启 PC1 客户端。

### 19.2 PC1 地址或控制端口变化

修改控制命令：

```bash
python3 control_direct.py status --host <PC1_IP> --port <MANAGER_PORT>
```

PC1 的 `manager_port` 必须与命令一致。

### 19.3 更换操作者主机

不需要迁移模型。新主机只要：

- 能访问 PC1 `8890`；
- 有 `control_direct.py` 及其 Python 依赖；
- 有对应的客户端配置 JSON；
- 能执行 `status/start/stop`。

推理和动作闭环仍全部位于 PC1/PC2。

## 20. 自动停止条件

当前客户端会在以下条件停止后续动作并进入 `error` 或 `idle`：

- 机器人状态反馈超过 1 秒未更新；
- 状态不是 `Idle` 或 `Running`；
- 任一电机不是 `OP`；
- 任一电机错误码非 0；
- 任一 server error 不是 `None`；
- 必需观测缓冲区缺失；
- Linker L6 反馈缺失、非有限或超出支持范围；
- 动作不是严格的 `100×19`；
- 动作包含 NaN/Inf；
- 前一 chunk 未执行完却收到下一 chunk；
- PC1/PC2 网络异常；
- 人工 `STOP`。

故障后不要绕过门禁。先停止、确认控制话题静默、检查机器人健康，再恢复目标姿态并重新部署。

## 21. 常用诊断命令

### 21.1 网络

```bash
nc -vz 192.168.20.21 8889
nc -vz 192.168.20.20 8890
```

### 21.2 PC2 服务

```bash
ssh dexforce@192.168.20.21 \
  'systemctl --user status w1-act-server-direct.service --no-pager -l; ss -ltnp | grep :8889'
```

### 21.3 PC1 服务

```bash
ssh dexforce@192.168.20.20 \
  'systemctl --user status w1-act-client-direct.service --no-pager -l; ss -ltnp | grep :8890'
```

### 21.4 推理状态

```bash
cd /home/wengyikun/workplace/popcorn
PYTHONPATH=w1_act-ljl-act_train python3 \
  w1_act-ljl-act_train/direct_runtime/control_direct.py status
```

### 21.5 ROS 2 发布者和订阅者

```bash
ros2 topic info -v /camera/left_eye_resize
ros2 topic info -v /camera/right_eye_resize
ros2 topic info -v /camera_l/color/image_rect_raw
ros2 topic info -v /camera_r/color/image_rect_raw
ros2 topic info -v /control/joint_position
ros2 topic info -v /control/ee/left
ros2 topic info -v /control/ee/right
```

不能只看 publisher endpoint 是否存在。真机验收必须同时查看消息频率、实际消息内容、目标与反馈误差以及机器人错误码。

## 22. 当前部署文件索引

本机维护副本：

```text
/home/wengyikun/workplace/popcorn/w1_act-ljl-act_train/direct_runtime/
/home/wengyikun/workplace/popcorn/w1_act-ljl-act_train/xwiz_act_server/
/home/wengyikun/workplace/popcorn/w1_act-ljl-act_train/xwiz_real_runtime/
```

PC2：

```text
/home/dexforce/w1/w1_act/xwiz_act_server/
/home/dexforce/w1/w1_act/start_pc2_direct_server.sh
/home/dexforce/.config/systemd/user/w1-act-server-direct.service
/home/dexforce/workspace/outputs/
```

PC1：

```text
/home/dexforce/w1/w1_act/xwiz_real_runtime/
/home/dexforce/w1/w1_act/start_pc1_direct.sh
/home/dexforce/.config/systemd/user/w1-act-client-direct.service
/home/dexforce/w1/w1_act/xwiz_real_runtime/client_runtime_direct_pc2.json
```

机器人姿态：

```text
/home/dexforce/w1/dexe_mobile_application/script/act_default_pose_runtime.json
/home/dexforce/w1/dexe_mobile_application/script/records.d/default/standby.json
/home/dexforce/w1/dexe_mobile_application/script/records.d/default/rest.json
```

## 23. 最小日常操作清单

```text
1. 确认 PC2 模型服务 active、8889 监听。
2. 确认 PC1 客户端 active、8890 监听。
3. 确认左右头目和左右腕真实图像正常，腕部各只有一个发布者。
4. 确认没有 Auto/Tele/ACT/drag/replay 等竞争控制源。
5. 确认 W1 Idle、20/20 OP、零错误。
6. 缓慢恢复 ACT 默认姿态并验收误差。
7. 执行 control_direct.py start。
8. 用 status 确认 PC1/PC2 均 running。
9. 需要停止时立即执行 control_direct.py stop。
10. 停止后选择 ACT、Standby 或 Rest，不自动复位。
```
