# W1 Popcorn 三模型 PC1/PC2 推理部署总说明

本文对应 `/home/wengyikun/workplace/joint_songling/start` 中的三套独立启动器：

1. `220000`：ACT-DINOv3，包含 EE pose/FK 辅助损失，relative arm joints，chunk 16；
2. `180000`：ACT-DINOv3，relative arm joints，chunk 16；
3. `500000`：旧 ACT absolute 版本。当前 PC2 `/home/dexforce/workspace/outputs/500000_pc2/config.json` 已核对为 chunk 16；历史源配置曾写成 100，不能仅凭目录名或旧文档猜测。

本文件只记录部署和验收，不自动启动真机。启动动作会向 W1 发布控制话题，必须由操作者在完成门禁后明确执行 `start`。

## 0. 迁移结论：哪些可直接复用，哪些必须匹配

当前 Popcorn W1 的 IP、安装路径、ROS 域/RMW、话题命名、20D/19D 关节顺序、身体限位、ACT 默认身体姿态和端口已作为通用基线写入 `start/profiles/popcorn_w1.json`。这些值在同一软件栈的 W1 上可直接复用，启动器也保留为默认值；如新 W1 的安装根目录或网络不同，可通过环境变量覆盖。

不能只复制目录就运行的项目是硬件和 artifact：腕部相机序列号/分辨率、末端执行器反馈名称、驱动版本、默认姿态实测值、模型目录内容、DINOv3 依赖和 PC2 GPU。新设备应复制 `start/profiles/new_w1_template.json`，完成序列号和差异填写，再执行只读预检：

```bash
python3 start/preflight_w1_profile.py --profile start/profiles/new_w1_template.json
```

模板中的占位序列号会使预检失败，这是有意的安全门禁。profile 通过后，再针对模型检查 checkpoint：

```bash
python3 start/preflight_w1_profile.py \
  --profile start/profiles/popcorn_w1.json \
  --model 220000 --checkpoint /home/wengyikun/workplace/popcorn/outputs/220000_pc2
```

预检只读文件和合同，不 source ROS、不连接 PC1/PC2、不发布控制消息；它不能替代新 W1 的真实相机画面、反馈频率、电机状态和唯一控制源检查。

## 1. 机器、网络和职责

| 主机 | 地址 | 职责 |
|---|---|---|
| PC1 | `192.168.20.20` | ROS 2/EtherCAT、读取真实反馈和图像、相对/绝对适配、限位、发布动作 |
| PC2 | `192.168.20.21` | 加载 checkpoint、CUDA 推理、返回动作 chunk；不直接控制电机 |
| 操作者主机 | `192.168.20.164`（历史记录） | 通过 PC1 manager 端口发送 status/start/stop |

PC1 和 PC2 通过 TCP 通信；PC1 才是唯一允许向机器人发布动作的一端。协议使用私有网络上的长度前缀 JSON/pickle，禁止暴露公网。

PC1 ROS 环境：

```bash
source /opt/ros/humble/setup.bash
source /home/dexforce/w1/install/setup.bash
export ROS_DOMAIN_ID=20
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

## 2. 模型与端口映射

| 模型 | PC2 checkpoint | PC2 端口 | PC1 manager | PC1 配置 |
|---|---|---:|---:|---|
| 220000 EE/FK | `/home/dexforce/workspace/outputs/220000_pc2` | 8889 | 8890 | `client_runtime_220000.json` |
| 180000 DINOv3 | `/home/dexforce/workspace/outputs/act_dinov3_popcorn_0827_19d_relative_arm_joints_nextstate_chunk16_b64_500k_gpu1_2_180000_pc2` | 8892 | 8891 | `client_runtime_180000.json` |
| 500000 ACT | `/home/dexforce/workspace/outputs/500000_pc2` | 8889 | 8890 | `client_runtime_pc2.json` |

本地已审计的 DINOv3 权重分别为 `/home/wengyikun/workplace/popcorn/outputs/220000_pc2` 和 `/home/wengyikun/workplace/popcorn/outputs/act_dinov3_popcorn_0827_19d_relative_arm_joints_nextstate_chunk16_b64_500k_gpu1_2_180000_pc2`。500000 的历史源位于 `/home/wengyikun/workplace/popcorn/act_popcorn_45w/pretrained_model`；本机当前不以 `outputs/500000` 作为已存在权重目录，使用前必须确认 `/home/dexforce/workspace/outputs/500000_pc2` 已完整部署并核对 `config.json`、权重和预处理文件。

220000 和 180000 是同一类 DINOv3 relative 协议的不同迭代 checkpoint，不能因为迭代号不同而混用端口、统计文件或配置。500000 使用旧 `xwiz_act_server.server`，与 protocol-v2 相对链路不同。

## 3. 启动顺序（每套模型）

### 3.1 PC2：只启动模型服务

在 PC2 终端：

```bash
cd /home/wengyikun/workplace/joint_songling/start
./pc2_server_220000_ee_loss_relative.sh       # 或 180000 / 500000 对应脚本
```

服务脚本在启动前检查 `config.json`、`model.safetensors` 及 relative q01/q99（DINOv3）文件；500000 还必须确认其目录真实存在并检查自身配置。另开终端用 `ss -ltnp | rg ':(8889|8892)'` 和 CUDA 日志确认端口、权重、GPU。

### 3.2 PC1：启动 ROS 客户端

在 PC1 终端：

```bash
cd /home/wengyikun/workplace/joint_songling/start
./pc1_client_220000_ee_loss_relative.sh       # 或 180000 / 500000
```

归档脚本引用 PC1 `/home/dexforce/w1/w1_act` 中的运行时模块；若将归档复制到 PC1，必须同时放置对应的 `direct_runtime`、`xwiz_real_runtime` 和 vendor overlay，不能只复制一个 `.sh`。

### 3.3 先 status/dry-run，再 start

在操作者主机或能访问 PC1 的终端：

```bash
python3 control_220000_ee_loss_relative.py status
python3 control_220000_ee_loss_relative.py start   # 仅在门禁通过且得到明确授权后执行
python3 control_220000_ee_loss_relative.py stop
```

180000 使用其对应控制器；500000 使用 `control_500000_absolute.py`。dry-run 只验证网络、checkpoint 加载和观测协议，必须保证 `start_infer=false`、不发布 `/control/*`。

## 4. 真机安全门禁

启动前逐项确认：

- 机器人状态 `Idle`（推理运行后才允许 `Running`）；20 个电机均为 `OP`；20 个错误码为 0；20 个服务错误为 `None`；
- 没有 Tele、ACT、回放、拖拽、XWiz inference manager 或旧客户端占用控制话题；对每个具体 unit/PID 定向停止，禁止 `pkill -f python/camera/xwiz`；
- `/feedback/robot_server_state`、双手反馈、三路图像均有唯一发布者且时间戳新鲜（建议 <1 s）；
- 身体姿态已恢复 ACT 默认姿态，双手目标和实际反馈按下节协议核对；
- PC2 服务端已完成 finite CUDA smoke test，PC1↔PC2 dry-run 成功；
- 明确知道本次模型的端口、chunk 长度、归一化和相对/绝对模式。

默认身体 20D（机器人反馈顺序）为：

```text
ANKLE,KNEE,BUTTOCK,WAIST,NECK1,NECK2,
LEFT_J1..LEFT_J7,RIGHT_J1..RIGHT_J7
```

## 5. 相机身份、话题和图像转换

USB 序列号是唯一身份依据，不能按 `camera_l/r` 字面猜左右：

| 物理位置 | 序列号 | ROS 原始彩色话题 | 模型字段 |
|---|---|---|---|
| 左腕部 | `412622271335` | `/camera_r/color/image_rect_raw` | `cam_hand_left` |
| 右腕部 | `412622273406` | `/camera_l/color/image_rect_raw` | `cam_hand_right` |

头部使用 `/camera/right_eye_resize`（模型 `cam_high_right`）；左头图若链路需要可作为就绪检查，但三模型 checkpoint 的输入 key 以各自 `config.json` 为准。

### DINOv3 180000/220000（当前相对链路）

客户端订阅原始腕图并在 PC1 转换，避免重复 resize：

```text
头部 960×540 → 上下黑边补为 960×960（上下各 210）→ 224×224
物理左腕 camera_r 640×360 → 直接拉伸 360×360 → 224×224
物理右腕 camera_l 640×480 → 直接拉伸 480×480 → 224×224
```

转换后为 RGB、`uint8`、`224×224×3`，再由 PC2 转成模型需要的 `CHW`。数据集已经是 224×224 时不要再次处理。必须分别查看左右腕转换结果，确认画面中的手与物理左右一致。

### 500000 旧 ACT

旧链路可能订阅 resize 话题并由旧 runtime 完成处理；不要把 DINOv3 的 raw-topic 配置直接套到 500000。其视觉/特征名（如 `cam_high_left`）和预处理以部署目录的 `config.json`、preprocessor 为准。

## 6. 19D state 语义与相对位姿

模型 19D 顺序固定为：

```text
0 WAIST
1..7 LEFT_J1..LEFT_J7
8..9 NECK1,NECK2
10..16 RIGHT_J1..RIGHT_J7
17 LEFT_GRIPPER
18 RIGHT_GRIPPER
```

PC1 从 20D 机器人反馈组装：

```python
state19 = positions[3:4] + positions[6:13] + positions[4:6] \
          + positions[13:20] + [left_scalar, right_scalar]
```

220000/180000 的 `joint_representation=relative` 只作用于左右臂索引 `1..7`、`10..16`。输入必须使用**相邻的两帧真实反馈**：

```text
relative_state[t, arm] = current_real_feedback[t, arm]
                         - previous_real_feedback[t-1, arm]
```

上一帧是紧邻的真实机器人反馈，不是上一次推理 state，也不是累计的模型动作。WAIST、NECK1/2、左右开合度索引 `[0,8,9,17,18]` 保持当前绝对值。

模型输出先按 relative q01/q99 反归一化，然后：

```text
absolute_action[arm] = current_real_feedback[arm] + predicted_relative_action[arm]
absolute_action[0,8,9,17,18] = predicted_absolute_action[0,8,9,17,18]
```

得到绝对 19D 后，PC1 在发布前对每个身体关节按 `runtime_w1_contract.py` 的物理最小/最大值裁剪；非有限值直接拒绝。手部标量再由 PC1 转为六关节 POSITION。500000 没有 relative processor，state/action 均按绝对值处理；不要对它再做增量还原。

## 7. q01/q99、MEAN_STD 和手部协议

- 220000/180000：`normalization_mapping.STATE/ACTION=QUANTILES`，使用各 checkpoint `relative_stats/relative_*_q01_q99.json`，并启用 `clip_quantiles=true`。视觉为 RGB identity（模型内部按其配置处理）。
- 500000：旧 checkpoint 的 active mapping 是 `MEAN_STD`；即使目录里存在 q01/q99，也不能擅自切换到 quantiles，必须跟随其 preprocessor/postprocessor。

19D 的第 17、18 维不是六关节数组，而是 `0..100` 开合度标量。约定为“0=任务闭合姿势，100=打开”：

```text
ACT 默认打开（左右相同）：[0,70,0,0,0,0]
左 scalar=0： [0,100,35,45,47,37]
右 scalar=0： [65,100,70,75,100,100]
```

六维顺序为 `T_MCP, T_CMC_YAW, IF_MCP_PITCH, MF_MCP_PITCH, RF_MCP_PITCH, LF_MCP_PITCH`；第二项 `T_CMC_YAW=70` 是拇指根部旋转。PC1 只接收 scalar 并内部线性映射到六关节，不能在模型 payload 中额外发送一组手指数组。历史硬件反馈可能出现 `0.4/69.4` 等近似端点值，按反馈实测处理，不改变上述端点定义。

如启用裁剪策略 `clip_gripper_openness`：模型输出 `<95` 直接归为 scalar `0`，否则保留并裁剪到 `0..100`。每次改动后必须现场验证左、右闭合姿势，尤其是左手握杯姿态和右手无名指是否完全打开。

只改变双手而不发送身体关节时，可在已 source ROS 环境的 PC1 使用（参数仍然只有 scalar）：

```bash
python3 set_hand_scalar.py --left 100 --right 100   # 两手打开/ACT 默认手
python3 set_hand_scalar.py --left 0 --right 0       # 按左右各自端点闭合
```

该工具内部才展开六关节数组；PC2 模型网络 payload 只携带第 17、18 维 scalar，不发送具体六关节数组。

## 8. chunk、频率和“无限连续”

220000/180000 客户端严格要求每个动作 chunk 为 `16×19`，不插值、不补成 100 步。执行完当前 16 帧、动作队列清空，并收到更新的真实反馈后，才请求下一个 chunk；不得在 chunk 中途请求。`execution_mode=continuous` 只表示按该规则无限循环。

配置默认控制/采集频率为 30 Hz；如需慢速验收，将两者同时改为 10 Hz，并先 dry-run。当前已核对的 500000 PC2 部署 artifact 为 `chunk_size=16,n_action_steps=16`；历史 `/act_popcorn_45w` 源配置中的 100 步不代表当前转换产物。若替换 artifact，仍必须重新读取实际 `config.json` 并让 PC1/PC2 两端一致。

## 9. 停止、恢复与故障处理

正常停止顺序：

```bash
python3 control_<model>.py stop
# 确认 PC1 manager 无活动、PC2 端口服务已停止或回到 idle
# 再按现场 SOP 将身体恢复 ACT 默认位姿、双手恢复 [0,70,0,0,0,0]
```

若出现 stale feedback、相机超时、非零电机错误、队列残留、协议版本/shape 不符，客户端必须停止发布；先修复根因再重启。不要通过 `skip_default_pose_check` 绕过健康门禁；只有经过明确风险授权且仍保持电机/反馈/限位检查时才可例外。

## 10. 启动前最小验收命令

```bash
# 两端网络
ping -c 1 192.168.20.20
ping -c 1 192.168.20.21

# PC2 端口和进程
ss -ltnp | rg ':(8889|8892)'

# PC1 ROS 话题（在已 source 环境的 PC1）
ros2 topic list | rg 'robot_server_state|camera_(r|l)|feedback/ee|control/'
ros2 topic hz /feedback/robot_server_state
ros2 topic hz /camera_r/color/image_rect_raw
ros2 topic hz /camera_l/color/image_rect_raw

# 本归档静态校验
for f in /home/wengyikun/workplace/joint_songling/start/*.sh; do bash -n "$f"; done
python3 -m json.tool /home/wengyikun/workplace/joint_songling/start/config_220000_ee_loss_relative.json >/dev/null
```

静态通过不代表允许运动；仍需核对实际 checkpoint 文件、相机画面、控制话题发布者和机器人状态后再发送 `start`。
