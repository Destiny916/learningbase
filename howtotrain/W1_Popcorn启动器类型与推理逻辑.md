# W1 Popcorn 启动器类型与推理逻辑

本文是 `start/` 目录的操作索引。它补充模型合同、相机转换、手部协议、PC1/PC2 职责和各启动器的差异。真机动作属于高风险操作：先完成只读 dry-run、反馈/电机/控制权门禁，再由操作者明确执行 `start`。

## 1. 两台机器的职责

| 主机 | 地址 | 允许做什么 |
|---|---|---|
| PC1 | `192.168.20.20` | ROS/EtherCAT、读取真实反馈和图像、构造 state、相对量适配、限位、发布绝对动作 |
| PC2 | `192.168.20.21` | 加载 GPU checkpoint、执行推理、返回 action chunk；不直接发布电机命令 |

默认 ROS 环境：`ROS_DOMAIN_ID=20`、`RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`。启动器都支持 `W1_ACT_ROOT`、`CONFIG_PATH`、`POLICY_PATH` 等环境变量覆盖，迁移到新 W1 时不要把固定路径当成硬编码契约。

## 2. 启动器分类

### 2.1 DINOv3 relative（180000、220000）

PC2：`pc2_server_180000_dinov3_relative.sh`（8892）或 `pc2_server_220000_ee_loss_relative.sh`（8889）。
PC1：对应 `pc1_client_*_relative.sh`，控制器分别为 `control_180000_dinov3.py`、`control_220000_ee_loss_relative.py`。

两者都是 19D、chunk16、q01/q99；220000 的 EE/FK 只属于训练辅助损失，不会产生额外的机器人输出。左右臂索引使用相邻真实反馈计算相对 state，动作反归一化后以当前真实 state 还原成绝对目标，再按每个关节物理限位裁剪。

### 2.2 旧 ACT absolute（500000）

`pc2_server_500000_absolute.sh` + `pc1_client_500000_absolute.sh` 使用旧 ACT processor。state/action 均按绝对值解释，不得套用 DINOv3 的 relative 差分或 q01/q99。若使用 `*_chunk100` 变体，必须先读取该 checkpoint 的 `config.json`，确认 `chunk_size=100` 和 `n_action_steps=100`。

### 2.3 ACT 200000 chunk100

- 同步：`pc1_client_200000_sync100.sh` 或 `pc1_client_200000_chunk100.sh`。100 个策略点直接执行，当前块完成后才请求下一块。
- 异步插值：`pc1_client_200000_async100.sh`。100 策略点按 `sample_factor=2` 生成 200 控制点；剩 30 控制点时请求下一块，按绝对 `control_step` 对齐，身体最多 30 点线性 LIPO；左右手开合度不融合，直接采用新块。
- 异步无插值：`pc1_client_200000_nointerp_async.sh`。100 策略点直接执行；剩 15 点时请求下一块，身体最多 15 点 LIPO；手部 scalar 直接采用新块。

新会话必须清空旧队列、transition 和 pending 结果。延迟到达的 chunk 先丢弃已经过去的时间步，只执行尚未过期的后缀；不能把上一会话的 chunk 融入第一个新 chunk。

## 3. 观测与动作合同

模型 19D 顺序：`WAIST, LEFT_J1..J7, NECK1, NECK2, RIGHT_J1..J7, LEFT_GRIPPER, RIGHT_GRIPPER`。其中第 17、18 维是 `0..100` 开合度 scalar，不是六维手指数组。

相对模型只对左右臂索引 `1..7`、`10..16` 做：

```text
relative_state[t] = actual_feedback[t] - actual_feedback[t-1]
```

`actual_feedback[t-1]` 必须是紧邻的上一帧真实反馈，不能用上一条命令或上一次推理 state。模型输出反归一化后，臂关节加回当前真实反馈；腰部、颈部和双手 scalar 保持模型规定的绝对语义。最终所有身体关节做非有限值检查和物理上下限裁剪。

## 4. 相机映射与转换

USB 序列号决定物理左右，不能按话题字母猜测：

| 物理位置 | 序列号 | 原始话题 | 模型 key | 转换 |
|---|---|---|---|---|
| 左腕 | `412622271335` | `/camera_r/color/image_rect_raw` | `cam_hand_left` | `640×360 → 拉伸 360×360 → resize 224×224` |
| 右腕 | `412622273406` | `/camera_l/color/image_rect_raw` | `cam_hand_right` | `640×480 → 拉伸 480×480 → resize 224×224` |
| 头部 | — | `/camera/left_eye_resize`、`/camera/right_eye_resize` | 以 checkpoint `input_features` 为准 | `960×540 → 上下补黑边为 960×960 → resize 224×224` |

输出必须是 RGB、`uint8`、`224×224×3`，再由模型 processor 转 CHW。数据集已经是 224×224 时禁止二次处理。每次更换驱动或模型都应保存原图和转换后图像各一帧，核对物理左右是否一致。

## 5. 手部协议

六关节顺序固定为 `T_MCP, T_CMC_YAW, IF_MCP_PITCH, MF_MCP_PITCH, RF_MCP_PITCH, LF_MCP_PITCH`。ACT 默认打开姿态为 `[0,70,0,0,0,0]`，第二项 70 是拇指根部旋转。

```text
左 scalar=0:   [0,100,35,45,47,37]
左 scalar=100: [0,70,0,0,0,0]
右 scalar=0:   [65,100,70,75,100,100]
右 scalar=100: [0,70,0,0,0,0]
```

网络 payload 只传第 17、18 维 scalar；PC1 的硬件适配器才将 scalar 展开为六维并发布 `/control/ee/left`、`/control/ee/right`，反馈从 `/feedback/ee/left`、`/feedback/ee/right` 读取。`/feedback/hand/*` 不是当前协议。手部维度在异步 LIPO 中不与旧 chunk 融合。

## 6. 推荐启动顺序

1. PC2 运行对应 `pc2_server_*.sh`，确认端口、checkpoint、GPU 和 finite smoke test。
2. PC1 运行对应 `pc1_client_*.sh`，只读检查相机、反馈、state/action shape 和控制发布者。
3. 用 `control_*.py status` 查看 manager；确认机器人 `Idle`、20 电机 `OP`、错误码全 0、无其他控制源。
4. 先做 PC1↔PC2 dry-run（不发布 `/control/*`），操作者确认急停和工作区后才执行 `start`。
5. 停止时先 `control_*.py stop`，确认客户端停止发布，再按现场要求恢复 ACT 默认身体姿态和双手 scalar=100。

常用检查：

```bash
for f in start/*.sh; do bash -n "$f"; done
PYTHONPATH=. pytest -q tests/test_async_chunk100_runtime.py tests/test_chunk100_deployment_contract.py
ros2 topic list | rg 'feedback/(ee|robot)|camera_(r|l)|control/'
```

静态检查通过不等于允许运动；模型加载、图像内容、反馈新鲜度和控制权必须在现场再次确认。
