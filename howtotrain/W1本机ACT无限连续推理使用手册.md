# W1 本机 ACT 无限连续推理使用手册

## 1. 运行结构

本方案不依赖 PC2 和 XWiz GUI：本机 `192.168.20.164` 运行 ACT 模型服务，PC1 `192.168.20.20` 读取机器人观测、请求推理并发布真机动作。

```text
W1 左右头部相机、20D 机器人反馈、双手反馈、双腕真实彩色图
  -> PC1:8890 观测与安全执行器
  -> 本机:8889 ACT 模型
  -> 返回 100x19 动作 chunk
  -> PC1 以 10 Hz 发布到 W1
```

PC1 空闲时只缓存观测，不保存训练数据、不调用模型、不发布动作。

左右头图都会进入 PC1 的观测缓冲区并参与观测就绪检查，但当前
`act_popcorn_45w` checkpoint 的实际视觉输入只包含左头目
`observation.images.cam_high_left`，右头目不输入该模型。

## 2. 当前 checkpoint 的训练/推理合同

### 2.1 19D 状态与动作

`observation.state` 和每一帧 `action` 都是绝对值 19D，不是相对上一帧的增量。
顺序必须严格为：

| 索引 | 含义 |
| --- | --- |
| 0 | `WAIST` |
| 1..7 | `LEFT_J1..LEFT_J7` |
| 8..9 | `NECK1, NECK2` |
| 10..16 | `RIGHT_J1..RIGHT_J7` |
| 17 | 左手开合度 |
| 18 | 右手开合度 |

前 17 维是关节位置；不包含 `ANKLE`、`KNEE`、`BUTTOCK`。当前真机适配器将
夹爪标量解释为 `0..100` 开合度：`0` 是任务定义的闭合端，`100` 是张开端，
再分别线性映射为左右 Linker L6 各 6 个手指关节命令。若以后重新制作数据集，
夹爪的数值范围和方向必须与这个适配合同一致。

### 2.2 图像输入

当前 checkpoint 固定需要三个图像 key：

- `observation.images.cam_high_left`：左头目真实图像。
- `observation.images.cam_hand_left`：左腕真实彩色图像。
- `observation.images.cam_hand_right`：右腕真实彩色图像。

三路图像的合同尺寸均为 `3x360x640`。PC1 向模型服务发送 `640x360x3` BGR
字节，模型服务先转为 RGB，再交给 checkpoint 自带的 processor。

当前运行时直接订阅 `/camera_l/color/image_rect_raw` 和
`/camera_r/color/image_rect_raw`。checkpoint 的腕图归一化统计为正常彩色图尺度，
但真机任务效果仍需通过受控试运行验证，不能只凭接口形状判断策略质量。

### 2.3 归一化与输出

- checkpoint 自带 `policy_preprocessor.json` 及其统计参数，对图像、状态做预处理/归一化。
- 模型每次生成有限的 `100x19` 绝对动作。
- checkpoint 自带 `policy_postprocessor.json` 及其统计参数，将动作反归一化回真机适配器使用的数值空间。
- 不得绕过 checkpoint processor，也不得将其他数据集的统计文件混用到该 checkpoint。

### 2.4 “符合训练要求”的判定边界

当前已能确认运行时符合 `act_popcorn_45w` 的推理合同：19D 顺序、绝对值语义、
三个图像 key、图像尺寸、processor 和 `100x19` 输出一致。这不能单独证明新采集数据
已达到训练质量要求。

用于新训练前，还必须单独验收：

- episode 边界和末尾 padding 不跨 episode；
- 状态、动作和三路图像的时间同步及实际 FPS；
- 19D 字段顺序、单位、绝对值语义和夹爪开合度一致；
- 图像非空、无大量重复/损坏帧，色彩顺序、相机视角与训练配置一致；
- 训练使用由该数据集生成的 normalization statistics，不沿用不相干数据集的统计。

## 3. 连续执行语义

ACT 单次推理固定生成 `100x19`：100 帧动作，每帧为 17 个机身关节和左右手开合度。一个 chunk 在 10 Hz 下约执行 10 秒。

```text
执行 chunk N 的第 1..100 帧
  -> 确认第 100 帧已经发布
  -> 等待该时刻之后的新机器人状态反馈
  -> 清除 chunk 边界前的旧观测
  -> 用最新观测生成 chunk N+1
  -> 执行 chunk N+1
```

连续模式固定使用：

```text
action_horizon=100
chunk_size_threshold=0.0
sample_factor=1.0
```

`100` 是模型每次输出的 chunk 长度，不是总运行帧数。运行时使用原厂线程能接受的极大有限 `max_steps` 兼容值，实际由人工 `stop` 或安全门禁失败终止。

## 4. 文件与日志

本机代码工作树：

```text
/home/wengyikun/workplace/popcorn/.worktrees/w1-local-pc1-executor
```

连续任务：

```text
w1_act-ljl-act_train/xwiz_real_runtime/task_3_continuous_real.json
```

PC1 已部署运行时：

```text
/home/dexforce/w1/w1_act/xwiz_real_runtime
```

日志：

```text
PC1: /home/dexforce/xwiz_real_client.log
本机: journalctl --user -u xwiz-act-server.service
```

## 5. 全部重启后的启动顺序

### 5.1 本机模型服务

```bash
systemctl --user start xwiz-act-server.service
systemctl --user is-active xwiz-act-server.service
ss -ltnp 'sport = :8889'
```

要求服务为 `active`，本机 TCP `8889` 正在监听。

### 5.2 PC1 观测与执行器

```bash
ssh dexforce@192.168.20.20
/home/dexforce/w1/w1_act/xwiz_real_runtime/start_pc1_runtime.sh status
/home/dexforce/w1/w1_act/xwiz_real_runtime/start_pc1_runtime.sh start
```

若 `status` 已显示 `client_service` 正在运行且 `8890` 正在监听，不要重复启动。

### 5.3 启动前门禁

必须同时满足：

- 现场无人和障碍，物理急停可立即触达。
- Auto、Tele、ACT、Map、Auto timer 均未运行，`act_ros2` 未运行。
- 机器人为 `Idle`，20 个电机全为 `OP`，电机和服务错误码全为 0/`None`。
- 机器人位于 ACT 默认姿势，20D 最大误差不超过 `0.05 rad`。
- 左右头图、机器人状态、左右手反馈、左右腕真实彩色图共 7 类观测齐全。
- 三个真机控制话题没有其他持续控制流。

PC1 会在部署时再次检查姿势、健康和观测，不满足时拒绝启动。

## 6. 恢复 ACT 默认姿势

这是真机运动，先停止所有推理与其他控制源，再在 PC1 执行：

```bash
export ROS_DOMAIN_ID=20
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
source /opt/ros/humble/setup.bash
source /home/dexforce/w1/install/setup.bash
cd /home/dexforce/w1/dexe_mobile_application/script
python3 slowly_move_to.py act_default_pose_runtime.json 1000
```

约 20 秒完成。完成不等于验收通过，仍需从 `/feedback/robot_server_state` 核对 `Idle`、20/20 `OP`、零错误和姿态误差。

## 7. 启动无限连续真机推理

在本机功能工作树执行：

```bash
cd /home/wengyikun/workplace/popcorn/.worktrees/w1-local-pc1-executor
PYTHONPATH=w1_act-ljl-act_train python3 -m xwiz_real_runtime.control_cli \
  --client-host 192.168.20.20 start-continuous-real \
  --task w1_act-ljl-act_train/xwiz_real_runtime/task_3_continuous_real.json \
  --confirm EXECUTE_CONTINUOUS_REAL_FRAMES
```

查看状态：

```bash
PYTHONPATH=w1_act-ljl-act_train python3 -m xwiz_real_runtime.control_cli \
  --client-host 192.168.20.20 status
```

正常连续日志应反复出现：

```text
Validated finite ACT action chunk N with shape (100, 19)
Executed guarded real action ... chunk=N frame=100/100
Continuous real chunk N completed; waiting for a fresh observation ...
Validated finite ACT action chunk N+1 with shape (100, 19)
Executed guarded real action ... chunk=N+1 frame=1/100
```

## 8. 停止

正常软件停止：

```bash
cd /home/wengyikun/workplace/popcorn/.worktrees/w1-local-pc1-executor
PYTHONPATH=w1_act-ljl-act_train python3 -m xwiz_real_runtime.control_cli \
  --client-host 192.168.20.20 stop
```

停止后不自动复位，机器人保持最后一帧姿势。紧急情况优先使用物理急停，软件 `stop` 不能替代急停。

`stop` 成功且已确认命令话题静默后，如果本次请求没有事先指定返回姿态，
必须先询问操作者选择：ACT 默认姿态、站立 Standby 默认姿态或蹲姿 Rest 默认姿态。
在获得选择前不发布复位动作；返回姿态是新的真机运动，需重新执行运动前安全检查。

如需停止所有相关软件组件：

```bash
PYTHONPATH=w1_act-ljl-act_train python3 -m xwiz_real_runtime.control_cli \
  --client-host 192.168.20.20 stop
ssh dexforce@192.168.20.20 \
  /home/dexforce/w1/w1_act/xwiz_real_runtime/start_pc1_runtime.sh stop
systemctl --user stop xwiz-act-server.service
```

## 9. 自动停止条件

任一条件触发时会停止推理和新动作发布，并保持最后姿势：

- 机器人状态反馈超过 1 秒未更新。
- 机器人状态不是 `Idle` 或 `Running`。
- 任一电机不是 `OP`，或出现电机/服务错误码。
- 动作 chunk 不是有限的 `100x19`，或帧内包含 NaN/Inf。
- 前一 chunk 未执行完却提前收到下一 chunk。
- 模型连接、观测链路异常或人工停止。

故障后不要绕过门禁。先检查日志与机器人状态，确认控制话题静默，再恢复 ACT 默认姿势并重新部署。

## 10. 2026-08-21 真机验收记录

启动前实测：机器人 `Idle`、20/20 电机 `OP`、错误码为零，ACT 默认姿势最大误差小于 `0.001 rad`；7/7 观测到齐，双头图有效，双腕图 `min=0/max=0/nonzero=0`。

连续执行实测跨越多个 chunk：chunk 1 完成后才用新反馈请求 chunk 2，chunk 2、3 同样完整执行到 `100/100` 后生成下一块。客户端、模型服务保持 `running`，机器人运行中仍为 20/20 `OP` 且错误码为零。
