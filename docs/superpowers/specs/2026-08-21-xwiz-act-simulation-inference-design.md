# XWiz W1 ACT 仿真推理设计

日期：2026-08-21

## 目标

用户在本机 XWiz 点击“仿真推理”后，系统加载 `act_popcorn_45w`，使用真实头部左目图像、左右腕部黑图和 W1 的 19 维状态完成 ACT 推理，并把生成的 `100×19` 绝对关节动作块送入仿真控制链。整个流程不得复位或控制真机。

## 方案比较与决策

### 方案 A：本机运行模型 server，PC2 运行安全 client（当前采用）

- 本机 `192.168.20.164` 使用 RTX 5060 加载模型并监听 TCP `8889`。
- PC2 `192.168.20.21` 采集 ROS 观测、连接 server，并只发布仿真控制话题。
- 优点：模型和完整训练代码已在本机，GPU 更适合先完成协议适配与调试。
- 代价：运行时依赖本机和机器人局域网连接。

### 方案 B：模型 server 与 client 均运行在 PC2（后续迁移目标）

- 保持 server 协议和配置不变，把模型环境、模型文件和启动单元迁移到 PC2。
- 优点：机器人内部闭环，不依赖外部工作站。
- 代价：需重新验证 Jetson CUDA、PyTorch、LeRobot 依赖、显存和推理时延。

### 方案 C：修改 XWiz 或 PC1 manager 直接调用 LeRobot（不采用）

- 会让模型细节侵入 XWiz/机器人管理层，扩大真机侧改动和安全风险，也不利于后续迁移。

决策：先实施方案 A；server 保持 ROS 无关、配置驱动，使方案 B 只需要迁移运行环境和修改地址。

## 系统边界

### 本机模型 server

- 绑定 `0.0.0.0:8889`，加载 XWiz 模型目录中的 `model.safetensors`、preprocessor 和 postprocessor。
- 兼容 PC2 旧 client 的网络协议：4 字节大端消息长度加 Python pickle。
- 处理 `SETUP_CONFIG`、`OBSERVATION`、`GET_ACTIONS`、`STATUS`、`STOP` 和 `SHUTDOWN`。
- 将旧 client 观测映射为 LeRobot 输入，再把 `[100,19]` 输出映射为旧 client 所需的 `actions.qpos` 分组结构。
- server 不导入 ROS 控制接口，不发布任何机器人话题。

### PC2 安全 client

- 延续现有 `8890` 管理监听和当前 W1 接口兼容层。
- 强制覆盖任何 XWiz 下发配置：`mode=1`、`home_position=''`、`chunk_size_threshold=0.0`。
- 头部输入使用 `/camera/left_eye_resize`；腕部输入使用两路 640×360 `bgr8` 全零图像。
- 动作发布目标只能是 `/mj_sim/control/*`；配置中出现真机模式或真机控制话题时拒绝启动。

### PC1 InferenceManager 与 XWiz

- XWiz 任务的 server 地址改为 `192.168.20.164:8889`，动作长度改为 100。
- 用户点击“仿真推理”后，PC1 manager 把任务配置传给 PC2 client；安全 client 再连接本机 server。
- XWiz 的“停止推理”应停止计算与仿真发送，但保留 8890 管理监听，方便再次启动。

## 数据契约

### 输入

- `observation.state`：19 维，顺序为 `WAIST`、左臂 J1–J7、`NECK1`、`NECK2`、右臂 J1–J7、左右夹爪。
- `observation.images.cam_high_left`：真实头部左目图像，3×360×640。
- `observation.images.cam_hand_left`：全黑腕部图像，3×360×640。
- `observation.images.cam_hand_right`：全黑腕部图像，3×360×640。
- 图像从 ROS/OpenCV 的 BGR HWC 转为模型要求的 RGB CHW，数值范围遵循保存的 preprocessor。

### 输出

- 模型原始动作块经 postprocessor 反归一化后为 `[100,19]` 绝对目标。
- 旧协议响应包含成功状态、观测时间戳、时间步和 `actions.qpos`。
- 19 维动作按 `WAISTQPOS`、`LEFT_ARMQPOS`、`HEADQPOS`、`RIGHT_ARMQPOS`、左右 gripper 分组，维数总和必须严格为 19。

## 安全机制

- server 和 client 启动前验证模式必须为 simulation。
- 忽略并清空 XWiz task 中的 `home_position`，禁止调用 `_exec_home()`。
- 端到端验收前记录 `/control/joint_position`、`/control/ee/left`、`/control/ee/right` 的发布者基线；验收后不得新增来自推理 client 的真机发布者。
- 仿真动作先由审计订阅器验证维数、有限值、频率和来源，不启动真实机器人控制模式。
- 模型加载、输入缺失、维数错误或非有限输出均返回明确失败状态并停止该次推理，不能降级到真机路径。

## 错误处理

- 模型加载失败：server 保持监听，`STATUS` 返回 error 和具体异常，不能伪报 running。
- 相机或状态超时：PC2 client 不发送不完整观测，并向 manager 报错。
- server 断开：client 停止推理线程和仿真动作发布，保留管理服务等待重新配置。
- 协议或动作维数错误：拒绝响应/执行并记录 request id、实际形状与期望形状。

## 测试与验收

1. 单元测试覆盖长度前缀 pickle 协议、观测键映射、BGR→RGB/CHW、19 维状态映射、`100×19` 动作分组和安全配置覆盖。
2. 使用合成观测直接调用 server，验证真实 checkpoint 能在 CUDA 上加载并输出有限的 `100×19` 动作。
3. 在不调用 XWiz 的情况下连接 PC2 client，验证三路图像和状态可产生一次完整推理响应。
4. 从 XWiz 点击“仿真推理”，验证界面进入运行状态、server 收到观测、模型产生动作、PC2 仅发布 `/mj_sim/control/*`。
5. 验证机器人没有发生运动，真机控制话题没有新增推理发布者；随后从 XWiz 停止推理并确认计算线程退出。

## 后续迁移到 PC2

- server 的监听地址、模型路径、设备和日志路径通过命令行或配置文件提供，不能硬编码本机路径。
- 建立环境检查命令，报告 CUDA、PyTorch、LeRobot、checkpoint 和可用显存。
- 迁移时保持 TCP 协议与测试向量不变，在 PC2 先完成离线单次推理，再把 XWiz task 的 server 地址改为 `192.168.20.21`。
