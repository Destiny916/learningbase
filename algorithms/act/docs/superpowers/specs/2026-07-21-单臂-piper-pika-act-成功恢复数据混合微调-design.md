# 单臂 Piper/Pika ACT 成功恢复数据混合微调设计

## 目标

在不引入在线 SAC、不改写现有 ACT 模型结构的前提下，利用单臂 Piper/Pika 真机运行中产生的人为接管数据，补足 ACT 在常见偏离状态下的恢复能力。

产物是可与现有基线直接比较的新 ACT checkpoint。成功标准是：在禁止人工接管的相同评估条件下，新 checkpoint 保持或提升原任务成功率，并提升已知偏离状态下的自主恢复能力。

## 范围

本设计仅覆盖一个单臂任务，采用“成功恢复片段混合行为克隆（BC）”。

包含：

- ACT 自主运行期间的人机接管数据记录；
- Pika 双击触发的单臂策略/遥操控制权切换；
- ROS2 遥操输入到 SDK 真机控制的单向桥接与动作仲裁；
- 独立于遥操切换的回合采集、即时结果标注和数据落盘；
- 回合成功、失败、超时与接管元数据；
- 成功恢复片段的确定性筛选；
- 与原始成功示范混合的 ACT 微调；
- 不接管的回归与恢复评估；
- checkpoint 版本与回退规则。

不包含：

- HIL-SERL/SAC 在线强化学习、奖励分类器和数值奖励训练；
- 双臂任务；
- ACT + SAC 残差策略；
- 修改 Piper/Pika 的底层 CAN、ROS 控制或遥操作实现；
- 同时运行现有 Piper ROS 控制器与 SDK 控制器。

## 设计原则

1. ACT 是监督式行为克隆策略。本方案以成功动作标签训练，不将奖励传入 ACT 损失。
2. 人接管表示策略动作不可信，但不表示接管后的动作无效；人工恢复并最终成功的动作是核心训练样本。
3. 原始成功示范始终保留，防止模型只学习局部纠错而遗忘正常任务流程。
4. 失败回合不作为 ACT 的动作监督标签；它们仅用于错误归因、覆盖率统计和无干预评估。
5. 训练、部署和评估 checkpoint 严格版本化；未通过无接管评估不得替换部署基线。
6. SDK 是 Piper/Pika 硬件动作的唯一写入者。ROS2 仅提供 Pika 遥操观测和控制权切换触发，绝不向 Piper 控制 topic 发布动作。
7. 本方案的 `success` 与 `failure` 是数据集结果标签，不是 ACT 训练时的数值奖励；每个回合结束时立即写入，不在整批采集后追溯猜测。

## 数据流

```text
Pika 双击 → ROS2 /teleop_trigger 服务 → TeleopBridge 切换控制模式

ACT-vN 自主执行
  ├── 正常成功 ────────────────> 可选自主成功正样本
  ├── 发生偏离 → 人接管 → 恢复 → 成功
  │                              └── 成功恢复正样本
  └── 失败 / 超时 / 急停 ───────> 诊断与评估数据

原始成功示范 + 成功恢复正样本 + 可选自主成功正样本
  └── 混合训练集 ───> ACT-vN+1
                         └── 无接管评估 ───> 通过：候选部署；失败：回退 ACT-vN
```

## 单臂 SDK 控制与 Pika 双击切换

真机部署进程直接通过 Piper/Pika SDK 控制硬件，不通过 ROS 发送 Piper 控制命令。ROS2 与 SDK 的边界如下：

```text
Pika 单臂定位、夹爪与双击事件（ROS2）
                 ↓
         PikaRosTeleopBridge
                 ↓
  控制模式状态机：POLICY / TELEOP
                 ↓
ACT 动作或 Pika 遥操动作 → 安全限幅与动作仲裁 → SDK → Piper/Pika 硬件
```

`PikaRosTeleopBridge` 是 SDK policy runner 的内部组件；若 ROS2 与 SDK 依赖无法在同一 Python 环境加载，该组件可独立为 ROS2 进程，并仅通过本机 IPC 向 SDK policy runner 发送规范化遥操输入和模式变更事件。无论进程布局如何，动作仲裁和 SDK 调用必须在 SDK policy runner 内执行。

### 现有双击机制的单臂复用

当前 Pika `serial_gripper_imu` 节点会在串口 `Command` 值发生变化时调用 ROS2 服务 `/teleop_trigger`。该 `Command` 变化是现有 Pika 双击触发机制在 ROS2 侧的表现。

单臂运行采用无后缀服务名 `/teleop_trigger`：

1. `PikaRosTeleopBridge` 提供 `/teleop_trigger` 的 `std_srvs/srv/Trigger` 服务。
2. Pika 双击触发该服务后，bridge 请求 SDK policy runner 切换控制模式。
3. `POLICY → TELEOP`：冻结当前 SDK 实际末端目标为遥操参考；等待获得有效的 Pika 位姿与 SDK 反馈后，才设置 `is_intervention=true` 并开始发送 Pika 相对动作。
4. `TELEOP → POLICY`：停止采用 Pika 动作，重新以最新 SDK 实际状态作为策略动作的参考状态，设置 `is_intervention=false`，由 ACT 继续控制。
5. 任一参考状态、Pika 位姿或 SDK 反馈无效时，拒绝切换并保持当前安全状态；不得以默认零位姿切换。

同一次双击只产生一次模式翻转。bridge 必须对服务请求去抖并记录切换时间、切换前后模式和失败原因。

### 单臂 ROS2 启动边界

单臂接管仅启动提供 Pika 定位、夹爪、相机和双击服务调用能力的单臂 sensor/locator 路径，例如现有 `sensor_tools` 的 `open_single_sensor` 或与其等价的最小启动组合。

不得启动 `teleop_rand_single_piper.launch.py` 或任何双臂 `teleop_rand_multi_piper.launch.py`：这些 launch 会同时启动 Piper FK、IK 和 Piper 控制器，与 SDK 控制硬件冲突。bridge 也不得向 `/piper_IK/ctrl_end_pose`、`/joint_states` 或其他 Piper ROS 控制 topic 发布动作。

## 采集状态机、结果标注与数据落盘

### 结果标签不是 ACT 奖励

本方案训练 ACT 使用行为克隆损失，不读取逐步数值 reward。操作者在每个 episode 结束时立即给出任务结果标签：

| 结束事件 | `episode_outcome` | 是否产生 ACT 正样本 |
|---|---|---|
| 任务客观完成 | `success` | 是；若含接管则提取成功恢复片段 |
| 明确判定无法完成 | `failure` | 否 |
| 超过任务时间上限 | `timeout` | 否 |
| 安全急停或保护停止 | `safety_stop` | 否 |
| 重录/放弃当前回合 | `discarded` | 否 |

任务成功不区分最终由策略还是人工完成。`is_intervention` 与 `control_mode` 另行记录，用于区分自主成功、协作成功和恢复样本。若未来启用 HIL-SERL/SAC，才在同一结束事件上额外使用 `success=1`、其余结束结果为 `0` 的稀疏 reward；该数值 reward 不属于本 ACT 微调方案。

### 采集控制接口

SDK policy runner 提供独立的 `CollectionControl` 接口，包含五个原子命令：

| 命令 | 合法状态 | 效果 |
|---|---|---|
| `ARM_EPISODE` | `IDLE` | 校验 SDK、Pika、相机和任务初始状态后进入 `RECORDING`，创建临时 episode |
| `SUCCESS` | `RECORDING` | 立即写入 `success`，停止追加帧并原子落盘 |
| `FAILURE` | `RECORDING` | 立即写入 `failure`，停止追加帧并原子落盘 |
| `DISCARD` | `RECORDING` | 停止追加帧，写入 `discarded`；保留诊断原始文件但不导入训练集 |
| `STOP_SESSION` | `IDLE` | 结束当前采集会话；若仍在 `RECORDING` 则先执行 `DISCARD` |

控制接口的传输方式可为 SDK runner 本地键盘、CLI 或受鉴权的本机服务，但它们必须映射到同一组原子命令并记录操作者、时间和命令来源。Pika 双击不属于 `CollectionControl`，不得隐式调用 `ARM_EPISODE`、`SUCCESS`、`FAILURE` 或 `DISCARD`。

### 单个 episode 的边界

```text
IDLE
  └── 完成 reset、物体摆放和反馈健康检查
        └── ARM_EPISODE → RECORDING
              ├── POLICY ↔ TELEOP：Pika 双击，仅改变动作来源，继续记录
              ├── SUCCESS → 落盘 success episode → IDLE
              ├── FAILURE / 超时 → 落盘非成功 episode → IDLE
              ├── safety_stop → 落盘 safety_stop episode → IDLE
              └── DISCARD → 落盘诊断文件，不导入数据集 → IDLE
```

采集从 `ARM_EPISODE` 成功后的第一帧开始，到结束命令或自动结束事件发生的最后一个已执行动作帧为止。reset、回零、物体摆放、手动调整场景和 episode 结束后的清理过程不得写入 ACT 训练 episode。

每个临时 episode 必须先写入独立临时目录；结束标签、帧数、时间戳和数据完整性校验全部成功后才原子重命名为已完成 episode。进程异常退出留下的临时目录标为 `incomplete`，不得自动进入训练集。

### 与现有 Pika 采集服务的隔离

现有单臂 Pika `serial_gripper_imu` 在检测到 `Command` 变化时，除调用 `/teleop_trigger` 外还会调用 `/data_tools_dataCapture/capture_service`。SDK policy runner 不得将该服务调用解释为本设计的 episode 开始或结束。

单臂接管部署必须满足以下之一：

1. 运行最小 Pika sensor/locator 组合，不暴露或不连接该 capture service；或
2. 由 bridge 提供隔离实现，使该 capture service 仅记录兼容性事件，不改变 `CollectionControl` 状态。

无论选择哪一种，只有 `CollectionControl` 可以创建、结束或丢弃 ACT 训练 episode。

## 运行时事件与数据契约

每一帧必须使用与现有 ACT 数据集完全一致的观测键、动作维度、单位、坐标系、相机颜色顺序和时间基准。单臂动作标签必须是最终发往 Piper/Pika 的实际动作，不能是被接管覆盖前的 ACT 预测动作。

每个 episode 和 frame 至少保存以下字段：

| 字段 | 粒度 | 含义 |
|---|---|---|
| `observation.*` | 帧 | 与原 ACT 一致的状态和图像观测 |
| `action` | 帧 | 实际执行的单臂动作 |
| `is_intervention` | 帧 | 该帧是否由人工接管控制 |
| `episode_outcome` | 回合 | `success`、`failure`、`timeout` 或 `safety_stop` |
| `intervention_start` | 帧 | 人工接管开始帧 |
| `intervention_end` | 帧 | 人工交还策略或成功结束前的最后接管帧 |
| `control_mode` | 帧 | `policy` 或 `teleop`，是 `is_intervention` 的可读表示 |
| `teleop_toggle_id` | 帧 | 所属 Pika 双击切换事件的单调递增编号；非接管帧为空 |
| `collection_command` | 回合 | 创建或结束该 episode 的 `CollectionControl` 命令与时间戳 |
| `dataset_integrity` | 回合 | `complete`、`discarded` 或 `incomplete`，决定是否允许导入训练集 |
| `policy_checkpoint` | 回合 | 产生该回合的 ACT 版本 |
| `task_id` | 回合 | 单臂任务标识 |

成功与失败是任务结果标签，而非控制来源标签：无论最终由人还是策略完成，只要任务客观达成，`episode_outcome=success`。是否自主完成由 `is_intervention` 另行统计。

## 成功恢复片段筛选

对于每个 `success` 回合：

1. 若从未接管，可作为可选的自主成功片段。
2. 若发生接管，对每次接管取从接管开始前的固定短观测上下文开始，到任务成功结束的连续片段。
3. 片段中保留人工恢复和后续成功执行的实际动作；接管前明显导致偏离的纯策略动作不作为监督动作标签。
4. 若同一回合有多次接管，分别产生片段；重叠区只保留一次。
5. `failure`、`timeout` 与 `safety_stop` 回合不产生训练片段。
6. 数据筛选输出必须可复现：输入 episode id 与配置后，片段边界和样本数固定。

接管前的短上下文是必要的，因为它让 ACT 学到“看见即将失败的状态时应如何恢复”，而不仅是模仿任务末端动作。

## 训练集构成与微调策略

训练集由三类正样本组成：

```text
D_train = D_base + D_recovery + D_auto_success（可选）
```

- `D_base`：原始人工成功示范，必须始终参与训练。
- `D_recovery`：人为接管后最终成功的恢复片段，是新增数据的重点。
- `D_auto_success`：当前 ACT 无接管成功回合，可用于巩固已有能力。

微调保持现有 ACT 的输入输出、action chunk 定义和 BC 损失不变。训练过程采用保守设置：从 ACT-vN 初始化、低学习率、短训练轮次、频繁保存 checkpoint。训练配置必须记录三类数据的采样比例、源数据集版本、随机种子和 ACT 基础 checkpoint。

不得以仅包含 `D_recovery` 的数据训练，也不得将失败回合直接作为正向 BC 标签。

## 评估与发布门槛

每个候选 checkpoint 使用与 ACT-vN 相同的任务定义、相机配置、初始姿态集合和物体扰动集合，且评估时禁止人工接管。

必须报告：

- 原任务无接管成功率；
- 已知偏离场景的无接管恢复成功率；
- 平均完成时间；
- 安全事件数（越界、碰撞、急停）；
- 若在训练采集阶段统计，人工接管帧占比和接管回合占比。

只有同时满足以下条件，ACT-vN+1 才可成为候选部署版本：

1. 原任务无接管成功率不低于 ACT-vN；
2. 恢复成功率高于 ACT-vN，或在 ACT-vN 尚无恢复能力时达到预先定义的最低可用水平；
3. 安全事件不增加；
4. 评估日志、配置、数据版本和 checkpoint 可追溯。

任何一项不满足时，保持 ACT-vN 为部署版本，并将候选版本、数据组成和失败模式归档以便下一轮采集调整。

## 安全与操作约束

- 接管用于避免不可逆或危险行为，不能以收集失败数据为目的延迟接管。
- 训练数据采集期间，Piper 的工作空间、单步动作、关节和夹爪限制保持启用。
- 同一时刻只允许 SDK policy runner 向 Piper/Pika 硬件发布实际动作；接管时仅切换该 runner 内的动作来源。
- Pika 双击服务只负责改变控制模式，不能直接绕过动作仲裁或安全限制。
- 禁止与 SDK runner 并发启动 Piper ROS controller、IK 或现有全链路 single/multi teleop launch。
- 真机评估从低风险初始状态开始，并保留人工急停能力；任何安全终止均记为非成功。

## 验收标准

该设计完成实现后，应能证明：

1. 每个成功恢复片段都可追溯到原始 episode、接管边界和实际动作；
2. 训练集同时含原始成功示范与成功恢复样本；
3. 失败数据不进入 ACT 的正向 BC 监督；
4. 新旧 ACT 的无接管评估可重复对比；
5. 未通过门槛的 checkpoint 不会覆盖当前部署版本。
6. Pika 双击仅切换 `POLICY` 与 `TELEOP`，不创建或结束 episode；
7. 每个已完成 episode 在结束时已有不可变的结果标签和完整性状态。
