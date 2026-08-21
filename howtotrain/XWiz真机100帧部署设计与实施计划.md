# XWiz 真机100帧部署设计与实施计划

> **执行要求：** 实施时使用测试驱动开发；任何真机运动只能由用户在 XWiz 中点击所选任务的“部署”触发。测试和 dry-run 均不得发布真机动作。

**目标：** 让本机 XWiz 在仿真/真机之间切换；真机模式一次完整执行 `act_popcorn_45w` 输出的一个 `100×19` 绝对位置动作块，其中17维控制机身，末两维分别控制左右 Linker L6 开合度。

**架构：** 本机继续运行 CUDA ACT 模型服务，PC1 运行 XWiz ROS 推理管理器，PC2 运行观测与动作执行客户端。XWiz 的“部署”携带模型和任务ID但不携带模式，因此仿真与真机使用两个独立任务；PC1 从所选任务配置读取 `mode`，点击部署后立即向 PC2 下发并启动。PC2 在真机模式启动前检查当前 ACT 默认姿势、机器人状态、相机和动作合同，之后仅消费一个100帧块并自动停止。

**技术栈：** Python 3、ROS 2 Humble、CycloneDDS、DexForce 自定义消息、pytest、XWiz 长度前缀 TCP 协议。

---

## 已确认合同

- 模型动作顺序：`WAIST`、左臂 J1–J7、`NECK1`、`NECK2`、右臂 J1–J7、`LEFT_GRIPPER`、`RIGHT_GRIPPER`。
- 动作为绝对位置；输出严格为 `100×19`，所有值必须有限。
- 夹爪标量含义：`0=完全闭合`、`100=完全张开`，超出范围先裁剪。
- Linker L6 命令顺序：`T_CMC_YAW`、`T_MCP`、`IF_MCP_PITCH`、`MF_MCP_PITCH`、`RF_MCP_PITCH`、`LF_MCP_PITCH`。
- 左右手均按配置的闭合端点与张开端点逐关节线性插值；不把标量直接作为弧度或单关节值发送。
- 腕部相机继续使用左右黑图；头部输入使用真实 `/camera/left_eye_resize`，模型目前不消费右头图。
- 真机控制频率为10 Hz，`sample_factor=1`，一次最多执行100帧，即约10秒；`chunk_size_threshold=0`，不提前请求和拼接第二块。

## 安全门禁

开始真机推理前必须同时满足：

1. 机器人状态为 `Idle`，20个电机均为 `OP`，电机及服务器错误码全为零。
2. 当前20维反馈与 ACT 默认姿势最大误差不超过 `0.05 rad`。
3. 左头图、右头图、两路腕部黑图和机器人反馈均已收到。
4. 模型输出形状严格为 `100×19` 且全部有限。
5. 17维机身输出逐关节裁剪至 W1 已知限位；夹爪输出裁剪至 `[0,100]`。
6. 客户端累计发布100帧后立即停止，不自动执行下一块。
7. 仿真模式只发布 `/mj_sim/control/*`；真机模式只发布 `/control/joint_position` 与 `/control/ee/{left,right}`。

## Task 1：纯函数与失败测试

**文件：**

- 创建：`w1_act-ljl-act_train/xwiz_real_runtime/runtime.py`
- 创建：`tests/xwiz_real_runtime/test_runtime.py`

- [x] 先测试 `0/50/100` 夹爪标量到左右6维手势的映射、范围裁剪和非有限值拒绝。
- [x] 先测试 `100×19` 动作合同、逐帧机身限位和恰好100帧停止条件。
- [x] 先测试机器人状态、ACT 默认姿势误差和观测缓冲门禁。
- [x] 运行测试并确认因函数尚不存在而失败。
- [x] 实现最小纯函数并确认测试全部通过。

## Task 2：模型服务支持仿真/真机标签

**文件：**

- 修改：`w1_act-ljl-act_train/xwiz_act_server/server.py`
- 修改：`tests/xwiz_act_server/test_server.py`

- [x] 先将原“拒绝 real”测试改为接受 `simulation` 和 `real`、拒绝其他值，并确认失败。
- [x] 只修改配置标签校验；两种模式仍调用同一已加载 checkpoint，不在服务端发布 ROS 控制。
- [x] 运行 XWiz 模型服务测试并确认通过。

## Task 3：PC1 按所选任务部署并立即启动

**文件：**

- 创建：`w1_act-ljl-act_train/xwiz_real_runtime/manager_service.py`
- 创建：`tests/xwiz_real_runtime/test_manager_runtime.py`

- [x] 先测试“部署”使用请求中的模型ID和任务ID，并从任务配置读取 `mode` 后立即调用 PC2 `setup_config`。
- [x] 先测试开始仿真时动态设置 `mode=1,data_type=simulation`，开始真机时设置 `mode=2,data_type=real`。
- [x] 强制两种模式均为 `action_horizon=100,max_steps=100,sample_factor=1,chunk_size_threshold=0,home_position=''`。
- [x] 保留现有 ROS graph 看门狗容错逻辑。

## Task 4：PC2 双模式单块客户端

**文件：**

- 创建：`w1_act-ljl-act_train/xwiz_real_runtime/client_service.py`
- 创建：`w1_act-ljl-act_train/xwiz_real_runtime/client_runtime.json`
- 创建：`tests/xwiz_real_runtime/test_client_runtime.py`

- [x] 仿真模式沿用 `/mj_sim/control/*`，真机模式切换真实控制话题。
- [x] 从 Linker L6 反馈按关节名重排并反算左右夹爪开合度，构造模型19维状态。
- [x] 将模型末两维按已确认端点转换成左右6维手命令。
- [x] 真机启动前执行全部安全门禁；失败时返回明确错误且不创建动作流。
- [x] 每帧发布前再次检查有限值、机身限位与帧计数；第100帧后停止。

## Task 5：部署与无动作验证

- [x] 将 `xwiz_real_runtime` 同步到 PC1、PC2 的 `/home/dexforce/w1/w1_act/`。
- [x] 启动本机模型服务、PC1 管理器、PC2 双模式客户端和腕部黑图。
- [x] 不调用 `/inference/deploy` 或 `/inference/start_inference`，验证 `/control/joint_position`、`/control/ee/left`、`/control/ee/right` 均无消息流。
- [x] 调用纯模型 dry-run，确认 checkpoint 输出 `100×19` 有限值。
- [x] 分别验证仿真任务和真机任务配置能够通过部署前检查，但不在自动测试中调用部署服务。
- [x] 等待用户亲自选择任务并点击“部署”；该点击将立即启动所选模式推理。

## 验收与停止

- XWiz 通过独立仿真/真机任务切换；点击所选任务“部署”即启动推理。
- 真机开始时若不在 ACT 默认姿势或机器人异常，界面返回失败。
- 合法真机开始仅执行100帧，然后 PC2 状态回到 Idle。
- 点击“停止推理”、XWiz 退出或看门狗失联时停止客户端动作循环。
- 紧急情况优先使用机器人物理急停；软件停止不能替代物理急停。
