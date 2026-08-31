# W1 ACT-DINOv3 160000 PC2 直连部署设计

日期：2026-08-31

## 1. 目标与隔离边界

为 `/home/wengyikun/workplace/popcorn/outputs/160000` 新建一套 PC2 模型服务、PC1 真机客户端配置和启动器。新链路不得覆盖或复用现有 500000 启动入口的默认模型路径，构建、推送和 dry-run 期间不得启动真机推理或发布动作。

目标数据流为：

```text
PC1 相邻两帧真实绝对 19D 反馈
-> PC2 仅将左右臂关节转换为相对 state
-> state q01/q99 归一化
-> ACT-DINOv3 生成 16x19 action
-> action q01/q99 反归一化
-> PC2 仅将左右臂 action 恢复为绝对目标
-> PC1 接收可直接执行的绝对 16x19
```

PC1 与 PC2 的外部运动边界保持绝对 19D。模型表示转换只在 PC2 的 160000 adapter 中执行一次。

## 2. 采用方案

采用独立协议 v2 和独立启动器：

- 新增 160000 专用 PC2 server 启动器和 systemd 单元。
- 新增 160000 专用 PC1 client 配置、启动器和 systemd 单元。
- 新增相对模型 adapter，不修改现有绝对 500000 adapter 的默认行为。
- v2 observation 明确携带相邻两帧反馈元数据；旧协议不允许启动 160000 真机推理。

不采用以下方案：

- 直接修改共用 500000 入口：容易改变当前已验证链路。
- 由 PC1 完成 q01/q99 和相对 action 转换：模型语义会分散在两台机器，难以证明没有重复转换。

## 3. 19D 合同

字段顺序固定为：

```text
0       WAIST
1..7    LEFT_J1..LEFT_J7
8..9    NECK1..NECK2
10..16  RIGHT_J1..RIGHT_J7
17      LEFT_GRIPPER
18      RIGHT_GRIPPER
```

相对维度仅为 `1..7,10..16`。绝对维度为 `0,8,9,17,18`。所有 state 和 action 必须是有限 `float32`，不得交换字段顺序或单位。

灵巧手标量保持现有真机合同：`0=闭合`、`100=打开`。PC1 发布前继续应用已确认的 `<95 -> 0` 规则；否则限制在 `0..100`。该门限是硬件 adapter 规则，不得写入或改变 q01/q99 统计。

## 4. 相邻真实反馈协议

PC1 的机器人反馈回调维护至少两帧只读快照。每一帧包含完整绝对 19D、单调递增序号、反馈源时间戳和本机接收时间。推理请求必须发送：

```text
previous_state
previous_sequence
previous_timestamp
current_state
current_sequence
current_timestamp
```

`previous_state` 必须是反馈线程中 `current_state` 的真实上一帧，绝不能使用上一次推理请求、上一 chunk 起点、上一条命令或模型预测 action。

PC2 在推理前必须验证：

- 两个 state 均为 19D 有限值。
- `current_sequence == previous_sequence + 1`。
- `current_timestamp > previous_timestamp`。
- 当前反馈没有超过配置的新鲜度上限。
- `current_state` 同时作为 action 恢复绝对值的唯一锚点。

任一条件失败时返回协议错误，不运行模型、不缓存该观测、不返回动作块。首个请求必须等待两帧真实反馈，不能用全零 delta 或复制当前帧代替上一帧。

## 5. PC2 模型 adapter

adapter 从绝对反馈构造模型 state：

```text
model_state[current relative indices] = current_state - previous_state
model_state[current absolute indices] = current_state
```

随后使用 160000 PC2 副本中的 state q01/q99 进行量化归一化：

```text
normalized = clip(2 * (value - q01) / (q99 - q01) - 1, -1, 1)
```

state 和 action 使用各自独立统计：

- `relative_stats/relative_state_q01_q99.json`
- `relative_stats/relative_action_chunk16_q01_q99.json`

模型输出为 `16x19` 归一化 action。postprocessor 先按 action q01/q99 反归一化，再只对 `1..7,10..16` 加上本次请求的 `current_state` 锚点。`0,8,9,17,18` 保持模型反归一化后的绝对值。server 返回绝对 `16x19`，PC1 不再做相对转绝对。

adapter 必须避免调用会把“上一次推理请求”当作 previous state 的在线缓存逻辑；相对 state 由本次 v2 payload 显式计算。

## 6. 图像合同

三个模型 key 和物理来源固定如下：

- `cam_high_right`：头部 `960x540`，上下补黑边为 `960x960`，再 resize 为 `224x224`。
- `cam_hand_left`：物理左腕 `/camera_r`，`640x360` 直接拉伸为 `360x360`，再 resize 为 `224x224`。
- `cam_hand_right`：物理右腕 `/camera_l`，`640x480` 直接拉伸为 `480x480`，再 resize 为 `224x224`。

图像仅转换一次。server adapter 接收已经按上述步骤生成的 RGB 224 图像，不能再次裁剪、补边或交换左右 key。

## 7. 连续 chunk 调度

模型固定输出并执行 16 个策略点，不插值、不重采样、不融合：

- `chunk_size=16`
- `action_horizon=16`
- `sample_factor=1`
- `control_frequency=30 Hz`
- `collect_frequency=30 Hz`
- `chunk_size_threshold=0`

PC1 完整执行 chunk N 的 16 帧后，清空动作队列并等待新的相邻两帧真实反馈和新图像，再请求 chunk N+1。执行期间不得提前请求、并行重规划或复用旧观测。

## 8. 启动器和部署路径

启动器和单元名必须包含 `160000`，并显式绑定：

- PC2 checkpoint：`/home/dexforce/workspace/outputs/160000_pc2`
- PC2 算法源码：`/home/dexforce/workspace/act_dinov3_c8c674b`
- PC2 runtime 依赖：`/home/dexforce/workspace/act_dinov3_runtime_deps`
- PC2 模型端口：`8889`
- PC1 控制端口：`8890`

新启动器不得从环境缺省回退到 `500000_pc2`。checkpoint、统计文件、协议版本、动作 shape 或模型类型不匹配时必须启动失败。

推送后保持 PC2 server 和 PC1 client 停止。现有服务文件和启动器不删除、不覆盖，除非只添加兼容且默认关闭的共享代码路径。

## 9. 故障处理与真机门禁

以下任一条件立即拒绝当前 chunk，并停止后续动作发布：

- 相邻反馈序号或时间戳不合法。
- state、图像或 action shape 不匹配。
- q01/q99 维度、字段名或统计来源不匹配。
- action 包含 NaN/Inf 或超出机身安全限制。
- 反馈、图像、server 连接或模型推理超时。
- 控制源冲突、机器人状态错误或用户停止。

服务启动和模型加载不等于授权运动。首次真机执行仍需单独进行实时门禁检查和用户授权。

## 10. 验证

实现按以下层次验收：

1. 单元测试验证 v2 payload、相邻序号/时间戳拒绝和 19D 字段顺序。
2. 数值测试验证仅臂关节做 `current-previous`，绝对维度原样保留。
3. 数值测试分别验证 state q01/q99 归一化、action q01/q99 反归一化和基于 current state 的绝对恢复。
4. 图像测试验证三个物理来源、左右 key、补边/拉伸步骤及最终 `224x224`。
5. 调度测试验证一个 16 帧 chunk 完整消费后才允许下一请求，且无插值。
6. PC2 CUDA strict load 和单次离线 `16x19` dry-run。
7. PC1/PC2 网络 dry-run，只传观测并接收动作，明确禁用所有动作 publisher。
8. 推送后核对两个新服务均为 inactive、端口未监听、PC1 没有 ACT 命令发布。

完成标准是所有 dry-run 和协议测试通过，新启动器已分别部署到 PC1/PC2 且保持停止，现有 500000 链路未被改变。真机推理启动不属于本次构建步骤。
