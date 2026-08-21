# W1 本机推理与 PC1 安全执行器设计

## 目标

不依赖 PC2 或 XWiz GUI。本机使用 RTX GPU 接收序列化的 W1 观测并运行 `act_popcorn_45w`；PC1 在 ROS 2 Humble/Domain 20 内采集观测、执行安全检查，并根据显式选择发布仿真或真机动作。

## 架构

```text
PC1 ROS 2 Humble
  双目 + 20关节状态 + 双手反馈 + 本地腕部黑图
                         |
                         | TCP framed-pickle :8889
                         v
本机 192.168.20.164      ACT 100x19 推理
                         |
                         | TCP action response
                         v
PC1 安全执行器
  mode=1 -> /mj_sim/control/*
  mode=2 -> /control/joint_position + /control/ee/{left,right}
```

本机不加入机器人 DDS graph，从而避开本机 Jazzy 与机器人 Humble 已出现的 graph 元数据兼容问题。PC1 复用现有 `xwiz_real_runtime.client_service`，而不是新增第二套动作映射。

## 组件

### 本机模型服务

- 继续使用用户服务 `xwiz-act-server.service`，监听 `0.0.0.0:8889`。
- 输入严格为 19 维状态、真实双目和两路 640x360 黑图。
- 输出必须为有限的 `100x19` 绝对动作块。

### PC1 观测与执行服务

- `client_service` 监听 `0.0.0.0:8890`，接收本机 CLI 的配置、停止和状态命令。
- `black_wrist_images` 在 PC1 发布 `/camera_l/color/image_rect_raw` 和 `/camera_r/color/image_rect_raw`，PC2 不再发布腕图。
- 仿真模式只创建 `/mj_sim/control/*` publisher。
- 真机模式只在全部门禁通过后创建和使用真实控制 publisher。

### 本机控制 CLI

提供四个明确命令：`start-sim`、`start-real`、`status`、`stop`。`start-real` 额外要求精确确认串 `EXECUTE_100_REAL_FRAMES`；缺少或错误时不得向 PC1 发送部署命令。

## 真机安全门禁

真机启动必须同时满足：机器人 `Idle`，20 个电机均为 `OP`，所有电机和 server 错误码为零，当前姿势与 `ACT_DEFAULT_20` 最大误差不超过 0.05 rad，双目、双腕黑图、机器人状态和双手反馈均已到齐，模型输出严格为有限的 `100x19`。

动作执行期间每帧重新检查机器人健康状态；17 维身体动作按 W1 关节限位裁剪，两个开合标量映射到 Linker L6 六关节命令。第 100 帧后 client 向模型服务发送 STOP 并回到 Idle。任一异常立即停止，不进行自动复位，也不自动把机器人移动到 ACT 默认姿势。

## 测试与验收

1. 单元测试验证 CLI 模式匹配、真机确认串、配置收敛为单个 100 帧块以及 stop/status 转发。
2. `bash -n` 验证 PC1 生命周期脚本；进程管理只使用 PID 文件和精确命令行，不模糊杀进程。
3. 先启动模型服务、PC1 client 和 PC1 黑图，验证端口、四路观测和黑图像素。
4. 只执行 `start-sim`，确认模型产生一次 `100x19` 动作且真实控制话题没有新增 client publisher。
5. 真机路径只做门禁 dry-run；没有用户新的明确执行命令时不发送 `start-real`。

