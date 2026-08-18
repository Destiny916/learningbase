# 单臂 Piper/Pika ACT 成功恢复数据实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在真机 `kw@192.168.10.82` 上，为现有单臂 7D ACT 策略增加 Pika 双击接管、独立 episode 控制、成功恢复数据导出、混合微调和无接管评估，同时保证 Piper 与机器人侧 Pika 夹爪始终只由 SDK runner 写入。

**Architecture:** ROS2 侧新增一个无硬件写权限的单臂 Pika bridge，只读取 `/pika_pose`、`/gripper/data` 并承接 `/teleop_trigger`，再通过 Unix datagram 把位姿、夹爪和双击事件送给 LeRobot。LeRobot 的 `RobotClient` 是唯一动作仲裁与 SDK 写入者：策略模式执行 ACT 的 7D 绝对关节目标，遥操模式把 Pika 相对位姿经 Piper IK 转成相同的 7D 绝对关节目标，两条路径经过相同限幅后才写硬件。采集先写可审计的原始 episode，结束时立即写结果并原子提交；离线工具只从成功 episode 导出可训练恢复片段，再与基础成功数据物理聚合后微调 ACT。

**Tech Stack:** Python 3、LeRobot `jn_dev`、PyTorch/ACT、gRPC async inference、Piper SDK、Pika SDK、ROS2 Humble (`rclpy`)、Unix domain datagram、NumPy/Pillow、LeRobotDataset v3、pytest、colcon。

---

## 实施边界与固定约定

- 实现目标是远端 `/home/kw/workspace/lerobot` 当前 `jn_dev` 分支和 `/home/kw/workspace/pika_ros`，本文档本身保存在本地 LeRobot 工作区。
- 基线 checkpoint 固定为 `/data/wengyikun/outputs/act_0714_relative_state_chunk16_20260716_152500/train_out/checkpoints/020000/pretrained_model`；基线运行配置为 `/home/kw/workspace/lerobot/runs/act_relative_joint_piper_normal.yaml`。
- ACT 输入输出契约保持不变：`observation.state=[joint_0..joint_5, gripper]`、`action.shape=(7,)`、`right_fisheye` RGB `480x640x3`、30 FPS、chunk 16。
- 机器人侧夹爪继续由 async client 中的 `PikaGripper` 直接调用 Pika SDK；ROS2 `/gripper/data` 是操作者 Pika 的输入，只读。
- `/teleop_trigger` 只切换 `POLICY/TELEOP`。采集命令固定为 `ARM_EPISODE/SUCCESS/FAILURE/DISCARD/STOP_SESSION`，通过另一个本机控制 socket 提交。
- 遥操动作必须先经过 IK 变成 6 关节绝对目标，再附加夹爪宽度，随后复用现有 `ActRelativeJointPiperAdapter` 和 `PiperFollower.send_action()`；不使用 EE 指令冒充 ACT 的关节动作标签。
- 首版 HIL 采集要求 `async_observation: false`，避免后台相机读取与 episode 帧时序分叉。
- 不启动 `teleop_rand_single_piper.launch.py`、任何 `teleop_rand_multi_piper*.launch.py`、Piper ROS controller、`piper_IK.py` 或 `teleop_piper_publish.py`。
- 本方案是 ACT 行为克隆微调，不给 ACT 写数值 reward，也不安装或启动 `lerobot[hilserl]` 的 SAC 训练链。

## Task 1：建立 HIL 状态、事件与配置契约

**Files:**

- Create: `/home/kw/workspace/lerobot/src/lerobot/async_inference/hil_recovery/types.py`
- Create: `/home/kw/workspace/lerobot/src/lerobot/async_inference/hil_recovery/__init__.py`
- Modify: `/home/kw/workspace/lerobot/src/lerobot/async_inference/configs.py`
- Test: `/home/kw/workspace/lerobot/tests/async_inference/test_hil_recovery_types.py`
- Test: `/home/kw/workspace/lerobot/tests/async_inference/test_hil_recovery_config.py`

- [ ] **Step 1：先写枚举和状态转换失败测试**

```python
def test_collection_commands_have_explicit_legal_transitions():
    machine = CollectionStateMachine()
    with pytest.raises(IllegalCollectionTransition):
        machine.apply(CollectionCommand.SUCCESS)
    assert machine.apply(CollectionCommand.ARM_EPISODE).state is CollectionState.RECORDING
    assert machine.apply(CollectionCommand.SUCCESS).outcome is EpisodeOutcome.SUCCESS
    assert machine.state is CollectionState.IDLE


def test_stop_session_discards_open_episode_before_stopping():
    machine = CollectionStateMachine()
    machine.apply(CollectionCommand.ARM_EPISODE)
    transition = machine.apply(CollectionCommand.STOP_SESSION)
    assert transition.outcome is EpisodeOutcome.DISCARDED
    assert transition.session_stopped is True
```

- [ ] **Step 2：运行测试并确认红灯**

Run: `cd /home/kw/workspace/lerobot && pytest -q tests/async_inference/test_hil_recovery_types.py`

Expected: collection 类型尚不存在，测试以 import error 失败。

- [ ] **Step 3：实现不可变事件类型和纯状态机**

```python
class ControlMode(StrEnum):
    POLICY = "policy"
    TELEOP = "teleop"


class EpisodeOutcome(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    SAFETY_STOP = "safety_stop"
    DISCARDED = "discarded"


class CollectionCommand(StrEnum):
    ARM_EPISODE = "arm_episode"
    SUCCESS = "success"
    FAILURE = "failure"
    DISCARD = "discard"
    STOP_SESSION = "stop_session"


@dataclass(frozen=True)
class PikaSample:
    sequence: int
    monotonic_ns: int
    position_m: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    gripper_width_m: float


@dataclass(frozen=True)
class ToggleRequest:
    toggle_id: int
    monotonic_ns: int
```

状态机只接受上述五个命令；超时和安全停止调用专用 `finish(EpisodeOutcome)`，不能伪装成键盘命令。

- [ ] **Step 4：给 `RobotClientConfig` 增加并校验 HIL 字段**

```yaml
hil_recovery_enabled: true
pika_bridge_socket: /tmp/lerobot-pika-bridge.sock
pika_runner_socket: /tmp/lerobot-pika-runner.sock
collection_control_socket: /tmp/lerobot-recovery-control.sock
pika_sample_timeout_s: 0.20
teleop_toggle_debounce_s: 0.35
teleop_translation_scale: 1.0
teleop_rotation_scale: 1.0
teleop_ik_urdf: /home/kw/workspace/pika_ros/src/PikaAnyArm/piper/piper_ros/piper_description/urdf/piper_description.urdf
recovery_raw_root: /data/wengyikun/lerobot/recovery_raw/grasp_bread_single_arm_v1
policy_checkpoint_id: act_0714_relative_state_chunk16_step020000
task_id: grasp_bread_single_arm
episode_timeout_s: 60.0
```

校验规则：只允许 `robot.type=piper_follower`、`policy_type=act`、`act_relative_joint_piper_adapter=true`、`actions_per_chunk=16`、`async_observation=false`、`execute_robot_actions` 为显式布尔值，并拒绝重复 socket 路径、非正 timeout、空 task/checkpoint id。

- [ ] **Step 5：运行定向测试**

Run: `cd /home/kw/workspace/lerobot && pytest -q tests/async_inference/test_hil_recovery_types.py tests/async_inference/test_hil_recovery_config.py`

Expected: 全部通过。

- [ ] **Step 6：提交 Task 1**

```bash
cd /home/kw/workspace/lerobot
git add src/lerobot/async_inference/hil_recovery src/lerobot/async_inference/configs.py tests/async_inference/test_hil_recovery_types.py tests/async_inference/test_hil_recovery_config.py
git commit -m "feat(hil): define recovery runtime contracts"
```

## Task 2：新增只读 ROS2 Pika bridge，并隔离旧 capture service

**Files:**

- Create: `/home/kw/workspace/pika_ros/src/pika_hil_bridge/package.xml`
- Create: `/home/kw/workspace/pika_ros/src/pika_hil_bridge/setup.py`
- Create: `/home/kw/workspace/pika_ros/src/pika_hil_bridge/setup.cfg`
- Create: `/home/kw/workspace/pika_ros/src/pika_hil_bridge/resource/pika_hil_bridge`
- Create: `/home/kw/workspace/pika_ros/src/pika_hil_bridge/pika_hil_bridge/__init__.py`
- Create: `/home/kw/workspace/pika_ros/src/pika_hil_bridge/pika_hil_bridge/node.py`
- Create: `/home/kw/workspace/pika_ros/src/pika_hil_bridge/launch/single_pika_hil_bridge.launch.py`
- Create: `/home/kw/workspace/pika_ros/src/pika_hil_bridge/launch/single_pika_hil_input.launch.py`
- Test: `/home/kw/workspace/pika_ros/src/pika_hil_bridge/test/test_protocol.py`
- Test: `/home/kw/workspace/pika_ros/src/pika_hil_bridge/test/test_node_contract.py`

- [ ] **Step 1：先写 bridge 协议测试**

测试必须证明：PoseStamped 和 `data_msgs/msg/Gripper.distance` 被合并成带序号的 `pika_sample`；一次 Trigger 只发一个 `toggle_request`；350 ms 内重复请求被拒绝；没有 runner ACK 时服务返回 `success=false`；代码中不存在 Piper 控制 topic 发布者和 capture service 服务端。

```python
def test_bridge_serializes_single_arm_sample():
    payload = encode_sample(sequence=7, pose=POSE, gripper_distance_mm=42.0, monotonic_ns=99)
    assert payload == {
        "version": 1,
        "kind": "pika_sample",
        "sequence": 7,
        "monotonic_ns": 99,
        "position_m": [0.1, 0.2, 0.3],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        "gripper_width_m": 0.042,
    }
```

- [ ] **Step 2：运行测试并确认红灯**

Run: `cd /home/kw/workspace/pika_ros && python3 -m pytest -q src/pika_hil_bridge/test`

Expected: `pika_hil_bridge` 尚不存在。

- [ ] **Step 3：实现 ROS2 节点和 Unix datagram 协议**

节点只包含：

```text
subscribe  /pika_pose      geometry_msgs/msg/PoseStamped
subscribe  /gripper/data   data_msgs/msg/Gripper
serve      /teleop_trigger std_srvs/srv/Trigger
send IPC   pika_sample | toggle_request
recv IPC   toggle_ack
```

`toggle_ack` 必须包含相同 `toggle_id`、`accepted`、`mode` 和 `reason`。Trigger 回调只有收到匹配 ACK 才返回成功。退出时删除 bridge 自己绑定的 socket，不删除 runner socket。

- [ ] **Step 4：提供 bridge-only 测试 launch 和无相机的正式输入 launch**

`single_pika_hil_bridge.launch.py` 只启动 `pika_hil_bridge`，并把两个 socket、topic 和 debounce 暴露为参数。`single_pika_hil_input.launch.py` 只组合现有 `pika_single_locator`、单臂 `serial_gripper_imu` 和该 bridge，不启动 RealSense、Pika fisheye、Piper FK/IK/controller。这样 LeRobot 继续独占读取 `right_fisheye` 设备，也不需要当前单臂 sensor 脚本中并不存在的 `--no-depth/--no-fisheye` 参数。两个 launch 都不能 include `teleop_rand_single_piper.launch.py`。

- [ ] **Step 5：构建和测试 ROS2 包**

Run: `cd /home/kw/workspace/pika_ros && source /opt/ros/humble/setup.bash && colcon build --packages-select pika_hil_bridge --symlink-install`

Expected: 包构建成功。

Run: `cd /home/kw/workspace/pika_ros && source install/setup.bash && colcon test --packages-select pika_hil_bridge --event-handlers console_direct+ && colcon test-result --verbose`

Expected: 0 failures。

- [ ] **Step 6：以无数据采集节点的最小拓扑验证服务隔离**

Run: `ros2 launch pika_hil_bridge single_pika_hil_input.launch.py serial_port:=/dev/ttyUSB50`

Run: `ros2 service list | sort`

Expected: 存在 `/pika_pose`、`/gripper/data` 和 `/teleop_trigger`，不存在 `/data_tools_dataCapture/capture_service`；Pika 双击对不存在的旧 capture service 的请求不能改变 LeRobot collection 状态，也没有进程占用 LeRobot 使用的 fisheye 设备。

- [ ] **Step 7：提交 Task 2**

```bash
cd /home/kw/workspace/pika_ros
git add src/pika_hil_bridge
git commit -m "feat(hil): add read-only single-arm Pika bridge"
```

## Task 3：实现 Pika IPC 接收、双击模式状态机和单写者锁

**Files:**

- Create: `/home/kw/workspace/lerobot/src/lerobot/async_inference/hil_recovery/ipc.py`
- Create: `/home/kw/workspace/lerobot/src/lerobot/async_inference/hil_recovery/mode_controller.py`
- Create: `/home/kw/workspace/lerobot/src/lerobot/async_inference/hil_recovery/writer_guard.py`
- Test: `/home/kw/workspace/lerobot/tests/async_inference/test_hil_ipc.py`
- Test: `/home/kw/workspace/lerobot/tests/async_inference/test_hil_mode_controller.py`
- Test: `/home/kw/workspace/lerobot/tests/async_inference/test_hardware_writer_guard.py`

- [ ] **Step 1：先写模式切换和单写者测试**

覆盖：无 Pika 样本、样本过期、无 SDK 位姿反馈、重复 toggle id 都拒绝；合法 `POLICY→TELEOP` 保存两端参考位姿；`TELEOP→POLICY` 生成 `require_fresh_policy_chunk=true`；第二个进程不能获得同一 `can1` 锁。

- [ ] **Step 2：运行测试并确认红灯**

Run: `cd /home/kw/workspace/lerobot && pytest -q tests/async_inference/test_hil_ipc.py tests/async_inference/test_hil_mode_controller.py tests/async_inference/test_hardware_writer_guard.py`

Expected: 新模块尚不存在。

- [ ] **Step 3：实现非阻塞 IPC 收包器**

`PikaIpcReceiver` 使用后台线程读取 datagram，仅在锁内替换 latest sample/toggle；控制循环调用 `drain_events()` 和 `latest_sample(max_age_s)`，绝不在 30 Hz 主循环中等待 ROS。

- [ ] **Step 4：实现模式状态机**

进入 TELEOP 时按以下顺序执行：校验新鲜 Pika 样本 → 读取 SDK 当前末端和关节状态 → 保存 `T_pika_start/T_piper_start` → 清空 ACT queue → 增加 `toggle_id` → ACK。退出时：停止采用 Pika 动作 → 清空 queue → 清除 in-flight request → 设置 `must_go` → 记录 fresh-chunk 下界 → ACK。TELEOP 中样本过期时保持当前关节目标并报告 `teleop_input_stale`，不自动恢复旧策略 chunk。

- [ ] **Step 5：实现 `fcntl.flock` 单写者守卫**

锁文件固定为 `/tmp/lerobot-piper-can1.lock`，内容写 PID、启动时间和配置路径；必须在构造 `PiperFollower` 之前获得，正常/异常退出均释放。锁冲突直接终止，不能降级成无锁运行。

- [ ] **Step 6：运行测试**

Run: `cd /home/kw/workspace/lerobot && pytest -q tests/async_inference/test_hil_ipc.py tests/async_inference/test_hil_mode_controller.py tests/async_inference/test_hardware_writer_guard.py`

Expected: 全部通过。

- [ ] **Step 7：提交 Task 3**

```bash
cd /home/kw/workspace/lerobot
git add src/lerobot/async_inference/hil_recovery tests/async_inference/test_hil_ipc.py tests/async_inference/test_hil_mode_controller.py tests/async_inference/test_hardware_writer_guard.py
git commit -m "feat(hil): add teleop mode arbitration"
```

## Task 4：实现与现有单臂遥操一致的 Pika→Piper 关节动作适配

**Files:**

- Create: `/home/kw/workspace/lerobot/src/lerobot/async_inference/adapters/pika_teleop_piper.py`
- Modify: `/home/kw/workspace/lerobot/src/lerobot/async_inference/adapters/__init__.py`
- Test: `/home/kw/workspace/lerobot/tests/async_inference/test_pika_teleop_piper.py`
- Create: `/home/kw/workspace/lerobot/tests/async_inference/verify_pika_teleop_ik_parity.py`

- [ ] **Step 1：用 fake IK 写变换、限幅和失败测试**

```python
def test_relative_pose_uses_existing_teleop_transform_order():
    adapter.begin(pika_pose=PIKA_START, piper_pose=PIPER_START, joints=JOINTS)
    action = adapter.compute(sample=PIKA_MOVED, current_joints=JOINTS)
    expected_target = T_PIPER_START @ np.linalg.inv(T_PIKA_START) @ T_PIKA_MOVED
    fake_ik.assert_called_once_with(expected_target, seed=JOINTS)
    assert tuple(action) == (*SOLVED_JOINTS, PIKA_MOVED.gripper_width_m)
```

另测 quaternion 归一化、旋转顺序 `Rz*Ry*Rx`、平移/旋转 scale、IK 无解、关节跳变、夹爪范围、输入过期。

- [ ] **Step 2：运行测试并确认红灯**

Run: `cd /home/kw/workspace/lerobot && pytest -q tests/async_inference/test_pika_teleop_piper.py`

Expected: adapter 尚不存在。

- [ ] **Step 3：从现有 `Arm_IK` 提取纯计算后端**

只复用 `/home/kw/workspace/pika_ros/src/PikaAnyArm/piper/pika_remote_piper/scripts/piper_IK.py` 中的模型、关节限制和求解逻辑；不得导入 `rospy/rclpy`、创建 publisher 或调用 Piper SDK。URDF 路径来自 config，seed 必须使用 SDK 当前 6 关节反馈。

- [ ] **Step 4：实现相同的相对位姿映射并输出 7D 绝对关节动作**

```python
target = T_piper_start @ np.linalg.inv(T_pika_start) @ T_pika_current
joint_target = ik.solve(target, seed=current_joint_state)
action_7d = np.concatenate([joint_target, [leader_gripper_width_m]])
```

适配器只返回 7D tensor，不直接写硬件；后续必须复用 `ActRelativeJointPiperAdapter.convert_action()`，保证 ACT 和人类动作使用同一单位、key、单步限幅和夹爪拆分逻辑。

- [ ] **Step 5：运行离线 IK parity 检查**

`verify_pika_teleop_ik_parity.py` 对 20 个固定、可达、无硬件的位姿同时调用旧 IK 与新纯后端。关节解最大差必须 `<1e-4 rad`，FK 位置差 `<1 mm`，姿态差 `<0.5°`。

Run: `cd /home/kw/workspace/lerobot && python tests/async_inference/verify_pika_teleop_ik_parity.py`

Expected: 打印 `20/20 parity cases passed`，不打开 CAN。

- [ ] **Step 6：运行单测并提交**

Run: `cd /home/kw/workspace/lerobot && pytest -q tests/async_inference/test_pika_teleop_piper.py tests/async_inference/test_act_relative_joint_piper.py`

```bash
cd /home/kw/workspace/lerobot
git add src/lerobot/async_inference/adapters tests/async_inference/test_pika_teleop_piper.py tests/async_inference/verify_pika_teleop_ik_parity.py
git commit -m "feat(hil): convert Pika teleop to Piper joint targets"
```

## Task 5：实现独立 collection 控制和原子原始 episode 存储

**Files:**

- Create: `/home/kw/workspace/lerobot/src/lerobot/async_inference/hil_recovery/episode_store.py`
- Create: `/home/kw/workspace/lerobot/src/lerobot/async_inference/hil_recovery/collection_server.py`
- Create: `/home/kw/workspace/lerobot/src/lerobot/scripts/lerobot_recovery_control.py`
- Modify: `/home/kw/workspace/lerobot/pyproject.toml`
- Test: `/home/kw/workspace/lerobot/tests/async_inference/test_recovery_episode_store.py`
- Test: `/home/kw/workspace/lerobot/tests/async_inference/test_collection_server.py`
- Test: `/home/kw/workspace/lerobot/tests/scripts/test_lerobot_recovery_control.py`

- [ ] **Step 1：先写原子提交和命令测试**

覆盖：`ARM_EPISODE` 前不写帧；启动后第一帧序号为 0；SUCCESS/FAILURE 立即停止追加；进程崩溃留下 `.incomplete`；只有 `os.replace()` 后出现 `complete/episode-*`；DISCARD 保留审计但不标 complete；超时和 safety stop 标签正确；Pika toggle 不触发 collection 命令。

- [ ] **Step 2：运行测试并确认红灯**

Run: `cd /home/kw/workspace/lerobot && pytest -q tests/async_inference/test_recovery_episode_store.py tests/async_inference/test_collection_server.py tests/scripts/test_lerobot_recovery_control.py`

Expected: 新模块和 CLI 尚不存在。

- [ ] **Step 3：实现有背压的磁盘写入器**

每个 episode 先写：

```text
recovery_raw/grasp_bread_single_arm_v1/
  .recording/episode-<uuid>/
    manifest.pending.json
    frames.jsonl
    images/right_fisheye/000000.png
  complete/episode-<uuid>/...
  rejected/episode-<uuid>/...
  incomplete/episode-<uuid>/...
```

帧 writer 使用有界队列和单后台线程，图像写 PNG，`frames.jsonl` 每行含 observation state、7D `action`、`control_mode`、`is_intervention`、start/end 标记、toggle id、单调时钟、SDK 返回动作和完整性字段。队列满或写盘错误立即触发 `safety_stop`，不能悄悄丢帧。

- [ ] **Step 4：结束时立即写不可变结果并原子落盘**

结束命令先停止接收新帧，再 drain writer，校验帧连续、图像数量、7D shape、时间单调和至少一帧，最后写带 SHA256 的 `manifest.json` 并 fsync，再把临时目录原子 rename 到 `complete` 或 `rejected`。`success/failure/timeout/safety_stop/discarded` 必须在 episode 结束时确定。

- [ ] **Step 5：实现独立控制 socket 与 CLI**

```bash
lerobot-recovery-control --socket /tmp/lerobot-recovery-control.sock arm_episode
lerobot-recovery-control --socket /tmp/lerobot-recovery-control.sock success
lerobot-recovery-control --socket /tmp/lerobot-recovery-control.sock failure
lerobot-recovery-control --socket /tmp/lerobot-recovery-control.sock discard
lerobot-recovery-control --socket /tmp/lerobot-recovery-control.sock stop_session
```

CLI 打印 `accepted/state/episode_id/outcome`，非零退出表示命令未执行。每条命令保存操作者 `$USER`、PID、来源和时间；不复用 `/teleop_trigger`。

- [ ] **Step 6：运行测试并提交**

Run: `cd /home/kw/workspace/lerobot && pytest -q tests/async_inference/test_recovery_episode_store.py tests/async_inference/test_collection_server.py tests/scripts/test_lerobot_recovery_control.py`

```bash
cd /home/kw/workspace/lerobot
git add src/lerobot/async_inference/hil_recovery src/lerobot/scripts/lerobot_recovery_control.py tests/async_inference/test_recovery_episode_store.py tests/async_inference/test_collection_server.py tests/scripts/test_lerobot_recovery_control.py pyproject.toml
git commit -m "feat(hil): record outcome-labeled recovery episodes"
```

## Task 6：把仲裁、真实动作和采集接入 async RobotClient

**Files:**

- Modify: `/home/kw/workspace/lerobot/src/lerobot/async_inference/robot_client.py`
- Create: `/home/kw/workspace/lerobot/runs/act_relative_joint_piper_recovery_collect.yaml`
- Modify: `/home/kw/workspace/lerobot/tests/async_inference/test_robot_client.py`
- Create: `/home/kw/workspace/lerobot/tests/async_inference/test_robot_client_hil_recovery.py`

- [ ] **Step 1：先写端到端 fake robot 测试**

测试序列固定为：POLICY 执行两帧 → ARM_EPISODE → POLICY 一帧 → toggle TELEOP → 人工两帧 → toggle POLICY → 丢弃旧 chunk → 新 request id 的策略一帧 → SUCCESS。断言硬件 writer 调用顺序、记录的 7D 动作等于 `send_action()` 返回值、接管边界、toggle id、最终 outcome 和 episode 原子目录。

- [ ] **Step 2：运行测试并确认红灯**

Run: `cd /home/kw/workspace/lerobot && pytest -q tests/async_inference/test_robot_client_hil_recovery.py`

Expected: RobotClient 尚未接入 HIL 组件。

- [ ] **Step 3：在连接硬件前获取 writer lock，并统一动作执行入口**

重构出 `_execute_7d_action(action, source)`：ACT tensor 和 Pika IK tensor 都先经过 `_act_relative_joint_adapter.convert_action()`，再 `_split_pika_gripper_action()`，最后依次调用 `robot.send_action()` 与 `PikaGripper.execute_width()`。只有该函数能在 HIL 模式写硬件并返回实际发送的完整 7D action。

- [ ] **Step 4：在 30 Hz 控制循环中加入模式仲裁**

每 tick 只做一次可记录 observation capture。POLICY 时消费合格 ACT chunk；TELEOP 时不发送 policy observation、不执行 policy queue，只用最新 Pika sample 生成 7D target。退出 TELEOP 后必须丢弃退出前收到的所有 action，并等待退出后 request id 对应的新 chunk。

- [ ] **Step 5：接入 episode recorder 与自动结束**

只在 `CollectionState.RECORDING` 时追加 `{observation, performed_action, mode metadata}`。计时超过 60 s 自动 `timeout`；SDK/IK/IPC/写盘错误或安全保护触发 `safety_stop`；SIGINT/STOP_SESSION 在录制中执行 `discarded`。reset 和物体摆放发生在 `ARM_EPISODE` 之前，不会进数据。

- [ ] **Step 6：创建采集配置**

复制当前 normal YAML 的基线 checkpoint、相机、CAN、夹爪和服务器参数，加入 Task 1 固定 HIL 字段。`execute_robot_actions` 默认仍为 `false`；只有完成 Task 10 的人工安全门后，通过 CLI 显式覆盖成 true。

- [ ] **Step 7：运行回归测试**

Run: `cd /home/kw/workspace/lerobot && pytest -q tests/async_inference/test_robot_client.py tests/async_inference/test_robot_client_hil_recovery.py tests/async_inference/test_act_relative_joint_piper.py tests/async_inference/test_e2e.py`

Expected: 全部通过，原 normal/dry-run 路径行为不变。

- [ ] **Step 8：提交 Task 6**

```bash
cd /home/kw/workspace/lerobot
git add src/lerobot/async_inference/robot_client.py runs/act_relative_joint_piper_recovery_collect.yaml tests/async_inference/test_robot_client.py tests/async_inference/test_robot_client_hil_recovery.py
git commit -m "feat(hil): integrate recovery collection into async client"
```

## Task 7：从成功 episode 确定性导出 ACT 恢复数据

**Files:**

- Create: `/home/kw/workspace/lerobot/src/lerobot/scripts/lerobot_prepare_act_recovery_dataset.py`
- Test: `/home/kw/workspace/lerobot/tests/scripts/test_prepare_act_recovery_dataset.py`

- [ ] **Step 1：先写成功筛选、边界和确定性测试**

构造自主成功、一次接管成功、多次接管成功、失败、超时、安全停止、discard、incomplete 共 8 类 fixture。断言只有成功回合导出；接管片段从 `intervention_start-15` 帧保留上下文；纯策略偏离帧 `trainable=false`；从首个人工动作开始 `trainable=true`；重叠片段合并；同输入两次输出 manifest SHA256 一致。

- [ ] **Step 2：运行测试并确认红灯**

Run: `cd /home/kw/workspace/lerobot && pytest -q tests/scripts/test_prepare_act_recovery_dataset.py`

Expected: 导出脚本尚不存在。

- [ ] **Step 3：实现两阶段导出**

第一阶段生成审计切片 manifest，保留 15 帧（0.5 秒）接管前观测上下文，但把这些导致偏离的 policy action 标为不可训练。第二阶段生成 LeRobotDataset v3 时，只把 `trainable=true` 的帧作为样本起点；上下文仍保留在切片审计目录，不把错误 policy action 当标签。每个导出帧的 `observation.*` 和 7D action 必须与基线 metadata 精确一致。

- [ ] **Step 4：增加兼容性硬门**

脚本从固定基线 checkpoint 的 `config.json`、pre/postprocessor 和训练 metadata 自动解析基础数据契约；若无法解析基础 dataset repo/root、camera key、action names、FPS 或 chunk size，直接失败并打印缺失字段，不靠人工猜测。

- [ ] **Step 5：运行导出测试和 CLI smoke test**

```bash
cd /home/kw/workspace/lerobot
python -m lerobot.scripts.lerobot_prepare_act_recovery_dataset \
  --raw-root /data/wengyikun/lerobot/recovery_raw/grasp_bread_single_arm_v1 \
  --base-checkpoint /data/wengyikun/outputs/act_0714_relative_state_chunk16_20260716_152500/train_out/checkpoints/020000/pretrained_model \
  --output-root /data/wengyikun/lerobot/datasets/grasp_bread_recovery_v1 \
  --repo-id local/grasp_bread_recovery_v1 \
  --context-frames 15 \
  --seed 20260721 \
  --validate-only
```

Expected: 在还没有完整采集时报告 0 个可导出 episode，但契约校验通过；不得创建伪训练数据。

- [ ] **Step 6：提交 Task 7**

```bash
cd /home/kw/workspace/lerobot
git add src/lerobot/scripts/lerobot_prepare_act_recovery_dataset.py tests/scripts/test_prepare_act_recovery_dataset.py
git commit -m "feat(data): export successful ACT recovery segments"
```

## Task 8：构建基础数据 + 恢复数据的可复现混合训练集

**Files:**

- Create: `/home/kw/workspace/lerobot/src/lerobot/scripts/lerobot_build_act_recovery_mix.py`
- Test: `/home/kw/workspace/lerobot/tests/scripts/test_build_act_recovery_mix.py`

- [ ] **Step 1：先写混合比例和兼容性测试**

测试要求：基础成功数据始终存在；recovery 为空时拒绝产出候选集；失败 raw episode 永不出现；相机/action/schema 不一致立即失败；固定 seed 下 episode 复制/抽样顺序一致；最终训练帧目标占比为基础 80%、恢复 20%，误差不超过一个 recovery episode 的帧数。

- [ ] **Step 2：运行测试并确认红灯**

Run: `cd /home/kw/workspace/lerobot && pytest -q tests/scripts/test_build_act_recovery_mix.py`

Expected: 混合脚本尚不存在。

- [ ] **Step 3：物理生成单一 LeRobotDataset v3**

当前 `make_dataset()` 明确禁用 `MultiLeRobotDataset`，所以脚本使用现有 `aggregate_datasets()` 思路生成单一输出。它从基线 checkpoint 自动找到原始成功数据，以 episode 为单位确定性复制较小的 recovery 数据，接近 80:20 的训练帧比例，并生成：

```text
/data/wengyikun/lerobot/datasets/grasp_bread_base_recovery_mix_v1/
  meta/...
  data/...
  videos/...
  mix_manifest.json
```

`mix_manifest.json` 保存基础/恢复源版本、episode id、SHA256、实际帧比例、seed、基线 checkpoint 和生成命令。

- [ ] **Step 4：运行真实数据 validate-only**

```bash
cd /home/kw/workspace/lerobot
python -m lerobot.scripts.lerobot_build_act_recovery_mix \
  --base-checkpoint /data/wengyikun/outputs/act_0714_relative_state_chunk16_20260716_152500/train_out/checkpoints/020000/pretrained_model \
  --recovery-root /data/wengyikun/lerobot/datasets/grasp_bread_recovery_v1 \
  --output-root /data/wengyikun/lerobot/datasets/grasp_bread_base_recovery_mix_v1 \
  --repo-id local/grasp_bread_base_recovery_mix_v1 \
  --base-fraction 0.80 \
  --seed 20260721 \
  --validate-only
```

Expected: 打印源 schema、episode/frame 数和预计实际比例；任何失败/不完整 episode 计数必须为 0。

- [ ] **Step 5：运行测试并提交**

Run: `cd /home/kw/workspace/lerobot && pytest -q tests/scripts/test_build_act_recovery_mix.py tests/datasets/test_aggregate.py`

```bash
cd /home/kw/workspace/lerobot
git add src/lerobot/scripts/lerobot_build_act_recovery_mix.py tests/scripts/test_build_act_recovery_mix.py
git commit -m "feat(data): build reproducible ACT recovery mix"
```

## Task 9：增加保守 ACT 微调配置和无接管评估门

**Files:**

- Create: `/home/kw/workspace/lerobot/runs/act_relative_joint_piper_recovery_finetune.yaml`
- Create: `/home/kw/workspace/lerobot/src/lerobot/scripts/lerobot_compare_recovery_eval.py`
- Create: `/home/kw/workspace/lerobot/runs/eval/grasp_bread_single_arm_v1.yaml`
- Test: `/home/kw/workspace/lerobot/tests/scripts/test_compare_recovery_eval.py`

- [ ] **Step 1：先写评估门测试**

```python
def test_candidate_passes_only_when_baseline_preserved_and_recovery_improves():
    baseline = Metrics(original_success=8, original_trials=10, recovery_success=2, recovery_trials=10, safety=0)
    candidate = Metrics(original_success=8, original_trials=10, recovery_success=6, recovery_trials=10, safety=0)
    assert compare(baseline, candidate).promote is True
```

另测原任务下降、恢复不升、安全事件增加、trial 缺失都拒绝 promotion。

- [ ] **Step 2：运行测试并确认红灯**

Run: `cd /home/kw/workspace/lerobot && pytest -q tests/scripts/test_compare_recovery_eval.py`

Expected: 评估工具尚不存在。

- [ ] **Step 3：创建固定微调配置**

从基线 020000 checkpoint 初始化，保持原 ACT 输入输出和损失不变；训练数据固定为 mix v1；首轮使用 learning rate `1e-5`、5,000 steps、每 1,000 steps 保存、seed `20260721`、不覆盖基线输出目录。输出为 `/data/wengyikun/outputs/act_grasp_bread_recovery_v1/`。

- [ ] **Step 4：创建单臂评估清单和结果格式**

`grasp_bread_single_arm_v1.yaml` 固定两组各 10 回合：10 个原任务初始状态、10 个已记录偏离状态。评估配置必须设 `hil_recovery_enabled=false`，并由独立 collection CLI 在每回合结束即时标注 success/failure/timeout/safety_stop。Pika 双击在评估期间不启用，任何人工动作都使该回合无效并重录。

- [ ] **Step 5：实现比较报告与 promotion 文件**

```bash
python -m lerobot.scripts.lerobot_compare_recovery_eval \
  --baseline runs/eval/results/act_0714_step020000.json \
  --candidate runs/eval/results/act_grasp_bread_recovery_v1_step005000.json \
  --output runs/eval/results/comparison_recovery_v1.json
```

只有原任务成功率不降、恢复成功率提高、安全事件不增加时才写 `promotion_approved: true`；工具不复制、不删除、不覆盖任何 checkpoint。

- [ ] **Step 6：运行测试并提交**

Run: `cd /home/kw/workspace/lerobot && pytest -q tests/scripts/test_compare_recovery_eval.py`

```bash
cd /home/kw/workspace/lerobot
git add runs/act_relative_joint_piper_recovery_finetune.yaml runs/eval/grasp_bread_single_arm_v1.yaml src/lerobot/scripts/lerobot_compare_recovery_eval.py tests/scripts/test_compare_recovery_eval.py
git commit -m "feat(train): gate ACT recovery fine-tuning"
```

## Task 10：执行无硬件验证和分级真机验收

**Files:**

- Create: `/home/kw/workspace/lerobot/docs/real_robot/single_arm_act_recovery_runbook.md`
- Modify: `/home/kw/workspace/lerobot/runs/act_relative_joint_piper_recovery_collect.yaml`

- [ ] **Step 1：运行两个仓库的完整相关测试**

```bash
cd /home/kw/workspace/lerobot
pytest -q tests/async_inference tests/scripts/test_prepare_act_recovery_dataset.py tests/scripts/test_build_act_recovery_mix.py tests/scripts/test_compare_recovery_eval.py
```

Expected: 0 failures。

```bash
cd /home/kw/workspace/pika_ros
source /opt/ros/humble/setup.bash
source install/setup.bash
colcon test --packages-select pika_hil_bridge --event-handlers console_direct+
colcon test-result --verbose
```

Expected: 0 failures。

- [ ] **Step 2：检查禁止并发的进程和 ROS 节点**

```bash
pgrep -af 'piper_ctrl|piper_IK|teleop_piper_publish|teleop_rand_.*piper'
ros2 node list
```

Expected: `pgrep` 无输出；ROS node list 只有 sensor/locator、相机、`serial_gripper_imu` 和 `pika_hil_bridge`，没有 Piper controller/IK。

- [ ] **Step 3：网络与传感器 dry-run，不使能、不发动作**

启动 sensor + bridge 后，运行 recovery collect 配置并覆盖：

```bash
python -m lerobot.async_inference.robot_client \
  --config_path=runs/act_relative_joint_piper_recovery_collect.yaml \
  --execute_robot_actions=false \
  --robot.enable_on_connect=false \
  --pika_enable_on_connect=false
```

Expected: ACT observation/response正常；双击只在 POLICY/TELEOP 间切换；日志显示 dry-run 7D 动作；collection CLI 可完成一次 success 和一次 failure，只有 success 可被导出；CAN 与两个夹爪均未收到使能或动作命令。

- [ ] **Step 4：经操作者明确确认后做低速小范围真机检查**

这一步是安全门，执行者不能自行越过。确认急停、工作区净空、`can1` 唯一 writer、Pika 输入方向、夹爪端口后，才把 `execute_robot_actions=true`，并暂时设 `move_spd_rate_ctrl=10`、`act_relative_joint_max_step_rad=0.02`、`act_relative_gripper_max_step_m=0.005`。只验证一次进入 TELEOP、每轴小位移、退出后等待 fresh ACT chunk；出现方向错误、跳变、IK 失败或 stale input 立即急停并记录 `safety_stop`。

- [ ] **Step 5：按正式 episode 生命周期采集三种样本**

依次验证：无接管成功；策略偏离→双击接管→人工直接完成→SUCCESS；策略偏离→接管恢复到可继续状态→双击交还→策略完成→SUCCESS。另做一次主动 FAILURE，确认它只留在 rejected 审计目录。每回合 reset 后才执行 `ARM_EPISODE`，结束后立即标注。

- [ ] **Step 6：运行数据完整性和导出验证**

Expected：每个 complete episode 有 immutable outcome、checkpoint、task id、连续 frame、实际 7D action 和接管边界；failure/timeout/safety/discard/incomplete 的训练导出数均为 0；混合集始终包含基础示范。

- [ ] **Step 7：完成训练、20+20 无接管评估和回退判断**

先评估基线，再训练 candidate，再按同一 20 回合清单评估 candidate。仅当比较工具输出 `promotion_approved: true` 才把 candidate 记录为候选；基线 checkpoint 始终原地保留，部署配置的 checkpoint 路径不由脚本自动修改。

- [ ] **Step 8：写 runbook 并提交最终文档**

runbook 必须给出三个终端的启动/停止顺序、collection CLI、双击含义、禁止 launch、故障标签和紧急停止步骤。

```bash
cd /home/kw/workspace/lerobot
git add docs/real_robot/single_arm_act_recovery_runbook.md runs/act_relative_joint_piper_recovery_collect.yaml
git commit -m "docs(hil): add single-arm recovery runbook"
```

## 最终完成定义

- 单元、集成、ROS2 包测试全部通过，旧 async ACT 路径无回归。
- Pika 双击只改变动作来源，不能创建、结束或标注 episode。
- POLICY 与 TELEOP 都只通过同一个 SDK 动作入口写 Piper/Pika，第二 writer 被锁拒绝。
- 遥操保存的 action 是经过 IK、限幅后实际提交给 SDK 的 7D 绝对关节/夹爪动作，不是 ACT 被覆盖的预测，也不是 EE pose。
- 每个 episode 在结束时已有 outcome；异常残留为 incomplete，不能进入训练集。
- 失败类数据只用于审计和评估，ACT 训练集只含基础成功示范、成功恢复和可选自主成功。
- candidate 只有通过相同条件、完全无人工接管的回归与恢复评估才可晋级；原 checkpoint 永不被自动覆盖。
