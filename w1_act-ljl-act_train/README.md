# W1 ACT 训练与仿真项目

本目录包含 W1 机器人的 ACT 训练框架、真机推理代码和 MuJoCo 仿真验证平台。

当前训练版本在标准 ACT 动作重建损失基础上增加了两种 W1 末端位姿辅助损失：

- L1：19 维关节动作重建。
- EE：从 ACT decoder 特征直接预测双臂末端位姿。
- FK：将预测动作输入可微正运动学后计算末端位姿误差。

严格来说，启用 ACT VAE 时总损失还包含 KL 项。

## 代码入口

训练实际使用的 ACT 源码是：

```text
w1_lerobot/src/lerobot/policies/act/
├── configuration_act.py
├── modeling_act.py
├── processor_act.py
└── kinematics.py
```

根目录下的 `w1_lerobot/src/lerobot/policies/act_w1_origin` 是历史或对照版本，不是当前`w1_lerobot` 安装后的训练入口。

主要文件：

| 路径 | 作用 |
| --- | --- |
| `w1_lerobot/` | 当前 LeRobot 训练源码 |
| `train_act_popcorn_ee_fk.sh` | W1 ACT L1+EE+FK 训练模板 |
| `tb_monitor.py` | 从训练日志生成 TensorBoard 标量 |
| `w1_simulation/` | ACT raw/bridge MuJoCo 仿真验证 |
| `inference_codes/` | 真机推理相关代码 |
| `assets/w1/` | W1 URDF 资产 |

## 训练数据流

```text
相机图像 + 19D W1 状态
          │
          ▼
    ACT encoder/decoder
          │
          ├── action_head ──> 100×19 动作
          │                       │
          │                       ├── L1 loss
          │                       └── 可微 FK ──> FK loss
          │
          └── ee_pose_head ──> 双臂末端位姿 ──> EE loss

真实动作 ──> 反归一化 ──> URDF FK ──> 目标双臂末端位姿
```

EE/FK 的目标位姿直接由数据集真实动作和 URDF 在线计算，不要求数据集中额外保存末端位姿标签。

辅助损失只用于训练。推理输出仍然是原来的 `100×19` ACT action chunk，
不会增加末端位姿输出，也不会改变 raw、bridge 或真机推理接口。

## 19D W1 动作契约

启用 EE 或 FK loss 后，动作维度必须严格为 19，数据集中的动作名称和顺序必须是：

```text
WAIST
LEFT_J1
LEFT_J2
LEFT_J3
LEFT_J4
LEFT_J5
LEFT_J6
LEFT_J7
NECK1
NECK2
RIGHT_J1
RIGHT_J2
RIGHT_J3
RIGHT_J4
RIGHT_J5
RIGHT_J6
RIGHT_J7
LEFT_GRIPPER
RIGHT_GRIPPER
```

其中：

- FK 左臂链使用 `WAIST` 和 `LEFT_J1～LEFT_J7`。
- FK 右臂链使用 `WAIST` 和 `RIGHT_J1～RIGHT_J7`。
- `NECK1`、`NECK2` 和两个 gripper 不进入双臂 FK，但仍由 L1 loss 监督。
- 当前实现没有单独的 gripper loss 权重；19 个动作通道在归一化 L1 中统一处理。
- `ANKLE`、`KNEE`、`BUTTOCK` 不允许进入以 `buttock` 为参考坐标系的双臂 FK 链。

如果 EE 和 FK 权重都为 0，ACT 会恢复为通用动作维度，不启用 W1 19D 契约检查。

## 总损失

训练总损失为：

```text
L_total =
    L_l1
    + use_vae × kl_weight × L_kl
    + ee_pose_loss_weight × L_ee
    + fk_loss_weight × L_fk
```

当前默认值：

| 参数 | 默认值 |
| --- | ---: |
| `kl_weight` | `10.0` |
| `ee_pose_loss_weight` | `0.0` |
| `fk_loss_weight` | `0.0` |
| `ee_position_scale_m` | `0.1` |
| `ee_rotation_loss_weight` | `0.25` |

`train_act_popcorn_ee_fk.sh` 当前使用：

```text
ee_pose_loss_weight = 0.05
fk_loss_weight      = 0.10
ee_position_scale_m = 0.1
ee_rotation_weight  = 0.25
```

### L1 loss

L1 在经过 `MEAN_STD` 归一化的动作空间中计算：

```text
L_l1 = mean(abs(action_normalized - prediction_normalized) × valid_mask)
```

padding 位置会乘零，但当前实现最后直接调用 `.mean()`，因此 padding 元素仍包含在分母中。

### EE loss

启用 `ee_pose_loss_weight > 0` 后，模型会增加一个 `ee_pose_head`。

每个时间步输出：

```text
2 hands × (3D position + 6D rotation) = 18D
```

6D rotation 会转换为 `3×3` 旋转矩阵。EE loss 为：

```text
L_ee = L_ee_position
       + ee_rotation_loss_weight × L_ee_rotation
```

位置使用缩放后的 Smooth L1：

```text
SmoothL1(predicted_position / ee_position_scale_m,
         target_position / ee_position_scale_m)
```

旋转使用旋转矩阵元素的均方误差。

EE loss 主要约束 decoder 表征具备直接恢复双臂末端位姿的能力。

### FK loss

预测动作首先通过数据集统计量反归一化：

```text
physical_action = normalized_action × action_std + action_mean
```

随后输入基于 URDF 的可微正运动学：

```text
predicted_action ──> differentiable FK ──> predicted EE pose
target_action    ──> FK, no_grad       ──> target EE pose
```

FK loss 为：

```text
L_fk = L_fk_position
       + ee_rotation_loss_weight × L_fk_rotation
```

它直接将末端空间误差的梯度传回 ACT action head。

### 数值精度

EE/FK 辅助损失在关闭 autocast 的区域内使用 FP32 计算，即使外围训练启用了混合精度，
运动学计算也不会使用 FP16/BF16。

当前训练脚本使用：

```text
--mixed_precision=no
```

即训练不使用 FP16/BF16 AMP；训练入口仍允许 CUDA matmul 使用 TF32。

## URDF 要求

启用任意辅助损失时必须提供：

```text
--policy.kinematics_urdf_path=<urdf>
```

可选但推荐同时提供：

```text
--policy.kinematics_urdf_sha256=<sha256>
```

计算哈希：

```bash
sha256sum <urdf>
```

训练首次计算辅助损失时会验证实际 URDF 哈希。路径正确但哈希不一致时会直接终止，
避免训练过程中静默使用错误的机器人结构。

URDF 必须满足：

- reference link 是左右末端 link 的祖先。
- 双臂链包含 `WAIST`。
- 左右链分别包含 `LEFT_J1` 和 `RIGHT_J1`。
- 非固定关节必须能映射到 19D W1 动作。
- 关节类型只能是 `fixed`、`revolute` 或 `continuous`。
- 从 `buttock` 出发时，链中不能包含 `ANKLE`、`KNEE`、`BUTTOCK` 关节。

当前训练模板使用：

```text
reference link: buttock
left EE:        left_ee
right EE:       right_ee
```

`left_ee/right_ee` 与 `left_hand_base_link/right_hand_base_link` 具有不同的末端偏移。
训练、验证和仿真比较时应明确使用同一组末端定义。

当前目录内 URDF 的哈希不等于训练模板中记录的 `8cf751...`。不能在保留旧哈希的同时
直接替换为 `assets/w1` 或 `w1_simulation/urdf` 中的其他 URDF。

## 数据集要求

训练数据使用 LeRobot 数据集接口，EE/FK 模式额外要求：

- action shape 为 19。
- action names 与上述顺序完全一致。
- action normalization 为 `MEAN_STD`。
- 数据集包含有限的 action mean/std。
- mean/std shape 必须为 `(19,)`。
- action std 不允许为负数。

训练框架会将 `dataset.meta.stats` 和 `dataset.meta` 传入 `ACTPolicy`，用于：

- 校验动作名称和顺序。
- 将预测动作与目标动作恢复到物理关节空间。
- 为训练预处理器配置动作归一化。

## 安装

以下命令均从 `w1_act` 根目录执行：

```bash
python -m pip install -e ./w1_lerobot
python -m pip install tensorboard
```

确认加载的是当前源码：

```bash
python -c "import lerobot; from lerobot.policies.act.modeling_act import ACTPolicy; print(lerobot.__file__)"
```

输出路径应指向当前目录中的：

```text
w1_lerobot/src/lerobot
```

## 启动训练

当前训练模板：

```text
train_act_popcorn_ee_fk.sh
```

模板默认配置：

| 配置 | 值 |
| --- | ---: |
| chunk size | 100 |
| inference action steps | 100 |
| vision backbone | ResNet |
| vision trainable | 是 |
| batch size | 8 |
| workers | 28 |
| training steps | 5,000,000 |
| checkpoint interval | 50,000 |
| log interval | 100 |
| mixed precision | 关闭 |
| processes | 1 |

脚本包含训练机相关的绝对路径。执行前必须检查：

- Conda 初始化脚本。
- LeRobot 工作目录。
- `lerobot_train.py` 路径。
- 数据集目录。
- 运动学 URDF 路径和 SHA256。
- 输出目录。
- TensorBoard 地址。

确认后执行：

```bash
bash ./train_act_popcorn_ee_fk.sh
```

## Loss 模式切换

### 标准 ACT：L1 + KL

```text
--policy.ee_pose_loss_weight=0
--policy.fk_loss_weight=0
```

不加载 URDF，不创建 EE head，也不限制 action 必须为 W1 19D。

### L1 + KL + FK

```text
--policy.ee_pose_loss_weight=0
--policy.fk_loss_weight=0.10
```

直接约束动作预测经过 FK 后的双臂末端位姿，不增加额外推理输出头。

### L1 + KL + EE

```text
--policy.ee_pose_loss_weight=0.05
--policy.fk_loss_weight=0
```

增加 EE pose head，用末端位姿辅助监督 decoder 表征。

### L1 + KL + EE + FK

```text
--policy.ee_pose_loss_weight=0.05
--policy.fk_loss_weight=0.10
```

这是当前 `train_act_popcorn_ee_fk.sh` 使用的方案。

## 训练指标

ACT forward 会生成：

```text
l1_loss
kld_loss
ee_pose_loss
ee_position_loss
ee_rotation_loss
fk_loss
fk_position_loss
fk_rotation_loss
auxiliary_loss
total_loss
```

其中：

```text
auxiliary_loss =
    ee_pose_loss_weight × ee_pose_loss
    + fk_loss_weight × fk_loss
```

当前 `lerobot_train.py` 的控制台 `MetricsTracker` 只记录：

```text
loss
grdn
lr
updt_s
data_s
```

详细的 `l1_loss`、`ee_pose_loss`、`fk_loss` 等位于 policy 返回的 `output_dict` 中，
当前只有启用 W&B 时才会被统一写出。

因此，在 `wandb.enable=false` 时，`tb_monitor.py` 即使配置了
`l1,ee,fk,aux` 标签，也无法从现有控制台日志中解析出这些详细指标；TensorBoard 中可靠可见的是
控制台实际输出的 `loss`、`grdn` 等字段。

## Checkpoint 与推理

EE/FK 训练不会改变 ACT 的动作接口：

```text
input:  图像 + 19D state
output: chunk_size × 19D action
```

推理时：

- `predict_action_chunk()` 只返回 action head 的结果。
- EE pose head 不参与控制输出。
- 不需要向真机或仿真器发布 EE pose。
- URDF 不参与常规 action chunk 推理。
- 使用相同 checkpoint config 加载时会保留训练时的辅助损失参数。

启用了 EE head 的 checkpoint 比标准 ACT 多一组 `ee_pose_head` 参数。修改
`ee_pose_loss_weight` 后再加载不同结构的 checkpoint 时，需要检查 missing/unexpected keys。

## start_infer.sh 真机推理

根目录的 `start_infer.sh` 用于同时启动：

```text
policy_infer_act.py
    └── ACT 模型服务、预处理、后处理和共享内存推理

policy_bridge_act_lipo.py
    └── ROS 观测采集、动作队列、异步重规划、LIPO 和话题发布
```

启动脚本通过本机回环地址连接两个进程：

```text
相机 + 关节反馈
        │
        ▼
policy_bridge_act_lipo.py
        │  共享内存 + 本机 IPC
        ▼
policy_infer_act.py
        │
        ▼
ACT action chunk
        │
        ▼
插值、异步重规划、身体 LIPO
        │
        ├── body action topic
        ├── left hand action topic
        └── right hand action topic
```

### 当前可运行状态

当前版本不能直接视为完整可运行入口。

`policy_bridge_act_lipo.py` 包含：

```python
import policy_bridge_act as blocking
```

但根目录和 Python 环境中目前不存在 `policy_bridge_act.py`。启动 bridge 时会出现：

```text
ModuleNotFoundError: No module named 'policy_bridge_act'
```

此外，`start_infer.sh` 中的 checkpoint 和 URDF 仍是训练机或部署机相关的固定绝对路径。
整体移动 `w1_act` 后必须先修复这些路径，或者改为根据脚本位置动态解析。

在上述问题修复前，不应在真机上执行：

```bash
bash ./start_infer.sh
```

### 启动过程

`start_infer.sh` 的预期流程是：

1. 将模型服务固定到 CPU 3。
2. 在 `cuda:0` 加载 ACT checkpoint。
3. 等待模型服务监听本机端口 `8888`，最长等待 90 秒。
4. 将 LIPO bridge 固定到 CPU 5。
5. bridge 连接模型服务并初始化共享内存。
6. 等待身体动作话题出现订阅者。
7. 收到新订阅会话后重置动作队列和 LIPO 状态。
8. 获取三路图像与机器人关节反馈。
9. 执行首个 action chunk 推理。
10. 在旧轨迹剩余点数达到触发条件时提交异步重规划。
11. 新轨迹返回后按绝对控制 step 对齐，并执行身体 LIPO。

任意子进程提前退出时，脚本会终止另一个子进程。按 `Ctrl+C` 时也会统一清理模型服务和 bridge。

### 模型服务

`policy_infer_act.py` 使用：

- `TrainPipelineConfig.from_pretrained()` 加载训练配置。
- `ACTPolicy.from_pretrained(..., strict=True)` 加载权重。
- `local_files_only=True`，不会联网下载缺失文件。
- checkpoint 中的 `policy_preprocessor.json`。
- checkpoint 中的 `policy_postprocessor.json`。
- `predict_action_chunk()` 输出完整 ACT chunk。
- 后处理器将动作恢复到数据集物理空间。

如果请求 CUDA 但 CUDA 不可用，模型服务会直接退出，不会静默回退 CPU。

模型服务和 bridge 使用：

```text
address: 127.0.0.1
port:    8888
authkey: w1_act_secret
```

图像、状态和动作通过共享内存交换，IPC 连接主要负责发送初始化、推理、重置和模型切换命令。

虽然模型服务支持多个 model ID 和 `SWITCH_MODEL`，当前启动脚本只配置一个模型。

### 当前输入话题

| 数据 | 话题 |
| --- | --- |
| 机器人关节反馈 | `/feedback/robot_server_state` |
| 头部左相机 | `/camera/left_eye_resize` |
| 左手相机 | `/camera_l/color/image_rect_raw` |
| 右手相机 | `/camera_r/color/image_rect_raw` |

模型输入图像 key 为：

```text
observation.images.cam_high_left
observation.images.cam_hand_left
observation.images.cam_hand_right
```

图像会调整为：

```text
head: 640×360
hand: 640×360
```

### 当前输出话题

| 控制对象 | 话题 |
| --- | --- |
| 身体动作 | `/feedback/body_act` |
| 左手动作 | `/feedback/hand/left_act` |
| 右手动作 | `/feedback/hand/right_act` |

这些是当前脚本配置的 ACT 中间话题，不是
`/control/joint_position`、`/control/hand/left`、`/control/hand/right`
标准控制终点。

`shadow_mode=false` 表示 bridge 会实际发布动作，而不仅记录推理结果。

### 身体输出

当前 `selected_body_names` 包含 17 个模型身体关节：

```text
WAIST
LEFT_J1
LEFT_J2
LEFT_J3
LEFT_J4
LEFT_J5
LEFT_J6
LEFT_J7
NECK1
NECK2
RIGHT_J1
RIGHT_J2
RIGHT_J3
RIGHT_J4
RIGHT_J5
RIGHT_J6
RIGHT_J7
```

`ANKLE`、`KNEE`、`BUTTOCK` 不在当前推理发布集合中。

### 手部输出

当前手部输入模式为：

```text
hand_input_mode=scalar
hand_sides=[left, right]
```

ACT 的：

```text
LEFT_GRIPPER
RIGHT_GRIPPER
```

会分别转换为左右手 6 维位置。

当前左右手都启用了 gripper 方向反转，并分别使用脚本中的
`left_hand_start/end` 和 `right_hand_start/end` 作为手势插值端点。

身体维度参与 LIPO，左右 gripper 维度被排除在 LIPO mask 外，因此手部不会跟随身体执行相同的线性融合。

### 当前动作频率与插值

脚本参数为：

```text
policy_hz=20
sample_factor=2
```

设计意图是将每个策略点机械插值为两个控制点：

```text
100 policy points
    × sample_factor 2
    = 200 control points
```

对应控制频率为：

```text
20 Hz × 2 = 40 Hz
```

`sample_factor` 同时缩放轨迹长度、重规划触发点数和 LIPO 点数，不改变它们在原始策略时间轴上的比例。

### 当前异步重规划与 LIPO

当前根目录版本使用固定点数：

```text
lipo_trigger_points=15
lipo_blend_points=6
```

在 `sample_factor=2` 时转换为：

```text
触发重规划：剩余 30 个 control points
LIPO 长度：12 个 control points
```

这不是基于比例阈值的动态重规划。当前脚本没有：

```text
replan_threshold
BRIDGE_MODE=sync
BRIDGE_MODE=async
```

因此它与 `w1_simulation` 当前使用的“100 步、剩余比例 0.5 时重规划、5 个策略点 LIPO”
不是同一套调度参数。

异步推理期间旧轨迹继续执行。新轨迹返回后：

- 新轨迹以提交推理时的 control step 为时间原点。
- 已经过期的新轨迹前缀会被丢弃。
- 新旧轨迹按相同的绝对 control step 对齐。
- 身体维度在有效重叠区间内线性过渡。
- 如果新结果在整个时间范围失效后才返回，该结果会被丢弃。
- 如果旧轨迹耗尽但新结果仍未返回，则保持最后一次动作。

### 启停和会话重置

bridge 只有检测到动作话题订阅者后才开始推理。

订阅会话发生变化时会重置：

- 当前轨迹块。
- 待安装推理结果。
- LIPO transition。
- control step。
- hold 和重规划日志状态。

跨会话返回的旧推理结果会被丢弃，防止停止后再次启动时沿用上一次会话的轨迹。

### 运行前检查

修复缺失依赖后，至少需要确认：

```bash
python3 -c "import policy_bridge_act"
python3 -c "from lerobot.policies.act.modeling_act import ACTPolicy; print(ACTPolicy.name)"
bash -n ./start_infer.sh
```

同时确认：

- ROS 2 环境已经加载。
- 身体和灵巧手消息包可以导入。
- checkpoint、URDF 均存在。
- checkpoint 包含预处理器和后处理器配置。
- 三路相机话题存在且尺寸正确。
- 关节反馈话题持续更新。
- 端口 `8888` 未被其他进程占用。
- 输出话题连接到预期的下游控制节点。

## 基础验证

检查 ACT 模块可导入：

```bash
PYTHONPATH=./w1_lerobot/src \
python -c "from lerobot.policies.act.modeling_act import ACTPolicy; print(ACTPolicy.name)"
```

检查语法：

```bash
python -m compileall -q ./w1_lerobot/src/lerobot/policies/act
```

检查 URDF 哈希：

```bash
sha256sum <urdf>
```

完整训练前建议先使用较小参数进行 smoke test：

```text
--steps=2
--batch_size=1
--num_workers=0
--save_freq=2
```

smoke test 应至少确认：

- 数据集 action names 通过 19D 顺序校验。
- action mean/std 成功载入。
- URDF link 和关节链通过校验。
- URDF SHA256 匹配。
- forward 和 backward 能完成。
- `l1_loss`、`ee_pose_loss`、`fk_loss` 和总 loss 均为有限值。

## 仿真验证

训练完成后使用：

```text
w1_simulation/run_act_sim.sh
w1_simulation/run_act_sim_bridge.sh
```

分别验证：

- raw ACT chunk 推理。
- 异步动态重规划与 LIPO。
- 19D 动作到 MuJoCo 的映射。
- 末端轨迹、动作质量和实时性。

仿真框架的安装、运行和验收说明见：

```text
w1_simulation/README.md
```
