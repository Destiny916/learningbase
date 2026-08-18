# W1 ACT 仿真验证平台

## 项目定位

`w1_simulation` 是独立的 ACT 仿真验证项目，内置 MuJoCo 仿真、ACT 推理服务、raw/bridge 控制、
可视化、质量评估和验证代码，不依赖 `w1_sim`、`w1_rl` 或仓库外的 sim2real 源码。平台从
`w1_simulation/urdf/dexforce_w1_v024_brainco_revo1_r/robot_with_ee.urdf` 构建含双灵巧手的 W1 MuJoCo
模型，并在原生 Rerun 窗口中同步显示仿真视角、模型实际输入图像和运行指标。
两个策略仿真入口默认在运行时 MJCF 的 `eyes` 链接下安装 MuJoCo 相机，以标准30 FPS离屏渲染
1280×720 第一人称画面。左侧主视图默认由眼部画面替代第三视角，右侧三路数据集相机始终保留；
眼部画面仅用于可视化，不进入 ACT 输入或控制路径。

MuJoCo 3.3.7 不原生读取当前 URDF 的 GLB 视觉网格。构建运行时模型时，项目会在本次运行的
`generated/urdf_visual_meshes` 下将 GLB 几何无损适配为 MuJoCo 二进制 MSH；源 URDF、源 GLB、
URDF 材质配置和独立碰撞 OBJ 均不修改。眼部相机渲染 visual group，碰撞 group 不参与画面。

origin 数据提供同步相机图像，并在 `reset` 时提供一次初始姿态。Bridge 默认使用状态对齐回放：
录制关节只用于在局部向前窗口内选择图像帧，不作为ACT状态或动作。启用动作质量指标时，录制关节还会
进入独立评估器作为参考轨迹。复位完成后，raw 的19维
状态来自MuJoCo；bridge首轮保存MuJoCo状态，后续仅用实际下发值更新所选身体关节和双手标量，
未发布身体关节始终保持首轮状态。该平台不负责训练ACT，也不会启动ROS
或向真机发布命令。

```text
origin 同步图像 ─┐
                 ├─> ACT（100×19 动作块）─> raw / bridge ─> 19→29 映射 ─> MuJoCo
MuJoCo 19D 反馈 ─┘                                                    │
       ↑──── raw使用最新反馈；bridge使用首轮基线和已发布子集 ────────────┘

图像始终30 Hz；bridge默认动作时间线为policy_hz×sample_factor=60 Hz
```

## 两种验证模式

| 启动脚本 | 调度方式 | 动作处理 | 适用目的 |
| --- | --- | --- | --- |
| `run_act_sim.sh`（纯 ACT） | 每 100 步用最新图像和最新 MuJoCo 状态同步推理；新块在同一步从索引 0 替换旧块 | 恒等处理，ACT 输出逐位不变 | 判断模型本身是否可用，复现引入 bridge 前的流畅链路 |
| `run_act_sim_bridge.sh` | 首块同步；100点剩余50个策略点时异步推理，最多一个请求在途 | 100点线性插值为200点；身体17维用10个控制点LIPO交接，夹具直接采用新块 | 验证阈值动态重规划、推理预算与LIPO交接 |

数据集图像在两种模式下都保持 30 Hz。raw 按原始时间回放；bridge 默认按原始时间回放，显式启用
状态对齐时才由当前MuJoCo反馈在录像局部窗口内选择最接近的图像帧。raw 的动作输出固定 30 Hz；bridge 的动作频率为
`BRIDGE_POLICY_HZ × BRIDGE_SAMPLE_FACTOR` Hz，默认 `30×2=60 Hz`。bridge 只处理动作：同一源图像可供多个动作周期使用，但不会插值、
重解码或提高图像帧率。Rerun 每张源图像只记录一次，并显示该时刻动作执行后的真实 MuJoCo 状态；
MuJoCo眼部相机默认按标准30 FPS渲染；完整的高频动作、状态和目标同时保存在NPZ与TensorBoard中。
显式设置 `EYE_CAMERA_FPS=0` 时相机才跟随控制频率，bridge此时为60 FPS。

### bridge 时间对齐

每个候选动作的目标 step 固定为 `submit_step + action_index`。例如在 step 100 提交、step 112 返回时，
索引 0–11 已经过期，当前只允许从索引 12 开始参与控制。新旧块按同一绝对step对齐；身体前17维在
10个控制点内以 `alpha=1/10…10/10` 线性过渡，夹具两维直接使用新块值。每次请求复制提交时刻的
最新30 Hz图像，并以首轮状态基线和最后实际下发的选中关节构造模型状态；控制循环不会等待GPU。

## 快速启动

所有入口都位于 `w1_simulation/`。两个策略仿真脚本拥有各自完整配置，不需要也不接受通过
`ACTION_PIPELINE` 环境变量切换类别：

```bash
# 在 w1_act 根目录执行
./w1_simulation/run_act_sim.sh
./w1_simulation/run_act_sim_bridge.sh
./w1_simulation/run_ee_pose_validation.sh
```

前两个入口分别验证19维 ACT 的 raw 与 bridge 控制链路。是否使用 EE/FK loss 训练不会改变
checkpoint 的19维关节输出接口，因此不再提供独立的 EE/FK 策略仿真入口；这类模型直接由
`run_act_sim_bridge.sh` 加载。`run_ee_pose_validation.sh` 不加载策略，只独立检查数据集动作到
URDF 末端坐标的 FK 一致性。

常用启动方式：

```bash
./w1_simulation/run_act_sim.sh                                      # 纯 ACT 模型验证
./w1_simulation/run_act_sim_bridge.sh                               # Bridge及EE/FK训练模型验证
./w1_simulation/run_ee_pose_validation.sh --no-open-viewer           # 独立 EE 坐标验证
BRIDGE_POLICY_HZ=20 BRIDGE_SAMPLE_FACTOR=2 \
  ./w1_simulation/run_act_sim_bridge.sh                             # 真机同款40Hz动作消费
MAX_FRAMES=29 KEEP_OPEN=0 ./w1_simulation/run_act_sim.sh            # 纯 ACT 快速检查
STRICT_VERIFICATION=1 KEEP_OPEN=0 \
  ./w1_simulation/run_act_sim_bridge.sh                             # Bridge 严格门禁
CONTROL_MODE=dynamic ./w1_simulation/run_act_sim_bridge.sh          # MuJoCo 动力学跟踪
QUALITY_END_EFFECTOR=0 QUALITY_AMPLITUDE=0 \
  ./w1_simulation/run_act_sim_bridge.sh                             # 只看姿态和运动方向质量
SCORE_SMOOTHNESS=0 SCORE_REALTIME=0 \
  ./w1_simulation/run_act_sim_bridge.sh                             # 关闭末尾平滑性和实时性分项
RERUN_VIEW_MODE=eye ./w1_simulation/run_act_sim_bridge.sh           # 左侧眼部画面，右侧保留数据集相机
EYE_CAMERA_FPS=30 ./w1_simulation/run_act_sim_bridge.sh             # 将眼部相机限制为30 FPS
RERUN_VIEW_MODE=standard ./w1_simulation/run_act_sim.sh             # 恢复旧版机器人+模型输入布局
RERUN_VIEW_MODE=both ./w1_simulation/run_act_sim.sh                 # 左侧叠放第三视角和眼部画面，右侧保留数据集相机
```

脚本优先调用现有 LeRobot Conda 环境中的 Python；找不到时使用 `python3`。Rerun 和
TensorBoard 端口默认自动选择；完整运行后窗口保持打开，按 `Ctrl+C` 统一关闭。
三个入口会根据自身位置优先加载同级 `w1_lerobot/src`，目录移动后无需修改源码路径。
仿真运行时不依赖父项目中的 `inference_codes`。

项目使用同级 `w1_lerobot` 作为 LeRobot 源码依赖。以下命令从 `w1_act` 根目录执行；
`requirements.txt` 通过 `../w1_lerobot` 解析依赖，因此整个 `w1_act` 目录可以移动：

```bash
cd w1_simulation
python -m pip install --no-build-isolation -r requirements.txt
python -m pip install --no-build-isolation --no-deps -e .
python -c "import lerobot, w1_simulation; from lerobot.policies.act.modeling_act import ACTPolicy; print(lerobot.__file__)"
```

最后一条命令应显示同级 `w1_lerobot/src/lerobot` 中的包路径。

源码按职责分为 `robot/`、`inference/`、`control/`、`replay/`、`simulation/`、
`evaluation/`、`runtime/` 和 `observability/`。旧顶层兼容模块已删除；验证入口为
`python -m w1_simulation.evaluation.verify`，EE/FK入口为 `python -m w1_simulation.evaluation.ee_pose`。

URDF/mesh 位于 `w1_simulation/urdf`。checkpoint 和 origin 数据默认从项目父目录查找，也可通过
`W1_SIMULATION_ASSET_ROOT` 统一指定资产根目录；`W1_SIMULATION_URDF` 可单独覆盖 URDF。
运行产物默认仍写入兼容目录 `w1_simulation/artifacts`，该目录已从 Git 排除；可设置
`W1_SIMULATION_ARTIFACT_ROOT=runs/w1_simulation` 将所有新产物放到 `w1_act/runs/w1_simulation`，
也可继续使用优先级更高的 `ARTIFACT_ROOT` 针对单次启动覆盖。

## 配置说明

W1、ACT、仿真和真机Bridge默认配置统一位于 `w1_simulation/configs/w1_popcorn_v1.json`。两个仿真脚本
只锁定raw或bridge类别，并通过 `--profile` 读取默认值；环境变量和命令行参数仍可进行单次覆盖。

| 配置 | 默认值 | 说明 |
| --- | --- | --- |
| `CHECKPOINT` | raw、bridge：`checkpoints/Act/popcorn/0450000/pretrained_model` | LeRobot ACT 权重目录；EE/FK loss 不改变推理接口 |
| `ORIGIN_ROOT` | 脚本内配置 | 图像、reset姿态及可选质量参考；raw与bridge可分别指定 |
| `POLICY_BACKEND` | `script` | `script` 运行真机同源推理服务；`direct` 在进程内加载同一权重 |
| `CONTROL_MODE` | `kinematic` | 精确执行目标，隔离策略误差；`dynamic` 验证动力学跟踪 |
| `RAW_EXECUTION_HORIZON` | `100` | raw 模式实际返回和执行的动作步数；`0` 使用 checkpoint 默认值 |
| `RAW_REPLAN_INTERVAL` | `0` | raw 模式重新推理间隔；`0` 自动使用运行时 execution horizon |
| `BRIDGE_SIMULATED_INFERENCE_MS` | `200` | 仿真中注入的最短端到端推理时间；不属于真机推理参数 |
| `BRIDGE_INFERENCE_BUDGET_MS` | `300` | 重规划安全校验采用的端到端推理预算 |
| `RERUN_VIEW_MODE` | `eye` | 右侧始终保留数据集相机；`standard`左侧显示第三视角，`eye`左侧显示眼部画面，`both`在左侧叠放两者 |
| `EYE_CAMERA_WIDTH/HEIGHT` | `1280/720` | MuJoCo离屏相机和Rerun图像分辨率 |
| `EYE_CAMERA_FPS` | `30` | 标准视角默认30 FPS；`0`跟随控制频率，也可设置不高于控制频率的正数 |
| `EYE_CAMERA_FOVY` | `70` | 可视化相机垂直视场角，当前没有冒充真实标定内参 |
| `EYE_CAMERA_SCENE` | `grid` | `grid`增加无碰撞网格地面；`robot`只渲染机器人和skybox |
| `BRIDGE_EXECUTION_HORIZON` | `100` | bridge 每次推理返回的策略点数；`0` 使用 checkpoint 默认值 |
| `BRIDGE_POLICY_HZ` | `30` | 策略点基础消费频率 |
| `BRIDGE_SAMPLE_FACTOR` | `2` | 将100个策略点线性插值为200个控制点 |
| `BRIDGE_REPLAN_THRESHOLD` | `0.5` | 剩余策略点达到执行长度的50%时触发异步推理 |
| `BRIDGE_LIPO_BLEND_POLICY_POINTS` | `5` | 身体LIPO交接长度，默认对应10个控制点 |
| `BRIDGE_REPLAN_MARGIN_POLICY_POINTS` | `2` | 推理预算和LIPO之外保留的策略点安全余量 |
| `IMAGE_REPLAY_MODE` | `time` | Bridge 默认按数据集时间戳固定30 Hz回放；`state` 仅保留为实验模式 |
| `IMAGE_SEARCH_AHEAD_FRAMES` | `15` | 状态对齐时向前搜索的录像帧数 |
| `IMAGE_MAX_ADVANCE_FRAMES` | `2` | 每个30 Hz图像周期最多向前推进的帧数 |
| `IMAGE_MATCH_THRESHOLD` | `0.18` | 归一化姿态距离超过该值时冻结图像 |
| `IMAGE_SIMILARITY_SLACK` | `0.005` | 多个姿态近似相同时优先选择更靠后的帧 |
| `START_FRAME` / `MAX_FRAMES` | `0` / `0` | 起始同步帧；`0` 表示运行至数据结束 |
| `DEVICE` | `cuda:0` | 推理设备；CUDA 不可用时拒绝静默回退 CPU |
| `REALTIME` / `KEEP_OPEN` | `1` / `1` | 是否按实际动作频率运行，以及结束后是否保留界面 |
| `QUALITY_POSE` | `1` | 关节姿态归一化误差和身体/夹爪误差 |
| `QUALITY_END_EFFECTOR` | `1` | 双手末端位置及方向误差 |
| `QUALITY_MOTION_DIRECTION` | `1` | 当前运动方向与示范的一致率 |
| `QUALITY_AMPLITUDE` | `1` | 累计运动幅度及腰部幅度覆盖率 |
| `SCORE_SMOOTHNESS` | `1` | 末尾综合评分是否包含速度/加速度连续性 |
| `SCORE_REALTIME` | `1` | 末尾综合评分是否包含帧率、周期、miss和遥测flush |

### 过程质量与末尾综合评分

四个 `QUALITY_*` 开关均为 `1` 或 `0`，两个启动脚本默认全部开启。过程质量由旁路线程计算，控制循环
只投递当前MuJoCo状态，不等待末端运动学或质量统计。终端中的 `score_step` 表示评分线程已经处理到的
控制步。若还启用了 `SCORE_SMOOTHNESS=1`，即使关闭四个过程指标，末尾评分仍会读取参考轨迹；要完全
停用参考轨迹需同时关闭平滑性：

```bash
QUALITY_POSE=0 QUALITY_END_EFFECTOR=0 \
QUALITY_MOTION_DIRECTION=0 QUALITY_AMPLITUDE=0 \
SCORE_SMOOTHNESS=0 \
./w1_simulation/run_act_sim_bridge.sh
```

综合 `quality` 为0～100分，默认权重依次为40%、30%、20%、10%；关闭某项后，其余所选权重自动
归一化。`gain` 表示综合分相对“机器人始终保持初始姿态”基线的提升。终端每秒输出示例：

```text
track_rmse=0.000000 quality=82.6 gain=+31.4 pose=0.094 ee=5.0cm dir=88% amp=79%
```

`track_rmse` 只衡量MuJoCo是否跟上命令；`quality` 才衡量仿真动作与参考姿态的接近程度。时间回放使用
同时间戳姿态，状态对齐回放使用当前实际显示图像帧对应的姿态。
姿态分在归一化RMSE达到0.25时降为0；末端分结合平均位置误差和方向误差，位置30 cm或方向90°
对应0分；方向分统计活动关节运动方向；幅度分同时惩罚幅度不足和过冲。参考关节只进入评估器，
summary固定记录 `reference_use=evaluation_only_not_policy_input`，verifier会独立重算全部质量序列。

回放结束后，程序基于完整轨迹输出正式平均分：

```text
总分 = 动作复现×70% + 平滑性×10% + 幅度×10% + 实时性×10%
ACT_SIM_RUN_SCORE=90.5 motion_reproduction=92.8 smoothness=98.4 amplitude=59.7 realtime=97.7
```

动作复现汇总姿态、双手末端和运动方向；平滑性比较实际与参考轨迹的归一化速度及加速度；幅度同时
惩罚动作不足和过冲；实时性综合有效FPS、p95周期、deadline miss和遥测flush。`REALTIME=0` 时实时性
自动退出评分，其余权重重新归一化，并标记为不可与完整四项分直接横向比较。关节越限或Bridge绝对
step错位会使正式平均分无效。summary、NPZ和TensorBoard保存所有分项，verifier会从轨迹独立重算。

### 推理触发与 sample_factor

bridge不使用固定推理频率。触发策略点数按
`ceil(execution_horizon × BRIDGE_REPLAN_THRESHOLD)` 动态计算，控制频率为
`BRIDGE_POLICY_HZ × BRIDGE_SAMPLE_FACTOR`。默认100个策略点、阈值0.5、sample factor 2，因此60 Hz
消费200个控制点轨迹，在剩余50个策略点（100个控制点）时提交异步推理。旧轨迹在等待期间继续
执行；新轨迹返回后跳过已经过期的绝对step前缀，并使用5个策略点（10个控制点）完成身体LIPO交接。

启动时还会校验触发余量不小于
`ceil(BRIDGE_INFERENCE_BUDGET_MS / 1000 × BRIDGE_POLICY_HZ) + BRIDGE_LIPO_BLEND_POLICY_POINTS + BRIDGE_REPLAN_MARGIN_POLICY_POINTS`。
默认值对应 `9 + 5 + 2 = 16` 个必需策略点，50点触发余量中还剩34点。仿真注入的200 ms延迟仅用于
复现异步等待；推理较慢或较快时，实际跳过数量仍按install step自动变化。

Bridge 默认使用源图像的相对时间戳，以固定30 Hz顺序回放三路图像；机器人状态不再改变图像游标。
回放按实际控制时长播放，不会因为 `sample_factor` 增大而将图像或机器人动作拉慢。中文图解位于
持久化运行目录或用户指定的输出路径，可由 `python -m w1_simulation.explain_sample_factor_cn --overwrite`
重新生成。

### 单独指定每路图像

三路相机始终共用同一个数据集时间位置，图像内容和30 Hz输出频率保持不变。状态对齐试验代码仍保留
为非默认诊断选项；只有显式设置以下参数时才会启用：

```bash
IMAGE_REPLAY_MODE=state ./w1_simulation/run_act_sim_bridge.sh
```

两个脚本内 `CAMERA_SOURCES` 的每项格式为 `模型输入键=origin 数据源`：

```bash
CAMERA_SOURCES=(
  "observation.images.cam_high_left=head_left"
  "observation.images.cam_hand_left=hand_left"
  "observation.images.cam_hand_right=hand_right"
)
```

数据源可写 `metadata.jsonl` 中的 `camera_type`、相对图像目录或绝对目录。相机路数可以随 checkpoint
变化，但输入键集合必须与权重声明完全一致；缺路、重复键、不同步或错误尺寸会直接失败。默认Rerun
布局为左侧W1三维视图、右侧MuJoCo眼部画面；`standard`模式恢复左侧机器人、右侧模型输入图像。
也可以不编辑脚本，重复传入参数；一旦
出现命令行 `--camera-source`，脚本内三路默认值将全部停用：

```bash
./w1_simulation/run_act_sim_bridge.sh \
  --camera-source observation.images.cam_high_left=head_left \
  --camera-source observation.images.cam_hand_left=hand_left \
  --camera-source observation.images.cam_hand_right=hand_right
```

## 状态、动作与灵巧手映射

ACT 状态和动作均为 19 维：17 个身体关节加 `LEFT_GRIPPER`、`RIGHT_GRIPPER`。每个单步ACT动作先转换为
标准W1位置命令：身体使用 `/control/joint_position` 的稀疏 `name/position` 契约，实际维度和顺序由
profile 的 `commands.body_order` 决定；左右灵巧手分别使用
`/control/hand/left`、`/control/hand/right` 的6维 `POSITION name/value` 契约。身体单位为弧度，手指
值域为0～100；MuJoCo在同一个控制step原子消费身体和双手命令，但不依赖ROS消息包或topic通信。

当前仿真控制为29维：17个身体主动关节加左右手各6个主动关节。`ANKLE`、`KNEE`、`BUTTOCK`
由MuJoCo模型构建层固定，不进入推理命令。未出现在身体命令中的活动关节保持上一控制目标；8个DIP
关节由URDF mimic约束驱动。仿真后端按名称从标准命令更新29维目标并裁剪到URDF限位，不接收ACT
chunk、checkpoint或裸19维动作。

真机 `policy_bridge.py` 使用 `ordered_body_names` 固定模型身体顺序，使用 `selected_body_names` 决定
`/feedback/body_act` 实际发布的名称和值。后者可以是前者的任意无重复子集和顺序，不会改变ACT维度、
左右手索引、重规划阈值或LIPO时机。未发布身体关节不补零、不补锁定值，也不会把未执行的模型预测
写回下一轮状态。

`w1_simulation/configs/w1_popcorn_v1.json` 分别定义左右手在 `gripper=0` 和 `gripper=100` 时的
6 维设备姿态，中间值逐维线性插值。设备顺序是 `THUMB1, THUMB2, INDEX, MIDDLE, RING, PINKY`；
映射到 URDF 时使用 `THUMB1 → T_MCP`、`THUMB2 → T_CMC_YAW`，其余四维顺序不变。初始化、动作
下发和状态反馈共用同一个双向映射，配置内容及 SHA256 会写入运行摘要。

## 动作块处理器

动作处理器由模式固定选择：raw使用恒等处理器，bridge使用线性插值处理器。运行时不再加载
`MODULE:FACTORY` 插件，避免仿真验收被外部处理代码改变。`raw` 必须保持
`(execution_horizon, 19)`；处理后动作必须
保持19维，内置 bridge 的长度为 `(execution_horizon × sample_factor, 19)`。所有数值必须有限，阶段名称必须可审计。处理器
不会接触图像、MuJoCo 状态、推理时钟或队列调度。`raw` 模式强制使用恒等处理器，防止模型验证
结果被后处理污染。

## 可视化、产物与验收

仿真默认使用 `SAVE_ARTIFACTS=false`。运行期间，TensorBoard、运行模型、轨迹和服务日志位于
`/dev/shm` 临时目录；Rerun只保留Viewer中的live数据。关闭可视化服务后临时目录整体清理，
`w1_simulation/artifacts` 不产生文件。

需要离线保留和回放时显式启用：

```bash
SAVE_ARTIFACTS=true ./w1_simulation/run_act_sim.sh
SAVE_ARTIFACTS=true ./w1_simulation/run_act_sim_bridge.sh
```

一次运行的全部产物保存在 `artifacts/runs/<timestamp>_<run_name>/`：

- `summary.json`：权重、脚本、输入源、调度、映射、时延、综合评分和哈希清单。
- `trajectory.npz`：状态、动作、原始/处理后 chunk、阶段结果和来源索引。
- `recording.rrd`：控制频率下的机器人状态和MuJoCo眼部画面，以及30 Hz模型输入图像、目标关节、
  SHA256、相机内参、相机视锥和相对秒时间线。
- `tensorboard/`：推理时延、周期、队列、跟踪误差、过程质量、末尾综合评分、限位和GPU指标。
- `generated/`：本次运行使用的MuJoCo XML、适配URDF和转换报告。
- `logs/`：本次运行的Rerun和TensorBoard服务日志。
- `verification.json`：独立重放和完整性验收结果。

验证器检查最新状态反馈、30 Hz 图像同源、19→29 映射、关节限位、raw 恒等性、bridge 各处理阶段、
绝对step对齐、过期前缀裁剪、单请求、身体LIPO、夹具直通、RRD行数、TensorBoard tags和模型哈希。
实时门禁要求bridge动作接近60 Hz；Rerun分别验证30 Hz模型输入行数、控制频率机器人状态行数和
MuJoCo眼部相机行数，默认不允许眼部渲染丢帧。

启用 `QUALITY_*` 后，运行中会按控制时间戳插值origin pose并实时输出质量；参考pose不参与推理。
完整kinematic录像仍执行离线行为门禁，短smoke只验证链路合同。
非严格模式下，行为或性能不达标会打印 `ACT_SIM_VERIFICATION_WARNING` 并保留可视化；严格模式会
以非零状态退出。平台门禁通过表示链路可复现且数据可信，不等同于模型已经复现示范动作。

## 故障判断

- CUDA 或预/后处理器加载失败：推理脚本会立即终止，不会退化为 CPU 或未归一化推理。
- 右侧图像缺失或黑屏：先检查 `CAMERA_SOURCES` 的键、源目录和同步帧；提高 `sample_factor` 不会
  生成新图像，相邻动作周期看到同一张图是预期行为。
- MuJoCo眼部画面缺失：检查summary中的 `eye_camera.enabled`、`frames_rendered`、`frames_dropped`，以及
  verification中的 `world/robot/eyes/w1_eye_camera/rgb` 行数。
- 图像和机器人一起慢放：检查 summary 中 `timing.effective_fps` 是否接近 `control_hz`，以及
  `visualization_frames <= source_frames`。当前30 Hz可视化不应反压60 Hz动作循环。
- raw 动作抖动：确认摘要中 `inference_schedule=synchronous_latest`、`asynchronous=false`。
- bridge 与 raw 差异大：先检查 `target_step_error` 是否全为0，再比较 `chunk_submit_step`、
  `chunk_install_step`、`blend_alpha`、`raw_candidate_chunks` 和 `processor_stage_*`。
- 运行结束出现 verification warning：查看对应运行目录中的 `verification.json`，区分模型行为偏差、时延
  超限与数据完整性错误。

## 后续可扩展方向

1. **策略适配器**：复用状态、相机、映射和遥测契约，增加 Diffusion、SmolVLA 或其他 LeRobot
   policy 的 chunk runtime。
2. **输入源适配器**：在不改变模型输入键与时间线的前提下，支持视频文件、rosbag、实时相机或网络流，
   并继续记录源文件与解码后图像哈希。
3. **真机—仿真一致性**：采集真机 bridge 的输入、阶段动作和反馈，进行确定性重放与逐阶段差分。
4. **动力学标定**：根据真机辨识关节增益、延迟、摩擦和手部行程，并加入噪声、掉帧及域随机化测试。
5. **任务与场景**：向 MuJoCo 添加操作物体、碰撞、任务成功条件和相机渲染，使平台从策略回放扩展为
   可量化任务评测；RL 训练环境应作为独立模块接入，而不是混入 ACT 验证链路。
6. **自动对比报告**：同一输入一键运行 raw/bridge，多维比较关节误差、末端轨迹、jerk、队列时延和
   夹具状态，并输出可分享的视频与 HTML 报告。
7. **安全插件**：在动作处理器边界加入速度/加速度限制、工作空间约束、碰撞预测和急停门禁，同时保持
   原始 ACT 输出可追溯。
