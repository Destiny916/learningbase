# ACT-DINOv3 新数据训练通用说明

本文用于 Popcorn/W1 的 LeRobot v3.0 数据集。目标是把一个新数据集无覆盖地转换为正确的监督合同，独立计算 state/action q01/q99，再使用 ACT-DINOv3 单卡或多卡训练。

当前代码与远端服务器：

```text
本地代码: /home/wengyikun/workplace/popcorn/algorithms/act_dinov3
远端代码: /data/wengyikun/popcorn/algorithms/act_dinov3
服务器:   ssh -p 50210 wengyikun@183.230.224.121
镜像:     lerobot-pi05-train:20260706
```

不要覆盖原始数据集、已有 stats 或已有输出目录。新数据、新统计和新训练各使用一个新目录。

## 1. 训练数据合同

### 1.1 时间对齐

原始数据必须派生为：

```text
action[t] = observation.state[t+1]
```

每个 episode 的最后一帧没有 `t+1`，派生数据令最后一帧 action 重复最后一帧 state。构造未来 action chunk 时，超出 episode 的位置必须由 padding mask 屏蔽，不能跨 episode 取目标。

这与相对动作并不冲突。先把 parquet 中 action 对齐为下一帧绝对状态，训练 processor 再把机械臂目标转换为相对当前帧的动作：

```text
relative_action[t,k] = action[t+k, arm_joint] - state[t, arm_joint]
```

### 1.2 0827 的 19D 顺序

```text
0      WAIST
1..7   LEFT_J1 .. LEFT_J7
8..9   NECK1, NECK2
10..16 RIGHT_J1 .. RIGHT_J7
17     LEFT_GRIPPER
18     RIGHT_GRIPPER
```

只有左右机械臂关节使用相对表示：

```text
relative indices = [1,2,3,4,5,6,7,10,11,12,13,14,15,16]
absolute indices = [0,8,9,17,18]
```

对应训练配置：

```text
joint_representation=relative
state_absolute_indices=[0,8,9,17,18]
action_absolute_indices=[0,8,9,17,18]
state_gripper_indices=[17,18]
gripper_indices=[17,18]
```

因此模型的实际 state/action 语义是：

```text
state 机械臂: q[t] - q[t-1]
state 腰/颈/夹爪: 当前帧绝对值

action 机械臂: q[t+k+1] - q[t]
action 腰/颈/夹爪: t+k 对应 action 中的绝对目标值，即下一状态序列
```

state q01/q99 和 action q01/q99 必须分别计算，不能互相复用。换数据集、改维度顺序、改相对/绝对索引或改 chunk size 后都必须重新计算。

## 2. 新数据集预检查

至少检查 `meta/info.json`、parquet、视频和 feature names：

```bash
export SOURCE_ROOT=/data/wengyikun/datasets/popcorn/<new_dataset>

python3 - "$SOURCE_ROOT" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
info = json.loads((root / "meta/info.json").read_text())
features = info["features"]
print("version:", info.get("codebase_version"))
print("episodes:", info.get("total_episodes"))
print("frames:", info.get("total_frames"))
print("fps:", info.get("fps"))
print("state:", features["observation.state"])
print("action:", features["action"])
print("images:", [key for key in features if key.startswith("observation.images.")])
assert info.get("codebase_version") == "v3.0"
assert tuple(features["observation.state"]["shape"]) == (19,)
assert tuple(features["action"]["shape"]) == (19,)
PY
```

还必须确认：

```text
episode_index 连续且范围正确
每个 episode 内 frame_index 从 0 连续递增
state/action 全部 finite，无 NaN/Inf
state/action names 与物理顺序一致
三路视频存在、可解码、帧数与 parquet 对齐
fps 是采集时间语义，不用于替代 action 对齐检查
```

## 3. 派生 next-state action 数据集

本地脚本：

```text
/home/wengyikun/workplace/popcorn/scripts/derive_next_state_actions.py
```

远端可从同步后的 Popcorn 仓库运行。输出目录必须不存在：

```bash
export SOURCE_ROOT=/data/wengyikun/datasets/popcorn/<new_dataset>
export DATA_ROOT=/data/wengyikun/datasets/popcorn/<new_dataset>_action_nextstate

python3 /data/wengyikun/popcorn/scripts/derive_next_state_actions.py \
  "$SOURCE_ROOT" "$DATA_ROOT"
```

脚本行为：

```text
复制 meta 和 videos
重写 parquet 的 action 列
非尾帧 action[t] = state[t+1]
尾帧 action[t] = state[t]
写入 meta/derived_action_contract.json
拒绝覆盖已有输出目录
```

派生后做数值验证：

```bash
python3 - "$DATA_ROOT" <<'PY'
import numpy as np
import pyarrow.parquet as pq
import sys
from pathlib import Path

root = Path(sys.argv[1])
grouped = {}
for path in sorted((root / "data").glob("**/*.parquet")):
    table = pq.read_table(path, columns=["observation.state", "action", "episode_index", "frame_index"])
    for state, action, episode, frame in zip(
        table["observation.state"].to_pylist(),
        table["action"].to_pylist(),
        table["episode_index"].to_pylist(),
        table["frame_index"].to_pylist(),
        strict=True,
    ):
        grouped.setdefault(int(episode), []).append(
            (int(frame), np.asarray(state), np.asarray(action))
        )

max_error = 0.0
for episode, rows in grouped.items():
    rows.sort(key=lambda row: row[0])
    assert [row[0] for row in rows] == list(range(len(rows)))
    for index in range(len(rows) - 1):
        max_error = max(max_error, float(np.max(np.abs(rows[index][2] - rows[index + 1][1]))))
print("NEXT_STATE_ALIGNMENT_OK episodes=", len(grouped), "max_error=", max_error)
assert max_error <= 1e-6
PY
```

## 4. 计算独立 q01/q99

Popcorn 专用统计脚本：

```text
/home/wengyikun/workplace/popcorn/algorithms/act_dinov3/src/lerobot/scripts/compute_popcorn_relative_joint_stats.py
```

计算命令：

```bash
export DATA_ROOT=/data/wengyikun/datasets/popcorn/<new_dataset>_action_nextstate
export STATS_ROOT=/data/wengyikun/act_stats/<new_dataset>_19d_relative_arm_joints_chunk16

cd /data/wengyikun/popcorn/algorithms/act_dinov3
PYTHONPATH=$PWD/src python3 -m lerobot.scripts.compute_popcorn_relative_joint_stats \
  --dataset-root="$DATA_ROOT" \
  --output-dir="$STATS_ROOT" \
  --horizon=16
```

action quantile 的样本定义是：

```text
对每个 chunk 起点 t：
机械臂 = action[t:t+16] - state[t]
腰/颈/夹爪 = action[t:t+16] 原始绝对值
```

统计目录必须包含：

```text
relative_state_q01_q99.json
relative_action_chunk16_q01_q99.json
relative_stats_manifest.json
full_dataset_manifest.json
```

启动前检查：

```bash
python3 - "$DATA_ROOT" "$STATS_ROOT" <<'PY'
import json, sys
from pathlib import Path

data, stats = map(Path, sys.argv[1:])
manifest = json.loads((stats / "relative_stats_manifest.json").read_text())
assert manifest["format_version"] == 4
assert manifest["horizons"] == [16]
assert manifest["source_dataset_root"] == str(data.resolve())
assert manifest["state_absolute_indices"] == [0,8,9,17,18]
assert manifest["action_absolute_indices"] == [0,8,9,17,18]
assert manifest["state_gripper_indices"] == [17,18]
assert manifest["gripper_indices"] == [17,18]
for name in ("relative_state_q01_q99.json", "relative_action_chunk16_q01_q99.json"):
    payload = json.loads((stats / name).read_text())
    assert len(payload["q01"]) == 19
    assert len(payload["q99"]) == 19
    assert all(low <= high for low, high in zip(payload["q01"], payload["q99"]))
print("DATA_AND_STATS_OK")
PY
```

`clip_quantiles=true` 会把归一化前超出 q01/q99 的值截到边界，用于降低极端值影响。它不会把 parquet 原始数据改掉，但训练 target 会按该边界截断。

## 5. 图像和 DINOv3

0827 使用三路 224x224 RGB：

```text
observation.images.cam_high_right
observation.images.cam_hand_left
observation.images.cam_hand_right
```

新数据必须在 `meta/info.json` 中保持明确且稳定的 key/name 顺序，并在训练配置生成后核对 `policy.input_features`。不能只凭三路 tensor shape 相同就认为相机顺序正确。

当前训练关闭随机图像增强：

```text
dataset.image_transforms.enable=false
```

这只关闭随机 transforms；数据集视频中已经是 224x224，训练仍会执行 tensor 转换和 DINOv3 所需归一化。

DINOv3 权重：

```text
/data/wengyikun/models/turbovla_joint_songling/dinov3-vitl16-pretrain-lvd1689m
model.safetensors 约 1.21 GB
```

启动日志必须显示每个 DDP 进程完整加载：

```text
415/415 weights loaded
```

训练参数：

```text
dinov3_gradient_checkpointing=true
dinov3_autocast_dtype=bfloat16
DINOv3 lr=1e-6
全部 DINOv3 参数参与反向传播
```

## 6. 通用 launcher

0827 已验证 launcher：

```text
本地: /home/wengyikun/workplace/popcorn/algorithms/act_dinov3/run_scripts/launch_act_dinov3_popcorn_0827_relative_joints.sh
远端: /data/wengyikun/popcorn/algorithms/act_dinov3/run_scripts/launch_act_dinov3_popcorn_0827_relative_joints.sh
```

新数据可复制一份 launcher，并至少修改：

```text
DATA_ROOT
STATS_ROOT
OUTPUT_DIR
JOB_NAME
数据集 episode/state/action/camera preflight
state/action feature names
相对/绝对索引
```

默认训练参数：

```text
policy.type=act_dinov3
DDP GPUs=2
per-GPU batch=32
global batch=64
gradient_accumulation_steps=1
num_workers=16 per process
chunk_size=16
n_action_steps=16
dropout=0.1
use_vae=true
kl_weight=10
optimizer=AdamW
main lr=1e-5
DINOv3 lr=1e-6
weight_decay=1e-4
grad_clip_norm=10
warmup=25000
cosine decay=500000
decay lr=1e-6
steps=500000
save_freq=20000
eval_steps=0
state/gripper noise=0
BF16 DDP
```

当前推荐由 `ACTDINOv3Config` 提供 optimizer/scheduler preset，因此 launcher 应保持：

```bash
--use_policy_training_preset=true
```

并通过 `policy.optimizer_lr`、`policy.scheduler_warmup_steps`、`policy.scheduler_decay_steps` 和 `policy.scheduler_decay_lr` 设置本任务参数。此路径不会出现 `scheduler=None`，也不需要同时传入通用的 `--optimizer.*` 和 `--scheduler.*`。

## 7. 双卡 Docker 启动

该服务器使用 NVIDIA CDI，使用 `--device nvidia.com/gpu=N`，不要写旧式 `--gpus device=...`。

```bash
export CONTAINER_NAME=act_dinov3_<dataset>_gpu1_2
export DATA_ROOT=/data/wengyikun/datasets/popcorn/<dataset>_action_nextstate
export STATS_ROOT=/data/wengyikun/act_stats/<dataset>_19d_relative_arm_joints_chunk16
export OUTPUT_DIR=/data/wengyikun/outputs/<run_name>/train_out
export JOB_NAME=<run_name>

docker run -d \
  --name "$CONTAINER_NAME" \
  --device nvidia.com/gpu=1 \
  --device nvidia.com/gpu=2 \
  --user 1009:1008 \
  --ipc=host \
  --shm-size=16g \
  -e PYTHONPATH=/data/wengyikun/popcorn/algorithms/act_dinov3/src \
  -e HOME=/home/wengyikun \
  -e USER=wengyikun \
  -e HF_HOME=/data/wengyikun/.cache/huggingface \
  -e TORCHDYNAMO_DISABLE=1 \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e DATA_ROOT="$DATA_ROOT" \
  -e STATS_ROOT="$STATS_ROOT" \
  -e OUTPUT_DIR="$OUTPUT_DIR" \
  -e JOB_NAME="$JOB_NAME" \
  -v /data/wengyikun:/data/wengyikun \
  -w /data/wengyikun/popcorn/algorithms/act_dinov3 \
  lerobot-pi05-train:20260706 \
  bash /data/wengyikun/popcorn/algorithms/act_dinov3/run_scripts/<launcher>.sh
```

launcher 内 `BATCH_SIZE=32` 是每个进程的 batch；双卡全局 batch 为 `32 x 2 = 64`。

## 8. 单卡启动

单卡时不能原样保留 `accelerate --multi_gpu --num_processes=2`。建议让 launcher 支持：

```text
NUM_PROCESSES=1
CUDA_VISIBLE_DEVICES=0
每卡 batch 根据显存设置
```

Docker 只暴露一张卡：

```bash
--device nvidia.com/gpu=<physical_gpu>
-e CUDA_VISIBLE_DEVICES=0
```

若 launcher 固定使用 `--multi_gpu`，应为单卡单独建立 launcher 并在启动前做一步 smoke test。

## 9. 启动后严格验证

容器存活不等于训练正确。至少检查：

```bash
docker ps --filter name="$CONTAINER_NAME" --format 'table {{.Names}}\t{{.Status}}'
docker logs --tail 200 "$CONTAINER_NAME"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

日志需要确认：

```text
dataset preflight OK
relative stats preflight OK
state/action 维度与绝对索引正确
两个 DDP rank 均加载 415/415 DINOv3 权重
Effective batch size 与预期一致
use_policy_training_preset=true
scheduler 为 cosine_decay_with_warmup，而不是 None
step 持续增长
loss、grad norm 都是 finite
两张 GPU 均有显存和利用率
```

学习率检查：

```text
warmup 阶段 lr 应从低值逐步增加到 1e-5
25000 步后开始 cosine decay
500000 步到约 1e-6
```

以上三项指主网络参数组。DINOv3 参数组始终保持其 10 倍更小的独立学习率，因此对应为约 `4e-11 -> 1e-6 -> 1e-7`。

若从第 0 步起一直显示 `lr:1.0e-05`，检查 launcher 是否确实使用 `ACTDINOv3Config` 的 preset、是否保存了 scheduler 配置，以及日志中的学习率是否来自主网络参数组。主网络应在 warmup 内从约 `4e-10` 增加到 `1e-5`，DINOv3 参数组按相同比例从约 `4e-11` 增加到 `1e-6`。

## 10. loss 的含义

ACT 训练 loss 由 action reconstruction、VAE/KL 等路径组成，优化的是经过相对转换和 q01/q99 归一化后的 action chunk，不是直接以物理单位表示的关节误差。

因此：

```text
loss 下降只能证明归一化训练目标拟合改善
loss=0.01 不能直接解释为 0.01 rad 或 0.01 m
物理 MSE 必须把预测和标签反归一化后，按各维度分别计算
padding 位置不能进入 loss/MSE
```

本任务 `eval_steps=0`，没有验证集指标，最终效果需要使用固定 checkpoint 做离线回放或真机前的只推理检查。

## 11. CPU/GPU 性能检查

```bash
docker stats --no-stream "$CONTAINER_NAME"
mpstat 1 5
nvidia-smi dmon -s pucm -c 5
```

0827 双卡参考值：

```text
远端 96 logical CPUs
容器约使用 35-38 个逻辑核
系统约 38% idle，约 37 个逻辑核空闲
data_s 约 0.009-0.017 s
update_s 约 1.4-1.9 s
```

`data_s` 很小而 `update_s` 较大时，瓶颈主要在模型计算，不应盲目继续提高 workers。

## 12. checkpoint 与续训

保存目录：

```text
<OUTPUT_DIR>/checkpoints/<step>
```

真正续训必须同时恢复：

```text
模型参数
optimizer state
scheduler state
global step
随机状态
```

且必须保持：

```text
同一数据合同和 feature 顺序
同一相对/绝对索引
同一 q01/q99
同一 chunk size
兼容的模型结构和 DINOv3 权重
```

只把 checkpoint 当 `pretrained_path` 加载后从 step 0 开始，是 warm-start，不是严格续训。不要把新 run 写回旧输出目录。

## 13. 常见错误

### scheduler=None

原因：没有使用 `ACTDINOv3Config` 的 scheduler preset，或启动器没有传入 `policy.scheduler_*` 参数。

修复：

```bash
--use_policy_training_preset=true
```

### action[t] 仍等于 state[t]

原因：未运行派生脚本，或 stats 指向原始数据集。

检查 `meta/derived_action_contract.json`，并重新执行 next-state 数值验证。

### q01/q99 看似能加载但语义错误

原因通常是复用了旧数据统计、action chunk 没有相对 chunk 起点、绝对索引错误或 manifest 指向其他数据集。

必须核对 `source_dataset_root`、`horizons`、维度和四组 indices。

### DINOv3 没有真正加载

不能只看目录存在。必须确认 `model.safetensors` 约 1.21 GB，并在两个 rank 日志中看到 `415/415 weights loaded`。

### 相机顺序错误

不能只看三张图 shape 都是 224x224。检查 `meta/info.json`、训练生成的 `input_features`，并抽样解码显示每个 key 的真实画面。

### 容器运行但没有训练

检查日志是否出现 finite step/loss/grad norm，同时确认两张 GPU 的显存、利用率和训练进程。

## 14. 0827 已验证示例

```text
dataset:
/data/wengyikun/datasets/popcorn/0827_lerobot_v30_action_nextstate
50 episodes / 95725 frames / 30 FPS

stats:
/data/wengyikun/act_stats/popcorn_0827_19d_relative_arm_joints_absolute_waist_neck_grippers_chunk16

output:
/data/wengyikun/outputs/act_dinov3_popcorn_0827_19d_relative_arm_joints_nextstate_chunk16_b64_500k_gpu1_2/train_out

container:
act_dinov3_popcorn_0827_gpu1_2
```

截至本文修订时，旧的 GPU1+GPU2 ACT-DINOv3 训练已经停止。launcher 已改为使用 `ACTDINOv3Config` 的 optimizer/scheduler preset，并按 25000-step warmup、到 500000 step 完成 cosine decay；当前未启动新训练。
