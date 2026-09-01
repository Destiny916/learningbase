# JEPA-WAM 适配 DexForce W1 与 LeRobot v3.0 训练说明

本文对应 `/home/wengyikun/workplace/popcorn/JEPA_WAM`，记录当前 Popcorn 项目中 JEPA-WAM 的 W1 数据适配、LeRobot v3.0 读取、归一化、loss 和远程 Docker 训练方法。

## 1. 路径与数据

远程主机：`183.230.224.121:50210`，用户：`wengyikun`。

```text
数据集: /data/wengyikun/datasets/popcorn/0827_lerobot_v30_action_nextstate
模型:   /data/wengyikun/models/jepa_wam_w1
输出:   /data/wengyikun/runs/jepa_wam_w1_500k
配置:   JEPA_WAM/configs/w1_500k.yaml
```

模型目录包含 Qwen2.5-0.5B、V-JEPA-L/384、预训练 VLM 和 W1 的 state/action q01/q99 统计。训练输出与基础模型必须分目录保存。

## 2. W1 的 19D 合同

state 和 action 都是 19D，顺序固定：

```text
0       WAIST
1..7    LEFT_J1..LEFT_J7
8..9    NECK1, NECK2
10..16  RIGHT_J1..RIGHT_J7
17      LEFT_GRIPPER
18      RIGHT_GRIPPER
```

相对关节索引为 `[1..7, 10..16]`；绝对维度为 `[0, 8, 9, 17, 18]`。

- state：左右臂关节做时间差分，首帧相对关节为 0；腰部、颈部、夹爪保持绝对值。
- action：左右臂关节目标减去当前 state 的关节 anchor；腰部、颈部、夹爪目标保持绝对值。
- 推理：先按 action 自己的 q01/q99 反归一化，再把相对关节加回当前绝对关节状态；绝对维度直接使用。

不能把 19D 全部当成相对量，也不能交换不同语义维度的统计量。

## 3. LeRobot v3.0 转换和读取

原始数据转换脚本：

```text
/home/wengyikun/workplace/popcorn/scripts/convert_popcorn_0827_to_lerobot.py
```

转换必须写入新目录，不能覆盖原始数据。转换后的 `meta/info.json` 必须满足：

```text
codebase_version = v3.0
observation.state.shape = [19]
action.shape = [19]
```

三路图像 key 和顺序固定为：

```text
observation.images.cam_high_right
observation.images.cam_hand_left
observation.images.cam_hand_right
```

每个 parquet 行还应包含 `episode_index`、`frame_index` 和全局 `index`；每个 episode 的 frame index 必须从 0 连续递增，视频帧、parquet 行和 episode 边界必须对齐。

训练读取器为 `JEPA_WAM/prismatic/vla/datasets/lerobot_w1.py`，负责 metadata、19D shape、三路图像、episode 行、normalized state/action、pair index 和 `action_valid_mask`；视频由训练环境中的 LeRobot/PyAV 解码。

数据 metadata 是 RGB `3×224×224`（CHW）。当前 V-JEPA-L/384 pipeline 会把图像 resize 为实际的 `384×384` 输入，以匹配 V-JEPA checkpoint 的 token 网格和位置编码。因此 224 是数据合同，384 是 backbone 的实际模型输入。

## 4. 独立 q01/q99

必须分别使用：

```text
/data/wengyikun/models/jepa_wam_w1/stats/state_q01_q99.json
/data/wengyikun/models/jepa_wam_w1/stats/action_q01_q99.json
```

两套文件都必须是 19D、有限值，每维满足 `q99 >= q01`。公式：

```text
x_clip = clip(x, q01, q99)
x_norm = 2 * (x_clip - q01) / (q99 - q01) - 1
```

当某维 `q99 == q01` 时，该维归一化为 0，避免 NaN。state q01/q99 不能用于 action，action q01/q99 不能用于 state。manifest 应记录 19D state、19D action、20 步 horizon、独立统计文件、相对索引和绝对索引。

## 5. action chunk、padding 和 loss

训练 `action_horizon=20`，运行时执行 `n_action_steps=16`。episode 尾部不足 20 帧时，用当前 episode 最后一帧 action 填充，并生成长度为 20 的 `action_valid_mask`。mask 会传入 Flow-GR00T flow-matching loss，padding timestep 不参与 action MSE；不能跨 episode 填充。

视觉 pair 使用当前帧和目标帧，`visual_token_pair_offset=31`，超出 episode 长度时钳制在当前 episode 最后一帧。

当前模型组件：

1. 冻结的 V-JEPA-L/384 视觉 backbone，提取三路图像 token。
2. Qwen2.5-0.5B 语言 backbone，使用当前 LoRA/训练配置。
3. Flow-GR00T action head，输入视觉/语言条件、19D proprio 和 20×19 action target，预测 flow-matching 速度。
4. Visual token cosine head，预测当前帧到目标帧的视觉 token 变化。

总 loss：

```text
total_loss = masked_flow_action_loss + 0.5 * visual_token_cosine_loss
```

action loss 在 q01/q99 归一化后的 action 空间计算。物理单位误差必须在预测和标签都按 action q01/q99 反归一化后再比较。

## 6. 500k 训练配置

- 单卡 GPU5 或 GPU6，`num_processes=1`，BF16。
- 三路 RGB，random image augmentation 关闭。
- `action_horizon=20`，`n_action_steps=16`，`condition_on_state=true`，dropout=0.1。
- learning rate=`1e-5`，weight decay=`1e-4`，betas=`[0.9,0.999]`，eps=`1e-8`，grad clip=`10`。
- batch size=16，gradient accumulation=1，effective batch=16。
- num_workers=16，prefetch_factor=4，persistent_workers=true。
- max_steps=500000，每 20000 步保存，log_freq=10，验证关闭，WandB/SwanLab 关闭。
- 0～25000 步线性 warmup，之后 cosine decay 到 `1e-6`。

## 7. Docker 单卡启动模板

先检查 GPU，禁止使用 `--gpus all`：

```bash
ssh -p 50210 wengyikun@183.230.224.121 \
  'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits'
```

GPU5 启动的核心挂载和命令：

```bash
docker run -d --name jepa_wam_w1_train_500k_gpu5 \
  --gpus "device=5" --ipc=host \
  -e PYTHONPATH=/workspace/JEPA_WAM:/home/wengyikun/lerobot/src \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v /home/wengyikun/workplace/popcorn/JEPA_WAM:/workspace/JEPA_WAM:rw \
  -v /home/wengyikun/lerobot:/home/wengyikun/lerobot:ro \
  -v /data/wengyikun/datasets:/data/datasets:ro \
  -v /data/wengyikun/models:/data/models:ro \
  -v /data/wengyikun/runs/jepa_wam_w1_500k:/data/runs:rw \
  -w /workspace/JEPA_WAM jepa-wam-w1:20260828 \
  bash -lc 'torchrun --standalone --nnodes=1 --nproc-per-node=1 \
    --module prismatic.training.train \
    --config_path=/workspace/JEPA_WAM/configs/w1_500k.yaml \
    --initial_checkpoint=/data/runs/<初始化run>/checkpoints/latest-checkpoint.pt \
    --run_id=w1_jepa_wam_500k_gpu5'
```

以 `wengyikun` 用户运行时设置可写 `HOME=/tmp` 和 `XDG_CACHE_HOME=/tmp/.cache`，避免可选库写入 `/.swanlab`。

## 8. smoke test、监控和恢复

smoke test 必须确认 Docker 只看到目标 GPU，dataset 成功加载，batch 为 `pixel_values=[B,3,3,384,384]`、`actions=[B,20,19]`、`proprio=[B,19]`，mask 正确，三个 loss 有限且无 OOM/NaN/Inf/Traceback。

```bash
docker inspect -f '{{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}}' jepa_wam_w1_train_500k_gpu5
docker logs --tail 100 jepa_wam_w1_train_500k_gpu5
nvidia-smi -i 5 --query-compute-apps=pid,used_memory --format=csv,noheader,nounits
```

FSDP 保存 full state 后会释放临时 state dict，并执行 `gc.collect()` 和 `torch.cuda.empty_cache()`，应额外观察保存后的下一步 forward。当前 `load_vla()` 只恢复模型权重；未保存 optimizer、scheduler、global step 和 dataloader 状态时，从 checkpoint 启动属于 warm initialization，不是严格断点续训。

## 9. 最终检查表

```text
[ ] LeRobot v3.0；state/action 均为 19D
[ ] 三路图像 key、顺序和 RGB 语义正确
[ ] parquet、视频和 episode/frame index 对齐
[ ] 左右臂关节仅相对；腰部、颈部、夹爪绝对
[ ] state/action 使用不同 q01/q99 文件
[ ] horizon=20；n_action_steps=16；尾部 padding 有 mask
[ ] batch=16；单卡 Docker 只暴露一张 GPU
[ ] smoke test loss finite
[ ] checkpoint 保存后下一步仍能 forward
[ ] 训练输出目录与基础模型目录分离
```

