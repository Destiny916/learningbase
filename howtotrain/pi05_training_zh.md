# LeRobot PI05 双臂 14D 训练说明

本文记录 2026-08-05 当前代码和远端服务器上实际运行的 PI05 训练方法。内容面向以下数据集：

```text
/data/wengyikun/datasets/joint_songling/
0724_0727_doublefripper_top_grippebread_combined_full_99episodes_task_en
```

本文只描述 `policy.type=pi05`，不描述 PI052，也不把 ACT 的训练逻辑混入 PI05。

## 1. 当前结论

当前同时训练两种 state 表示，但两者的 action 语义相同：

| 训练 | state | action | GPU | 状态 |
|---|---|---|---|---|
| relative-state | 机械臂关节相对，夹爪绝对 | 机械臂关节相对当前绝对关节，夹爪绝对 | 3 | 已通过 step 100，继续运行 |
| absolute-state | 14D 全部使用当前绝对值 | 机械臂关节相对当前绝对关节，夹爪绝对 | 7 | step 190 首次记录 NaN，已停止 |

GPU3 在 step 100 的现场日志：

```text
step:100 loss:0.159 grdn:1.726 grip:0.160 mem_gb:19.19
```

从 step 10 到 step 100，`loss` 和 `grdn` 均为有限值，没有 `NaN`、`Inf`、
`FloatingPointError`、OOM 或 traceback。继续监督到 step 590 时，GPU3 仍为有限值：

```text
step:590 loss:0.116 grdn:0.649 grip:0.090 mem_gb:19.19
```

`grdn` 是裁剪前返回的总梯度范数；优化器实际使用 `grad_clip_norm=1.0`，所以日志中
`grdn > 1` 不表示梯度裁剪失效。

GPU7 absolute-state 在 step 180 仍正常，但在下一次每 10 步日志点首次记录非有限值：

```text
step:180 loss:0.167 grdn:1.771 grip:0.149
step:190 loss:nan grdn:nan grip:nan
```

这只能把首次异常定位在 optimizer step 181 到 190 之间，不能进一步断言恰好是第 190 个
batch 触发。该任务已经停止；不能使用其被 NaN 污染后的参数或把它当作有效训练结果。

## 2. 数据集契约

数据集是 LeRobot v3.0：

```text
episodes: 99
frames: 26339
fps: 30
```

任务文本只有一条：

```text
Pick up the bread with the right gripper, transfer it to the left gripper,
and place it in the bowl.
```

### 2.1 14D state/action 顺序

```text
0  left_joint_0
1  left_joint_1
2  left_joint_2
3  left_joint_3
4  left_joint_4
5  left_joint_5
6  left_gripper
7  right_joint_0
8  right_joint_1
9  right_joint_2
10 right_joint_3
11 right_joint_4
12 right_joint_5
13 right_gripper
```

夹爪维索引必须保持为：

```text
[6, 13]
```

### 2.2 图像特征

```text
observation.images.top           [405, 720, 3]
observation.images.gripper_left  [480, 640, 3]
observation.images.gripper_right [480, 640, 3]
```

输入模型的固定顺序为：

```text
top -> gripper_left -> gripper_right
```

启动参数必须显式设置：

```bash
--policy.image_feature_order='["observation.images.top","observation.images.gripper_left","observation.images.gripper_right"]'
```

`PI05Config.validate_features()` 会检查该列表无重复、没有遗漏，并按照该列表重新排列
`input_features`。因此当前训练不是依赖 parquet 或字典的偶然顺序。

## 3. relative-state 训练语义

当前远端 GPU3 的 14D 路径使用 `Pi05JointRepresentationProcessorStep`，不是仅匹配 7D
输入的 `RelativeJointProcessorStep`，也不是旧的 `use_relative_actions=true` 分支。

在时刻 `t`：

```text
state arm joints = q_t - q_(t-1)
state grippers   = g_t

action arm joints at future k = q_(t+k) - q_t
action grippers at future k   = g_(t+k)
```

因此：

- 左右各 6 个机械臂关节是相对值。
- 第 6、13 维夹爪始终是绝对开口值。
- action chunk 的每一帧都相对同一个当前 `q_t`，不是相邻 action 之间逐帧累加的 delta。
- episode 第一帧没有合法前一帧时，state 的机械臂相对量置零；夹爪仍保留当前绝对值。
- 推理后先反归一化，再把相对机械臂 action 加回当前绝对关节 state，得到可执行的绝对关节目标。

对应统计文件：

```text
normalization/relative_state_q01_q99.json
normalization/relative_action_chunk50_q01_q99.json
```

已核验统计数量：

```text
relative state count          = 26339
relative action chunk50 count = 1190725
action horizon                = 50
gripper indices               = [6, 13]
```

## 4. absolute-state 训练语义

GPU7 使用：

```text
state = [q_t, g_t]，全部是当前绝对值
action arm joints = q_(t+k) - q_t
action grippers   = g_(t+k)
```

它与 relative-state 训练相比，只改变 state 表示；action 定义相同。实现上先由
`Pi05JointRepresentationProcessorStep(joint_representation="absolute")` 保留绝对 state，
再由 `RelativeActionsProcessorStep` 只对非夹爪 action 维做相对转换。

对应统计文件分别是：

```text
normalization_absolute/absolute_state_q01_q99.json
normalization/relative_action_chunk50_q01_q99.json
```

这里 state 和 action 必须使用各自统计，不能让 action 复用 state 的 q01/q99。

## 5. q01/q99 归一化

当前两路训练都设置：

```bash
--policy.clip_quantiles=true
```

处理顺序是：

```text
原始绝对数据
-> 构造 relative state/action（如果该训练需要）
-> 按各自 q01/q99 做 quantile normalization
-> 将 q01/q99 外的数据裁剪到归一化边界
-> 输入模型
```

训练 loss 使用归一化后的 action，不会先反归一化。反归一化只用于推理输出和物理单位指标。

本次从 `clip_quantiles=false` 改为 `true` 后，20 步诊断以及两路正式训练的前 100 步均未出现
`grdn:inf`。但 GPU7 absolute-state 随后在 step 181 到 190 之间产生 NaN，因此
`clip_quantiles=true` 只能限制归一化输入范围，不能单独保证视觉反向或整个训练长期数值稳定。
GPU3 relative-state 到 step 300 仍正常，也不能据此推断完整 100000 步必然稳定。

PI05 的 state 不走独立的连续 state projection。归一化 state 会在 `[-1,1]` 上离散为 256 个
bin，然后与任务文本拼成：

```text
Task: <task>, State: <14个离散bin>;\nAction:
```

再交给 PaliGemma tokenizer。`clip_quantiles=true` 可保证 state 在离散前位于预期范围内。
action 不做这种文本离散，仍以连续归一化张量进入 flow-matching action expert。

## 6. 图像预处理

训练脚本设置：

```bash
--dataset.image_transforms.enable=false
```

因此没有随机颜色、仿射等 dataset augmentation。PI05 自己仍会执行确定性的缩放、补边和归一化。

### 6.1 top 相机

`resize_with_pad_torch()` 保持宽高比并居中补零：

```text
405 x 720
-> resize 到 126 x 224
-> 上 49 px、下 49 px 黑边
-> 224 x 224
```

这与先把 405x720 居中补成约 720x720、再缩放到 224x224 的几何效果等价。

### 6.2 左右夹爪相机

```text
480 x 640
-> resize 到 168 x 224
-> 上 28 px、下 28 px 黑边
-> 224 x 224
```

图像进入 resize 前是 `[0,1]` float32，补边值为 `0.0`。随后执行：

```text
image = image * 2 - 1
```

因此模型看到的黑边数值为 `-1.0`。三张真实图的 mask 都是 `true`；当前
`empty_cameras=0`，不会额外创建空相机。

## 7. 视觉输入如何进入 PI05

完整路径如下：

```text
LeRobot batch 三张真实图
-> 按 top/left/right 固定顺序读取
-> resize + centered pad + [-1,1]
-> 每张图分别进入 PaliGemma 的 SigLIP vision tower
-> 得到三组 image embeddings
-> 按 top/left/right 顺序拼接
-> 再拼接 task/state tokenizer 的语言 token
-> 形成统一 prefix
-> prefix 与 noisy action suffix 一起进入 PI05 前向
-> flow-matching loss
-> backward 回传到视觉塔、projector 和 action expert
```

三相机不是三个完全隔离的动作头。三组视觉 token 拼接进同一个 prefix 后，会在 VLM prefix
self-attention 中发生跨相机 token 交互，并共同条件化 action expert。

当前没有先把数据集键名重命名成 `base_0_rgb/left_wrist_0_rgb/right_wrist_0_rgb`；
`image_feature_order` 的三个位置直接承担这三个固定视觉槽位。也就是说，当前实际输入槽位是：

```text
slot 0 = observation.images.top
slot 1 = observation.images.gripper_left
slot 2 = observation.images.gripper_right
```

当前运行时参数审计：

```text
language_model:       total 3B,   trainable 0
vision_encoder:       total 412M, trainable 412M
multimodal_projector: total 2M,   trainable 2M
action_expert_other:  total 1B,   trainable 1B
```

所以当前确实训练视觉编码器，不是仅训练 action expert。

## 8. 视觉精度与 gradient checkpointing

启动配置：

```text
policy.dtype=bfloat16
gradient_checkpointing=true
freeze_language_model=true
freeze_vision_encoder=false
train_expert_only=false
```

当前代码的实际 dtype 行为：

- PI05 主体使用 BF16。
- `vision_tower` 和 `multi_modal_projector` 参数被保留为 FP32。
- 图像在进入 `get_image_features()` 前显式转成 FP32。
- 但整个 policy forward 位于 `accelerator.autocast()` 中，`embed_image()` 内没有单独关闭
  autocast。因此不能写成“视觉所有算子严格 FP32”；更准确的说法是“视觉参数和输入是 FP32，
  视觉算子仍受外层 BF16 autocast 管理”。
- 视觉 embedding 在进入 BF16 language/action 主体前会按主模型需要转为 BF16。

当前 gradient checkpointing 同时开启：

- language model checkpoint flag；
- vision tower 内部 checkpoint；
- action expert checkpoint；
- 外层 image embedding 和联合 forward checkpoint。

它节省显存，但当前实现存在视觉塔内部与外层 image checkpoint 嵌套。前 100 步数值稳定，
不代表该嵌套设计已经完成长期性能验证。

远端当前没有在 `embed_image()` 内额外使用 `autocast(enabled=False)`。因此这是沿用当前
PI05 的混合精度路径，而不是此前试验过的“视觉算子强制全 FP32”路径。

## 9. 视觉 Base 权重加载审查

基础权重路径：

```text
/data/wengyikun/openpi/lerobot_pi05_base/model.safetensors
```

启动日志会出现两条：

```text
Vision embedding key might need handling: ...patch_embedding.bias
Vision embedding key might need handling: ...patch_embedding.weight
```

该 warning 是 `_fix_pytorch_state_dict_keys()` 对所有 `patch_embedding` key 主动打印的提示，
当前日志没有同时出现 `Missing pi05 checkpoint keys` 或 `Unexpected pi05 checkpoint keys`。
因此它本身不是“视觉权重未加载”的直接证据，但仍应保留为待核验风险，不能仅凭参数数量断言
checkpoint 每个视觉 tensor 都与官方键映射完全一致。

## 10. PI05 flow-matching loss

模型 action 维度先从 14D pad 到内部 32D，chunk 形状为：

```text
[batch, 50, 32]
```

训练采样高斯噪声 `noise` 和时间 `t`：

```text
x_t = t * noise + (1 - t) * action
u_t = noise - action
v_t = PI05(images, task/state tokens, x_t, t)
loss_element = (v_t - u_t)^2
```

随后只保留真实 14D action 的 loss，内部 padding 的后 18D 不参与最终 loss。episode 尾部
补齐的 action step 通过 `action_is_pad` 屏蔽，也不参与均值。

总体 loss 是所有有效 timestep 和 14 个真实维度上的平均 flow-matching MSE。夹爪 loss 只统计
第 6、13 维，但它只是独立日志指标；总体 loss 仍包含夹爪维。

注意：这里的 loss 不是“预测关节角与标签关节角的物理单位 MSE”。loss 两边都是归一化 action
空间中的 flow velocity。要得到 rad、deg 或 m 单位的 MSE，必须完成 action sampling、反归一化、
相对转绝对后再计算。

## 11. 当前正式训练参数

两路共同参数：

```text
policy.type                  pi05
pretrained_path              /data/wengyikun/openpi/lerobot_pi05_base
dtype                        bfloat16
batch_size                   8
gradient_accumulation_steps  1
effective_batch_size         8
num_workers                  8
chunk_size                   50
n_action_steps               50
empty_cameras                0
clip_quantiles               true
state_noise_std_rad          0
gripper_noise_std_m          0
freeze_language_model        true
freeze_vision_encoder        false
train_expert_only            false
gradient_checkpointing       true
optimizer                    AdamW
learning_rate                2.5e-5
betas                        [0.9, 0.95]
eps                          1e-8
weight_decay                 0.01
grad_clip_norm               1.0
warmup_steps                 5000
decay_steps                  100000
decay_lr                     2.5e-6
steps                        100000
save_freq                    10000
log_freq                     10
eval_steps                   0
wandb                        disabled
```

全量数据每个 epoch 约为：

```text
26339 / 8 = 3292.375 optimizer steps
```

所以 100000 steps 约为 30.37 epochs，每 10000 steps 保存约相隔 3.04 epochs。

当前没有 validation dataset，`eval_steps=0`。因此训练期间只有 train loss、gripper loss、
gradient norm、学习率、吞吐和显存日志，没有 valid loss/MSE。

## 12. 启动 relative-state 训练

核心 policy 参数如下。容器外层应使用当前服务器的单 GPU wrapper，并设置
`LEROBOT_GPUS=device=3`：

```bash
cd /home/wengyikun/lerobot

LEROBOT_GPUS=device=3 ./run_scripts/remote_pi05_libero_smoke_container.sh \
  accelerate launch --num_processes=1 --mixed_precision=bf16 \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/0724_0727_doublefripper_top_grippebread_combined_full_99episodes_task_en \
  --dataset.root=/data/wengyikun/datasets/joint_songling/0724_0727_doublefripper_top_grippebread_combined_full_99episodes_task_en \
  --dataset.image_transforms.enable=false \
  --policy.type=pi05 \
  --policy.pretrained_path=/data/wengyikun/openpi/lerobot_pi05_base \
  --policy.dtype=bfloat16 --policy.device=cuda \
  --policy.compile_model=false --policy.gradient_checkpointing=true \
  --policy.freeze_language_model=true --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.chunk_size=50 --policy.n_action_steps=50 --policy.empty_cameras=0 \
  --policy.image_feature_order='["observation.images.top","observation.images.gripper_left","observation.images.gripper_right"]' \
  --policy.joint_representation=relative \
  --policy.joint_gripper_indices='[6,13]' \
  --policy.condition_on_state=true \
  --policy.relative_state_stats_path=/data/wengyikun/datasets/joint_songling/0724_0727_doublefripper_top_grippebread_combined_full_99episodes_task_en/normalization/relative_state_q01_q99.json \
  --policy.relative_action_stats_path=/data/wengyikun/datasets/joint_songling/0724_0727_doublefripper_top_grippebread_combined_full_99episodes_task_en/normalization/relative_action_chunk50_q01_q99.json \
  --policy.clip_quantiles=true \
  --policy.state_noise_std_rad=0 --policy.gripper_noise_std_m=0 \
  --policy.scheduler_warmup_steps=5000 --policy.scheduler_decay_steps=100000 \
  --batch_size=8 --gradient_accumulation_steps=1 --num_workers=8 \
  --steps=100000 --save_checkpoint=true --save_freq=10000 \
  --log_freq=10 --eval_steps=0 --wandb.enable=false \
  --output_dir=/data/wengyikun/outputs/NEW_RELATIVE_STATE_RUN/train_out \
  --job_name=NEW_RELATIVE_STATE_RUN
```

每次新训练必须使用新的 `output_dir`，不要在旧目录中不带 `--resume=true` 从 step 0 重跑。

## 13. 启动 absolute-state + relative-action 训练

与上一条命令相比，只替换以下参数：

```bash
--policy.joint_representation=absolute \
--policy.use_relative_actions=true \
--policy.relative_exclude_joints='["gripper"]' \
--policy.absolute_state_stats_path=/data/wengyikun/datasets/joint_songling/0724_0727_doublefripper_top_grippebread_combined_full_99episodes_task_en/normalization_absolute/absolute_state_q01_q99.json \
--policy.relative_action_stats_path=/data/wengyikun/datasets/joint_songling/0724_0727_doublefripper_top_grippebread_combined_full_99episodes_task_en/normalization/relative_action_chunk50_q01_q99.json
```

并将 GPU 和输出目录改为新的 absolute-state 任务，例如 `device=7`。

## 14. 当前输出目录

relative-state：

```text
/data/wengyikun/outputs/
pi05_0724_0727_full99_relative_state_relative_action_cliptrue_b8_100k_gpu3_20260805/
train_out
```

absolute-state：

```text
/data/wengyikun/outputs/
pi05_0724_0727_full99_absolute_state_relative_action_cliptrue_b8_100k_gpu7_20260805/
train_out
```

absolute-state 目录仅保留故障分析用途。该次运行在第一个 checkpoint 保存点 10000 之前已停止，
不应作为推理模型或续训起点。

## 15. 监控命令

查看 GPU：

```bash
nvidia-smi
```

查看容器：

```bash
docker ps --format '{{.Names}} {{.Status}}'
```

查看 relative-state 日志：

```bash
tail -f /tmp/pi05_0724_0727_relative_state_cliptrue_b8_gpu3_20260805.log
```

检查非有限值和异常：

```bash
grep -Ei 'step:|grdn:inf|nan|FloatingPointError|Traceback|out of memory' \
  /tmp/pi05_0724_0727_relative_state_cliptrue_b8_gpu3_20260805.log | tail -50
```

监控脚本必须把 `loss:nan`、`grdn:nan`、`grdn:inf` 都视为立即停止条件。正式诊断时可设置：

```bash
export LEROBOT_PI05_FINITE_DEBUG=1
```

它会在 backward 后逐参数检查梯度并报告首个非有限参数及 batch 的 episode/frame 元数据，
代价是每步多一次完整梯度有限性扫描，不建议在未评估开销时永久开启。

检查保存结果：

```bash
find /data/wengyikun/outputs/REPLACE_RUN/train_out/checkpoints \
  -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort -n
```

## 16. 训练前检查清单

1. 数据集必须为 v3.0、99 episodes、26339 frames。
2. 14D 顺序必须是左 7D 后右 7D，夹爪索引必须为 `[6,13]`。
3. 三相机必须真实存在，顺序必须是 top、left、right。
4. relative state/action 必须使用独立 q01/q99 文件。
5. `clip_quantiles=true`。
6. state/gripper noise 当前都为 0。
7. `freeze_vision_encoder=false`，确认视觉塔参数 trainable。
8. `freeze_language_model=true`，语言模型参数不更新。
9. `steps=100000`，避免短诊断触发 scheduler 自动缩放。
10. 新训练使用新输出目录；续训必须使用匹配 checkpoint 和 `--resume=true`。
11. 确认目标 GPU 空闲，不停止 GPU1/GPU2 等无关任务。
12. 观察至少前 100 步的 `loss/grdn`；100 步只是最低检查点，之后仍需持续监控。
13. 发现 `NaN/Inf` 立即停止，不能让优化器继续更新或保存被污染的权重。

## 17. 当前剩余风险

1. `patch_embedding` key 会打印映射 warning。当前没有 missing/unexpected key 日志，但仍建议后续做
   checkpoint tensor 级别的键名和数值对照。
2. 视觉参数和输入为 FP32，但视觉前向仍位于 BF16 autocast 中；不能宣称视觉计算严格全 FP32。
3. vision tower 内部与外层 image embedding 都启用 checkpoint，存在嵌套重计算；GPU3 目前只
   验证到 step 590 数值稳定，GPU7 absolute-state 已在 step 190 日志点出现 NaN。
4. 当前不做 validation；训练 loss 下降不等价于真机推理一定改善。
5. `clip_quantiles=true` 会把 q01/q99 外的真实极值压到边界，这是当前为数值稳定性选择的训练定义。
6. `apply_action_limits=true` 主要作用于推理 postprocessor，不参与 flow-matching loss；部署时仍需
   独立验证真实 Piper/Pika 限位和单位。

## 18. 主要代码依据

```text
src/lerobot/policies/pi05/configuration_pi05.py
  image_feature_order、relative/absolute 配置、q01/q99 路径和训练超参数

src/lerobot/policies/pi05/processor_pi05.py
  relative/absolute processor 选择、归一化顺序、state 离散 prompt、反归一化

src/lerobot/processor/relative_joint_processor.py
  q_t-q_(t-1)、q_(t+k)-q_t、绝对夹爪和推理恢复绝对 action

src/lerobot/policies/pi05/modeling_pi05.py
  三相机预处理、SigLIP embedding、prefix 拼接、gradient checkpoint、flow-matching loss

src/lerobot/policies/common/vla_utils.py
  OpenPI 风格的保持宽高比 resize 和居中补边

src/lerobot/scripts/lerobot_train.py
  autocast、backward、gradient clipping、optimizer/scheduler、日志和 checkpoint 时机
```

## 19. 实际代码副本与复现边界

本次运行容器挂载的是远端：

```text
/home/wengyikun/lerobot
```

它不是可查询 commit 的 Git checkout。2026-08-05 审计时，远端与本地工作区文件关系如下：

```text
完全一致：modeling_pi05.py
完全一致：relative_joint_processor.py
完全一致：vla_utils.py
存在差异：configuration_pi05.py
存在差异：processor_pi05.py
存在差异：lerobot_train.py
```

与本次 14D 训练直接相关的差异是：远端 `processor_pi05.py` 只在 state/action 都为 7D 时直接
选择 `RelativeJointProcessorStep`；14D 会选择 `Pi05JointRepresentationProcessorStep`。
本文描述的是远端实际运行路径。后续同步代码后必须重新做 source hash、processor pipeline 和
一步 batch 审计，不能假设本地修改会自动进入正在运行的容器。

## 20. GPU2 与 GPU7 的对照边界

GPU2 当前运行的是 PI052 absolute-state，GPU7 故障任务运行的是 PI05 absolute-state。两者
虽然共享数据集、三相机顺序、14D state/action 的物理语义和 BF16 外层混合精度，但训练逻辑
不能视为几乎相同：

| 项目 | GPU7 故障任务 | GPU2 当前任务 |
|---|---|---|
| policy | `pi05` | `pi052` |
| 基础权重 | `/data/wengyikun/openpi/lerobot_pi05_base` | `/data/wengyikun/openpi/lerobot_pi052_base` |
| action loss | PI05 flow-matching velocity MSE | PI052 的 flow loss + FAST action loss + text loss |
| flow 重复次数 | PI05 默认路径 | `flow_num_repeats=5` |
| batch size | 8 | 4 |
| q01/q99 clipping | `true` | `false` |
| 语言模型 | frozen | frozen，且 `lm_head` 也保持 frozen |
| 视觉塔 | trainable | trainable |
| gradient checkpoint | true | true |

所以 GPU2 的 loss 正常或异常都不能直接解释 GPU7 的 PI05 NaN，尤其不能据此排除视觉反向
数值问题。要做有效 A/B，必须固定 `policy.type`、基础 checkpoint、batch、clipping、optimizer、
随机种子和同一 state/action processor，只改变一个变量。当前 GPU7 的诊断复现尚未进入首个
optimizer step，因此 NaN 的具体参数级根因仍未完成定位；不能把它归因给视觉塔或 q01/q99 中
任意一个单独因素。
