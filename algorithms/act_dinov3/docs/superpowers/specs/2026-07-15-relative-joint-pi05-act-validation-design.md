# PI0.5 与 ACT 相对关节训练及验证设计

## 目标

使用已经拆分完成的 0714 右眼鱼眼数据集训练四个策略。训练统一采用相对关节目标、仅由训练集计算的 q01/q99 归一化统计，并且每完成 200 个 optimizer step，使用完整测试集计算 validation loss 与反归一化真实动作空间 MSE。

本地数据集：

```text
/data/joint_songling/0714_gripper_bread_combined_split_seed42/train
/data/joint_songling/0714_gripper_bread_combined_split_seed42/test
```

远端训练服务器及目标路径：

```text
ssh -p 50210 wengyikun@183.230.224.121
/home/wengyikun/lerobot
/data/wengyikun/datasets/joint_songling/0714_gripper_bread_combined_split_seed42/train
/data/wengyikun/datasets/joint_songling/0714_gripper_bread_combined_split_seed42/test
```

当前 train 和 test 已经是两个独立、有效的 LeRobot v3.0 数据集。后续只需原样传输到远端，不再执行数据切分或格式转换。

## 四个实验

| 实验 | 模型实际接收的条件 | 预测目标 | Action chunk |
| --- | --- | --- | ---: |
| PI0.5 相对 state | 任务 prompt、右眼鱼眼图像、相对 state | 锚定到当前帧的相对 action | 50 |
| PI0.5 image-only | 任务 prompt、右眼鱼眼图像 | 锚定到当前帧的相对 action | 50 |
| ACT 相对 state | 右眼鱼眼图像、相对 state | 锚定到当前帧的相对 action | 16 |
| ACT image-only | 右眼鱼眼图像 | 锚定到当前帧的相对 action | 16 |

PI0.5 image-only 保留任务语言 prompt，但 prompt 中不包含 `State:`，模型也不接收任何其他 state 表示。ACT 不包含语言模块，因此任务字符串只作为数据集元信息，不进入 ACT。

两个 PI0.5 实验都冻结语言模型，包括 PI0.5 相对 state 和 PI0.5 image-only。两者都保持视觉编码器、multimodal projector、Action Expert 和 action projection 可训练。远端使用以下权重：

```text
PI0.5 base: /data/wengyikun/openpi/lerobot_pi05_base
VISTA 视觉权重: /data/wengyikun/models/TeleEmbodied_VISTA/pretrained_model/model.safetensors
PaliGemma 缓存根目录: /home/wengyikun/.cache
```

## 关节语义

七个维度依次为 `joint_0..joint_5` 和索引 6 的 `gripper`。六个手臂关节使用相对位姿，夹爪始终使用绝对开合量。

对于 episode 内第 `t` 帧：

```text
state_arm[t] = q_arm[t] - q_arm[t-1]
state_gripper[t] = q_gripper[t]

action_arm[t,k] = q_arm[t+k] - q_arm[t]
action_gripper[t,k] = q_gripper[t+k]
```

PI0.5 使用 `k=1..50`，ACT 使用 `k=1..16`。

当前数据集每一行已经满足 `action[t] = q[t+1]`。因此 loader 使用偏移 `0..H-1` 后，能够取得 `q[t+1]..q[t+H]`。相对关节 processor 再将所有未来手臂目标统一减去当前 `q[t]`。

每个 episode 第一帧的六个手臂相对 state 都置为 0，夹爪保留当前绝对值 `q_gripper[0]`，该样本继续参与训练和验证。

episode 末尾超出范围的未来位置属于 padding，不是真实动作目标。PI0.5 和 ACT 的 loss 都必须排除 `action_is_pad`。ACT 当前已经进行 mask，PI0.5 需要补充相同语义。action q01/q99 统计同样排除 padding。

## 运行时相对关节 Processor

保持 train/test 的物理存储格式不变，继续保存单步绝对 state/action。共享 processor 在 batch 进入 PI0.5 或 ACT 之前完成转换：

```text
原始 state/action
-> 取得同一 episode 的当前和上一帧绝对 state
-> condition_on_state=true 时生成相对 state
-> 生成锚定到当前 q_t 的相对 action chunk
-> image-only 模式删除模型输入中的 state
-> 使用训练集统计分别归一化 state 和 action
-> 模型前向
-> action 反归一化
-> 推理时给手臂 action 加回当前 q_t，恢复真机绝对命令
```

PI0.5 和 ACT 共用相同的数学转换逻辑，避免两套实现对相对 state、相对 action、夹爪、第一帧和 padding 产生不同解释。

image-only 模式下，确定性数据适配器仍需使用 `q_t` 构造相对训练标签，并在推理后恢复绝对命令；但 `q_t` 不得进入 PI0.5 token、ACT state projection 或任何其他可学习的模型输入。模型边界测试必须证明 state 已被删除。

在线连续推理时，processor 可以临时缓存当前绝对 `q_t`。episode 或真机控制会话 reset 时必须清除缓存，否则新 episode 第一帧可能错误使用上一个 episode 的最后状态。离线训练必须从同一个 episode 的样本中取得 `q[t-1]` 和 `q[t]`，不能依赖跨 batch 缓存。

## 仅由训练集计算 q01/q99

所有归一化统计只允许来自 143 个 episode 的 train。16 个 episode 的 test 不能参与统计。

需要保存三组独立分布：

```text
relative_state_q01_q99
relative_action_chunk50_q01_q99
relative_action_chunk16_q01_q99
```

state 统计使用应用第一帧规则后的全部训练帧。action 统计使用所有有效 `(t,k)` 组合，将样本轴和 horizon 轴展平，但保留七个独立关节维度。padding 不进入统计。chunk50 和 chunk16 必须分别计算，因为不同预测长度对应的未来位移分布不同。

三组统计中的六个手臂维度都是相对位移，夹爪维度都是绝对开合量。该数据规模允许使用精确的逐维 quantile。state 和 action 分别执行：

```text
z = 2 * (x - q01) / max(q99 - q01, eps) - 1
z = clip(z, -1, 1)
```

test 必须复用 train 统计。远端统计文件保存到：

```text
/data/wengyikun/datasets/joint_songling/0714_gripper_bread_combined_split_seed42/normalization/
```

统计文件记录训练集路径、`split_manifest.json` 哈希、horizon、特征名、夹爪索引、q01/q99 数组、有效样本数、公式版本和生成命令。

三套统计不能只保存在训练数据目录中，还必须写入每个 checkpoint：

```text
relative_state_q01_q99
relative_action_chunk50_q01_q99
relative_action_chunk16_q01_q99
```

每个 checkpoint 都复制保存上述三套统计及其统一 provenance manifest，同时标记当前 policy 实际使用的 action horizon。PI0.5 processor 使用 chunk50 action 统计，ACT processor 使用 chunk16 action 统计；未被当前模型使用的另一套 action 统计仍作为实验来源记录保留。image-only checkpoint 虽然不把 state 输入模型，也同样保留 state 统计来源信息和 action 后处理所需的完整配置。

后续推理必须优先从 checkpoint 内加载这些固定统计，禁止根据推理数据重新计算。`relative_state_q01_q99` 用于相对 state 输入归一化；对应 horizon 的 `relative_action_*_q01_q99` 用于模型输出反归一化，然后再给六个手臂维度加回当前 `q_t`。这样 checkpoint 可以在没有训练数据集的情况下独立完成正确推理。

## 模型输入边界

PI0.5 相对 state 模式将归一化后的相对 state 量化并写入 prompt。PI0.5 image-only 只生成任务 prompt，不读取或拼接 `State:`。PI0.5 模型只消费图像和语言 token，processor 边界需要保证不存在其他 state 输入。

ACT 相对 state 模式创建 robot-state projection，并输入归一化后的相对 state。ACT image-only 不创建 state feature 和 state projection。当前 ACT 实现中两处无条件读取 `observation.state` 来取得 device，需要改为从配置的图像或 action tensor 获取 device 和 batch size。在 ACT forward 之前，通过模型 batch 白名单移除完成相对标签计算后的 state。

两种模型使用统一配置语义：

```text
joint_representation=relative
condition_on_state=true|false
gripper_indices=[6]
relative_state_stats_path=/data/wengyikun/datasets/joint_songling/0714_gripper_bread_combined_split_seed42/normalization/relative_state_q01_q99.json
relative_action_stats_path=/data/wengyikun/datasets/joint_songling/0714_gripper_bread_combined_split_seed42/normalization/relative_action_chunk50_q01_q99.json  # PI0.5
relative_action_stats_path=/data/wengyikun/datasets/joint_songling/0714_gripper_bread_combined_split_seed42/normalization/relative_action_chunk16_q01_q99.json  # ACT
```

PI0.5 的 `chunk_size=50`，ACT 的 `chunk_size=16`。policy chunk、loader horizon、统计 horizon、loss mask 形状和 checkpoint processor horizon 必须完全一致，不一致时训练前直接报错。

## 六个训练与验证指标

不增加 gripper 权重，不改变原模型用于反向传播的 loss。训练阶段保留整体和夹爪 loss；validation 同时记录模型内部 loss，以及模型实际生成 action chunk 后在反归一化真实动作空间中的 MSE。

六个指标为：

```text
train/loss
train/gripper_loss
valid/loss
valid/gripper_loss
valid/action_mse
valid/gripper_mse
```

`train/loss` 是模型原本的整体训练损失，也是唯一用于反向传播的损失：

- PI0.5：对全部七个动作维度计算、排除 padding 后的原始 flow-matching MSE。
- ACT：原始训练目标，即全部七个动作维度的 padding-masked L1，加上配置的 KL 项。

`train/gripper_loss` 只用于日志，不参与反向传播：

- PI0.5：仅第 7 维夹爪的 padding-masked flow-matching MSE。
- ACT：仅第 7 维夹爪的 padding-masked L1，不包含 KL。

`valid/loss` 保留整体验证损失：

- PI0.5：全部七维、固定 validation noise/time、padding-masked flow-matching MSE。
- ACT：eval/inference 模式下全部七维的 padding-masked L1，不包含 KL。

`valid/gripper_loss` 是新增的夹爪验证指标：

- PI0.5：仅第 7 维、固定 validation noise/time、padding-masked flow-matching MSE。
- ACT：eval/inference 模式下仅第 7 维的 padding-masked L1。

`valid/action_mse` 和 `valid/gripper_mse` 不复用上述 loss，而是调用策略的实际 action chunk 推理路径：

```text
normalized predicted chunk
-> 仅执行 action q01/q99 反归一化
-> predicted physical relative action
-> 与反归一化后的 target physical relative action 比较
```

这里的“physical relative action”表示六个手臂维度为 `q[t+k] - q[t]` 的真实关节单位，夹爪维度为 `q_gripper[t+k]` 的真实绝对开合量。MSE 在加回当前 `q_t` 之前计算；不能经过 `AbsoluteJointActionProcessorStep`，否则会把动作表示转换为真机绝对命令。定义为：

```text
valid/action_mse = 所有非 padding 时间步、全部 7 个维度的平方误差总和 / 有效标量数量
valid/gripper_mse = 所有非 padding 时间步、index 6 的平方误差总和 / 有效时间步数量
```

PI0.5 的 MSE 使用 `predict_action_chunk(..., noise=fixed_noise)` 完整执行 flow-matching 去噪。固定初始高斯噪声符合 PI0.5 的生成机制，仅用于消除 validation 的采样波动；噪声必须由 `validation_seed + episode_index + frame_index` 稳定生成，不能依赖 batch 顺序或 distributed rank。这样同一样本在不同 checkpoint、单 GPU 和多 GPU 下使用相同初始噪声。ACT 在 `eval()` 下不使用 VAE encoder 的随机 latent，而使用零 latent，因此 `predict_action_chunk()` 是确定性的。

夹爪维度由配置和数据特征名共同确认，必须同时满足索引 6 且名称为 `gripper`，不能静默选择错误维度。`best_validation` 依据 `valid/action_mse` 的最低值更新；其他五个指标全部保存和记录，但不改变最佳模型判定标准。

## 使用独立 Test 进行 Validation

训练代码当前只支持从同一个数据集 root 内部再次切分 held-out episode。需要增加独立 validation dataset 配置，直接读取已经存在的 test：

```text
dataset.root=/data/wengyikun/datasets/joint_songling/0714_gripper_bread_combined_split_seed42/train
validation_dataset.root=/data/wengyikun/datasets/joint_songling/0714_gripper_bread_combined_split_seed42/test
eval_steps=200
max_eval_samples=0
```

该改动只影响训练器加载路径，不切分、不转换、不重写 train/test。

每完成 200 个 optimizer update 后运行一次 validation，而不是每 200 个 gradient-accumulation micro-step。validation 使用完整 test、关闭图像增强，并且严格复用 train processor 和 train q01/q99。

所有分布式 rank 必须同时参与 validation。每项指标都必须返回逐样本 numerator 和有效元素 denominator，再使用 `accelerator.gather_for_metrics()` 去除分布式 sampler 补齐的重复样本，最终以 `sum(numerator) / sum(denominator)` 计算严格全局均值。不能先平均每个 batch 或每个样本再取平均，因为 episode 尾部样本的有效 action 数量不同。

validation 必须保存并恢复 CPU/CUDA RNG 状态，不能影响之后的训练随机序列。PI0.5 的 `valid/loss` 使用按样本固定的 validation noise/time，`valid/action_mse` 使用按样本固定的初始推理噪声，保证不同 checkpoint 可比较。ACT 的 `valid/loss` 是确定性的 inference-mode masked L1，不包含训练时的 KL，因此 ACT 的 train/loss 和 valid/loss 不能直接比较数值绝对大小，只观察各自趋势。

保留常规 checkpoint 保存周期。当 `valid/action_mse` 降低时，保存或更新 `best_validation` checkpoint，其中包括模型、optimizer/scheduler、policy 配置、processor、相对统计、step 和全部六个指标。暂不自动 early stopping，通过最佳 checkpoint 和六条指标曲线判断过拟合。

## 远端执行流程

1. 在本地实现和测试，不覆盖工作区中无关的已有修改。
2. 通过 50210 端口将代码同步到远端 `/home/wengyikun/lerobot`。
3. 将 train、test 和 `split_manifest.json` 传输到远端 `/data/wengyikun/datasets/joint_songling/`，校验文件哈希和 LeRobot 元数据。
4. 在远端只使用 train 计算三组 quantile，并保存统计及来源信息。
5. 对四个实验分别执行单 batch 预处理语义断言。
6. 对四个实验分别执行短时单 GPU 训练加 validation 冒烟测试。
7. 执行双 GPU Accelerate validation 冒烟测试，验证六个指标的精确聚合和每 200 optimizer step 调度。
8. 生成四个明确的远端启动脚本；所有冒烟测试通过后才开始正式训练。

包括容器进程在内的远端命令必须以用户 `wengyikun` 运行，保证输出文件所有权正确。GPU 分配、batch size、梯度累积、训练总步数和最终任务名属于启动参数，不改变本设计中的数据语义。

## 失败条件与测试

以下情况必须在训练前失败：train/test 特征 schema 不一致；统计来源与 train manifest 不匹配；统计 horizon 与 policy chunk 不一致；夹爪索引或名称错误；validation root 与 train root 相同。

必须覆盖以下测试：

- 第一帧、中间帧和尾部帧的相对 state、锚定 action 精确公式；
- state 和所有 action horizon 中的绝对 gripper 语义；
- state、chunk50、chunk16 三套 q01/q99 相互独立且只来自 train；
- quantile clipping 和 q01=q99 时的稳定处理；
- PI0.5 与 ACT 都从整体 loss、gripper loss、action MSE 和 gripper MSE 中排除 padding；
- processor reset 后不会跨 episode 使用旧的 q_t；
- PI0.5 image-only prompt 不包含 state token；
- ACT image-only 不创建 state projection，模型 batch 不含 state；
- 独立 test root 可直接加载，且不进行数据转换或图像增强；
- 单 GPU 与多 GPU 下六个指标聚合正确，同一样本使用相同 PI0.5 初始噪声，validation 不改变训练 RNG；
- 两个 PI0.5 实验都冻结语言模型，同时视觉编码器和 projector 保持可训练；
- checkpoint 可在不访问训练数据集的情况下加载固定 q01/q99 并完成 state 归一化与 action 反归一化；
- `best_validation` 按最低 `valid/action_mse` 保存和恢复；
- 四个实验的单 batch 和短时训练/validation 冒烟测试。

以上检查全部通过前，不启动远端正式训练。
