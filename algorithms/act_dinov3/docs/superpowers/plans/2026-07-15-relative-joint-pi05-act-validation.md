# PI0.5 与 ACT 相对关节训练及验证实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改写现有 train/test 数据集的前提下，为 PI0.5 和 ACT 实现统一相对关节语义、train-only q01/q99、state/image-only 四种实验、每 1000 optimizer step 的独立 test validation，以及整体/夹爪 loss 和真实动作空间 MSE 六个指标。

**Architecture:** 新增共享相对关节 processor 和相对统计模块，PI0.5 与 ACT 只负责各自的模型输入边界、损失计算和 action chunk 推理。训练器加载独立 validation root，validation 全程 `eval + no_grad`，同时计算内部 loss 与反归一化真实动作空间 MSE，不参与梯度、optimizer 或归一化统计。三套 train-only q01/q99 保存在外部 `normalization/` 目录并由启动参数显式指定；所有定期 checkpoint 均保留，由用户根据 validation 指标手动选择。

**Tech Stack:** 本地 Python 3.12、远端容器 Python 3.11、PyTorch、Accelerate、LeRobotDataset v3.0、NumPy/PyArrow、Safetensors、pytest、Bash、Docker、SSH/rsync。

---

## 已确认的实施修订

- image-only 不删除 `observation.state`。relative label 仍使用真实绝对 `q_t` 构造；在归一化之后、进入 PI0.5 prompt 或 ACT model 之前，将完整 7 维 state 精确置为零。
- PI0.5 的 relative-state 和 image-only 两种实验都固定 `freeze_language_model=true`，同时保持 vision encoder 与 multimodal projector 可训练。
- 远端单卡训练先使用 PI0.5 `batch_size=16`、`gradient_accumulation_steps=1`；显存不足才降为 `batch_size=8`。ACT 使用 `batch_size=32`。正式 run 为 10,000 optimizer steps，`save_freq=2000`。
- 训练 `loss` 和 `valid/loss` 始终在归一化相对 action 空间计算，绝不反归一化。`valid/action_mse` 与 `valid/gripper_mse` 先用 q01/q99 反归一化预测和 target action，再给六个 arm 维度加回置零前 raw batch 的当前绝对 anchor `q_t`，在绝对关节空间比较；gripper 始终是绝对值，不加 anchor。
- 训练日志固定输出 `train/loss`、`train/gripper_loss`、`valid/loss`、`valid/gripper_loss`、`valid/action_mse`、`valid/gripper_mse`。前两项按 `log_freq` 输出，后四项每 `eval_steps=1000` 输出；正式脚本使用 `log_freq=10`，便于直接观察收敛。
- 每个步骤完成立即把本文件的 checkbox 标为 `[x]`。每个 Task 在测试、规格审查、质量审查后单独提交并推送 `chengdu/main`。

后续 Task 中与上述修订冲突的“删除 state / 删除 State prompt”文字，以本节为准。

---

## 文件边界

新增文件：

```text
src/lerobot/datasets/relative_joint_stats.py
src/lerobot/processor/relative_joint_processor.py
src/lerobot/common/offline_validation.py
src/lerobot/scripts/compute_relative_joint_stats.py
tests/datasets/test_relative_joint_stats.py
tests/processor/test_relative_joint_processor.py
tests/common/test_offline_validation.py
tests/datasets/test_train_eval_factory.py
run_scripts/launch_pi05_relative_state_0714.sh
run_scripts/launch_pi05_image_only_0714.sh
run_scripts/launch_act_relative_state_0714.sh
run_scripts/launch_act_image_only_0714.sh
run_scripts/remote_lerobot_container.sh
```

修改文件：

```text
src/lerobot/processor/__init__.py
src/lerobot/processor/normalize_processor.py
src/lerobot/configs/train.py
src/lerobot/datasets/factory.py
src/lerobot/policies/factory.py
src/lerobot/policies/pi05/configuration_pi05.py
src/lerobot/policies/pi05/joint_representation.py
src/lerobot/policies/pi05/processor_pi05.py
src/lerobot/policies/pi05/modeling_pi05.py
src/lerobot/policies/act/configuration_act.py
src/lerobot/policies/act/processor_act.py
src/lerobot/policies/act/modeling_act.py
src/lerobot/scripts/lerobot_train.py
src/lerobot/common/train_utils.py
tests/processor/test_normalize_processor.py
tests/policies/pi0_pi05/test_pi05_joint_representation.py
tests/processor/test_pi05_processor.py
tests/processor/test_act_processor.py
tests/training/test_multi_gpu.py
tests/utils/test_train_utils.py
```

### Task 1：为 q01/q99 增加显式 clipping

**Files:**
- Modify: `src/lerobot/processor/normalize_processor.py:39-401`
- Test: `tests/processor/test_normalize_processor.py`

- [x] **Step 1：先写失败测试**

```python
def test_quantiles_clip_to_unit_interval():
    step = NormalizerProcessorStep(
        features={OBS_STATE: PolicyFeature(FeatureType.STATE, (2,))},
        norm_map={FeatureType.STATE: NormalizationMode.QUANTILES},
        stats={OBS_STATE: {"q01": [-1.0, 0.0], "q99": [1.0, 2.0]}},
        clip_quantiles=True,
    )
    result = step(batch_to_transition({OBS_STATE: torch.tensor([[-2.0, 3.0]])}))
    torch.testing.assert_close(result[TransitionKey.OBSERVATION][OBS_STATE], torch.tensor([[-1.0, 1.0]]))


def test_quantile_inverse_uses_saved_q01_q99_without_clipping():
    step = UnnormalizerProcessorStep(
        features={ACTION: PolicyFeature(FeatureType.ACTION, (1,))},
        norm_map={FeatureType.ACTION: NormalizationMode.QUANTILES},
        stats={ACTION: {"q01": [0.02], "q99": [0.10]}},
        clip_quantiles=True,
    )
    result = step(batch_to_transition({ACTION: torch.tensor([[[0.0]]])}))
    torch.testing.assert_close(result[TransitionKey.ACTION], torch.tensor([[[0.06]]]))
```

- [x] **Step 2：运行测试并确认失败**

Run:

```bash
.venv/bin/pytest tests/processor/test_normalize_processor.py -k "quantiles_clip or quantile_inverse" -v
```

Expected: FAIL，`NormalizerProcessorStep` 尚无 `clip_quantiles`。

- [x] **Step 3：实现最小改动**

在 `_NormalizationMixin` 增加：

```python
clip_quantiles: bool = False
```

QUANTILES 正向分支改为：

```python
normalized = 2.0 * (tensor - q01) / denom - 1.0
return normalized.clamp(-1.0, 1.0) if self.clip_quantiles else normalized
```

反归一化分支保持原公式。`get_config()` 序列化 `clip_quantiles`，确保 checkpoint 恢复一致。

- [x] **Step 4：运行 processor 测试**

```bash
.venv/bin/pytest tests/processor/test_normalize_processor.py -v
```

Expected: 全部 PASS。

- [x] **Step 5：提交并推送**

```bash
git add src/lerobot/processor/normalize_processor.py tests/processor/test_normalize_processor.py
git commit -m "feat: support clipped quantile normalization"
```

### Task 2：计算并持久化三套 train-only 相对统计

**Files:**
- Create: `src/lerobot/datasets/relative_joint_stats.py`
- Create: `src/lerobot/scripts/compute_relative_joint_stats.py`
- Create: `tests/datasets/test_relative_joint_stats.py`
- Modify: `src/lerobot/datasets/__init__.py`

- [x] **Step 1：写相对 state/action 统计失败测试**

使用两个短 episode，明确验证第一帧、夹爪绝对值、有效 horizon 和 padding 排除：

```python
def test_compute_relative_joint_stats_separates_state_and_horizons():
    episodes = [
        torch.tensor([[0., 0., 0., 0., 0., 0., .02], [1., 0., 0., 0., 0., 0., .04], [3., 0., 0., 0., 0., 0., .08]]),
        torch.tensor([[10., 0., 0., 0., 0., 0., .03], [14., 0., 0., 0., 0., 0., .05]]),
    ]
    bundle = compute_relative_joint_stats_from_episodes(episodes, gripper_indices=[6], horizons=[2, 3])
    assert bundle.state.count == 5
    assert bundle.actions[2].count == 4
    assert bundle.actions[3].count == 4
    assert bundle.state.q01.shape == (7,)
    assert bundle.actions[2].q99.shape == (7,)
    expected_gripper_q01 = np.quantile([.02, .04, .08, .03, .05], 0.01)
    assert bundle.state.q01[6] == pytest.approx(expected_gripper_q01)
```

测试中的 action count 定义为全部有效 `(t,k)` 对：长度 3 episode 有 3 对，长度 2 episode 有 1 对，共 4；不会为尾部补齐虚假目标。

- [x] **Step 2：运行测试确认模块不存在**

```bash
.venv/bin/pytest tests/datasets/test_relative_joint_stats.py -v
```

Expected: FAIL，无法导入 `relative_joint_stats`。

- [x] **Step 3：实现统计数据结构和精确 quantile**

```python
@dataclass(frozen=True)
class QuantileStats:
    q01: np.ndarray
    q99: np.ndarray
    count: int


@dataclass(frozen=True)
class RelativeJointStatsBundle:
    state: QuantileStats
    actions: dict[int, QuantileStats]
    feature_names: list[str]
    gripper_indices: list[int]
    source_manifest_sha256: str
```

实现规则：

```python
relative_state[0, arm_mask] = 0
relative_state[1:, arm_mask] = q[1:, arm_mask] - q[:-1, arm_mask]
relative_state[:, gripper_mask] = q[:, gripper_mask]

for t in range(length):
    for k in range(1, min(horizon, length - 1 - t) + 1):
        target = q[t + k].copy()
        target[arm_mask] -= q[t, arm_mask]
        # gripper remains q[t+k]
```

用 `np.quantile(values, [0.01, 0.99], axis=0)` 计算精确统计。

- [x] **Step 4：实现 JSON 保存、加载和来源校验**

固定文件名：

```text
relative_state_q01_q99.json
relative_action_chunk50_q01_q99.json
relative_action_chunk16_q01_q99.json
relative_stats_manifest.json
```

`load_relative_joint_stats()` 必须验证 feature names、gripper index、manifest SHA256 和 requested horizon。

- [x] **Step 5：实现命令行脚本**

```bash
.venv/bin/python -m lerobot.scripts.compute_relative_joint_stats \
  --dataset-root=/data/joint_songling/0714_gripper_bread_combined_split_seed42/train \
  --split-manifest=/data/joint_songling/0714_gripper_bread_combined_split_seed42/split_manifest.json \
  --output-dir=/data/joint_songling/0714_gripper_bread_combined_split_seed42/normalization \
  --horizons='[50,16]' \
  --gripper-indices='[6]'
```

脚本拒绝 test root，并原子写入 `.tmp` 后 rename。

- [x] **Step 6：运行测试、提交并推送**

```bash
.venv/bin/pytest tests/datasets/test_relative_joint_stats.py -v
git add src/lerobot/datasets/relative_joint_stats.py src/lerobot/datasets/__init__.py \
  src/lerobot/scripts/compute_relative_joint_stats.py tests/datasets/test_relative_joint_stats.py
git commit -m "feat: compute train-only relative joint quantiles"
```

### Task 3：实现 PI0.5/ACT 共用的相对关节 Processor

**Files:**
- Create: `src/lerobot/processor/relative_joint_processor.py`
- Create: `tests/processor/test_relative_joint_processor.py`
- Modify: `src/lerobot/processor/__init__.py`

- [x] **Step 1：写公式、image-only、归一化后置零和 reset 失败测试**

```python
def test_relative_joint_processor_anchors_all_actions_to_current_state():
    state = torch.tensor([[[1., 2., 3., 4., 5., 6., .04], [2., 4., 6., 8., 10., 12., .05]]])
    action = torch.tensor([[[3., 5., 7., 9., 11., 13., .06], [4., 7., 8., 10., 12., 14., .08]]])
    step = RelativeJointProcessorStep(condition_on_state=True, gripper_indices=[6], action_names=ACTION_NAMES)
    out = step(batch_to_transition({OBS_STATE: state, ACTION: action}))
    torch.testing.assert_close(out[TransitionKey.OBSERVATION][OBS_STATE][..., :6], state[:, 1, :6] - state[:, 0, :6])
    torch.testing.assert_close(out[TransitionKey.ACTION][..., :6], action[..., :6] - state[:, None, 1, :6])
    torch.testing.assert_close(out[TransitionKey.ACTION][..., 6], action[..., 6])


def test_image_only_zeroes_state_after_normalization_and_target_conversion():
    step = RelativeJointProcessorStep(condition_on_state=False, gripper_indices=[6], action_names=ACTION_NAMES)
    out = step(batch_to_transition({OBS_STATE: state, ACTION: action}))
    # ZeroStateProcessorStep runs after NormalizerProcessorStep, so the model receives
    # exactly seven numeric zeroes while the processor can still form relative labels.
    assert OBS_STATE in out[TransitionKey.OBSERVATION]
    assert TransitionKey.ACTION in out


def test_reset_clears_online_absolute_state_cache():
    step = RelativeJointProcessorStep(condition_on_state=True, gripper_indices=[6], action_names=ACTION_NAMES)
    step(batch_to_transition({OBS_STATE: torch.ones(1, 7)}))
    assert step.get_cached_absolute_state() is not None
    step.reset()
    assert step.get_cached_absolute_state() is None
```

- [x] **Step 2：确认测试失败并实现 processor**

```bash
.venv/bin/pytest tests/processor/test_relative_joint_processor.py -v
```

新增：

```python
@ProcessorStepRegistry.register("relative_joint_processor")
@dataclass
class RelativeJointProcessorStep(ProcessorStep):
    enabled: bool = True
    condition_on_state: bool = True
    gripper_indices: list[int] = field(default_factory=lambda: [6])
    action_names: list[str] | None = None
    _last_absolute_state: Tensor | None = field(default=None, init=False, repr=False)

    def reset(self) -> None:
        self._last_absolute_state = None
```

配套 `AbsoluteJointActionProcessorStep` 仅给 arm mask 加回缓存 `q_t`，gripper 不相加。

- [x] **Step 3：保证序列化不保存运行时缓存，并在加载后重新绑定动作恢复步骤**

`get_config()` 只返回 enabled、condition_on_state、gripper_indices、action_names；`state_dict()` 不写 `_last_absolute_state`。

- [x] **Step 4：运行测试、提交并推送**

```bash
.venv/bin/pytest tests/processor/test_relative_joint_processor.py tests/policies/test_relative_actions.py -v
git add src/lerobot/processor/relative_joint_processor.py src/lerobot/processor/__init__.py \
  tests/processor/test_relative_joint_processor.py
git commit -m "feat: add shared relative joint processor"
```

### Task 4：给 PI0.5 与 ACT 增加一致的配置合同

**Files:**
- Modify: `src/lerobot/policies/pi05/configuration_pi05.py`
- Modify: `src/lerobot/policies/act/configuration_act.py`
- Modify: `src/lerobot/policies/factory.py:584-611`
- Test: `tests/policies/pi0_pi05/test_pi05_joint_representation.py`
- Test: `tests/processor/test_act_processor.py`

- [x] **Step 1：写配置失败测试**

```python
def test_relative_configs_request_previous_state_for_both_conditioning_modes():
    for condition_on_state in (True, False):
        pi05 = PI05Config(joint_representation="relative", condition_on_state=condition_on_state, chunk_size=50)
        act = ACTConfig(joint_representation="relative", condition_on_state=condition_on_state, chunk_size=16)
        assert pi05.observation_delta_indices == [-1, 0]
        assert act.observation_delta_indices == [-1, 0]
        assert pi05.action_delta_indices == list(range(50))
        assert act.action_delta_indices == list(range(16))
```

- [x] **Step 2：新增公共配置字段**

两个 config 都增加：

```python
joint_representation: str = "absolute"
condition_on_state: bool = True
gripper_indices: list[int] = field(default_factory=lambda: [6])
relative_state_stats_path: str | None = None
relative_action_stats_path: str | None = None
clip_quantiles: bool = True
```

`joint_representation=relative` 时强制统计路径存在、action stats horizon 等于 chunk size、feature index 6 名称为 `gripper`。

- [x] **Step 3：修正 image-only feature 推断**

按页首实施修订，`condition_on_state=False` 仍在 dataset/config 中保留
`observation.state`，供 processor 构造相对 state 和 action anchor。后续 Task 5/6 在归一化后将
完整 7D state 精确置零，并在各自模型输入边界不使用真实 state 值。

- [x] **Step 4：运行配置测试并提交**

```bash
.venv/bin/pytest tests/policies/pi0_pi05/test_pi05_joint_representation.py tests/processor/test_act_processor.py -v
git add src/lerobot/policies/pi05/configuration_pi05.py src/lerobot/policies/act/configuration_act.py \
  src/lerobot/policies/factory.py tests/policies/pi0_pi05/test_pi05_joint_representation.py \
  tests/processor/test_act_processor.py
git commit -m "feat: configure relative and image-only joint policies"
```

### Task 5：接入 PI0.5 processor、冻结规则和夹爪 loss

**Files:**
- Modify: `src/lerobot/policies/pi05/joint_representation.py`
- Modify: `src/lerobot/policies/pi05/processor_pi05.py`
- Modify: `src/lerobot/policies/pi05/modeling_pi05.py:1368-1406`
- Test: `tests/processor/test_pi05_processor.py`
- Test: `tests/policies/pi0_pi05/test_pi05_joint_representation.py`

- [x] **Step 1：写 PI0.5 image-only prompt 和 loss 失败测试**

```python
def test_pi05_image_only_prompt_keeps_fixed_zero_state_slot():
    normalized = normalizer(transition_with_task_and_state)
    zeroed = ZeroStateProcessorStep()(normalized)
    out = Pi05PrepareStateTokenizerProcessorStep()(zeroed)
    assert "State: 128 128 128 128 128 128 128" in out[TransitionKey.COMPLEMENTARY_DATA]["task"][0]


def test_pi05_reports_gripper_loss_and_masks_padding():
    loss, metrics = policy.forward(batch_with_last_20_steps_padded)
    assert loss.requires_grad
    assert metrics["gripper_loss"] == pytest.approx(expected_unpadded_dim6_mse)
    assert metrics["loss_count_per_sample"].tolist() == expected_valid_steps_times_7
    assert metrics["gripper_loss_count_per_sample"].tolist() == expected_valid_steps
```

- [x] **Step 2：PI0.5 processor 改用共享相对 processor**

顺序固定为：

```text
AddBatch -> RelativeJoint -> Normalizer(QUANTILES, clip) -> ZeroState(image-only) -> Prompt/Tokenizer -> Device
Unnormalizer -> AbsoluteJointAction -> CPU
```

保留旧 `pi05_joint_representation_processor` registry 名称作为兼容 wrapper，但内部委托共享实现。

- [x] **Step 3：实现固定零 state prompt**

image-only 不删除 State prompt。`ZeroStateProcessorStep` 在 Normalizer 之后将完整 7D state
精确置零，`Pi05PrepareStateTokenizerProcessorStep` 保持固定 prompt 结构：

```text
Task: {cleaned_task}, State: 128 128 128 128 128 128 128;
Action:
```

`condition_on_state=True` 保留量化相对 state prompt。两种模式拥有相同 prompt 槽位，但
image-only 模式永远不向模型暴露真实 state 值。

- [x] **Step 4：实现 PI0.5 padding-aware 整体和夹爪指标**

```python
valid = (~batch["action_is_pad"]).unsqueeze(-1)
action_losses = losses[:, :, :action_dim]
valid_all = valid.expand_as(action_losses)
loss = action_losses[valid_all].mean()
valid_steps = valid.squeeze(-1)
gripper_values = action_losses[:, :, gripper_index]
gripper_loss_per_sample = (gripper_values * valid_steps).sum(dim=1) / valid_steps.sum(dim=1).clamp_min(1)
gripper_loss = gripper_loss_per_sample.mean()
loss_dict["gripper_loss"] = gripper_loss.detach().item()
loss_dict["gripper_loss_per_sample"] = gripper_loss_per_sample.detach()
```

同时返回 validation 精确聚合需要的逐样本 `loss_sum_per_sample`、`loss_count_per_sample`、`gripper_loss_sum_per_sample` 和 `gripper_loss_count_per_sample`。`loss` 仍是唯一反向传播目标；`gripper_loss` 只进入 metrics。

`forward()` 增加仅供可复现 validation 使用的可选参数：

```python
def forward(
    self,
    batch: dict[str, Tensor],
    reduction: str = "mean",
    *,
    noise: Tensor | None = None,
    time: Tensor | None = None,
) -> tuple[Tensor, dict]:
    actions = self.prepare_action(batch)
    if noise is None:
        noise = self.model.sample_noise(actions.shape, actions.device)
    if time is None:
        time = self.model.sample_time(actions.shape[0], actions.device)
    losses = self.model.forward(images, img_masks, tokens, masks, actions, noise, time)
```

训练调用不传参数，继续随机采样；validation 显式传入按样本身份固定的 flow noise/time。`ActionSelectKwargs` 增加 `noise: Tensor | None`，使 `predict_action_chunk(batch, noise=fixed_initial_noise)` 与底层已经支持 `noise` 的 `sample_actions()` 类型合同一致。

- [x] **Step 5：验证两个 PI0.5 配置都冻结语言模型**

测试断言：

```python
assert all(not p.requires_grad for p in policy.model.paligemma.language_model.parameters())
assert any(p.requires_grad for p in policy.model.paligemma.vision_tower.parameters())
assert any(p.requires_grad for p in policy.model.paligemma.multi_modal_projector.parameters())
```

- [x] **Step 6：运行测试并提交**

```bash
.venv/bin/pytest tests/processor/test_pi05_processor.py \
  tests/policies/pi0_pi05/test_pi05_joint_representation.py -v
git add src/lerobot/policies/pi05 tests/processor/test_pi05_processor.py \
  tests/policies/pi0_pi05/test_pi05_joint_representation.py
git commit -m "feat: train PI05 with relative or image-only conditioning"
```

### Task 6：接入 ACT 相对 processor、image-only 和夹爪 loss

**Files:**
- Modify: `src/lerobot/policies/act/processor_act.py`
- Modify: `src/lerobot/policies/act/modeling_act.py:120-160,395-475`
- Test: `tests/processor/test_act_processor.py`

- [x] **Step 1：写 ACT image-only 和夹爪指标失败测试**

```python
def test_act_image_only_passes_only_zero_state_to_model():
    config = ACTConfig(condition_on_state=False, joint_representation="relative", chunk_size=16)
    policy = ACTPolicy(config)
    prepared = policy._prepare_model_batch(processed_batch)
    assert OBS_STATE in prepared
    assert torch.count_nonzero(prepared[OBS_STATE]) == 0
    assert hasattr(policy.model, "encoder_robot_state_input_proj")


def test_act_gripper_loss_is_monitor_only():
    loss, metrics = policy.forward(batch)
    assert metrics["gripper_loss"] == pytest.approx(expected_masked_gripper_l1)
    assert loss == pytest.approx(metrics["l1_loss"] + config.kl_weight * metrics["kld_loss"])
    assert metrics["loss_count_per_sample"].tolist() == expected_valid_steps_times_7
    assert metrics["gripper_loss_count_per_sample"].tolist() == expected_valid_steps
```

- [x] **Step 2：ACT processor 接入共享转换和独立统计**

预处理顺序与 PI0.5 相同，但无 tokenizer：

```text
AddBatch -> RelativeJoint -> Normalizer(QUANTILES, clip) -> ZeroState(image-only) -> Device
```

image-only 保留 `observation.state` 和 ACT state projection，但在 Normalizer 之后将完整 7D state
精确置零。真实 state 只用于构造相对 state/action anchor，不进入 ACT 模型。

- [x] **Step 3：修复 ACT 可选 state 配置的 device 访问**

新增：

```python
def _batch_reference_tensor(self, batch: dict[str, Tensor]) -> Tensor:
    if OBS_IMAGES in batch:
        return batch[OBS_IMAGES][0]
    if ACTION in batch:
        return batch[ACTION]
    return batch[OBS_ENV_STATE]
```

所有 latent、padding mask 的 device 从该 tensor 获取，不再无条件读取 `batch[OBS_STATE]`。

- [x] **Step 4：统一 `_prepare_model_batch()` 白名单**

`forward()` 和 `predict_action_chunk()` 共用它，只保留配置声明的图像/state、`ACTION`、`action_is_pad` 和内部 `OBS_IMAGES`。image-only 仍保留 `OBS_STATE`，但进入 ACTModel 前必须是归一化后的精确全零 7D tensor，不得携带真实 state 值。

- [x] **Step 5：增加 per-sample eval 输出**

`forward(batch, reduction="none")` 返回每个样本的 padding-masked L1，并在 details 中返回 `loss_sum_per_sample`、`loss_count_per_sample`、`gripper_loss_sum_per_sample` 和 `gripper_loss_count_per_sample`。ACT train 总 loss 保持 L1 + KL，不使用 gripper_loss 反向传播。

- [x] **Step 6：运行测试并提交**

```bash
.venv/bin/pytest tests/processor/test_act_processor.py -v
git add src/lerobot/policies/act tests/processor/test_act_processor.py
git commit -m "feat: support relative and image-only ACT training"
```

### Task 7：直接加载独立 test root

**Files:**
- Modify: `src/lerobot/configs/train.py:77-130,245-275`
- Modify: `src/lerobot/datasets/factory.py:146-221`
- Create: `tests/datasets/test_train_eval_factory.py`

- [x] **Step 1：写外部 validation dataset 失败测试**

```python
def test_make_train_eval_datasets_loads_external_root_without_resplitting(train_root, test_root):
    cfg = make_cfg(
        dataset=DatasetConfig(repo_id="local/train", root=str(train_root)),
        validation_dataset=DatasetConfig(repo_id="local/test", root=str(test_root)),
        eval_steps=1000,
    )
    train, valid = make_train_eval_datasets(cfg)
    assert train.root == train_root
    assert valid.root == test_root
    assert train.meta.total_episodes == 143
    assert valid.meta.total_episodes == 16
    assert valid.image_transforms is None
```

- [x] **Step 2：新增配置字段和互斥校验**

```python
validation_dataset: DatasetConfig | None = None
eval_steps: int = 0
max_eval_samples: int = 0
validation_seed: int = 42
```

规则：`eval_steps > 0` 时必须配置 `validation_dataset` 或 `dataset.eval_split`；两者不能同时启用；train/test resolved root 必须不同；`eval_steps=1000` 是正式实验值。

- [x] **Step 3：factory 加载 test，但只使用 train stats**

使用 train metadata 解析 delta timestamps，检查 test 的 FPS、state/action shape、feature names 和 camera keys 完全一致。test loader 不启用图像增强，不读取 test stats 作为 processor stats。

- [x] **Step 4：运行测试并提交**

```bash
.venv/bin/pytest tests/datasets/test_train_eval_factory.py -v
git add src/lerobot/configs/train.py src/lerobot/datasets/factory.py tests/datasets/test_train_eval_factory.py
git commit -m "feat: load an external validation dataset"
```

### Task 8：实现不会训练模型的精确 Validation、Loss 与真实动作 MSE

**Files:**
- Create: `src/lerobot/common/offline_validation.py`
- Create: `tests/common/test_offline_validation.py`
- Modify: `src/lerobot/scripts/lerobot_train.py:533-716`
- Modify: `src/lerobot/policies/pi05/modeling_pi05.py:63-68,798-870,1351-1406`
- Test: `tests/policies/pi0_pi05/test_pi05_joint_representation.py`

- [x] **Step 1：写 no-grad、RNG 和精确聚合失败测试**

```python
def test_validation_never_updates_parameters_or_optimizer():
    before = {k: v.clone() for k, v in policy.state_dict().items()}
    optimizer_steps_before = optimizer_step_count(optimizer)
    metrics = evaluate_offline(policy, loader, preprocessor, accelerator, seed=42)
    assert all(torch.equal(before[k], v) for k, v in policy.state_dict().items())
    assert optimizer_step_count(optimizer) == optimizer_steps_before
    assert metrics.keys() == {
        "valid/loss",
        "valid/gripper_loss",
        "valid/action_mse",
        "valid/gripper_mse",
    }


def test_validation_restores_rng_and_training_mode():
    policy.train()
    cpu_state = torch.random.get_rng_state().clone()
    evaluate_offline(policy, loader, preprocessor, accelerator, seed=42)
    assert policy.training
    assert torch.equal(torch.random.get_rng_state(), cpu_state)


def test_physical_action_mse_unnormalizes_and_excludes_padding():
    metrics = evaluate_offline(policy, loader_with_padded_tail, preprocessor, accelerator, seed=42)
    assert metrics["valid/action_mse"] == pytest.approx(expected_physical_7d_mse)
    assert metrics["valid/gripper_mse"] == pytest.approx(expected_physical_dim6_mse)


def test_pi05_noise_is_stable_by_sample_identity_across_batching():
    ids = [(0, 12), (4, 31)]
    *_, together = make_pi05_validation_randomness(
        policy, ids, seed=42, device="cpu"
    )
    separate = torch.cat([
        make_pi05_validation_randomness(
            policy, [sample_id], seed=42, device="cpu"
        )[2]
        for sample_id in ids
    ])
    torch.testing.assert_close(together, separate)
```

- [x] **Step 2：实现独立 evaluator**

核心边界必须是：

```python
was_training = policy.training
cpu_rng = torch.random.get_rng_state()
cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
policy.eval()
metric_parts: dict[str, list[torch.Tensor]] = defaultdict(list)
try:
    with torch.no_grad(), accelerator.autocast():
        for batch in eval_dataloader:
            processed_batch = preprocessor(batch)
            sample_ids = list(zip(batch["episode_index"], batch["frame_index"], strict=True))

            if policy.name == "pi05":
                flow_noise, flow_time, initial_noise = make_pi05_validation_randomness(
                    policy,
                    sample_ids,
                    seed=seed,
                    device=policy.config.device,
                )
                _, details = policy.forward(
                    processed_batch,
                    reduction="none",
                    noise=flow_noise,
                    time=flow_time,
                )
                predicted_normalized = policy.predict_action_chunk(
                    make_inference_batch(processed_batch),
                    noise=initial_noise,
                )
            else:
                _, details = policy.forward(processed_batch, reduction="none")
                predicted_normalized = policy.predict_action_chunk(make_inference_batch(processed_batch))

            target_normalized = processed_batch[ACTION]
            predicted_relative = action_unnormalizer(predicted_normalized)
            target_relative = action_unnormalizer(target_normalized)
            predicted_physical = restore_absolute_joint_actions(predicted_relative, raw_current_q_t)
            target_physical = restore_absolute_joint_actions(target_relative, raw_current_q_t)
            mse_parts = physical_action_mse_parts(
                predicted_physical,
                target_physical,
                processed_batch["action_is_pad"],
                gripper_index=6,
            )
            append_metric_parts(metric_parts, details, mse_parts)
finally:
    torch.random.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    policy.train(was_training)
```

禁止调用 `accelerator.backward()`、`optimizer.step()`、`optimizer.zero_grad()` 或 scheduler。

- [x] **Step 3：隔离标签并构造确定性 validation 随机量**

`make_inference_batch()` 必须移除 `ACTION` 和 `action_is_pad`，确保实际预测路径看不到标签。`action_unnormalizer` 只执行 q01/q99 反归一化，不执行有缓存副作用的在线 `AbsoluteJointActionProcessorStep`。evaluator 必须从置零前 raw batch 保存当前绝对 `q_t`，显式给预测和 target 的六个 arm 维度加回相同 anchor；gripper 保持反归一化后的绝对值。

PI0.5 固定随机量由 `validation_seed + episode_index + frame_index + purpose` 的稳定哈希生成，分别产生 flow-loss noise、flow-loss time 和 action-sampling initial noise。禁止使用 Python 内置 `hash()`，因为它跨进程不稳定；使用 SHA256 截取 64-bit seed。ACT 在 `eval()` 下使用零 latent，保持确定性。

```python
def stable_validation_seed(base_seed: int, episode: int, frame: int, purpose: str) -> int:
    payload = f"{base_seed}:{episode}:{frame}:{purpose}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)


def make_inference_batch(batch: dict[str, Tensor]) -> dict[str, Tensor]:
    return {key: value for key, value in batch.items() if key not in {ACTION, "action_is_pad"}}


def make_pi05_validation_randomness(policy, sample_ids, seed: int, device):
    flow_noise, flow_time, initial_noise = [], [], []
    shape = (policy.config.chunk_size, policy.config.max_action_dim)
    for episode, frame in sample_ids:
        episode, frame = int(episode), int(frame)
        flow_generator = torch.Generator(device=device).manual_seed(
            stable_validation_seed(seed, episode, frame, "flow_noise")
        )
        initial_generator = torch.Generator(device=device).manual_seed(
            stable_validation_seed(seed, episode, frame, "initial_noise")
        )
        flow_noise.append(torch.randn(shape, generator=flow_generator, device=device))
        initial_noise.append(torch.randn(shape, generator=initial_generator, device=device))

        # sample_time() 的 Beta 分布在 CPU 采样；fork_rng 防止改变训练 RNG。
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(stable_validation_seed(seed, episode, frame, "flow_time"))
            flow_time.append(policy.model.sample_time(1, device))

    return torch.stack(flow_noise), torch.cat(flow_time), torch.stack(initial_noise)
```

- [x] **Step 4：按有效元素数量严格聚合四个 validation 指标**

PI0.5/ACT 都返回逐样本 numerator/count。MSE 也返回逐样本 numerator/count：

```python
sq_error = (predicted_absolute - target_absolute).square()
valid_steps = ~action_is_pad
action_num = (sq_error * valid_steps.unsqueeze(-1)).sum(dim=(1, 2))
action_den = valid_steps.sum(dim=1) * sq_error.shape[-1]
gripper_num = (sq_error[..., gripper_index] * valid_steps).sum(dim=1)
gripper_den = valid_steps.sum(dim=1)
```

对每一项使用：

```python
all_num = accelerator.gather_for_metrics(per_sample_numerator)
all_den = accelerator.gather_for_metrics(per_sample_denominator)
metric = all_num.sum().double() / all_den.sum().clamp_min(1).double()
```

这既去除 Accelerate 为对齐 rank 添加的重复尾部样本，也避免把有效 horizon 不同的样本等权平均。

- [x] **Step 5：训练日志增加夹爪指标但不改 backward**

`update_policy()` 仍执行：

```python
loss, output_dict = policy.forward(batch)
accelerator.backward(loss)
```

只从 `output_dict["gripper_loss"]` 写入 `train/gripper_loss`。不得将其加到 `loss`。

梯度累积时每个 micro-batch 都更新独立的 `train_gripper_meter`，optimizer step 完成后再按样本数求均值并跨 rank reduce，不能只记录最后一个 micro-batch。

- [x] **Step 6：每 1000 optimizer step 调用**

保持 `step += 1` 在 optimizer update 后执行，条件固定：

```python
is_eval_step = cfg.eval_steps > 0 and step % cfg.eval_steps == 0
```

正式脚本使用 `--eval_steps=1000`。

- [x] **Step 7：运行测试并提交**

```bash
.venv/bin/pytest tests/common/test_offline_validation.py \
  tests/policies/pi0_pi05/test_pi05_joint_representation.py -v
git add src/lerobot/common/offline_validation.py src/lerobot/scripts/lerobot_train.py \
  src/lerobot/policies/pi05/modeling_pi05.py tests/common/test_offline_validation.py \
  tests/policies/pi0_pi05/test_pi05_joint_representation.py
git commit -m "feat: compute isolated offline validation metrics"
```

### Task 9：保存三套统计、保留全部定期 checkpoint 并按动作 MSE 标记 best_validation（按用户要求跳过）

> **2026-07-16 用户决策：** 不实现自动 `best_validation`、validation state resume 或 checkpoint 内嵌三套统计。现有数字 step checkpoint 全部保留，用户直接观察每次 validation 指标并手动选择模型。外部 `normalization/` q01/q99 文件仍是训练和推理必需输入。以下步骤仅保留为历史设计，不纳入完成判定。

- [x] **Task 决策：跳过自动 best checkpoint 与内嵌统计实现**

**Files:**
- Modify: `src/lerobot/common/train_utils.py:105-180`
- Modify: `src/lerobot/scripts/lerobot_train.py:695-755`
- Modify: `tests/utils/test_train_utils.py`
- Modify: `tests/common/test_offline_validation.py`

- [ ] **Step 1：写 checkpoint 持久化失败测试**

```python
def test_checkpoint_contains_all_relative_stats(tmp_path):
    save_checkpoint(
        checkpoint_dir=tmp_path,
        step=10,
        cfg=cfg,
        policy=policy,
        optimizer=optimizer,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        relative_stats_bundle=bundle,
    )
    root = tmp_path / "pretrained_model" / "relative_stats"
    assert (root / "relative_state_q01_q99.json").is_file()
    assert (root / "relative_action_chunk50_q01_q99.json").is_file()
    assert (root / "relative_action_chunk16_q01_q99.json").is_file()
    assert (root / "relative_stats_manifest.json").is_file()


def test_best_validation_uses_physical_action_mse(tmp_path):
    state = ValidationState(best_valid_action_mse=0.04, best_step=200)
    updated = maybe_save_best_validation(
        metrics={"valid/loss": 0.10, "valid/action_mse": 0.03},
        state=state,
        checkpoint_dir=tmp_path,
    )
    assert updated.best_valid_action_mse == pytest.approx(0.03)
    assert (tmp_path / "best_validation").is_dir()
    assert (tmp_path / "002000").is_dir()  # best 更新绝不删除任何数字 step checkpoint
```

- [ ] **Step 2：扩展 `save_checkpoint()`**

增加参数：

```python
relative_stats_bundle: RelativeJointStatsBundle | None = None
validation_metrics: dict[str, float] | None = None
```

写入 `pretrained_model/relative_stats/`，并把 active horizon 与全部六个指标写入 `training_state/validation_state.json`。

- [ ] **Step 3：推理优先加载 checkpoint stats**

processor `from_pretrained()` 从 checkpoint 内相对统计构建 Normalizer/Unnormalizer。若 config 是 relative 且 checkpoint 缺统计，直接报错，不回退到 dataset 或重新计算。

- [ ] **Step 4：实现原子 best checkpoint 替换**

当 `valid/action_mse < best_valid_action_mse` 时，所有 rank 参与 FSDP gather，rank0 保存到 `checkpoints/best_validation.tmp`，完成后原子替换额外的 `checkpoints/best_validation`。最佳判定只看反归一化并恢复绝对 arm 后的整体 MSE，不看 loss 或夹爪单项指标。`002000/004000/.../010000` 等所有数字 step checkpoint 必须永久保留；更新 `best_validation` 或 `last` 只更新额外快照/引用，禁止清理、覆盖或重命名数字 checkpoint。

- [ ] **Step 5：resume 恢复最佳值**

从 `validation_state.json` 恢复 `best_valid_action_mse` 和对应 step，避免 resume 后错误覆盖更好的模型。

- [ ] **Step 6：运行测试并提交**

```bash
.venv/bin/pytest tests/utils/test_train_utils.py tests/common/test_offline_validation.py -v
git add src/lerobot/common/train_utils.py src/lerobot/scripts/lerobot_train.py \
  tests/utils/test_train_utils.py tests/common/test_offline_validation.py
git commit -m "feat: persist relative stats and best validation checkpoint"
```

### Task 10：增加四实验集成测试和远端启动脚本

**Files:**
- Modify: `tests/training/test_multi_gpu.py`
- Create: `run_scripts/launch_pi05_relative_state_0714.sh`
- Create: `run_scripts/launch_pi05_image_only_0714.sh`
- Create: `run_scripts/launch_act_relative_state_0714.sh`
- Create: `run_scripts/launch_act_image_only_0714.sh`
- Create: `run_scripts/remote_lerobot_container.sh`

- [x] **Step 1：增加四配置单 batch 测试**

每个配置断言：

```text
PI0.5 action shape = [B,50,7]
ACT action shape = [B,16,7]
relative-state 模式模型输入含 7D relative state
image-only 模式模型输入含精确全零的 7D state
train loss 与 gripper loss 均有限
validation loss 与 validation gripper loss 均有限
反归一化并恢复绝对 arm 后的 validation action MSE 与 gripper MSE 均有限
所有数字 step checkpoint 均保留，由用户根据 validation action MSE 手动选择
```

- [x] **Step 2：增加双 GPU validation smoke**

在 `tests/training/test_multi_gpu.py` 添加 4-step 小数据测试，`eval_steps=2`。断言日志在 step 2、4 各出现一次 validation，且 rank0 指标等于单 GPU 精确均值。

- [x] **Step 3：创建当前用户容器入口**

远端已确认没有 `/home/wengyikun/lerobot/.venv`，使用镜像 `lerobot-pi05-train:20260706`。`remote_lerobot_container.sh` 固定为：

```bash
#!/usr/bin/env bash
set -euo pipefail

: "${LEROBOT_GPUS:?Set LEROBOT_GPUS to device=<gpu-id> before launching the container}"

exec docker run --rm \
  --gpus "${LEROBOT_GPUS}" \
  --user "$(id -u):$(id -g)" \
  --ipc=host \
  --shm-size=16g \
  -e HOME=/home/wengyikun \
  -e PYTHONPATH=/workspace/lerobot/src \
  -v /home/wengyikun/lerobot:/workspace/lerobot \
  -v /home/wengyikun/.cache:/home/wengyikun/.cache \
  -v /data/wengyikun:/data/wengyikun \
  -w /workspace/lerobot \
  lerobot-pi05-train:20260706 "$@"
```

- [x] **Step 4：创建 PI0.5 两个脚本**

共同关键参数：

```bash
accelerate launch --multi_gpu --num_processes="${NUM_PROCESSES:-2}" --mixed_precision=bf16 \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/0714_bread_train \
  --dataset.root=/data/wengyikun/datasets/joint_songling/0714_gripper_bread_combined_split_seed42/train \
  --validation_dataset.repo_id=local/0714_bread_test \
  --validation_dataset.root=/data/wengyikun/datasets/joint_songling/0714_gripper_bread_combined_split_seed42/test \
  --policy.type=pi05 \
  --policy.pretrained_path=/data/wengyikun/openpi/lerobot_pi05_base \
  --policy.visual_pretrained_path=/data/wengyikun/models/TeleEmbodied_VISTA/pretrained_model/model.safetensors \
  --policy.freeze_language_model=true \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.visual_pretrained_include_projector=true \
  --policy.joint_representation=relative \
  --policy.gripper_indices='[6]' \
  --policy.relative_state_stats_path=/data/wengyikun/datasets/joint_songling/0714_gripper_bread_combined_split_seed42/normalization/relative_state_q01_q99.json \
  --policy.relative_action_stats_path=/data/wengyikun/datasets/joint_songling/0714_gripper_bread_combined_split_seed42/normalization/relative_action_chunk50_q01_q99.json \
  --policy.clip_quantiles=true \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --eval_steps=1000 \
  --max_eval_samples=0 \
  --log_freq="${LOG_FREQ:-10}" \
  --batch_size="${BATCH_SIZE:-16}" \
  --gradient_accumulation_steps="${GRAD_ACC:-1}" \
  --steps="${STEPS:-10000}" \
  --save_freq="${SAVE_FREQ:-2000}"
```

`launch_pi05_relative_state_0714.sh` 必须写入：

```bash
--policy.condition_on_state=true
--job_name=pi05_0714_relative_state_chunk50
--output_dir=/data/wengyikun/outputs/pi05_0714_relative_state_chunk50/train_out
```

`launch_pi05_image_only_0714.sh` 必须写入：

```bash
--policy.condition_on_state=false
--job_name=pi05_0714_image_only_chunk50
--output_dir=/data/wengyikun/outputs/pi05_0714_image_only_chunk50/train_out
```

- [x] **Step 5：创建 ACT 两个脚本**

使用相同 train/test/eval 参数，并设置：

```bash
--policy.type=act
--policy.joint_representation=relative
--policy.gripper_indices='[6]'
--policy.relative_state_stats_path=/data/wengyikun/datasets/joint_songling/0714_gripper_bread_combined_split_seed42/normalization/relative_state_q01_q99.json
--policy.relative_action_stats_path=/data/wengyikun/datasets/joint_songling/0714_gripper_bread_combined_split_seed42/normalization/relative_action_chunk16_q01_q99.json
--policy.clip_quantiles=true
--policy.chunk_size=16
--policy.n_action_steps=16
--policy.normalization_mapping.STATE=QUANTILES
--policy.normalization_mapping.ACTION=QUANTILES
--batch_size="${BATCH_SIZE:-32}"
--gradient_accumulation_steps="${GRAD_ACC:-1}"
--steps="${STEPS:-10000}"
--save_freq="${SAVE_FREQ:-2000}"
```

`launch_act_relative_state_0714.sh` 必须写入：

```bash
--policy.condition_on_state=true
--job_name=act_0714_relative_state_chunk16
--output_dir=/data/wengyikun/outputs/act_0714_relative_state_chunk16/train_out
```

`launch_act_image_only_0714.sh` 必须写入：

```bash
--policy.condition_on_state=false
--job_name=act_0714_image_only_chunk16
--output_dir=/data/wengyikun/outputs/act_0714_image_only_chunk16/train_out
```

- [x] **Step 6：脚本语法检查和测试**

本机未安装 `shellcheck`；已用 `bash -n` 检查五个脚本，并运行 Task 1-10 相关回归测试。

```bash
shellcheck run_scripts/launch_*_0714.sh
.venv/bin/pytest tests/processor/test_relative_joint_processor.py \
  tests/datasets/test_relative_joint_stats.py \
  tests/common/test_offline_validation.py \
  tests/processor/test_pi05_processor.py \
  tests/processor/test_act_processor.py -v
git add run_scripts tests/training/test_multi_gpu.py
git commit -m "test: add four relative policy training workflows"
```

### Task 11：同步远端、生成统计并执行冒烟测试

**Files:**
- Read: `/home/wengyikun/workplace/joint_songling/lerobot/**`
- Write remote: `/home/wengyikun/lerobot/**`
- Read local: `/data/joint_songling/0714_gripper_bread_combined_split_seed42/**`
- Write remote: `/data/wengyikun/datasets/joint_songling/0714_gripper_bread_combined_split_seed42/**`

- [x] **Step 1：本地最终测试**

```bash
.venv/bin/pytest tests/processor/test_normalize_processor.py \
  tests/datasets/test_relative_joint_stats.py \
  tests/processor/test_relative_joint_processor.py \
  tests/processor/test_pi05_processor.py \
  tests/processor/test_act_processor.py \
  tests/datasets/test_train_eval_factory.py \
  tests/common/test_offline_validation.py \
  tests/utils/test_train_utils.py -v
```

Expected: 全部 PASS；允许的 skip 只能来自缺少 CUDA 的既有 GPU 测试。

- [x] **Step 2：同步代码和数据**

```bash
ssh -p 50210 wengyikun@183.230.224.121 \
  'mkdir -p /data/wengyikun/datasets/joint_songling/0714_gripper_bread_combined_split_seed42'

rsync -a -e 'ssh -p 50210' \
  --exclude='.git/' --exclude='.venv/' --exclude='outputs/' \
  /home/wengyikun/workplace/joint_songling/lerobot/ \
  wengyikun@183.230.224.121:/home/wengyikun/lerobot/

rsync -a -e 'ssh -p 50210' \
  /data/joint_songling/0714_gripper_bread_combined_split_seed42/ \
  wengyikun@183.230.224.121:/data/wengyikun/datasets/joint_songling/0714_gripper_bread_combined_split_seed42/
```

不得删除远端 `/data` 中其他数据，所有进程以 `wengyikun` 用户运行。

- [x] **Step 3：远端校验并计算统计**

远端使用可移植的相对 split root 从 train 重算。state、chunk16、chunk50 三个 q01/q99
文件与本地 SHA256 一致；provenance manifest 记录远端命令和远端数据根，因此其 SHA 与本地不同。

```bash
ssh -p 50210 wengyikun@183.230.224.121 \
  'cd /home/wengyikun/lerobot && ./run_scripts/remote_lerobot_container.sh \
    python -m lerobot.scripts.compute_relative_joint_stats \
    --dataset-root=/data/wengyikun/datasets/joint_songling/0714_gripper_bread_combined_split_seed42/train \
    --split-manifest=/data/wengyikun/datasets/joint_songling/0714_gripper_bread_combined_split_seed42/split_manifest.json \
    --output-dir=/data/wengyikun/datasets/joint_songling/0714_gripper_bread_combined_split_seed42/normalization \
    --horizons="[50,16]" --gripper-indices="[6]"'
```

确认三套 q01/q99 均为 7 维，来源只指向 train。

- [x] **Step 4：四个单 GPU train+validation 冒烟**

每个脚本覆盖环境变量 `NUM_PROCESSES=1 STEPS=1 EVAL_STEPS=1 MAX_EVAL_SAMPLES=4`，
确认真实 train step 和 validation 均完成、六个指标均输出、MSE 已反归一化并恢复绝对 arm。

- [x] **Step 5：按用户要求跳过双 GPU 冒烟**

本轮只验证 PI0.5 relative-state、PI0.5 image-only、ACT relative-state、ACT image-only
四个单 GPU 训练方案能正确运行，不执行远端双 GPU smoke。

- [x] **Step 6：完成四方案 smoke 审计，按用户要求不启动正式训练**

四个启动脚本均已打印并保存 GPU、有效 batch size、train/test root、stats SHA256、
condition_on_state、chunk size、冻结/可训练参数数量和输出目录。四个单 GPU smoke 均完成真实
train step、validation 六指标并正常结束。按用户最新要求，本轮只验证四个训练方案，未启动
10000-step 正式训练；正式脚本默认仍为 `steps=10000`、`save_freq=2000`、`eval_steps=1000`。

### Task 12：固定鱼眼裁剪与 train-only 7D 状态噪声

- [x] **Step 1：固定 PI0.5 与 ACT 图像路径**

两种策略统一执行 `480x640 -> center crop columns [80:560] -> 480x480 -> 224x224`，不使用随机图像增强或 padding。

- [x] **Step 2：实现仅训练态的 7D 状态噪声**

relative-state 训练在相对表示构造之后、q01/q99 归一化之前，对六个关节加入 `0.003 rad` 高斯噪声、对绝对夹爪加入 `0.001 m` 高斯噪声。action label 不变。image-only、validation loss 和 validation MSE 均关闭噪声，image-only 的 7D 始终为零。

- [x] **Step 3：测试、提交并推送**

新增裁剪和状态噪声单元测试，并回归 PI0.5、ACT processor、relative-joint 与启动脚本测试。

---

## 完成判定

以下条件全部满足才可宣布实施完成：

```text
1. 原 train/loss 是唯一 backward 目标。
2. train/gripper_loss、valid/loss、valid/gripper_loss、valid/action_mse、valid/gripper_mse 都不参与 backward。
3. test 不参与训练统计、图像增强、optimizer 或 scheduler。
4. 每 1000 optimizer step 使用完整 test 计算四个 validation 指标。
5. PI0.5/ACT 的 padding 不进入整体/夹爪 loss 或整体/夹爪 MSE。
6. 两个 PI0.5 实验都冻结语言模型，视觉 encoder/projector 可训练。
7. image-only 模型边界保留 7D state，且每个值精确为零。
8. 外部 normalization 目录保存三套 train-only q01/q99，训练和推理脚本显式指定其路径。
9. 所有数字 step checkpoint 均保留，不自动创建或替换 best_validation；用户根据 valid/action_mse 手动选择。
10. PI0.5/ACT 的 relative-state 与 image-only 四个远端单 GPU 冒烟全部通过；双 GPU 冒烟按用户要求跳过。
11. 四个 smoke 日志均出现 train/loss、train/gripper_loss 和四个 valid/* 指标并正常结束。
```
