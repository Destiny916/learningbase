# PI05 LIBERO 官方参考验证实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 使用官方 `lerobot/pi05_libero_finetuned_v044` 和
`HuggingFaceVLA/libero`，验证当前 LeRobot PI05 的数据处理、训练和闭环推理逻辑
是否与官方 LIBERO 契约一致。

**架构：** 分开验证三类结论：输入/输出语义是否一致、训练优化是否真正可学习、闭环
控制是否达到官方参考表现。官方 v044 checkpoint 仅作为只读金标准；候选模型从
`lerobot/pi05_libero` 初始化并在同一数据与同一评估协议下比较。所有运行使用现有
Docker 当前用户单 GPU runner；完整参考基准按 8 GPU/6000 step 另行执行。

**技术栈：** LeRobot PI05、`HuggingFaceVLA/libero`、LIBERO/MuJoCo EGL、
PyTorch BF16、Docker、pytest、JSON/MP4 evaluator artifacts。

---

## 通过标准

- 官方 v044 能在本机 fork 中严格加载，且 processor 输入输出特征与 checkpoint
  配置一致。
- 数据集原始 frame 经 processor 后，图像键、8D state、7D 相对 action、归一化
  统计和 action chunk 形状均满足 checkpoint 契约。
- 单 episode 过拟合训练的训练 loss 和物理量纲 action MSE 明显下降，并生成可重载
  checkpoint；这证明当前训练反向传播和目标构造正确。
- v044 在四个标准 LIBERO suite 各 10 episode 的评估产物完整；候选模型使用相同
  seed、suite、episode 数和 `n_action_steps=10` 评估。
- 报告明确区分：结构/语义正确、训练可学习、闭环任务成功率三种结论。

## Task 1：固定官方参考与运行契约

**文件：**
- Create: `run_scripts/eval_pi05_libero_reference_v044.sh`
- Create: `tests/training/test_pi05_libero_reference_validation.py`
- Create: `docs/superpowers/validation/2026-07-17-pi05-libero-reference-contract.md`

- [ ] **Step 1：写失败的 launcher 契约测试**

```python
def test_v044_reference_launcher_uses_official_checkpoint_and_protocol():
    script = _read_script("eval_pi05_libero_reference_v044.sh")
    assert 'POLICY_PATH="lerobot/pi05_libero_finetuned_v044"' in script
    assert '--env.type=libero' in script
    assert '--env.task=libero_spatial,libero_object,libero_goal,libero_10' in script
    assert '--eval.n_episodes=10' in script
    assert '--policy.n_action_steps=10' in script
    assert '--env.control_mode=relative' in script
```

- [ ] **Step 2：运行测试确认失败**

Run: `PYTHONPATH=src .venv/bin/pytest tests/training/test_pi05_libero_reference_validation.py -q`

Expected: FAIL，因为 launcher 尚不存在。

- [ ] **Step 3：实现只读官方参考 launcher**

创建脚本，固定 `POLICY_PATH="lerobot/pi05_libero_finetuned_v044"`，输出目录为
`/data/wengyikun/pi05_libero_reference_v044/eval_out`，并执行：

```bash
python -m lerobot.scripts.lerobot_eval \
  --policy.path="$POLICY_PATH" \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.n_action_steps=10 \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --env.control_mode=relative \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --output_dir="$OUTPUT_DIR"
```

- [ ] **Step 4：运行测试确认通过**

Run: `PYTHONPATH=src .venv/bin/pytest tests/training/test_pi05_libero_reference_validation.py -q`

Expected: PASS。

- [ ] **Step 5：记录 v044 不可变参考信息**

在验证文档记录模型 revision、`config.json`、processor JSON 的 SHA256、
`HuggingFaceVLA/libero` revision、输出 action 为 7D、state 为 8D、两路 256 图像、
`chunk_size=50`、推理 `n_action_steps=10`、以及官方 6000-step/8-H100 训练设置。

- [ ] **Step 6：提交**

```bash
git add run_scripts/eval_pi05_libero_reference_v044.sh \
  tests/training/test_pi05_libero_reference_validation.py \
  docs/superpowers/validation/2026-07-17-pi05-libero-reference-contract.md
git commit -m "test: add PI05 LIBERO reference protocol"
git push chengdu main
```

## Task 2：验证数据和 Processor 语义

**文件：**
- Create: `tests/policies/pi05/test_libero_reference_contract.py`
- Create: `src/lerobot/scripts/audit_pi05_libero_reference_contract.py`
- Create: `docs/superpowers/validation/2026-07-17-pi05-libero-processor-audit.md`

- [ ] **Step 1：写失败的 processor 契约测试**

测试从数据集读取一个 frame 后，经官方 checkpoint 的 preprocessor：

```python
assert processed["observation.state"].shape[-1] == 8
assert processed["action"].shape[-1] == 7
assert processed["observation.images.image"].shape[-2:] == (224, 224)
assert processed["observation.images.image2"].shape[-2:] == (224, 224)
assert torch.isfinite(processed["observation.state"]).all()
assert torch.isfinite(processed["action"]).all()
```

还要断言 action postprocessor 的反归一化结果和原始 7D action 在有效 token 上
`torch.allclose(..., atol=1e-5, rtol=1e-5)`。

- [ ] **Step 2：运行测试确认失败**

Run: `PYTHONPATH=src .venv/bin/pytest tests/policies/pi05/test_libero_reference_contract.py -q`

Expected: FAIL，因为审计脚本和 test fixture 尚不存在。

- [ ] **Step 3：实现审计脚本**

脚本加载 `lerobot/pi05_libero_finetuned_v044`、`HuggingFaceVLA/libero` episode 0，
打印原始和处理后的每个键的 shape、dtype、min/max，验证 action 的
`postprocess(preprocess(action))` 误差，并检查 `LiberoProcessorStep` 的图像 180 度
旋转路径在训练与环境评估两端都被使用。

- [ ] **Step 4：运行单元测试与远端审计**

Run:

```bash
PYTHONPATH=src .venv/bin/pytest tests/policies/pi05/test_libero_reference_contract.py -q
LEROBOT_GPUS=device=<idle_gpu> bash run_scripts/remote_pi05_libero_smoke_container.sh \
  python -m lerobot.scripts.audit_pi05_libero_reference_contract
```

Expected: 单元测试 PASS；远端日志包含 `REFERENCE_CONTRACT_PASS`，并输出所有
归一化往返最大误差。

- [ ] **Step 5：记录结果并提交**

```bash
git add tests/policies/pi05/test_libero_reference_contract.py \
  src/lerobot/scripts/audit_pi05_libero_reference_contract.py \
  docs/superpowers/validation/2026-07-17-pi05-libero-processor-audit.md
git commit -m "test: audit PI05 LIBERO processor contract"
git push chengdu main
```

## Task 3：小数据过拟合验证训练目标

**文件：**
- Create: `run_scripts/launch_pi05_libero_overfit.sh`
- Create: `src/lerobot/scripts/eval_pi05_libero_offline.py`
- Create: `tests/training/test_pi05_libero_overfit_launcher.py`
- Create: `docs/superpowers/validation/2026-07-17-pi05-libero-overfit.md`

- [ ] **Step 1：写失败 launcher 测试**

测试脚本固定：`dataset.episodes=[0]`、`steps=2000`、`batch_size=1`、
`gradient_accumulation_steps=1`、`log_freq=10`、`save_freq=500`、
`policy.chunk_size=50`、`policy.n_action_steps=10`、冻结语言模型但不冻结视觉编码器。

- [ ] **Step 2：运行测试确认失败**

Run: `PYTHONPATH=src .venv/bin/pytest tests/training/test_pi05_libero_overfit_launcher.py -q`

Expected: FAIL，因为 overfit launcher 尚不存在。

- [ ] **Step 3：实现过拟合 launcher**

从 `lerobot/pi05_libero` 初始化，使用与官方相同的 7D 相对动作和
`HuggingFaceVLA/libero`；输出到
`/data/wengyikun/pi05_libero_overfit/train_out`。显式使用 `--log_freq=10`，保证
记录训练 loss；不得设置图像随机增强。

- [ ] **Step 4：实现物理量纲离线评估**

评估脚本必须分别记录：归一化 action MSE、反归一化后 7D action MSE、每维 RMSE，
并对目标 chunk 和预测 chunk 使用同一个 postprocessor。输出 JSON，包含
`normalized_action_mse`、`physical_action_mse`、`per_dim_rmse`。

- [ ] **Step 5：运行并验收**

在单张空闲 GPU 运行 launcher，随后对 step 0（基础模型）和 step 2000 checkpoint
运行离线评估。验收条件：step 2000 的物理 action MSE 小于 step 0，训练 loss 在日志
中明显下降；若不满足，停止后先检查 Task 2 的数据契约，不进入正式训练。

- [ ] **Step 6：提交**

```bash
git add run_scripts/launch_pi05_libero_overfit.sh \
  src/lerobot/scripts/eval_pi05_libero_offline.py \
  tests/training/test_pi05_libero_overfit_launcher.py \
  docs/superpowers/validation/2026-07-17-pi05-libero-overfit.md
git commit -m "test: add PI05 LIBERO overfit validation"
git push chengdu main
```

## Task 4：官方 checkpoint 与候选 checkpoint 闭环对照

**文件：**
- Create: `run_scripts/eval_pi05_libero_candidate.sh`
- Create: `src/lerobot/scripts/compare_pi05_libero_metrics.py`
- Create: `tests/training/test_pi05_libero_candidate_eval.py`
- Create: `docs/superpowers/validation/2026-07-17-pi05-libero-comparison.md`

- [ ] **Step 1：写失败的候选评估测试**

测试 launcher 需要 `CANDIDATE_POLICY_PATH`，并固定四个 suite、每 task 10 episode、
相对控制、`n_action_steps=10`、单环境。测试比较脚本要求官方和候选的
`eval_info.json` 都含 `per_task`、`per_group`、`overall`。

- [ ] **Step 2：运行测试确认失败**

Run: `PYTHONPATH=src .venv/bin/pytest tests/training/test_pi05_libero_candidate_eval.py -q`

Expected: FAIL，因为候选评估与比较脚本尚不存在。

- [ ] **Step 3：实现候选评估与比较脚本**

候选 launcher 使用：

```bash
--env.task=libero_spatial,libero_object,libero_goal,libero_10
--env.control_mode=relative
--policy.n_action_steps=10
--eval.batch_size=1
--eval.n_episodes=10
--env.max_parallel_tasks=1
```

比较脚本读取两份 `eval_info.json`，按 suite 输出成功率、平均成功率、episode 数、
reward，并在任意 suite 缺少 100 个 episode 时返回非零状态。

- [ ] **Step 4：运行官方参考评估和候选评估**

先运行 v044 reference launcher，后运行候选 checkpoint；两者使用同一 Docker image、
GPU 类型、seed、LIBERO assets revision。保存 JSON 与全部视频路径，不覆盖旧输出。

- [ ] **Step 5：验收并提交**

验收官方 reference 的总 episode 数为 400，候选的总 episode 数也为 400。报告必须
把官方 v044、当前 1-step smoke、过拟合 checkpoint、正式候选 checkpoint 分开列出，
不得把不同训练量的 success 混为同一结论。

```bash
git add run_scripts/eval_pi05_libero_candidate.sh \
  src/lerobot/scripts/compare_pi05_libero_metrics.py \
  tests/training/test_pi05_libero_candidate_eval.py \
  docs/superpowers/validation/2026-07-17-pi05-libero-comparison.md
git commit -m "test: compare PI05 LIBERO reference rollout"
git push chengdu main
```

## Task 5：正式复现的资源门槛与最终结论

**文件：**
- Create: `docs/superpowers/validation/2026-07-17-pi05-libero-final-verdict.md`

- [ ] **Step 1：核对资源与正式复现实验参数**

正式官方对照训练使用 `steps=6000`、全局 batch size `256`、BF16、8 H100、
`chunk_size=50`、推理 `n_action_steps=10`。若当前硬件不足，不修改该定义；报告中
将其标为“未执行的官方规模复现”，并将小规模/过拟合结果标为逻辑验证而非性能复现。

- [ ] **Step 2：形成逐项结论**

最终文档分别给出以下结论：

1. 输入输出与归一化语义是否通过 Task 2。
2. 当前训练实现是否通过 Task 3 的可学习性验证。
3. 当前候选 checkpoint 是否通过 Task 4 的闭环完成性验证。
4. 是否已达到官方 v044/6000-step 性能复现标准。

- [ ] **Step 3：提交**

```bash
git add docs/superpowers/validation/2026-07-17-pi05-libero-final-verdict.md
git commit -m "docs: record PI05 LIBERO validation verdict"
git push chengdu main
```

## 不在本计划范围内

- 将 LIBERO 的 7D 相对末端动作直接用于 Piper/Pika 真机。
- 以 1-step 或单 episode 结果宣称模型能力。
- 修改已有 Piper/Pika 真实数据训练代码。
- 在未验证 action/state 语义前混合 LIBERO 与真实机器人数据。
