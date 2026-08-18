# TurboVLA 0806 Swap Stage-2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and launch a non-destructive 100,000-step TurboVLA stage-2 training run on GPU6 from the completed retry8 model checkpoint.

**Architecture:** Keep stage 1 immutable and introduce a separate stage-2 YAML and launcher. Reuse the established dataset overlay and representation contract, changing only the checkpoint source, learning-rate schedule, run identity, and output root.

**Tech Stack:** Bash, Hydra YAML, pytest, Docker, PyTorch/Accelerate, safetensors.

---

### Task 1: Specify the stage-2 configuration

**Files:**
- Modify: `tests/test_joint_songling_0806swap_config.py`
- Create: `experiments/joint_songling/configs/0806swap_3view_stage2.yaml`

- [ ] Add a test that requires `TURBOVLA_STAGE1_CKPT`, batch 16, 100,000 steps, 2,000 warmup steps, `1.0e-05` learning rates, cosine scheduling, and 20,000-step saves.
- [ ] Run `pytest -q tests/test_joint_songling_0806swap_config.py` and confirm the new test fails because the stage-2 YAML does not exist.
- [ ] Add the stage-2 YAML by copying the validated representation and dataset settings from `0806swap_3view.yaml` and changing only the stage-2 run/trainer settings.
- [ ] Run `pytest -q tests/test_joint_songling_0806swap_config.py` and confirm all tests pass.

### Task 2: Add the stage-2 launcher

**Files:**
- Modify: `tests/test_joint_songling_0806swap_config.py`
- Create: `scripts/joint_songling/train_0806swap_3view_stage2_gpu6.sh`

- [ ] Add a launcher test requiring GPU6 isolation, `TURBOVLA_STAGE1_CKPT`, overwrite protection, the fixed English task, and the stage-2 YAML path.
- [ ] Run the launcher test and confirm it fails because the stage-2 script does not exist.
- [ ] Derive the launcher from `train_0806swap_3view_gpu6.sh`, replace the checkpoint variable and config path, and retain all overlay validation and camera mappings.
- [ ] Run the complete 0806 swap test file and shell syntax validation.

### Task 3: Commit, synchronize, and launch

**Files:**
- Remote source: `/home/wengyikun/tmp/TurboVLA-joint-songling-relative`
- Remote output: `/data/wengyikun/outputs/turbovla_0806swap_gripper_fixed_3view_gpu6_stage2`

- [ ] Commit the tested local changes and push them to `chengdu/main`.
- [ ] Synchronize the changed files into the remote training source and verify their hashes.
- [ ] Verify the retry8 non-EMA checkpoint exists and stop the GPU6 inference-only container.
- [ ] Launch the stage-2 script in a named detached Docker training container with a new output root and launcher log.
- [ ] Inspect the live process, GPU6 usage, generated 20D/14D q01/q99 statistics, and first optimizer step; report any traceback or OOM instead of claiming success.
