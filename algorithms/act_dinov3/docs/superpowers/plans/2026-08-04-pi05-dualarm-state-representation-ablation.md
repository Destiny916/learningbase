# PI05 Dual-Arm State Representation Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Launch two verified PI05 trainings that differ only in relative versus absolute arm state while sharing relative arm actions and absolute grippers.

**Architecture:** Reuse the existing relative 14D processor for the relative-state run. Extend the PI05 configuration/statistics boundary so `joint_representation=absolute` and `use_relative_actions=true` can load absolute state quantiles together with relative action quantiles. Keep model, images, action labels, optimizer schedule, and dataset identical across launchers.

**Tech Stack:** Python, PyTorch, LeRobot processor pipelines, pytest, Bash, Docker, NVIDIA CUDA.

---

### Task 1: Specify mixed state/action statistics

**Files:**
- Modify: `tests/policies/pi0_pi05/test_pi05_joint_representation.py`
- Modify: `src/lerobot/policies/pi05/configuration_pi05.py`
- Modify: `src/lerobot/policies/pi05/joint_representation.py`

- [ ] Add a failing test that creates absolute state q01/q99 and relative
      action chunk50 q01/q99 for 14 named dimensions.
- [ ] Assert `PI05Config` accepts only the complete combination:
      `joint_representation=absolute`, `use_relative_actions=true`,
      `absolute_state_stats_path`, and `relative_action_stats_path`.
- [ ] Run the focused test and confirm it fails because mixed statistics are
      not yet loaded.
- [ ] Add the smallest validated mixed-stat loader and expose its bundle on the
      PI05 config.
- [ ] Make `merge_pi05_joint_stats` map absolute state q01/q99 to
      `observation.state` and relative action q01/q99 to `action`.
- [ ] Run the focused test and the existing PI05 joint-representation suite.

### Task 2: Prove the processor representation boundary

**Files:**
- Modify: `tests/policies/pi0_pi05/test_pi05_joint_representation.py`
- Modify: `src/lerobot/policies/pi05/processor_pi05.py`

- [ ] Add a failing test with one 14D state and a two-step absolute action
      chunk.
- [ ] Assert the absolute-state pipeline keeps state equal to `q_t`, converts
      arm actions to `q_(t+k)-q_t`, keeps grippers `6,13` absolute, and uses
      quantile normalization with clipping disabled.
- [ ] Assert postprocessing reconstructs absolute arm targets and leaves
      grippers unchanged.
- [ ] Make only the processor changes needed for the test.
- [ ] Run the focused test and all PI05 processor tests.

### Task 3: Add fixed launchers

**Files:**
- Create: `run_scripts/launch_pi05_relative_state_dualarm14d_0724_0727_full99_nonoise_noclip.sh`
- Create: `run_scripts/launch_pi05_absolute_state_relative_action_dualarm14d_0724_0727_full99_nonoise_noclip.sh`
- Modify: `tests/training/test_relative_policy_launch_scripts.py`

- [ ] Add failing launcher contract tests for the common dataset, camera order,
      PI05 Base weights, BF16 vision training, batch 16, 100k schedule, no
      validation, no noise, no clipping, and new output directories.
- [ ] Assert the relative-state launcher uses relative state statistics.
- [ ] Assert the absolute-state launcher uses absolute state statistics,
      `joint_representation=absolute`, `use_relative_actions=true`, and the same
      relative action statistics.
- [ ] Implement both launchers with mandatory file and output-directory guards.
- [ ] Run `bash -n` and the launcher contract tests.

### Task 4: Generate and audit remote statistics

**Files:**
- No repository files modified.

- [ ] Recompute absolute state statistics from all 99 raw episodes in the
      current LeRobot container environment.
- [ ] Independently compare count and q01/q99 against raw parquet
      `observation.state` values.
- [ ] Recompute relative action chunk50 statistics and compare them with the
      existing file, requiring zero numerical difference.
- [ ] Verify relative state first frames are zero and no episode boundary is
      crossed.

### Task 5: Commit, deploy, and launch

**Files:**
- No additional repository files modified.

- [ ] Run focused and regression tests plus `git diff --check`.
- [ ] Commit only files belonging to this feature and push `chengdu/main`.
- [ ] Synchronize the committed source to `/home/wengyikun/lerobot` without
      overwriting unrelated remote outputs.
- [ ] Verify GPU1 and GPU7 are idle and `/data` has enough initial capacity.
- [ ] Start the relative-state container on GPU1 and the absolute-state
      container on GPU7 as UID/GID `1009:1008`.
- [ ] Confirm each container sees one GPU, loads PI05 Base, receives all three
      cameras, freezes language, trains vision, and logs a finite first loss.
- [ ] Record container names, output directories, effective parameters, and
      first loss values.

