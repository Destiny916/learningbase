# PI05 Absolute Pose Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Configure isolated PI05 absolute-pose training for the 0714 joint7d and pose10d datasets with train-only action-specific quantiles and physical validation metrics.

**Architecture:** Add an opt-in absolute-stat bundle usable by both 7D joints and pose10d. PI05 selects the bundle only when the new absolute-stat option is enabled; pose10d reuses its pose-aware normalizer but skips all relative transformations. Offline validation dispatches by representation so absolute chunks are compared directly.

**Tech Stack:** Python, PyTorch, LeRobot processor pipelines, pytest, Bash, Accelerate.

---

### Task 1: Absolute Quantile Contract

**Files:**
- Create: `src/lerobot/datasets/absolute_action_stats.py`
- Create: `src/lerobot/scripts/compute_absolute_action_stats.py`
- Test: `tests/datasets/test_absolute_action_stats.py`

- [ ] Add failing tests for separate state and horizon action `q01/q99` values.
- [ ] Implement loadable, dimension-checked absolute statistic files.
- [ ] Add a CLI that reads only training episodes and writes stats under the split dataset.
- [ ] Run focused tests and commit.

### Task 2: PI05 Absolute Processor Configuration

**Files:**
- Modify: `src/lerobot/policies/pi05/configuration_pi05.py`
- Modify: `src/lerobot/policies/pi05/processor_pi05.py`
- Test: `tests/policies/pi0_pi05/test_pi05_joint_representation.py`
- Test: `tests/policies/test_end_effector_pose_pipelines.py`

- [ ] Add opt-in absolute state/action statistic paths and validation.
- [ ] Use generic quantile normalization for joint7d and pose-aware normalization for pose10d.
- [ ] Verify no relative transform is placed in either absolute pipeline.
- [ ] Run focused tests and commit.

### Task 3: Absolute Physical Validation

**Files:**
- Modify: `src/lerobot/common/offline_validation.py`
- Test: `tests/training/test_relative_policy_launch_scripts.py`

- [ ] Add failing direct-absolute action MSE test.
- [ ] Dispatch absolute joint and pose actions to direct de-normalized comparison.
- [ ] Preserve relative reconstruction and existing metrics unchanged.
- [ ] Run focused tests and commit.

### Task 4: Absolute Launch Configurations

**Files:**
- Create: `run_scripts/launch_pi05_absolute_joint7d_0714.sh`
- Create: `run_scripts/launch_pi05_absolute_end_effector_pose_0714.sh`
- Modify: `tests/training/test_relative_policy_launch_scripts.py`

- [ ] Add launch-contract tests first.
- [ ] Add isolated launchers using the split datasets, absolute stats, chunk size 50, and periodic offline validation.
- [ ] Verify static shell syntax and focused tests.
- [ ] Commit and push the task commits to `chengdu/main`.
