# PI05 Precomputed Relative Chunk Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the relative state and 20-step anchored action dataset and make PI05 train and infer with it without duplicate transforms.

**Architecture:** A dedicated converter copies the video assets and transforms only numeric LeRobot v3 data. A PI05 config flag disables dataset temporal lookup and processor transforms during training, while preserving the existing absolute-to-relative and relative-to-absolute inference path.

**Tech Stack:** Python 3.12, NumPy, PyArrow, LeRobot v3, PyTorch, pytest.

---

### Task 1: Define numerical transforms

**Files:**
- Create: `tests/scripts/test_preprocess_relative_chunk_dataset.py`
- Create: `src/lerobot/scripts/preprocess_relative_chunk_dataset.py`

- [x] Write failing tests for first-frame arm zeros, absolute grippers, future `q_{t+k}-q_t` arm targets, and endpoint padding.
- [x] Run the focused test and confirm the converter module is absent.
- [x] Implement pure array transformation helpers.
- [x] Run the focused test and confirm it passes.

### Task 2: Add precomputed PI05 mode

**Files:**
- Modify: `src/lerobot/policies/pi05/configuration_pi05.py`
- Modify: `src/lerobot/policies/pi05/joint_representation.py`
- Modify: `src/lerobot/policies/pi05/processor_pi05.py`
- Modify: `tests/policies/pi0_pi05/test_pi05_joint_representation.py`

- [x] Write failing tests for no training-time action lookup, no duplicate relative conversion, and online absolute-state conversion.
- [x] Implement `precomputed_relative_chunk` with raw-source action indices `0..chunk_size-1`.
- [x] Run the PI05 relative representation tests.

### Task 3: Build and verify the dataset

**Files:**
- Create: `/data/joint_songling/0704_bread_grasp_only_songling_robot_relative_chunk20`

- [x] Copy the two source videos without re-encoding.
- [x] Write transformed numeric parquet, episode metadata, info, and stats.
- [x] Verify 101 episodes, 9,762 samples, action shape `[20,14]`, and source-video hashes.
- [x] Verify representative numerical samples and a PI05 CPU preprocessing smoke batch.
