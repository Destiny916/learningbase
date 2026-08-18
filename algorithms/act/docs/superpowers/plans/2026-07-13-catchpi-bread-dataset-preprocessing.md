# CatchPI Bread Dataset Preprocessing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a tested converter that creates the no-top, first-two-thirds, gripper-named LeRobot v3 dataset.

**Architecture:** A focused CLI reads numeric rows and episode boundaries from the source parquet files, decodes the two source videos sequentially, and writes selected frames through `LeRobotDataset.create`. Pure helper functions own trimming, feature rewriting, and terminal-action behavior so they can be unit tested without processing the full dataset.

**Tech Stack:** Python 3.12, PyArrow, PyAV, NumPy, LeRobot v3, pytest, ffprobe.

---

### Task 1: Define preprocessing semantics with tests

**Files:**
- Create: `tests/scripts/test_preprocess_catchpi_dataset.py`
- Create: `src/lerobot/scripts/preprocess_catchpi_dataset.py`

- [x] Write tests for floor trimming, top-camera removal, gripper metadata names, and terminal actions.
- [x] Run the focused test and confirm it fails because the converter module is absent.
- [x] Implement the pure helper functions.
- [x] Run the focused test and confirm it passes.

### Task 2: Implement the LeRobot v3 conversion CLI

**Files:**
- Modify: `src/lerobot/scripts/preprocess_catchpi_dataset.py`

- [x] Validate source layout and reject an existing output path.
- [x] Read source parquet and episode metadata.
- [x] Decode both wrist videos in lockstep and write retained frames episode by episode.
- [x] Rewrite auxiliary conversion manifests without top-camera entries.
- [x] Emit a preprocessing summary containing source/output paths and old/new episode lengths.

### Task 3: Convert and validate the complete dataset

**Output:**
- Create: `/data/joint_songling/0704_bread_grasp_only_songling_robot`

- [x] Run the CLI against `0704_video`.
- [x] Verify 101 episodes and 9,762 total frames.
- [x] Verify both videos contain exactly 9,762 frames and no top-camera files remain.
- [x] Verify parquet indices, episode lengths, gripper names, statistics, and all terminal actions.
- [x] Confirm the source dataset is unchanged.
