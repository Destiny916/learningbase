# 0714 Stratified Train/Test Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce independent, validated LeRobot v3.0 train and test datasets from the 159-episode combined 0714 dataset using the approved fixed stratified split.

**Architecture:** Reuse `lerobot.datasets.split_dataset()` with explicit source episode lists. Generate the split lists deterministically before touching output data, let LeRobot rewrite Parquet/video metadata and episode indices, then write a root-level provenance manifest and validate both outputs against the source.

**Tech Stack:** Python 3, LeRobot v3 dataset APIs, Hugging Face Datasets/Parquet, PyAV, JSON.

---

### Task 1: Freeze And Check The Episode Selection

**Files:**
- Reference: `docs/superpowers/specs/2026-07-15-0714-stratified-train-test-split-design.md`
- Create during execution: `/data/joint_songling/0714_gripper_bread_combined_split_seed42/split_manifest.json`

- [ ] **Step 1: Generate the deterministic split in memory**

Use one `random.Random(42)` instance and call `sample` sequentially for `(0, 100, 10)`, `(101, 132, 3)`, and `(133, 158, 3)`.

- [ ] **Step 2: Assert the fixed expected test list**

```python
expected_test = [3, 13, 14, 17, 28, 31, 35, 81, 86, 94, 103, 106, 128, 133, 135, 139]
assert test_episodes == expected_test
assert len(train_episodes) == 143
assert set(train_episodes).isdisjoint(test_episodes)
assert set(train_episodes) | set(test_episodes) == set(range(159))
```

- [ ] **Step 3: Confirm the output root does not already exist**

Run:

```bash
test ! -e /data/joint_songling/0714_gripper_bread_combined_split_seed42
```

Expected: exit code 0. Do not overwrite an existing output.

### Task 2: Materialize The Two LeRobot Datasets

**Files:**
- Read: `/data/joint_songling/0714_gripper_bread_single_teleop_normal_differentplace_wrongplace_right_fisheye_combined_v30/**`
- Create: `/data/joint_songling/0714_gripper_bread_combined_split_seed42/train/**`
- Create: `/data/joint_songling/0714_gripper_bread_combined_split_seed42/test/**`
- Create: `/data/joint_songling/0714_gripper_bread_combined_split_seed42/split_manifest.json`

- [ ] **Step 1: Load and preflight the source**

```python
source = LeRobotDataset(
    repo_id="local/0714_gripper_bread_combined_v30",
    root=source_root,
)
assert source.meta.total_episodes == 159
assert source.meta.total_frames == 33631
assert source.meta.fps == 30
```

- [ ] **Step 2: Run the explicit episode-list split**

```python
outputs = split_dataset(
    source,
    splits={"train": train_episodes, "test": test_episodes},
    output_dir=output_root,
)
```

Expected: `train/` contains 143 reindexed episodes and `test/` contains 16 reindexed episodes. Video segments are decoded and re-encoded where the source video file contains episodes assigned to both outputs.

- [ ] **Step 3: Write the provenance manifest atomically**

Build old-to-new mappings by enumerating each sorted source episode list. Write JSON to `split_manifest.json.tmp`, flush and close it, then rename it to `split_manifest.json`. Include source/output paths, seed, strata, exact source episode lists, mappings, and source/output episode/frame totals.

### Task 3: Validate The Materialized Outputs

**Files:**
- Read: `/data/joint_songling/0714_gripper_bread_combined_split_seed42/**`

- [ ] **Step 1: Reload train and test datasets from disk**

```python
train = LeRobotDataset("local/0714_gripper_bread_combined_v30_train", root=output_root / "train")
test = LeRobotDataset("local/0714_gripper_bread_combined_v30_test", root=output_root / "test")
assert train.meta.total_episodes == 143
assert test.meta.total_episodes == 16
```

- [ ] **Step 2: Validate standard v3 metadata and index continuity**

For each output, assert `codebase_version == "v3.0"`, 30 FPS, state/action shape `[7]`, one right-fisheye video key, continuous output episode indices, continuous global frame indices, and `frame_index == 0..length-1` within every episode.

- [ ] **Step 3: Compare source and output trajectories**

For every old-to-new mapping, compare frame count and all per-frame `timestamp`, `frame_index`, `observation.state`, and `action` values. Resolve task indices to task strings and compare those strings rather than requiring numeric task indices to remain unchanged.

- [ ] **Step 4: Validate video references and decoded frame totals**

Confirm every referenced video exists. Decode each output video stream and assert the total decoded frames equal the output metadata's total frame count. Confirm every episode's video timestamp interval is present and ordered.

- [ ] **Step 5: Validate manifest coverage and totals**

Assert train/test source lists are disjoint, cover `0..158`, match the fixed test list, and that manifest frame totals match both output metadata objects.

- [ ] **Step 6: Report exact output paths, episode counts, frame counts, and test episode list**

Do not report the output as usable unless every validation above passes.
