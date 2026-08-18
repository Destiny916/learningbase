# 0714 Stratified Train/Test Split Design

## Goal

Split the existing 159-episode LeRobot v3.0 dataset into independent train and test datasets while preserving the source dataset unchanged.

Source dataset:

```text
/data/joint_songling/0714_gripper_bread_single_teleop_normal_differentplace_wrongplace_right_fisheye_combined_v30
```

Output root:

```text
/data/joint_songling/0714_gripper_bread_combined_split_seed42
```

The output contains `train/`, `test/`, and `split_manifest.json`.

## Episode Selection

The source has 159 continuous episode indices, `0..158`. Test episodes are sampled without replacement with one shared `random.Random(42)` instance, processing the strata in this order:

1. Sample 10 episodes from `0..100`.
2. Sample 3 episodes from `101..132`.
3. Sample 3 episodes from `133..158`.

The fixed test selection is:

```text
[3, 13, 14, 17, 28, 31, 35, 81, 86, 94,
 103, 106, 128,
 133, 135, 139]
```

The other 143 source episodes form the training set. The two sets must be disjoint and their union must equal `0..158`.

## Output Semantics

Use LeRobot's explicit episode-list split path so each output is a complete, directly loadable v3.0 dataset.

- Source episode content, timestamps, state, action, tasks, and video frames remain unchanged.
- Episodes in each output are sorted by source episode index and reindexed continuously from zero.
- Global frame `index`, `episode_index`, task indices, episode metadata, data paths, and video references are rewritten for each output.
- The training output contains 143 episodes; the test output contains 16 episodes.
- Feature statistics used for training are recomputed or aggregated for each output. Index-like metadata statistics are not treated as model feature statistics.
- The source `conversion_summary.json` and `alignment_reports/` are not copied into the split datasets because they are conversion artifacts rather than required LeRobot v3.0 files. Provenance is retained in the split manifest.

## Split Manifest

`split_manifest.json` records:

- source and output paths;
- random seed and the exact sampling algorithm;
- the three source strata and sample counts;
- sorted source episode lists for train and test;
- old-to-new episode mappings for both outputs;
- source and output episode/frame totals.

The exact test episode list above is authoritative. Regeneration must not silently produce a different split.

## Validation

Validation must complete before the split is accepted:

1. Load both outputs through `LeRobotDataset`.
2. Confirm train/test episode counts are 143 and 16.
3. Confirm the manifest lists are disjoint and cover all source episodes exactly once.
4. Confirm each output has continuous episode indices, global frame indices, and episode-local frame indices.
5. Compare every selected source episode with its output episode for frame count, timestamps, state, action, task text, and decoded video frame count.
6. Confirm all referenced Parquet and video files exist and that metadata frame totals match stored data.
7. Confirm both outputs retain `codebase_version: v3.0`, 30 FPS, one right-fisheye video feature, and 7D state/action features.

Any failed validation leaves the source untouched and the output must not be reported as usable.
