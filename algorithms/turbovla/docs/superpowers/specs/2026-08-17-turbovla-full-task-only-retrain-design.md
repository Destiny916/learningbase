# TurboVLA Full-Task-Only Fresh Retraining Design

## Objective

Retrain TurboVLA from scratch on only the complete bread-transfer task episodes so that inference with one fixed task instruction starts with the right arm grasping the bread while the left arm remains idle until handoff.

The source dataset is:

```text
/data/wengyikun/datasets/joint_songling/0812_binary_gripper_without_ep173_174
```

The fixed instruction is:

```text
first pick up the bread with the right hand, then hand it to the left hand at the middle point, then place the bread in the bowl with the left hand.
```

## Root-Cause Evidence

The source dataset contains five task groups. Their meanings are:

| Source task | Episodes | Frames | Meaning | Include |
|---|---:|---:|---|---|
| `0806swap` | 71 | 49,430 | Complete task | Yes |
| `0812` | 50 | 35,642 | Complete task | Yes |
| `0812_wrong1` | 50 | 31,304 | Complete task | Yes |
| `0812_wrong2` | 21 | 7,251 | Post-handoff placement stage | No |
| `0812_wrong3` | 10 | 1,931 | Placement stage | No |

Independent inspection showed that complete-task episodes have zero or near-zero initial left-arm motion while the right arm begins grasping. `wrong2` and `wrong3` begin with left-arm motion and little right-arm motion. Relabeling all five groups with one full-task instruction therefore introduced two competing initial action modes.

## Dataset Overlay

Create a new overlay without modifying the source dataset. It contains only source task indices `1`, `3`, and `4`, corresponding to `0806swap`, `0812`, and `0812_wrong1`.

The selected source episodes are already contiguous episode IDs `0..170`, so the overlay keeps those IDs. Expected totals are:

```text
total_episodes: 171
total_frames: 116376
total_tasks: 1
```

The overlay must:

- preserve every selected `observation.state` and `action` value byte-for-byte;
- preserve timestamps, frame indices, episode indices, global indices, and per-camera episode video routing;
- replace all selected `task_index` values with `0`;
- store only the fixed English instruction in `meta/tasks.parquet` and episode task metadata;
- link to the original video directory rather than re-encoding videos;
- record a contract containing source path, selected source task indices, totals, camera order, current-frame action anchor, binary absolute grippers, and image layout.

The new overlay path must be distinct from deleted or previous overlays, for example:

```text
/data/wengyikun/outputs/turbovla_0812_binary_gripper_fulltask_only_top_padded_overlay
```

## State and Action Contract

Keep the verified representation:

```yaml
state_mode: delta
state_mode_apply_keys: [left_joints, right_joints]
action_mode: rel
action_mode_apply_keys: [left_joints, right_joints]
```

Semantics:

- left/right joint state: `q[t] - q[t-1]` within each episode;
- endpoint xyz state: absolute;
- gripper state: absolute binary `0.0/0.1`;
- future joint actions: each of the 50 targets relative to the current state `q[t]`;
- gripper actions: absolute binary `0.0/0.1`.

Recompute q01/q99 from only the 171 selected episodes. Independently recompute state and current-anchor action quantiles and require maximum error `0` against the generated training statistics.

## Image Contract

Use three cameras in this order:

```text
top, gripper_left, gripper_right
```

Use `image_layout: joint_songling_top_padded`:

- top: `405x720`, pad 157 black rows above and 158 below to produce `720x720`, then resize to `224x224`;
- left/right wrist: retain the full `480x640` image and resize directly to `224x224`;
- DINOv3 processor: rescale by `1/255` and normalize with ImageNet mean/std.

No video file is rewritten in the overlay.

## Fresh Training Configuration

Run one fresh single-GPU training job. Do not load any prior TurboVLA full-model checkpoint. BERT and DINOv3 pretrained backbones remain the normal local backbone initialization.

Reuse the last verified training schedule:

```yaml
per_device_batch_size: 16
num_workers: 8
prefetch_factor: 2
persistent_workers: true
max_train_steps: 500000
num_warmup_steps: 25000
save_interval: 20000
eval_interval: 20000
learning_rate: 5.0e-5
lr_scheduler_type: cosine
ema_decay: 0.999
gradient_clipping: 1.0
```

Use a new run/output name containing `fulltask_only`, and refuse to overwrite an existing path. GPU1 is the preferred device if it remains free at launch. Other active GPU jobs must not be disturbed.

## Validation and Launch

Before formal training:

1. Validate overlay totals, selected task indices, task text, camera metadata, state/action dimensions, finite values, and binary grippers.
2. Confirm selected state/action arrays are unchanged from the source.
3. Independently verify state and action q01/q99 with zero maximum error.
4. Decode representative frames from each of the three source task groups and validate the top-padding and wrist-resize geometry.
5. Run a one-step smoke test with batch size 16 and the formal preprocessing path.

Formal training starts only after these checks pass. Completion of launch requires a live isolated GPU1 process, finite losses, no traceback/OOM/video decode errors, and creation of the new output configuration snapshots. Smoke-test outputs may be deleted after formal startup; the formal output and shared overlay are retained.

## Non-Goals

- Do not include `0812_wrong2` or `0812_wrong3` in this run.
- Do not use stage-specific prompts.
- Do not resume from retry8, EMA, non-EMA, or any recently deleted TurboVLA run.
- Do not modify the source dataset, ACT datasets, ACT training jobs, backbone weights, or unrelated GPU workloads.
