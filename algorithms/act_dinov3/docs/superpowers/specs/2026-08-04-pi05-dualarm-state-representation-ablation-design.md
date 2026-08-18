# PI05 Dual-Arm State Representation Ablation Design

## Goal

Train two new PI05 policies on the same full 99-episode 0724/0727 dataset. The
only representation difference between the policies is the arm portion of
`observation.state`. Both policies use the same OpenPI-style relative arm action
chunk and absolute grippers.

No existing output directory or running job is modified.

## Dataset Contract

Both runs use:

```text
/data/wengyikun/datasets/joint_songling/
0724_0727_doublefripper_top_grippebread_combined_full_99episodes_task_en
```

The dataset contract is fixed:

- LeRobot v3.0, 99 episodes, 26,339 frames, 30 FPS.
- All episodes are training data; there is no validation dataset.
- Images enter PI05 in this order:
  `top`, `gripper_left`, `gripper_right`.
- State/action dimension order is:
  left joints `0..5`, left gripper `6`, right joints `7..12`, right gripper `13`.
- Task text is:
  `Pick up the bread with the right gripper, transfer it to the left gripper, and place it in the bowl.`

## Representation Contracts

### Relative-State Run

```text
state arm      = q_t - q_(t-1)
state gripper  = g_t
action arm[k]  = q_(t+k) - q_t, k=1..50
action gripper = g_(t+k)
```

The first arm state in every episode is zero. No difference crosses an episode
boundary. This is the same state/action representation as the previous
relative-state training.

### Absolute-State Run

```text
state arm      = q_t
state gripper  = g_t
action arm[k]  = q_(t+k) - q_t, k=1..50
action gripper = g_(t+k)
```

This run uses OpenPI's absolute observation state plus `DeltaActions` behavior.
Only dimensions `6` and `13` are excluded from delta conversion, so both
grippers remain absolute. At inference, relative arm predictions are converted
back to absolute joint targets by adding the current absolute state after
unnormalization.

## Quantile Statistics

State and action never share q01/q99 values.

- Relative-state run:
  `relative_state_q01_q99.json` plus
  `relative_action_chunk50_q01_q99.json`.
- Absolute-state run:
  newly computed `absolute_state_q01_q99.json` plus the same
  `relative_action_chunk50_q01_q99.json` used by the relative-state run.

All statistics are computed from the same 99 episodes. The absolute state file
is computed directly from raw `observation.state`; it is not derived from the
relative state file. The relative action statistics use all valid future
targets for offsets 1 through 50, with arm targets anchored to `q_t` and
grippers left absolute.

Both runs use quantile normalization without clamping:

```text
normalized = 2 * (x - q01) / (q99 - q01) - 1
clip_quantiles = false
```

This matches official OpenPI normalization. Values outside q01/q99 remain
outside `[-1, 1]`; training labels are not rewritten by clipping.

## Shared Training Parameters

Both launchers use:

```text
pretrained_path=/data/wengyikun/openpi/lerobot_pi05_base
dtype=bfloat16
vision_encoder_dtype=bfloat16
freeze_language_model=true
freeze_vision_encoder=false
train_expert_only=false
gradient_checkpointing=true
chunk_size=50
n_action_steps=50
empty_cameras=0
batch_size=16
gradient_accumulation_steps=1
num_workers=8
steps=100000
save_freq=10000
log_freq=10
eval_steps=0
scheduler_warmup_steps=5000
scheduler_decay_steps=100000
state_noise_std_rad=0
gripper_noise_std_m=0
dataset.image_transforms.enable=false
wandb.enable=false
```

GPU1 runs the relative-state policy. GPU7 runs the absolute-state policy. Each
uses a new output directory and a single visible host GPU.

## Loss and Inference

PI05 flow-matching loss is computed over normalized 50x14 action targets. It
does not unnormalize actions before loss calculation. All 14 active dimensions
participate; internal padding to 32 dimensions is excluded from loss.

The representation choice changes only the state tokens. Both runs train on
the same normalized relative action labels. During inference, action output is
unnormalized with relative action q01/q99, then arm offsets are added to the
current absolute state while gripper values remain absolute.

## Verification Gates

Before launch:

1. Processor tests prove the relative-state and absolute-state inputs differ as
   specified while their relative action tensors are identical.
2. Tests prove indices `6` and `13` remain absolute during preprocessing and
   postprocessing.
3. Tests prove the absolute-state pipeline uses absolute state q01/q99 and
   relative action q01/q99 without clipping.
4. The two launchers are checked for identical shared parameters and distinct
   state settings/output directories.
5. Remote statistics are recomputed and numerically audited against the raw
   parquet data.
6. Each container must expose only its requested GPU and run as the current
   user.
7. Training is considered started only after each log shows a finite first
   `train/loss` and the parameter audit confirms a frozen language model and a
   trainable vision encoder.

