# TurboVLA Patch-Vision T2 Design

## Objective

Start a new TurboVLA training run on the 0806 Joint Songling dataset with a
two-frame, three-camera visual context while retaining the existing TurboVLA
state path and ACT action decoder.  Per-device batch size remains 16.

## Data contract

Each sample at step `t` contains camera frames for `t-1` and `t`, in the fixed
order `top`, `gripper_left`, `gripper_right`.  The data loader clips out-of-range
indices to the first frame of the same episode, so step zero is represented as
`[frame(0), frame(0)]` and no sample crosses episode boundaries.

The state and action contracts remain unchanged:

- state: 20D, with relative left/right joints and absolute endpoint xyz plus
  grippers;
- target: the 50-step, 14D normalized action chunk, with relative arm joints
  and absolute grippers;
- normalization: independently generated q01/q99 statistics in the new run.

## Architecture

The frozen DINOv3 ViT-L encodes all six images and returns dense patch tokens
with shape `[B, 2, 3, 196, H]`.  The existing visual projection, learned view
embedding, and learned patch-position embedding are reused for each time step.
A new trainable two-entry time embedding is added before flattening to
`[B, 1176, 256]`.  The existing text fusion, current-state projection, and
ACTDecoder then operate without structural changes and continue to produce
`[B, 50, 14]`.

The DINOv3 ViT-L and BERT remain frozen.  DINO gradient checkpointing is off,
because frozen DINO has no activation backward pass.  The official TurboVLA
release checkpoint initializes all compatible existing weights; the only
intentional new parameter is the time embedding, initialized randomly.

## Non-goals

This does not import Patch Policy's dataset, future-goal conditioning, VQ-BeT,
diffusion head, simulator evaluator, or min/max normalization.  It does not
modify the active `retry8` run, the source dataset, or its statistics.

## Validation

Tests must prove time-index ordering/padding, image packing and layout, 6D
input shape, 1176-token conditioning, frozen DINO gradients, nonzero time
embedding gradients, and action output shape.  A remote Docker smoke run must
read a real batch at batch size 16, perform a backward/optimizer step, verify
the saved q01/q99 statistics, then start a new independent training run.
