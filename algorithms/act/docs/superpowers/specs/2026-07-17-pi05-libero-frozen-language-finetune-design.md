# PI05 LIBERO Frozen-Language Fine-Tune Design

## Goal

Provide an isolated launcher for fine-tuning the unmodified official
`lerobot/pi05_libero` checkpoint on the complete `HuggingFaceVLA/libero`
dataset. This experiment freezes only the language model while keeping the
vision encoder, multimodal projector, Action Expert, and action projections
trainable.

## Training Contract

- Start from `lerobot/pi05_libero`, never `pi05_libero_finetuned` and never a
  real-robot checkpoint.
- Do not provide `visual_pretrained_path`; the checkpoint's own LIBERO visual
  weights remain the initialization.
- Preserve the official LIBERO processor contract: two 256x256 images, 8D
  state, 7D relative action, and checkpoint-provided mean/std normalization.
- Use `chunk_size=50`, `n_action_steps=50` during training, BF16, and
  gradient checkpointing.
- Default to one-process effective batch 16, `steps=250000`, and
  `save_freq=50000`. `BATCH_SIZE=8` is the fallback for an out-of-memory
  failure; it is an explicit relaunch, not an automatic mid-run change.
- Store artifacts under a new `/data/wengyikun/pi05_libero_frozen_language/`
  directory and refuse to overwrite an existing output directory.

## Validation

The launcher has a static contract test and Bash syntax check. Before a long
run, invoke it through the existing current-user, single-GPU LIBERO Docker
runner with a separate smoke output and a small step count. Verify the log
reports the requested freezing configuration and writes a reloadable
checkpoint. A full simulation evaluation must use `n_action_steps=10` and
`env.control_mode=relative`.
