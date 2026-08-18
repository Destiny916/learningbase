# PI052 Official Vision And Dual-Arm State Design

## Goal

Prepare two PI052 dual-arm training configurations without starting training:

- relative state: arm joints use `q_t - q_(t-1)` and grippers remain absolute;
- absolute state: arm joints and grippers use their absolute current values.

Both configurations use relative arm actions `q_(t+k) - q_t`, absolute grippers,
the same 99-episode dataset, and independent state/action q01/q99 statistics.

## Official vision boundary

Restore the PI05/PI052 vision implementation to the behavior in Hugging Face's
PI052 source branch `origin/codex/pi052-language-policy` at `f2c8867d`:

- the vision tower and multimodal projector parameters remain float32 when the
  rest of the policy uses bfloat16;
- image inputs are cast to float32 before the vision tower;
- remove the local `vision_encoder_dtype` option and the explicit nested
  `autocast(enabled=False)` override;
- PI052 enables Hugging Face layer-level SigLIP gradient checkpointing with
  `use_reentrant=False`;
- PI052 does not wrap the complete image embedding call in a second outer
  checkpoint.

This change restores official PI052 behavior. It does not change camera order,
image preprocessing, state/action semantics, or normalization statistics.

## PI052 data processing

PI052 must reuse the validated PI05 joint-representation step before recipe
rendering and FAST action tokenization. The processor order is:

1. rename and add batch dimension;
2. construct the requested 14D state representation and the shared relative
   action chunk while keeping gripper indices 6 and 13 absolute;
3. normalize state and action with their separate q01/q99-backed statistics;
4. render the PI052 recipe and tokenize the task text;
5. tokenize the already-normalized relative action for FAST supervision;
6. move the batch to the selected device.

The postprocessor reverses action normalization and reconstructs absolute joint
targets for inference. State is never reconstructed by the action postprocessor.

## Task-only language supervision

The dataset contains a task string but no subtask or memory annotations. Recipe
bindings marked `if_present` therefore skip high-level subtask targets and memory
updates. Training still receives:

- flow-matching action loss;
- FAST action-token loss;
- task text as conditioning context.

Text cross-entropy has no target on these samples and contributes zero. This
does not invalidate action training, but the resulting checkpoint is not trained
to generate reliable subtask or memory text.

## Verification boundary

This phase runs only static checks and CPU-capable unit tests. It must not start
Docker training containers, reserve GPUs, run a CUDA one-step test, or modify
remote training outputs. GPU validation and long training require a later
explicit instruction.
