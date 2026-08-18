# PI052 Official Vision And Dual-Arm State Implementation Plan

**Goal:** Restore official PI052 vision behavior and prepare tested relative-state and absolute-state 14D processors and launchers without starting training.

### Task 1: Lock official vision behavior with tests

- Update PI052 vision checkpoint tests to assert the official layer-level API.
- Update PI05 vision tests to assert hardcoded float32 vision input and remove the local dtype option.
- Run the tests first and record the expected failures against the current local implementation.

### Task 2: Restore official vision implementation

- Remove `vision_encoder_dtype` from PI05 configuration and model construction.
- Restore official float32 vision parameter/input handling without the local autocast override.
- Restore the class-flag-driven checkpoint implementation used by the official PI052 branch.
- Remove obsolete launcher arguments and update launcher contract tests.

### Task 3: Extend custom joint processing to PI052

- Add focused tests for PI052 relative-state/relative-action and absolute-state/relative-action pipelines.
- Merge custom joint q01/q99 stats for both `pi05` and `pi052` training configs.
- Compose joint representation, recipe rendering, text tokenization, FAST action tokenization, and the matching action postprocessor.

### Task 4: Add two non-running PI052 launchers

- Use official `/data/wengyikun/openpi/lerobot_pi052_base` weights.
- Use batch size 8, three real cameras, no validation, no noise, and no clipping.
- Configure 100k steps, 5k warmup, 100k decay, and 10k checkpoint frequency.
- Keep output directories new and refuse overwrite.

### Task 5: Verify without training

- Run formatting/static checks and targeted CPU tests in an environment with project dependencies.
- Inspect launcher commands and processor step order.
- Do not execute either launcher and do not touch remote GPU processes.
