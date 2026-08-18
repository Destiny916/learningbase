# PI052 full merge into Chengdu main

## Goal

Merge `origin/codex/pi05-pi052-training-time-rtc` at `12740f6b` into the current
`chengdu/main` at `e0bcc0e3`. The resulting branch must provide the complete
PI052 public API, including:

```python
from lerobot.policies.pi052.modeling_pi052 import PI052Policy
```

The merge also includes PI052 configuration, processors, tokenizer fitting,
runtime adapter, shared flow-matching utilities, factory/config registration,
dependency lock updates, and the source PI052 tests.

## Merge boundary

This is a three-way merge from common ancestor `e40b58a8`, not a directory copy.
The source branch is a descendant of `origin/codex/pi052-language-policy` and
adds the later training-time RTC fixes.

The source modifies framework-wide factory and processor behavior. Those changes
are required for PI052 checkpoint loading and processor restoration, so they
will be merged rather than replaced with ad-hoc PI052-only shims.

## Preservation requirements

The following current-project behavior must remain available after conflict
resolution and must be tested explicitly:

- PI05 14D left/right joint-plus-absolute-gripper representation.
- PI05 relative-state / relative-action and absolute-state / relative-action
  processor paths, including deserialization reconnect logic.
- PI05 visual initialization, frozen-language configuration, vision gradient
  checkpointing, finite-debug hooks, and vision dtype/autocast safeguards.
- Existing dual-arm PI05 and ACT launch scripts, datasets, converters, and
  custom dataset visualization utilities.
- Existing configured camera ordering and RGB/RGBD feature handling.

No checkpoint, dataset, training output, or runtime process is modified by this
code merge.

## Conflict-resolution policy

For each overlapping file, retain both independent behaviors when they operate
on different policy types. In particular, `policies/factory.py` must preserve
the source branch's convention-based PI052 discovery while retaining the local
PI05 processor reconnection special cases. PI05 model/config/processor conflicts
must retain current project fields and behavior unless the source implements a
strictly necessary framework compatibility correction.

Non-policy upstream changes are accepted only when they merge cleanly or are
necessary dependencies of the PI052 path. Any conflict outside the PI052,
factory, processor, shared-flow, configuration, CLI, or lockfile boundary is
resolved in favor of the current project unless a test demonstrates otherwise.

## Verification

1. Confirm clean merge and inspect every conflicted file.
2. Run import and registry checks for `PI052Policy` and `PI052Config`.
3. Run the source PI052 import/config/processor-focused tests that do not need
   model weights or a GPU.
4. Run PI05 import/config/processor regression checks, including the local
   14D relative/absolute processor paths.
5. Review the final diff against pre-merge `e0bcc0e3`, then push only after the
   requested verification succeeds.
