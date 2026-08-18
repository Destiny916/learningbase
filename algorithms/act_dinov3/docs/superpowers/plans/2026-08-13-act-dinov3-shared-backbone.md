# ACT-DINOv3 Shared Backbone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an isolated `act_dinov3` policy that uses one fully trainable DINOv3 ViT-L/16 for all ACT cameras without changing existing ACT behavior.

**Architecture:** A new config and policy subclass reuse ACT processing, VAE, transformer, decoder, and losses. A focused adapter converts DINOv3 patch tokens to `B x 1024 x 14 x 14`; the ACT model consumes those maps through its existing projection and positional encoding path. Factory registration is explicit and checkpoints include the complete shared DINOv3 module.

**Tech Stack:** Python, PyTorch, Hugging Face Transformers, LeRobot policy/processor factories, pytest, safetensors.

---

### Task 1: Register The Independent Configuration

**Files:**
- Create: `src/lerobot/policies/act_dinov3/__init__.py`
- Create: `src/lerobot/policies/act_dinov3/configuration_act_dinov3.py`
- Modify: `src/lerobot/policies/factory.py`
- Test: `tests/policies/act_dinov3/test_configuration_act_dinov3.py`

- [ ] Write a failing test that constructs `ACTDINOv3Config`, verifies `type == "act_dinov3"`, inherited ACT relative-joint fields, DINOv3 defaults, and validation of a missing initialization path for fresh construction.
- [ ] Run `pytest -q tests/policies/act_dinov3/test_configuration_act_dinov3.py` in the remote dependency container and verify import/registration failure.
- [ ] Implement `ACTDINOv3Config(ACTConfig)` with `@PreTrainedConfig.register_subclass("act_dinov3")` and fields for path, LR, checkpointing, autocast dtype, register-token count, and patch size.
- [ ] Add explicit factory config construction without modifying the existing ACT branch.
- [ ] Re-run the focused test and existing ACT processor tests.
- [ ] Commit `feat: register ACT DINOv3 configuration`.

### Task 2: Build The Shared Patch-Map Adapter

**Files:**
- Create: `src/lerobot/policies/act_dinov3/dinov3_backbone.py`
- Test: `tests/policies/act_dinov3/test_dinov3_backbone.py`

- [ ] Write a small DINOv3 test double whose output contains CLS, four register tokens, and 196 deterministic patch tokens.
- [ ] Write failing tests for one shared model instance, gradient-checkpointing enablement, exact special-token removal, `196 -> 14 x 14` reshape, input camera order, output dtype, and nonzero gradients.
- [ ] Run the tests and verify failure because the adapter does not exist.
- [ ] Implement `DINOv3SpatialBackbone` using injected models for unit tests and `AutoModel.from_pretrained(..., local_files_only=True)` for production.
- [ ] Keep normalization outside the adapter; accept already ImageNet-normalized tensors.
- [ ] Run focused tests and commit `feat: add shared DINOv3 spatial backbone`.

### Task 3: Integrate DINOv3 With ACT

**Files:**
- Create: `src/lerobot/policies/act_dinov3/modeling_act_dinov3.py`
- Modify: `src/lerobot/policies/factory.py`
- Test: `tests/policies/act_dinov3/test_modeling_act_dinov3.py`

- [ ] Write failing tests that construct the policy with a test adapter and verify three-camera order, output action shape, finite ACT loss, DINOv3 gradients, and two complete non-overlapping optimizer groups.
- [ ] Run the tests and verify the policy/factory branch is missing.
- [ ] Implement `ACTDINOv3Policy` and `ACTDINOv3` by reusing ACT policy methods and transformer modules while replacing only camera feature extraction.
- [ ] Use one 1 by 1 convolution from 1024 to `dim_model`; retain ACT 2D positional embeddings and token concatenation.
- [ ] Group DINOv3 parameters at `dinov3_learning_rate` and all remaining trainable parameters at the main optimizer LR.
- [ ] Add explicit `get_policy_class("act_dinov3")` and processor factory reuse through `make_act_pre_post_processors`.
- [ ] Run focused tests plus ACT processor/stereo tests and commit `feat: integrate DINOv3 with ACT`.

### Task 4: Verify Self-Contained Checkpoints And Regression Safety

**Files:**
- Test: `tests/policies/act_dinov3/test_act_dinov3_checkpoint.py`
- Modify only if required: `src/lerobot/policies/act_dinov3/modeling_act_dinov3.py`

- [ ] Write a failing save/load test using a tiny adapter-compatible model and remove access to its initialization path before loading.
- [ ] Verify the restored state dict and outputs match and that the checkpoint config remains `act_dinov3`.
- [ ] Make the minimal load-path adjustment needed so checkpoint weights are authoritative and no external DINOv3 reload overwrites them.
- [ ] Run all ACT-DINOv3 tests, existing ACT tests, factory tests, and `git diff --check`.
- [ ] Verify no launcher starts training and no remote active source directory is modified.
- [ ] Commit `test: verify ACT DINOv3 checkpoints and ACT regressions`.

### Task 5: Final Review

**Files:**
- Review all files changed on `feat/act-dinov3`.

- [ ] Compare implementation line by line with `docs/superpowers/specs/2026-08-13-act-dinov3-shared-backbone-design.md`.
- [ ] Run the complete focused test command in the remote container with fresh synced files.
- [ ] Inspect `git status`, `git diff --check`, and commit history.
- [ ] Confirm existing ACT training containers remain running and no ACT-DINOv3 training container exists.

