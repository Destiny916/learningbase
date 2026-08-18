# Popcorn Initial Training Stack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a versioned, non-training Popcorn workspace that pins ACT, ACT-DINOv3, TurboVLA, and TurboVLA-PatchVision sources and documents the existing 19D checkpoint contract.

**Architecture:** Keep upstream algorithms as Git submodules at known Chengdu commits, while storing Popcorn-specific documentation and validation in the parent repository. Snapshot only the small DexChain and training-guide files that are not independently versioned, excluding secrets and runtime state.

**Tech Stack:** Git submodules, Bash, Docker Compose static validation, Markdown.

---

### Task 1: Protect local data and document the repository

**Files:**
- Create: `.gitignore`
- Create: `README.md`
- Create: `docs/POPCORN_19D_CONTRACT.md`

- [ ] **Step 1:** Ignore local checkpoints, training state, datasets, outputs, virtual environments, caches, and `.env` files while allowing small checkpoint JSON metadata.
- [ ] **Step 2:** Document checkout with `git clone --recurse-submodules`, the four algorithms, and the explicit rule that this phase does not train.
- [ ] **Step 3:** Record the observed 19D absolute state/action contract, three camera keys, ACT horizon 100, and the boundary from Joint Songling's 20D/14D relative contract.

### Task 2: Pin Git-backed algorithm sources

**Files:**
- Create: `.gitmodules`
- Create: `algorithms/act/`
- Create: `algorithms/act_dinov3/`
- Create: `algorithms/turbovla/`
- Create: `algorithms/turbovla_patchvision/`

- [ ] **Step 1:** Add `chengdu/lerobot_joint.git` `main` as `algorithms/act`.
- [ ] **Step 2:** Add `chengdu/lerobot_joint.git` `feat/act-dinov3` as `algorithms/act_dinov3`.
- [ ] **Step 3:** Add `chengdu/turboVLA.git` `main` as `algorithms/turbovla` after publishing the relevant existing launcher fix.
- [ ] **Step 4:** Add `chengdu/turboVLA.git` `feature/turbovla-patchvision-t2-act` as `algorithms/turbovla_patchvision`.

### Task 3: Snapshot operational references

**Files:**
- Create: `dexchain/`
- Create: `howtotrain/`

- [ ] **Step 1:** Copy the DexChain Dockerfile, compose file, documentation, scripts, tests, `.dockerignore`, and `.env.example`; exclude `.env`.
- [ ] **Step 2:** Copy the ACT, TurboVLA, PatchVision, inference, and data-collection training notes from `joint_songling/howtotrain`.
- [ ] **Step 3:** Run DexChain's static test and `docker compose config --quiet` without starting or rebuilding a container.

### Task 4: Add a repository contract verifier using TDD

**Files:**
- Create: `tests/test_verify_stack.sh`
- Create: `scripts/verify_stack.sh`

- [ ] **Step 1:** Write a shell test that requires the verifier to confirm submodule paths, pinned Git links, checkpoint metadata, documentation, and absence of tracked secrets/weights.
- [ ] **Step 2:** Run `bash tests/test_verify_stack.sh` and confirm it fails because `scripts/verify_stack.sh` is absent.
- [ ] **Step 3:** Implement the minimal read-only verifier.
- [ ] **Step 4:** Run `bash tests/test_verify_stack.sh` and confirm it passes.

### Task 5: Verify, commit, merge, and publish

**Files:**
- Modify: plan checkboxes and Git metadata only.

- [ ] **Step 1:** Run `git diff --check`, the stack verifier, DexChain static validation, and shell syntax checks.
- [ ] **Step 2:** Confirm no `.env`, model weights, optimizer state, dataset, cache, or unrelated source changes are tracked.
- [ ] **Step 3:** Commit the focused initial stack on `feature/initial-training-stack`.
- [ ] **Step 4:** Merge the verified branch into `main` and push `main` plus the feature branch to `http://192.168.10.28:3000/chengdu/popcorn.git`.
