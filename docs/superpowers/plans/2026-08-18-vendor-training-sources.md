# Vendor Popcorn Training Sources Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace algorithm Git links with ordinary tracked source files so future Popcorn changes can be committed directly to the Popcorn repository.

**Architecture:** Copy each pinned upstream tree without its `.git` metadata, record its original URL/branch/commit in a manifest, and commit the existing W1 ACT source while excluding generated artifacts and weights. Update the repository verifier to reject gitlinks and require representative source files.

**Tech Stack:** Git, Bash, rsync, Markdown.

---

### Task 1: Define the vendored-source verification contract

- [ ] Extend `tests/test_verify_stack.sh` to require a vendored-source success marker.
- [ ] Run the test and confirm it fails against the submodule verifier.
- [ ] Update `scripts/verify_stack.sh` to reject mode `160000`, validate representative source files, and check `UPSTREAM_SOURCES.md`.

### Task 2: Convert four algorithm trees

- [ ] Remove the four gitlinks and top-level `.gitmodules`.
- [ ] Copy ACT, ACT-DINOv3, TurboVLA, and PatchVision at their pinned commits without Git metadata.
- [ ] Record each upstream URL, branch, and commit in `UPSTREAM_SOURCES.md`.

### Task 3: Add the existing W1 ACT source

- [ ] Replace the whole-directory ignore with targeted ignores for W1 checkpoints, logs, caches, and build outputs.
- [ ] Copy the W1 ACT source and URDF assets without generated artifacts.
- [ ] Document that the referenced `dexe_interfaces` submodule is absent from the source snapshot.

### Task 4: Verify and publish

- [ ] Run the repository verifier, shell syntax checks, DexChain static checks, `git diff --check`, and forbidden-file audit.
- [ ] Commit the vendored source, merge to `main`, rerun verification, and push to `chengdu/popcorn.git`.
