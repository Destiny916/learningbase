# ACT Five-Camera RGBD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and train an ACT visual mode that consumes one top RGB, two wrist RGB, and two wrist depth inputs.

**Architecture:** Add a `five_camera_rgbd` configuration mode and a focused RGBD visual encoder. RGB uses one ResNet18; depth uses a separate ResNet18 after metre-domain clipping and one-to-three-channel expansion. The encoder returns five maps, preserving one ACT token group per source image.

**Tech Stack:** PyTorch, torchvision ResNet18, LeRobot ACT, pytest, Docker CUDA training.

---

### Task 1: Define and verify the visual contract

**Files:**
- Modify: `src/lerobot/policies/act/configuration_act.py`
- Modify: `tests/policies/test_act_stereo_visual.py`

- [ ] Write a failing test that accepts exactly top RGB, left/right RGB, and left/right depth for `five_camera_rgbd`.
- [ ] Run `pytest tests/policies/test_act_stereo_visual.py -k five_camera -v` and verify it fails because the mode is unsupported.
- [ ] Add `five_camera_rgbd` to the accepted modes and validate the exact five keys.
- [ ] Re-run the focused test and commit the configuration change.

### Task 2: Emit five visual token sources

**Files:**
- Modify: `src/lerobot/policies/act/stereo_visual.py`
- Modify: `src/lerobot/policies/act/modeling_act.py`
- Modify: `tests/policies/test_act_stereo_visual.py`

- [ ] Write a failing test asserting five input images produce five `(B,512,H,W)` maps.
- [ ] Run the test and verify it fails.
- [ ] Add a separate depth ResNet18. Clamp metre depth to a finite range, scale to `[0,1]`, expand to three channels at its encoder boundary, and preserve configured key order.
- [ ] Run all `test_act_stereo_visual.py` tests and commit.

### Task 3: Add and verify the training launcher

**Files:**
- Create: `run_scripts/launch_act_five_camera_rgbd_730_subtask.sh`
- Modify: `tests/training/test_relative_policy_launch_scripts.py`

- [ ] Write a failing launcher test requiring `five_camera_rgbd`, `depth_output_unit=m`, 500k steps, 50k save frequency, and `eval_steps=0`.
- [ ] Run the test and verify it fails because the launcher is absent.
- [ ] Add the launcher with the current 14D relative q01/q99, noise, and batch-32 settings.
- [ ] Run focused tests plus a one-step GPU2 Docker smoke test that prints five image feature keys and finite loss.
- [ ] Commit and push each completed task to `origin/main`.

### Task 4: Restart long training

**Files:** None.

- [ ] Synchronize committed code to `/home/wengyikun/lerobot`.
- [ ] Launch GPU2 as UID:GID `1009:1008` with a new RGBD output directory.
- [ ] Verify `cfg.steps=500000`, `save_freq=50000`, `eval_steps=0`, five image features, and finite training loss.
