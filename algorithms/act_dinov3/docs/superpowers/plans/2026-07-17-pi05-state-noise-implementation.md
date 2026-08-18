# PI05 State Noise Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add state-only joint and pose noise augmentation to all four PI05 training variants.

**Architecture:** Reuse the existing joint noise step for both joint representations. Add a pose-specific processor that perturbs translation, SO(3) orientation, and gripper state before pose normalization. It is active only in training mode.

### Task 1: Pose State Noise

**Files:** `src/lerobot/processor/end_effector_pose_processor.py`, `tests/processor/test_end_effector_pose_processor.py`

1. Add failing tests proving training state changes but action does not, eval state does not change, and perturbed rot6d remains a valid rotation.
2. Implement `PoseStateNoiseProcessorStep` using xyz Gaussian noise, Gaussian axis-angle SO(3) noise, and gripper Gaussian noise.
3. Run `pytest tests/processor/test_end_effector_pose_processor.py -k pose_state_noise -v`.
4. Commit `feat: add PI05 pose state noise`.

### Task 2: Pipeline Wiring

**Files:** `src/lerobot/policies/pi05/configuration_pi05.py`, `src/lerobot/policies/pi05/processor_pi05.py`, related PI05 pipeline tests.

1. Add a failing test that configured pose pipelines contain the new noise step.
2. Add `state_position_noise_std_m`; validate all noise values are non-negative.
3. Insert joint noise after joint conversion and pose noise after pose conversion, before normalizers.
4. Run targeted processor and pipeline tests.
5. Commit `feat: apply PI05 state noise to all representations`.

### Task 3: Launch and Validate

**Files:** four PI05 0714 launchers.

1. Default and pass `state_noise_std_rad=0.003`, `state_position_noise_std_m=0.003`, and `gripper_noise_std_m=0.001`.
2. Run the PI05 processor, pose processor, and pipeline tests.
3. Commit and push to `main`.
4. Stop four pre-noise containers, deploy the pushed revision to a new remote directory, and restart all four from the PI05 base weights.
5. Verify each remote container has the intended noise parameters and completes one optimizer update.
