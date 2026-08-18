# TurboVLA Patch-Vision T2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a separate 0806 TurboVLA run using a frozen DINOv3 ViT-L and a two-frame, three-view dense-patch visual context while preserving the ACT action decoder and batch size 16.

**Architecture:** The loader returns camera frames `[t-1, t]` for each of the three views. The vision path projects dense DINO patches, adds the existing view/patch positions plus a new time embedding, and flattens six image patch grids into the unchanged TurboVLA language/state/ACT path.

**Tech Stack:** Python, PyTorch, Transformers DINOv3, StarVLA LeRobot loader, Accelerate, pytest, Docker.

---

### Task 1: Add a temporal two-frame Joint Songling data recipe

**Files:**
- Modify: `experiments/joint_songling/data_registry/data_config.py`
- Modify: `third_party/starvla_runtime/starVLA/dataloader/gr00t_lerobot/datasets.py`
- Test: `tests/test_joint_songling_patchvision_t2_data.py`

- [ ] **Step 1: Write failing loader tests**

```python
def test_t2_recipe_requests_previous_and_current_frames():
    assert TemporalConfig.observation_indices == [-1, 0]

def test_t2_packing_keeps_time_then_camera_order():
    assert packed["image"][0] == [top_t_minus_1, left_t_minus_1, right_t_minus_1]
    assert packed["image"][1] == [top_t, left_t, right_t]
```

- [ ] **Step 2: Run the tests and observe failure**

Run: `python3 -m pytest tests/test_joint_songling_patchvision_t2_data.py -q`

- [ ] **Step 3: Implement the temporal recipe and generic image packing**

Create a distinct data configuration with `observation_indices = [-1, 0]` and
emit nested image lists only when more than one image timestep was requested.
For every timestep, apply the existing Joint Songling camera preprocessing in
the fixed camera order.

- [ ] **Step 4: Run the loader tests**

Run: `python3 -m pytest tests/test_joint_songling_patchvision_t2_data.py -q`

### Task 2: Add dense patch time-position support without changing ACT

**Files:**
- Modify: `turbovla/models/configuration.py`
- Modify: `turbovla/models/vision_encoder.py`
- Modify: `turbovla/models/turbovla.py`
- Modify: `third_party/starvla_runtime/starVLA/model/framework/VLM4A/TurboVLA.py`
- Test: `tests/test_turbovla_patchvision_t2.py`

- [ ] **Step 1: Write failing temporal-vision tests**

```python
def test_two_frame_three_view_tokens_have_time_positions():
    condition = model.encode_condition([task], {"dinov3": images})
    assert condition.shape[1] == 2 * 3 * 196 + text_tokens

def test_t2_keeps_dino_frozen_but_time_embedding_trainable():
    loss.backward()
    assert all(p.grad is None for p in model.vision_encoder.backbone.parameters())
    assert model.time_embedding.grad is not None
```

- [ ] **Step 2: Run and observe failure**

Run: `python3 -m pytest tests/test_turbovla_patchvision_t2.py -q`

- [ ] **Step 3: Implement `temporal_window_size`**

Add `temporal_window_size` with default one to `VisionEncoderConfig`.  Validate
input time length, flatten `[B,T,V]` only for DINO execution, restore the time
axis, add a learnable time embedding, and flatten time/view/patch only after
the existing visual projection and positions.  Extend the framework image
adapter to require exactly `T` time groups of `V` images.

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_turbovla_patchvision_t2.py -q`

### Task 3: Add an isolated batch-16 recipe and release-load guard

**Files:**
- Create: `experiments/joint_songling/configs/0806swap_patchvision_t2_3view.yaml`
- Create: `scripts/joint_songling/train_0806swap_patchvision_t2_3view.sh`
- Modify: `third_party/starvla_runtime/starVLA/training/trainer_utils/trainer_tools.py`
- Test: `tests/test_joint_songling_patchvision_t2_config.py`
- Test: `tests/test_checkpoint_partial_load.py`

- [ ] **Step 1: Write failing configuration and checkpoint tests**

```python
assert "temporal_window_size: 2" in config
assert "freeze_vision_encoder: true" in config
assert "gradient_checkpointing: false" in config
assert "per_device_batch_size: 16" in config
assert "joint_songling_0806swap_endpoint20_t2" in config
```

- [ ] **Step 2: Run and observe failure**

Run: `python3 -m pytest tests/test_joint_songling_patchvision_t2_config.py -q`

- [ ] **Step 3: Add the recipe**

Clone the known-good launcher without overwriting its run root.  Preserve the
fixed task text and independent overlay creation.  Add a checkpoint guard that
reports incompatible keys and only permits the explicitly named new time
embedding to be absent from an otherwise compatible release initialization.

- [ ] **Step 4: Run focused tests**

Run: `python3 -m pytest tests/test_joint_songling_patchvision_t2_config.py tests/test_checkpoint_partial_load.py -q`

### Task 4: Validate remotely and start the independent run

**Files:**
- No additional source files.

- [ ] **Step 1: Run the full relevant test suite in the target Docker image**

Run: `python3 -m pytest tests/test_joint_songling_patchvision_t2*.py tests/test_joint_songling_0806swap_config.py tests/test_dinov3_gradient_checkpointing.py -q`

- [ ] **Step 2: Run a one-step real-data smoke test at batch size 16**

Set a new `RUN_ROOT_DIR`, run the launcher with `max_train_steps=1`, and verify
one optimizer step, 20D state/14D action statistics, q01<=q99, frozen DINO, and
the `50x14` output.

- [ ] **Step 3: Start training**

Select one idle GPU without disturbing the active GPU6 run.  Launch with batch
size 16, 200,000 max steps, 10,000 warmup steps, cosine decay, and 20,000-step
checkpoints. Record the run root, host GPU, container, and first logged step.
