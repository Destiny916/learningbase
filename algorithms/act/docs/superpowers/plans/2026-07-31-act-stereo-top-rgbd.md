# ACT Stereo-Top RGB-D Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and train isolated ACT StereoPolicy-style top-stereo RGB and RGB-D variants from the specified 0729 episode windows.

**Architecture:** A standalone converter writes a six-camera LeRobot v3 copy with fixed top stereo pairs, RGB-D wrists, and the existing 14D relative state/action contract. ACT gains an opt-in visual encoder that combines frozen DINOv2 and shared ResNet18 top features through a two-layer Stereo Transformer, while an optional independent depth ResNet fuses each wrist RGB-D pair.

**Tech Stack:** Python, PyTorch, torchvision ResNet18, torch.hub DINOv2 ViT-S/14, LeRobot v3, PyAV/HEVC depth videos, pytest, Docker/Accelerate.

---

### Task 1: Add Isolated Stereo RGB-D Dataset Conversion

**Files:**
- Create: `src/lerobot/scripts/convert_dual_arm_stereo_rgbd_to_lerobot_v30.py`
- Create: `tests/scripts/test_convert_dual_arm_stereo_rgbd_to_lerobot_v30.py`

- [x] **Step 1: Write failing converter tests**

```python
def test_convert_uses_inclusive_window_and_resets_episode_indices(tmp_path):
    report = convert_dataset(raw_root, output_root, windows={0: (1, 3)}, fps=25)
    assert report.total_kept_frames == 3
    assert load_frame_indices(output_root, episode=0) == [0, 1, 2]
    assert load_actions(output_root, episode=0)[:-1] == load_states(output_root, episode=0)[1:]

def test_convert_stores_stereo_and_camera_specific_clipped_depth(tmp_path):
    convert_dataset(raw_root, output_root, windows={0: (0, 1)}, fps=25)
    assert set(load_visual_keys(output_root)) == {
        "observation.images.top_left", "observation.images.top_right",
        "observation.images.gripper_left", "observation.images.gripper_right",
        "observation.images.gripper_left_depth", "observation.images.gripper_right_depth",
    }
    assert read_depth_max(output_root, "observation.images.gripper_right_depth") <= 0.60
    assert read_depth_max(output_root, "observation.images.gripper_left_depth") <= 0.90
```

- [x] **Step 2: Run the tests and verify they fail because the converter module is absent**

Run: `pytest tests/scripts/test_convert_dual_arm_stereo_rgbd_to_lerobot_v30.py -v`

Expected: collection failure for `convert_dual_arm_stereo_rgbd_to_lerobot_v30`.

- [x] **Step 3: Implement the independent converter**

Implement `convert_dataset(input_root, output_root, windows, fps, max_alignment_delta_sec)` with:

```python
TOP_LEFT_PATH = Path("camera/color/stereoLeft")
TOP_RIGHT_PATH = Path("camera/color/stereoRight")
RIGHT_DEPTH_RANGE_M = (0.07, 0.60)
LEFT_DEPTH_RANGE_M = (0.07, 0.90)

def clip_depth_m(depth: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return np.clip(depth.astype(np.float32), lower, upper)
```

Use `stereoRight` as the anchor, match `stereoLeft`, both RGB wrist cameras, both depth cameras, and both joint streams within 10 ms. Apply the configured inclusive window after alignment, save all six camera keys, and use a global depth video encoding range `[0.07, 0.90]` after camera-specific clipping. Write a conversion manifest containing source timestamps, source aligned indices, requested windows, and discarded counts.

- [ ] **Step 4: Run converter tests and the actual 0729 conversion**

Run:

```bash
pytest tests/scripts/test_convert_dual_arm_stereo_rgbd_to_lerobot_v30.py -v
python -m lerobot.scripts.convert_dual_arm_stereo_rgbd_to_lerobot_v30 \
  --input-root=/data/joint_songling/0729 \
  --output-root=/data/joint_songling/0729_dualarm14d_stereo_top_rgbd_subtask_v30 \
  --fps=25 --max-alignment-delta-sec=0.01
```

Expected: all tests pass; 39 episodes and 5,547 frames; all six visual keys are present.

- [x] **Step 5: Commit and push the converter task**

```bash
git add src/lerobot/scripts/convert_dual_arm_stereo_rgbd_to_lerobot_v30.py \
  tests/scripts/test_convert_dual_arm_stereo_rgbd_to_lerobot_v30.py
git commit -m "feat(dataset): add isolated stereo top RGBD converter"
git push chengdu main
```

### Task 2: Implement Stereo Top and RGB-D ACT Visual Modules

**Files:**
- Create: `src/lerobot/policies/act/stereo_visual.py`
- Modify: `src/lerobot/policies/act/configuration_act.py`
- Modify: `src/lerobot/policies/act/modeling_act.py`
- Create: `tests/policies/act/test_stereo_visual.py`

- [x] **Step 1: Write failing visual-module tests**

```python
def test_stereo_top_fusion_keeps_resnet_grid_and_backpropagates():
    module = StereoTopFusion(dim=512, nheads=8, num_layers=2)
    left = torch.randn(2, 512, 7, 7, requires_grad=True)
    right = torch.randn(2, 512, 7, 7, requires_grad=True)
    fused = module(left, right)
    assert fused.shape == (2, 512, 7, 7)
    fused.square().mean().backward()
    assert left.grad is not None and right.grad is not None

def test_depth_range_normalization_uses_distinct_wrist_limits():
    right = normalize_depth_m(torch.tensor([0.07, 0.60, 0.90]), 0.07, 0.60)
    left = normalize_depth_m(torch.tensor([0.07, 0.60, 0.90]), 0.07, 0.90)
    torch.testing.assert_close(right, torch.tensor([0.0, 1.0, 1.0]))
    torch.testing.assert_close(left, torch.tensor([0.0, (0.60 - 0.07) / 0.83, 1.0]))
```

- [x] **Step 2: Run the tests and verify the missing-module failure**

Run: `pytest tests/policies/act/test_stereo_visual.py -v`

Expected: collection failure for `lerobot.policies.act.stereo_visual`.

- [x] **Step 3: Implement the visual modules**

Create `FrozenDinoV2` loading `dinov2_vits14` with `torch.hub.load`, set every parameter to `requires_grad=False`, call it in `torch.no_grad()`, and preserve its cached weights through `TORCH_HOME=/data/wengyikun/models/torch`. Implement a DINO projection and interpolation to the ResNet `(7, 7)` grid. Implement two-layer, eight-head symmetric Stereo Transformer blocks with self-attention, bidirectional cross-attention, 2D RoPE on cross-attention Q/K, residual connections, and MLPs. Implement wrist RGB-D fusion with an independent ResNet18 depth backbone after per-camera depth normalization and channel repetition.

Add `visual_mode: Literal["standard_rgb", "stereo_top_rgb", "stereo_top_rgbd"]`, camera-key configuration, and per-wrist depth limits to `ACTConfig`. In `ACT`, select the new visual path only for the stereo modes; leave `standard_rgb` byte-for-byte behaviorally equivalent. Ensure DINOv2 stays in eval mode when `policy.train()` is called.

- [ ] **Step 4: Run focused tests and ACT forward/backward smoke test**

Run:

```bash
pytest tests/policies/act/test_stereo_visual.py tests/policies/act -v
pytest tests/policies/pi0_pi05/test_pi05_joint_representation.py -v
```

Expected: all focused ACT tests pass, standard RGB regressions remain green, and DINO parameters have no gradients.

- [x] **Step 5: Commit and push visual-model task**

```bash
git add src/lerobot/policies/act/stereo_visual.py \
  src/lerobot/policies/act/configuration_act.py \
  src/lerobot/policies/act/modeling_act.py \
  tests/policies/act/test_stereo_visual.py
git commit -m "feat(act): add frozen DINO stereo top RGBD encoder"
git push chengdu main
```

### Task 3: Add Full-Dataset Relative Statistics and Launchers

**Files:**
- Create: `run_scripts/launch_act_stereo_top_rgb_0729_subtask.sh`
- Create: `run_scripts/launch_act_stereo_top_rgbd_0729_subtask.sh`
- Modify: `tests/training/test_relative_policy_launch_scripts.py`

- [ ] **Step 1: Write failing launcher assertions**

```python
def test_stereo_rgbd_launcher_uses_full_dataset_and_500k_schedule():
    text = launcher_path.read_text()
    assert "--steps=\"$STEPS\"" in text
    assert "SAVE_FREQ=\"${SAVE_FREQ:-50000}\"" in text
    assert "--eval_steps=0" in text
    assert "--policy.visual_mode=stereo_top_rgbd" in text
```

- [ ] **Step 2: Run launcher test and verify it fails**

Run: `pytest tests/training/test_relative_policy_launch_scripts.py -k stereo -v`

Expected: FAIL because the stereo launchers do not exist.

- [ ] **Step 3: Implement the two launchers**

Both launchers use the isolated dataset, all-data q01/q99 files, 14D relative action/state configuration, chunk 16, state noise `0.003 rad`, gripper noise `0.001 m`, 500,000 steps, `save_freq=50000`, `eval_steps=0`, batch 32 unless OOM, eight workers, and distinct output directories. The RGB launcher sets `stereo_top_rgb`; RGB-D sets `stereo_top_rgbd`. Both set `TORCH_HOME=/data/wengyikun/models/torch`.

- [ ] **Step 4: Verify q01/q99 and launcher parsing**

Run:

```bash
python -m lerobot.scripts.compute_full_dataset_relative_joint_stats \
  --dataset-root=/data/joint_songling/0729_dualarm14d_stereo_top_rgbd_subtask_v30 \
  --output-dir=/data/joint_songling/0729_dualarm14d_stereo_top_rgbd_subtask_v30/normalization \
  --horizons='[16]' --gripper-indices='[6,13]'
pytest tests/training/test_relative_policy_launch_scripts.py -k stereo -v
```

Expected: state q01/q99 length 14 with count 5,547; action q01/q99 length 14 with count 83,448; launcher tests pass.

- [ ] **Step 5: Commit and push launcher task**

```bash
git add run_scripts/launch_act_stereo_top_rgb_0729_subtask.sh \
  run_scripts/launch_act_stereo_top_rgbd_0729_subtask.sh \
  tests/training/test_relative_policy_launch_scripts.py
git commit -m "chore(train): add 500k stereo top ACT launchers"
git push chengdu main
```

### Task 4: Remote Build and Training Confirmation

**Files:**
- Create: `/data/wengyikun/datasets/joint_songling/0729_dualarm14d_stereo_top_rgbd_subtask_v30`
- Create: `/data/wengyikun/outputs/act_0729_stereo_top_rgb_chunk16_b32_500k/train_out`
- Create: `/data/wengyikun/outputs/act_0729_stereo_top_rgbd_chunk16_b32_500k/train_out`

- [ ] **Step 1: Synchronize only the new code and isolated dataset to the remote server**

Run `rsync -a` for the new converter output and `git`-tracked stereo files; do not overwrite remote existing datasets or outputs.

- [ ] **Step 2: Run single-GPU Docker smoke tests as host user**

Run each launcher with `STEPS=1`, `SAVE_FREQ=0`, `--user 1009:1008`, one selected idle GPU, and `TORCH_HOME=/data/wengyikun/models/torch`. Verify DINO downloads/caches only once, CUDA is visible, no DINO parameter has gradients, and a finite `train/loss` is logged.

- [ ] **Step 3: Start the two 500k jobs**

Run each task in a separate Docker container as user `1009:1008`, with one idle GPU per container:

```bash
STEPS=500000 SAVE_FREQ=50000 BATCH_SIZE=32 \
docker run -d --user 1009:1008 --gpus "device=<GPU>" --ipc=host ...
```

Do not stop unrelated jobs. If either visual mode OOMs at 32, relaunch only that new job with `BATCH_SIZE=16` and `gradient_accumulation_steps=2` to retain effective batch 32.

- [ ] **Step 4: Verify real training start**

For each container, inspect logs for all of:

```text
cfg.steps=500000
save_freq=50000
eval_steps=0
dataset.num_episodes=39
dataset.num_frames=5547
step=10 ... train/loss=<finite>
```

- [ ] **Step 5: Record training launch and update the goal**

Report container names, GPUs, output paths, first finite losses, and q01/q99 evidence. Keep the goal active until both jobs are confirmed training.
