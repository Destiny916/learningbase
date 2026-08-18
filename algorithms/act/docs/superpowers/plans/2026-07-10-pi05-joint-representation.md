# pi05 Joint Representation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add selectable absolute and relative 14D Piper/Pika joint representations for pi05 training and inference.

**Architecture:** Keep a single `PI05Config` and pi05 policy type. Add a focused pi05 joint-representation processor that rewrites state/action semantics before normalization and reconstructs/clips actions after unnormalization. Override pi05 normalization stats with hardware limits so train and inference share the same scale.

**Tech Stack:** Python dataclasses, PyTorch tensors, LeRobot processor pipelines, pytest.

---

### Task 1: Joint Representation Unit Tests

**Files:**
- Create: `tests/policies/pi0_pi05/test_pi05_joint_representation.py`

- [ ] **Step 1: Write failing tests**

```python
def test_piper_pika_profile_marks_joint6_names_as_grippers():
    names = [f"left_joint_{i}" for i in range(7)] + [f"right_joint_{i}" for i in range(7)]
    assert build_arm_mask(14, gripper_indices=[6, 13], action_names=names).tolist() == [
        True, True, True, True, True, True, False,
        True, True, True, True, True, True, False,
    ]
```

Also test absolute pass-through, relative first-frame zero state, relative action chunk `q[t+k] - q[t]`, gripper absolute behavior, hardware min/max stats, and postprocess clipping.

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=src pytest tests/policies/pi0_pi05/test_pi05_joint_representation.py -q
```

Expected: fail because `lerobot.policies.pi05.joint_representation` does not exist.

### Task 2: Joint Representation Processor

**Files:**
- Create: `src/lerobot/policies/pi05/joint_representation.py`

- [ ] **Step 1: Implement the profile and tensor helpers**

Add:

```python
PIPER_PIKA_14D_ABSOLUTE_MIN = [...]
PIPER_PIKA_14D_ABSOLUTE_MAX = [...]
build_arm_mask(action_dim, gripper_indices, action_names=None)
make_pi05_joint_stats(mode, gripper_indices, profile)
```

For relative mode, arm dims use `[-(max-min), +(max-min)]`; gripper dims remain `[0.0, 0.10]`.

- [ ] **Step 2: Implement processor steps**

Add:

```python
Pi05JointRepresentationProcessorStep
Pi05AbsoluteActionProcessorStep
```

The preprocessor step rewrites `observation.state` and `action` according to the selected mode. The postprocessor step reconstructs relative arm actions to absolute commands using the cached current absolute state and clips to hardware limits.

- [ ] **Step 3: Run unit tests**

Run:

```bash
PYTHONPATH=src pytest tests/policies/pi0_pi05/test_pi05_joint_representation.py -q
```

Expected: pass.

### Task 3: PI05 Config And Pipeline Wiring

**Files:**
- Modify: `src/lerobot/policies/pi05/configuration_pi05.py`
- Modify: `src/lerobot/policies/pi05/processor_pi05.py`
- Modify: `src/lerobot/scripts/lerobot_train.py`

- [ ] **Step 1: Add config fields**

Add fields:

```python
joint_representation: str = "absolute"
joint_limit_profile: str = "piper_pika_14d"
joint_limit_path: str | None = None
joint_gripper_indices: list[int] = field(default_factory=lambda: [6, 13])
```

Validate `joint_representation in {"absolute", "relative"}`.

- [ ] **Step 2: Adjust delta indices**

For relative mode, expose previous/current observations via:

```python
observation_delta_indices == [-1, 0]
```

For both joint modes, action chunks should target `t1..tN`:

```python
action_delta_indices == list(range(1, self.chunk_size + 1))
```

- [ ] **Step 3: Wire processor pipeline**

In `make_pi05_pre_post_processors`, place the joint preprocessor before `NormalizerProcessorStep` and the joint postprocessor after `UnnormalizerProcessorStep`. Remove the old name-based relative step from pi05 for this mode.

- [ ] **Step 4: Support pretrained override reloads**

In `lerobot_train.py`, when `joint_representation == "relative"`, pass overrides for the new pi05 joint processor and absolute postprocessor.

### Task 4: Verification And Smoke

**Files:**
- No source edits expected.

- [ ] **Step 1: Run targeted tests**

Run:

```bash
PYTHONPATH=src pytest tests/policies/pi0_pi05/test_pi05_joint_representation.py tests/processor/test_pi05_processor.py -q
```

- [ ] **Step 2: Run a local dataset smoke check**

Run with writable cache:

```bash
HF_HOME=/tmp/joint_songling_hf_cache HF_DATASETS_CACHE=/tmp/joint_songling_hf_cache/datasets PYTHONPATH=src python - <<'PY'
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi05.configuration_pi05 import PI05Config

root = Path('/home/wengyikun/workplace/joint_songling/dataset/0704_video_224_crop_no_top_grasp_close20')
ds = LeRobotDataset('0704_video_224_crop_no_top_grasp_close20', root=root, delta_timestamps={
    'observation.state': [-1 / 31, 0],
    'action': [i / 31 for i in range(1, 11)],
})
cfg = PI05Config(joint_representation='relative', chunk_size=10, n_action_steps=10)
assert cfg.action_delta_indices == list(range(1, 11))
assert ds.meta.features['observation.state']['shape'] == [14]
assert ds.meta.features['action']['shape'] == [14]
assert sorted(ds.meta.camera_keys) == ['observation.images.left_wrist', 'observation.images.right_wrist']
sample = ds[0]
assert sample['action'].shape[-2:] == (10, 14)
print('dataset smoke ok')
PY
```

- [ ] **Step 3: Review and push**

Run:

```bash
git diff --stat
git diff --check
git status --short
git push chengdu main
```
