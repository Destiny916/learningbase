# Right-Arm Fisheye LeRobot v3 Conversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Download the two requested robot-capture roots, convert only their right 7D joint stream and right fisheye RGB frames into separate LeRobot v3 video datasets, and produce a verified combined dataset.

**Architecture:** A new converter will treat sorted `*.jpg` timestamps from `camera/color/pikaGripperFisheyeCamera_r` as the sampling clock. It will select the nearest sorted `*.json` sample in `arm/jointState/puppetRight`, reject camera frames whose absolute timestamp residual exceeds 10 ms, and write `[joint_0..joint_5, gripper]` state/action vectors through LeRobot's native dataset writer. The action is the next retained state and the terminal action repeats the terminal state. A separate combine step will use the repository's dataset tools so video and metadata stay consistent.

**Tech Stack:** Python 3.12, NumPy, Pillow, LeRobotDataset v3 writer, PyAV/H.264, pytest, rsync over SSH.

---

## File Structure

- Create: `src/lerobot/scripts/convert_right_arm_fisheye_to_lerobot_v30.py` -- timestamp discovery, nearest-neighbor alignment, v3 writing, and CLI.
- Create: `tests/scripts/test_convert_right_arm_fisheye_to_lerobot_v30.py` -- unit and small end-to-end conversion tests using synthetic timestamped JSON/JPEG data.
- Create: `docs/superpowers/plans/2026-07-14-right-arm-fisheye-v30-conversion.md` -- this implementation plan.
- Create at runtime: `/data/joint_songling/0714_gripper_bread_single_teleop_normal` and `/data/joint_songling/0714_gripper_bread_single_teleop_differentplace` -- immutable local raw copies of the two remote capture roots.
- Create at runtime: `/home/wengyikun/workplace/joint_songling/dataset/0714_gripper_bread_single_teleop_normal_right_fisheye_v30`, `/home/wengyikun/workplace/joint_songling/dataset/0714_gripper_bread_single_teleop_differentplace_right_fisheye_v30`, and a clearly named `_combined_v30` root -- derived LeRobot datasets.

### Task 1: Define and Test Timestamp Alignment

**Files:**
- Create: `tests/scripts/test_convert_right_arm_fisheye_to_lerobot_v30.py`
- Create: `src/lerobot/scripts/convert_right_arm_fisheye_to_lerobot_v30.py`

- [ ] **Step 1: Write failing alignment tests**

```python
def test_align_camera_frames_uses_nearest_joint_and_discards_large_residuals(tmp_path):
    joint_dir = tmp_path / "arm/jointState/puppetRight"
    camera_dir = tmp_path / "camera/color/pikaGripperFisheyeCamera_r"
    _write_joint(joint_dir / "1.000.json", [1, 2, 3, 4, 5, 6, 7])
    _write_joint(joint_dir / "1.030.json", [8, 9, 10, 11, 12, 13, 14])
    _write_rgb(camera_dir / "1.002.jpg")
    _write_rgb(camera_dir / "1.047.jpg")
    aligned = align_right_arm_to_camera(joint_dir, camera_dir, max_alignment_delta_sec=0.01)
    assert [item.camera_path.name for item in aligned] == ["1.002.jpg"]
    assert aligned[0].joint_path.name == "1.000.json"
    assert aligned[0].alignment_delta_sec == pytest.approx(0.002)


def test_build_state_and_next_action_clamps_terminal_frame():
    states = np.asarray([[1] * 7, [2] * 7], dtype=np.float32)
    actions = build_next_actions(states)
    np.testing.assert_array_equal(actions, np.asarray([[2] * 7, [2] * 7], dtype=np.float32))
```

- [ ] **Step 2: Verify tests fail because conversion functions do not exist**

Run: `uv run pytest tests/scripts/test_convert_right_arm_fisheye_to_lerobot_v30.py -q`

Expected: import failure for `align_right_arm_to_camera` and `build_next_actions`.

- [ ] **Step 3: Implement the minimal alignment domain API**

```python
@dataclass(frozen=True)
class AlignedFrame:
    camera_path: Path
    joint_path: Path
    camera_timestamp: float
    joint_timestamp: float

    @property
    def alignment_delta_sec(self) -> float:
        return abs(self.joint_timestamp - self.camera_timestamp)


def build_next_actions(states: np.ndarray) -> np.ndarray:
    if states.ndim != 2 or states.shape[1] != 7 or len(states) == 0:
        raise ValueError("states must have shape [N, 7] with N > 0")
    return np.concatenate([states[1:], states[-1:]], axis=0)
```

`align_right_arm_to_camera` must accept only numeric-stem `*.json` and `*.jpg` files, use sorted timestamps plus `np.searchsorted` for nearest selection, and preserve chronological camera ordering.

- [ ] **Step 4: Run alignment tests**

Run: `uv run pytest tests/scripts/test_convert_right_arm_fisheye_to_lerobot_v30.py -q`

Expected: alignment and terminal-action tests pass.

### Task 2: Write a Small v3 Dataset With H.264 Video

**Files:**
- Modify: `src/lerobot/scripts/convert_right_arm_fisheye_to_lerobot_v30.py`
- Modify: `tests/scripts/test_convert_right_arm_fisheye_to_lerobot_v30.py`

- [ ] **Step 1: Write a failing end-to-end test**

```python
def test_convert_episode_writes_video_and_q_t_to_q_t_plus_one(tmp_path):
    source = _make_episode(tmp_path / "source", timestamps=(1.0, 1.033, 1.066))
    output = tmp_path / "output"
    report = convert_episode(source, output, repo_id="local/test", fps=30, encoder_threads=1)
    dataset = LeRobotDataset("local/test", root=output)
    assert report.kept_frames == 3
    assert dataset.num_frames == 3
    assert (output / "videos").exists()
    np.testing.assert_array_equal(dataset[0]["observation.state"], np.arange(7, dtype=np.float32))
    np.testing.assert_array_equal(dataset[0]["action"], np.arange(7, dtype=np.float32) + 10)
    np.testing.assert_array_equal(dataset[2]["action"], np.arange(7, dtype=np.float32) + 20)
```

- [ ] **Step 2: Verify the end-to-end test fails**

Run: `uv run pytest tests/scripts/test_convert_right_arm_fisheye_to_lerobot_v30.py::test_convert_episode_writes_video_and_q_t_to_q_t_plus_one -q`

Expected: failure because `convert_episode` does not yet create a dataset.

- [ ] **Step 3: Implement conversion and report persistence**

Create dataset features exactly as:

```python
FEATURES = {
    "observation.images.right_fisheye": {"dtype": "video", "shape": (3, 480, 640), "names": None},
    "observation.state": {"dtype": "float32", "shape": (7,), "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"]},
    "action": {"dtype": "float32", "shape": (7,), "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"]},
}
```

Use `LeRobotDataset.create(..., use_videos=True, vcodec="h264", image_writer_threads=4, encoder_threads=encoder_threads, streaming_encoding=True, streaming_drop_frames=False)`. Add each RGB image, its aligned state, and precomputed next action in order, then call `save_episode(task=task)` once. Save a JSON conversion report containing raw counts, retained count, discarded counts, residual maximum/mean/p95, first/last timestamps, and the action semantics.

- [ ] **Step 4: Run the conversion test**

Run: `uv run pytest tests/scripts/test_convert_right_arm_fisheye_to_lerobot_v30.py -q`

Expected: all conversion tests pass, with a readable H.264 video.

### Task 3: Download Raw Sources Without Changing the Robot Host

**Files:**
- Runtime only: `/data/joint_songling/0714_gripper_bread_single_teleop_normal`
- Runtime only: `/data/joint_songling/0714_gripper_bread_single_teleop_differentplace`

- [ ] **Step 1: Inspect remote episode counts and source sizes**

Run a read-only SSH command against `kw@192.168.10.82` to list direct `episode*` directories and count only `puppetRight/*.json` plus `pikaGripperFisheyeCamera_r/*.jpg` for each source root.

- [ ] **Step 2: Download both roots to local raw storage**

Run two `rsync -a --partial --info=progress2` transfers from the remote roots to `/data/joint_songling/`. Authenticate interactively and do not delete remote files.

- [ ] **Step 3: Verify local raw copies**

Run `find`-based count and SHA-256 manifest checks for the required right-arm/right-camera files; confirm local episode directory names and counts match the remote inventory.

### Task 4: Convert the Individual Datasets

**Files:**
- Runtime: individual output roots under `/home/wengyikun/workplace/joint_songling/dataset/`

- [ ] **Step 1: Convert normal**

Run:

```bash
uv run python -m lerobot.scripts.convert_right_arm_fisheye_to_lerobot_v30 \
  --input-root /data/joint_songling/0714_gripper_bread_single_teleop_normal \
  --output-root /home/wengyikun/workplace/joint_songling/dataset/0714_gripper_bread_single_teleop_normal_right_fisheye_v30 \
  --repo-id local/0714_gripper_bread_single_teleop_normal_right_fisheye_v30 \
  --max-alignment-delta-sec 0.01 --fps 30 --vcodec h264 --encoder-threads 4
```

- [ ] **Step 2: Assert episode0 normal evidence**

Check its report shows 248 raw images, 247 retained frames, one discarded first image, and maximum retained residual at most 10 ms.

- [ ] **Step 3: Convert differentplace**

Run the same command with its `differentplace` input/output/repository identifiers. Preserve every valid episode and report per-episode skipped-frame counts.

### Task 5: Create and Verify the Combined Dataset

**Files:**
- Runtime: `/home/wengyikun/workplace/joint_songling/dataset/0714_gripper_bread_single_teleop_right_fisheye_combined_v30`

- [ ] **Step 1: Write a failing combine test**

```python
def test_combined_dataset_preserves_episode_order_and_frame_count(tmp_path):
    combined = combine_local_datasets([dataset_a, dataset_b], output_root=tmp_path / "combined", repo_id="local/combined")
    assert combined.meta.total_episodes == 2
    assert combined.num_frames == dataset_a.num_frames + dataset_b.num_frames
```

- [ ] **Step 2: Implement combine using the existing dataset-tools API**

Use the repository's metadata-aware dataset merge/copy helper; never concatenate Parquet rows or video files by hand. Reject inputs whose fps or feature schema differ.

- [ ] **Step 3: Run individual and combined validation**

Run a verifier that opens each `LeRobotDataset`, decodes the first/middle/last frame of every episode, checks numeric state/action equality against raw aligned inputs, checks terminal `action == state`, verifies all residuals are `<= 0.01`, and asserts combined episode/frame totals equal the sum of the individual datasets.

- [ ] **Step 4: Run focused tests and report artifacts**

Run: `uv run pytest tests/scripts/test_convert_right_arm_fisheye_to_lerobot_v30.py -q`

Expected: all new tests pass. Report raw roots, individual dataset roots, combined dataset root, retained/discarded counts, and video verification results.
