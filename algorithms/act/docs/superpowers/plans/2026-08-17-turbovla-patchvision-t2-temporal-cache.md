# TurboVLA PatchVision T2 Temporal Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Feed every PatchVision T2 inference request the latest two adjacent, synchronized three-camera samples while keeping dry-run incapable of sending hardware actions.

**Architecture:** A thread-safe temporal cache accepts timestamped frames from independent top, left, and right producer threads and emits synchronized three-view samples. The dry-run client reads the cache's latest valid adjacent pair at each inference boundary, while Piper/Pika state feedback remains on its existing independent sampler.

**Tech Stack:** Python 3.12, NumPy, threading/condition variables, pytest, pyrealsense2, PyAV/OpenCV, WebSocket TurboVLA runtime.

---

### Task 1: Pure Temporal Synchronization Cache

**Files:**
- Create: `temp/turbovla_patchvision_t2_180000_dryrun/temporal_image_cache.py`
- Create: `temp/turbovla_patchvision_t2_180000_dryrun/test_temporal_image_cache.py`

- [ ] **Step 1: Write failing tests for synchronized adjacent samples**

Create tests that add deterministic `uint8` frames for `top`, `gripper_left`, and `gripper_right`. Verify the cache returns sequences `(0,1)`, preserves view order, and reports an interval near 1/30 second.

```python
def test_latest_pair_uses_adjacent_synchronized_samples():
    cache = TemporalImageCache(max_camera_skew_s=0.05, min_pair_interval_s=0.015, max_pair_interval_s=0.060)
    add_triplet(cache, 1.000, marker=1)
    add_triplet(cache, 1.033, marker=2)
    pair = cache.latest_pair(timeout_s=0.01)
    assert (pair.previous.sequence, pair.current.sequence) == (0, 1)
    assert pair.interval_s == pytest.approx(0.033)
    assert [frame.camera for frame in pair.current.frames] == ["top", "gripper_left", "gripper_right"]
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
python3 -m pytest -q temp/turbovla_patchvision_t2_180000_dryrun/test_temporal_image_cache.py
```

Expected: collection fails because `temporal_image_cache` does not exist.

- [ ] **Step 3: Implement minimal timestamped frame, synchronized sample, pair, and cache types**

Implement:

```python
@dataclass(frozen=True)
class TimestampedFrame:
    camera: str
    sequence: int
    timestamp: float
    image: np.ndarray

@dataclass(frozen=True)
class TemporalViewSample:
    sequence: int
    timestamp: float
    frames: tuple[TimestampedFrame, TimestampedFrame, TimestampedFrame]
    max_camera_skew_s: float

@dataclass(frozen=True)
class TemporalPair:
    previous: TemporalViewSample
    current: TemporalViewSample
    interval_s: float
```

`TemporalImageCache.add_frame(camera, image, timestamp)` validates the camera
name, image type, and monotonic timestamp, then records the next camera-local
sequence. `TemporalImageCache.latest_pair(timeout_s)` waits on a condition and
returns the latest valid `TemporalPair`. The cache emits a sample only after all
three camera-local sequences have advanced and their latest timestamps fit the
skew limit.

- [ ] **Step 4: Add rejection tests**

Add tests proving the cache rejects excessive inter-camera skew, rejects pair intervals outside 15-60 ms, and never duplicates a frame to form a sample.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
python3 -m pytest -q temp/turbovla_patchvision_t2_180000_dryrun/test_temporal_image_cache.py
```

Expected: all tests pass.

Commit only the two Task 1 files.

### Task 2: Independent Camera Producers

**Files:**
- Modify: `temp/turbovla_patchvision_t2_180000_dryrun/pi052_reference_client.py`
- Modify: `temp/turbovla_patchvision_t2_180000_dryrun/temporal_image_cache.py`
- Modify: `temp/turbovla_patchvision_t2_180000_dryrun/test_temporal_image_cache.py`

- [ ] **Step 1: Write failing tests for producer freshness and shutdown**

Use deterministic blocking readers and verify each producer submits only newly returned frames, timestamps them with a monotonic clock, stops cleanly, and surfaces reader errors through `latest_pair()`.

```python
def test_camera_producer_adds_each_new_frame_once():
    reader = SequenceReader([frame(1), frame(2)])
    producer = CameraProducer("top", reader, cache, clock=SequenceClock([1.0, 1.033]))
    producer.start()
    producer.join(timeout_s=1.0)
    assert cache.camera_sequence("top") == 1
```

- [ ] **Step 2: Run tests and verify RED**

Expected: `CameraProducer` and producer error propagation are missing.

- [ ] **Step 3: Implement `CameraProducer` and top-camera new-frame access**

Add a producer thread that calls one camera reader exclusively and submits frames to the cache. Extend both top camera implementations with a condition-backed method:

```python
def read_right_after(self, sequence: int, timeout_s: float = 1.0) -> tuple[int, float, np.ndarray]:
    """Block until an actual camera frame newer than sequence is available."""
```

Increment the top sequence and record `time.monotonic()` only when the capture loop decodes a new frame. Preserve the existing `read_right()` API for normal TurboVLA clients.

- [ ] **Step 4: Extract state-only observation access**

Add `DualActHardware.get_state_observation()` containing only actual-state snapshot fields. Keep `get_observation()` behavior by extending the state-only result with camera images. This prevents the temporal client from competing with producer threads for D405 frames.

- [ ] **Step 5: Run tests and commit**

Run the temporal-cache tests and `python3 -m py_compile` on the modified runtime files. Commit only Task 2 files.

### Task 3: PatchVision T2 Dry-Run Client Integration

**Files:**
- Modify: `temp/turbovla_patchvision_t2_180000_dryrun/dual_turbovla_patchvision_t2_dryrun.py`
- Create: `temp/turbovla_patchvision_t2_180000_dryrun/test_patchvision_t2_dryrun.py`

- [ ] **Step 1: Write failing client-contract tests**

Test a helper that converts a `TemporalPair` into the exact nested model input:

```python
assert image_input == [
    [top_previous, left_previous, right_previous],
    [top_current, left_current, right_current],
]
```

Test that dry-run rejects all enable/action flags and that no `send_action()` call exists in the inference loop.

- [ ] **Step 2: Run tests and verify RED**

Expected: temporal-pair model-input helper and producer integration are missing.

- [ ] **Step 3: Integrate the cache and three producers**

Start one producer per camera after `hardware.connect()`. The top producer uses `read_right_after`; each D405 producer exclusively calls `read()`. At each request, call `hardware.get_state_observation()` and `cache.latest_pair()`, preprocess both samples, and send the nested two-time-step image list.

- [ ] **Step 4: Add diagnostics and strict dry-run teardown**

Print sample sequences, pair interval in milliseconds, both sample skews, image shapes, state sequences, server latency, decoded gripper ranges, and `DRY_RUN`. Stop producers before disconnecting hardware. Never call `hardware.send_action()`.

- [ ] **Step 5: Run all local tests and commit**

Run:

```bash
python3 -m pytest -q \
  temp/turbovla_patchvision_t2_180000_dryrun/test_temporal_image_cache.py \
  temp/turbovla_patchvision_t2_180000_dryrun/test_patchvision_t2_dryrun.py
python3 -m py_compile temp/turbovla_patchvision_t2_180000_dryrun/*.py
```

Expected: all tests pass and compilation exits zero.

### Task 4: Remote Two-Chunk Dry-Run Verification

**Files:**
- Copy updated runtime files to `/home/kw/runs/turbovla_patchvision_t2_180000_dryrun/` on `192.168.10.82`.
- Reuse server container `turbovla_patchvision_t2_180000_ema_dryrun_gpu3` on port `18067`.

- [ ] **Step 1: Verify no control processes and no enable flags**

Check robot processes, CAN state, and the exact client command. The command must omit `--enable-arms`, `--enable-grippers`, and `--execute-robot-actions`.

- [ ] **Step 2: Run two dry-run chunks**

Run the client with `--max-chunks 2`. Expected output for both requests includes adjacent image sample sequences, pair intervals from 15 to 60 ms, valid camera skews, and `DRY_RUN: action chunk was not sent to hardware`.

- [ ] **Step 3: Verify inference latency does not stretch image intervals**

Confirm the second request's temporal pair remains near 33 ms even when the first server inference takes hundreds of milliseconds or more.

- [ ] **Step 4: Verify teardown**

Confirm the dry-run process exits, both Pika serial devices and all cameras are released, no action-sending client remains, and the SSH tunnel is closed. Keep the GPU3 server loaded unless the user requests a stop.

- [ ] **Step 5: Report exact evidence**

Report checkpoint path, server port/GPU, both temporal sequence pairs, both intervals/skews, image shapes, state sequence pairs, latency, gripper predictions, and the explicit no-action evidence.

### Task 5: Separate Real Client, Deployment, And Push

**Files:**
- Create: `temp/turbovla_patchvision_t2_180000_dryrun/dual_turbovla_patchvision_t2_real.py`
- Create: `temp/turbovla_patchvision_t2_180000_dryrun/test_patchvision_t2_real.py`

- [ ] **Step 1: Write failing real-execution tests**

Test that real mode requires all three enable/action flags, sends exactly 50
ordered actions for a `[50,14]` chunk, and anchors every action row to the state
captured at that chunk's inference boundary.

- [ ] **Step 2: Implement the separate real client**

Reuse the validated temporal cache, camera producers, image conversion, state
normalization, and server protocol. Execute each full chunk at the configured
FPS, then request a new state observation and latest temporal pair.

- [ ] **Step 3: Run the complete local test suite**

Run all three focused test files, compile all runtime files, and verify the
dry-run source still contains no `hardware.send_action` call.

- [ ] **Step 4: Deploy and run the real client**

Verify no residual controller owns CAN or Pika devices, verify the GPU3 server
still serves the exact 180000 EMA checkpoint, establish the port 18067 tunnel,
and start the real client continuously with left `can1`, right `can0`, c4 left
Pika, and c6 right Pika.

- [ ] **Step 5: Commit and push**

Commit the real-client implementation and updated documentation, then push
`feature/patchvision-t2-temporal-cache` to the configured remote without
force-pushing.
