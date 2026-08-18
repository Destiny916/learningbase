# TurboVLA PatchVision T2 Temporal Cache Design

## Goal

Provide the PatchVision T2 checkpoint with the latest two adjacent, synchronized
three-camera observations at every inference boundary. Image capture must remain
independent of model latency and full action-chunk execution.

The read-only dry-run remains permanently incapable of enabling Piper arms or
Pika grippers. A separate real client owns action execution so validation and
motion entry points cannot be confused.

## Checkpoint Contract

- Temporal window: two image time steps, ordered `[t-1, t]`.
- View order at each time step: `top`, `gripper_left`, `gripper_right`.
- Top geometry: `405x720` RGB, padded by 157 rows above and 158 rows below to
  `720x720`.
- Wrist geometry: `480x640` RGB, center-cropped at `x=[80:560]` to `480x480`.
- The server image processor performs the final resize and pixel normalization
  for the DINOv3 vision encoder.
- State is 20D and action output is `[50,14]`.

## Architecture

Each camera has an independent producer that continuously captures frames and
stores timestamped frames in a bounded ring buffer. A temporal synchronizer
forms complete three-view samples by selecting frames with the smallest time
skew. Complete samples are stored in a second bounded ring buffer.

Inference requests read the latest two complete samples. The request is rejected
unless their sample sequences are adjacent and their timestamps satisfy the
configured temporal interval. The inference loop never carries an observation
from the previous action chunk forward as the model's previous image step.

The image sampler and the actual-state sampler are independent. State continues
to use the two latest Piper/Pika feedback samples at the inference boundary.

## Data Structures

`TimestampedFrame` contains:

- camera name
- camera-local sequence
- monotonic timestamp
- RGB `uint8` image

`TemporalViewSample` contains:

- synchronized-sample sequence
- monotonic timestamp
- top, left, and right timestamped frames
- maximum inter-camera skew
- preprocessed top, left, and right images

`TemporalPair` contains two adjacent `TemporalViewSample` instances and exposes
their interval and maximum camera skew for validation and logging.

## Synchronization Rules

- Producers run continuously while hardware is connected.
- The synchronizer emits samples in increasing timestamp order.
- A complete sample is valid only when the maximum timestamp difference among
  its three views is at most 50 ms.
- A temporal pair is valid only when sample sequences are adjacent.
- The pair interval must be between 15 ms and 60 ms, targeting 30 Hz.
- Frames and samples are never silently duplicated to satisfy the temporal
  window.
- A timeout or invalid pair prevents inference and produces a diagnostic error.

## Inference Flow

1. Start read-only Piper/Pika feedback and all three camera producers.
2. Wait until the synchronized-sample buffer contains two valid adjacent samples.
3. Read the latest actual 20D feedback pair and construct normalized state.
4. Read the latest valid temporal image pair.
5. Send `[[top,left,right] at t-1, [top,left,right] at t]` to the server.
6. Receive normalized `[50,14]` output and decode it only for diagnostics.
7. In dry-run, print diagnostics and discard the decoded chunk.

## Real Execution

The real client uses the same temporal cache and state/action conversion as the
validated dry-run client. It requires all three explicit motion flags and exits
if any are missing. It executes every decoded 50-step chunk completely at 30 Hz
before requesting the next inference.

The action chunk is converted from normalized relative joint actions to absolute
joint targets by anchoring every row to the measured state at that chunk's
inference boundary. Gripper channels remain absolute widths. The real client
adds no deployment-specific step clipping; Piper/Pika SDK physical limits remain
authoritative.

Camera producers continue running during inference and chunk execution. After a
chunk completes, the next request reads the cache's latest valid adjacent pair,
not the pair used by the preceding chunk.

Hardware routing is fixed to left Piper `can1`, right Piper `can0`, left Pika on
the c4 stable serial path, and right Pika on the c6 stable serial path.

## Diagnostics

Each request logs:

- request/chunk index
- previous/current synchronized-sample sequences
- temporal interval in milliseconds
- maximum camera skew for both samples
- image shapes for both time steps
- actual-state feedback sequence pair
- server latency
- decoded gripper ranges
- explicit confirmation that no action was sent

## Testing

Unit tests use deterministic fake camera streams and verify:

- view order and image geometry
- selection of the latest adjacent sample pair
- rejection of non-adjacent sequences
- rejection of stale or excessive temporal intervals
- rejection of excessive inter-camera skew
- inference latency and chunk execution do not change the selected frame interval
- no hardware action method is called in dry-run
- the real client sends exactly 50 ordered actions per inferred chunk
- the next real-client request uses a newer adjacent temporal pair after a full
  chunk execution delay

Integration verification runs at least two dry-run inference requests. Both must
show adjacent image sequences and valid 30 Hz intervals even though server
inference takes longer than one camera period.

## Out Of Scope

- Hardware-synchronized camera triggering
- Changes to the training dataset or checkpoint
- Changes to existing normal TurboVLA clients
