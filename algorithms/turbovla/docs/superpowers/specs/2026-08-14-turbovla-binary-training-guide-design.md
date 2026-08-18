# Generic TurboVLA Joint Songling Training Guide Design

## Goal

Update `/home/wengyikun/workplace/joint_songling/howtotrain/TurboVLA_0812训练说明.md` into a reusable training guide, while keeping the currently verified GPU1/GPU2 binary-gripper run as a copyable preset.

## Scope

- Store the final guide under `/home/wengyikun/workplace/joint_songling/howtotrain`.
- Put a generic variable-based workflow first. Parameterize the dataset root, overlay root, host GPU, training variant, initial checkpoint, run ID, container name, output root, and log path.
- Explain which values must change together when switching datasets, GPUs, or fresh/warm training.
- Keep shared verified defaults explicit: batch 16, workers 8, prefetch 2, persistent workers, PyAV single-thread decoding, 500000 steps, 25000 warmup, and 20000-step checkpoint interval.
- Add a complete current preset for `/data/wengyikun/datasets/joint_songling/0812_binary_gripper_without_ep173_174`: GPU1 fresh and GPU2 warm-start from retry8 non-EMA step 200000.
- Document the preset's dedicated top-padded overlay and corrected `binary_absolute_closed_zero` contract label.
- Keep a short migration note for the earlier `0812_closed_gripper_zero_without_ep173_174` workflow instead of retaining duplicate full instructions.

## Correctness Details

- State and action keep separate q01/q99 statistics: state 20D and action 14D.
- Joint state is relative to the previous frame; endpoint xyz and grippers stay absolute.
- Joint action chunks are relative to the current frame; grippers stay absolute binary physical values 0.0/0.1.
- Top images are padded from 405x720 to 720x720 with 157 rows above and 158 below, then resized to 224x224. Wrist images resize directly from 480x640 to 224x224.
- The guide must state that the remote mounted `video.py` sets `stream.codec_context.thread_count = 1`.
- Updating the overlay metadata label does not require retraining.
- The current launcher validates the 0812 dataset totals and shapes, so the generic guide must warn that materially different dataset metadata requires a matching launcher/config update rather than only changing an environment variable.

## Guide Structure

1. Verified model/data/image/state-action contract.
2. Generic environment-variable template.
3. Generic fresh and warm-start launch commands.
4. Current 0812 binary GPU1/GPU2 preset.
5. Monitoring, checkpoint inspection, stopping, and warm-start continuation.
6. Dataset-switch checklist and legacy-dataset note.

## Verification

- Search the guide for unintended hard-coded stale dataset, overlay, output, run ID, container, and `num_workers: 16` references.
- Verify the generic shell examples consistently use declared variables.
- Verify the binary preset uses the new dataset-specific paths and exact GPU1/GPU2 container names.
- Verify the documented parameters match the generated runtime `config.full.yaml` snapshots.
- Preserve the warning not to stop or modify GPU6/GPU7 and unrelated jobs.
