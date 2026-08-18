# PI05 Precomputed Relative Chunk Design

## Goal

Create `/data/joint_songling/0704_bread_grasp_only_songling_robot_relative_chunk20` from the absolute two-camera dataset. The new dataset stores the relative state at each frame and the full 20-step relative action target required by PI05 training.

## Dataset Semantics

For every episode-local frame `t`, write one sample with the same image and timestamp as the source frame.

- State shape: `[14]`.
  - Arm dimensions `0:6` and `7:13`: `q_t - q_{t-1}`.
  - At the first frame of each episode, those 12 arm dimensions are zero.
  - Gripper dimensions `6` and `13`: absolute `q_t` width in metres, including the first frame.
- Action shape: `[20, 14]`.
  - For horizon `k = 1..20`, the arm target is `q_{t+k} - q_t`.
  - The gripper target is absolute `q_{t+k}` width.
  - If `t+k` is beyond the episode, use the final episode state. This keeps all samples and matches LeRobot endpoint padding semantics.

The source action column is not used to build chunks because it already represents `q_{t+1}`. Future targets are built directly from the source absolute state column, preventing a one-frame shift.

## Training And Inference

`PI05Config.precomputed_relative_chunk=true` selects the new format.

- Training reads the stored `[20,14]` action directly: no temporal action lookup and no second relative subtraction.
- Training reads the stored relative state directly: no prior-state lookup and no second state subtraction.
- Online inference still receives absolute robot state. The existing processor computes relative arm state from `q_t-q_{t-1}`, retains absolute grippers, caches absolute `q_t`, and reconstructs absolute arm action targets from the predicted relative chunk.
- The standard absolute-data path changes its temporal action indices from `1..chunk_size` to `0..chunk_size-1`, because source `action[t]` is already `q_{t+1}`.

## Storage And Validation

The output copies source videos byte-for-byte, rewrites data/meta parquet and JSON, and recomputes numeric state/action stats. It retains 101 episodes and 9,762 samples. Validation checks formulas at first, interior, and endpoint frames, video hashes, metadata shapes, PI05 sampling configuration, and a CPU smoke batch through the relative processor.
