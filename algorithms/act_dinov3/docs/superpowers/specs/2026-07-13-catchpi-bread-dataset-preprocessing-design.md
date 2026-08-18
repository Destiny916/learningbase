# CatchPI Bread Dataset Preprocessing Design

## Goal

Create a new LeRobot v3 dataset at
`/data/joint_songling/0704_bread_grasp_only_songling_robot`
from `0704_video` while leaving the source dataset unchanged.

## Semantics

- Keep all 101 episodes.
- For each episode of length `N`, retain frames `0` through `floor(2N/3)-1`.
- Keep only `observation.images.left_wrist` and `observation.images.right_wrist`.
- Rename dimensions 6 and 13 in both state and action metadata from
  `left_joint_6`/`right_joint_6` to `left_gripper`/`right_gripper`.
- Preserve all retained state and action values, except set the final action in
  every trimmed episode equal to its final retained state.
- Rebuild parquet data, videos, episode metadata, and statistics through the
  LeRobot v3 writer.
- Preserve the source H.264 video codec for compatibility with the existing
  training environment.
- Rewrite the conversion summary and alignment manifests to describe only the
  retained frames and cameras.

## Safety And Validation

The conversion refuses to overwrite an existing output directory. Validation
checks 101 episodes, 9,762 frames, contiguous indices, exact per-episode
lengths, two video streams with 9,762 frames each, no top-camera metadata, the
two gripper names, and terminal action equality for every episode.
