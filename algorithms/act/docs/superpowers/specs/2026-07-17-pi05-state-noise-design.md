# PI05 State Noise Design

## Goal

Apply state-only training augmentation to all four PI05 runs while retaining deterministic validation and inference behavior.

## Noise Contract

- Joint7D: add independent zero-mean Gaussian noise with standard deviation `0.003 rad` to arm joints and `0.001 m` to the absolute gripper state.
- Pose10D: add independent zero-mean Gaussian translation noise with standard deviation `0.003 m`; apply a small axis-angle SO(3) perturbation with each axis-angle component having standard deviation `0.003 rad`; add `0.001 m` absolute gripper noise.
- Noise is applied after relative-state conversion and before normalization.
- Noise affects only `observation.state` during training. Actions, validation, inference, and images remain unchanged.
- Image preprocessing remains deterministic: 640x480 source image, horizontal center crop to 480x480, then bilinear resize to 224x224.

## Storage Contract

Each complete PI05 checkpoint is approximately 14 GB. Four runs save five checkpoints each, requiring approximately 280 GB. The remote data volume had 473 GB free before restart, leaving approximately 193 GB headroom.
