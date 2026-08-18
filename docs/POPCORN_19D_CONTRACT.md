# Popcorn W1 19D contract

This document records the contract observed in `act_popcorn_45w/pretrained_model/config.json` and the W1 ACT kinematics code. It is a compatibility boundary for later implementation, not a claim that every synchronized algorithm already trains against this dataset.

## Model input

- `observation.state`: one current 19-element state vector.
- `observation.images.cam_high_left`: RGB image with recorded shape `3 x 360 x 640`.
- `observation.images.cam_hand_left`: RGB image with recorded shape `3 x 360 x 640`.
- `observation.images.cam_hand_right`: RGB image with recorded shape `3 x 360 x 640`.

The state is an absolute joint-position representation, not `q_t - q_(t-1)`. Its semantic order is:

| Indices | Fields |
| --- | --- |
| 0 | `WAIST` |
| 1-7 | `LEFT_J1` through `LEFT_J7` |
| 8-9 | `NECK1`, `NECK2` |
| 10-16 | `RIGHT_J1` through `RIGHT_J7` |
| 17 | `LEFT_GRIPPER` |
| 18 | `RIGHT_GRIPPER` |

The first 17 values are robot joint positions; the last two are scalar gripper positions. Exact units and valid ranges must be taken from the dataset metadata/runtime adapter before future training or deployment.

## Model output

- Feature: `action`, shape 19 per step.
- ACT chunk size: 100.
- ACT action steps: 100.
- Training target: a future sequence shaped `100 x 19` before batching.
- Semantics: absolute target joint/gripper positions in the same order as the state, not relative deltas.

State and action are normalized with dataset mean and standard deviation for the network. Postprocessing maps normalized predictions back to the physical-value space using the recorded statistics.

## Auxiliary end-effector supervision

The checkpoint enables end-effector-pose and differentiable forward-kinematics losses. FK converts 19D joint action targets or predictions into left/right end-effector poses relative to the configured `buttock` reference link. These pose tensors supervise training losses; they do not replace the three image inputs, the 19D state input, or the `100 x 19` action output.

The configured reference and target links are:

- Reference link: `buttock` (the robot torso/pelvis reference link name in the URDF).
- Left target link: `left_ee`.
- Right target link: `right_ee`.

## Compatibility warning

The synchronized Joint Songling recipes commonly use:

- 20D state: relative arm state plus endpoint position and grippers.
- 14D action: relative arm action plus absolute binary grippers.
- Camera names/order: `top`, `gripper_left`, `gripper_right`.
- TurboVLA action horizon: 50.

Those recipes must not be pointed at Popcorn data unchanged. Later design work must explicitly adapt field order, camera mapping, normalization, absolute action semantics, padding, horizon, and any FK/URDF assumptions.
