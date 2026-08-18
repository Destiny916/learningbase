# Right End-Effector Pose Training Design

**Goal:** Build a separate LeRobot v3 dataset and relative-pose training path for ACT and PI0.5 from the three 0714 right-arm bread teleoperation captures, without changing existing joint-space datasets or their training semantics.

## Scope

- Inputs are the even-numbered source episodes from `normal`, `differentplace`, and `wrongplace`.
- Each source group is split independently with `seed=42`: 80 percent train and 20 percent test.
- The new dataset root is `/data/joint_songling/0714_gripper_bread_single_teleop_normal_differentplace_wrongplace_endpose_pose10d_right_fisheye_combined_v30_split_seed42`.
- `observation.state` and `action` are 10D absolute poses: `[x, y, z, rot6d(6), gripper]`.
- `action[t]` is the aligned absolute pose at `t+1`; the final action repeats the terminal pose.

## Source Synchronization

For every right-fisheye JPEG timestamp, conversion independently selects the nearest samples within 10 ms from:

- `arm/endPose/puppetRight` for `x`, `y`, `z`, `roll`, `pitch`, and `yaw`.
- `arm/jointState/puppetRight` for `position[6]`, the absolute Pika gripper width.

Euler orientation uses the verified Piper convention:

```text
R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
rot6d = [R[:, 0], R[:, 1]]
```

`rot6d -> R` orthonormalizes its two input columns before composition. This prevents a model prediction from being used as a non-rigid transform at inference.

## Relative Representation

Let `P_t = [T_t, g_t]` be an absolute pose and gripper width.

```text
relative state at t:
  [pose10d(inv(T_(t-1)) @ T_t), g_t]
  first frame arm pose = identity; g_0 remains absolute

relative chunk target at t, horizon k >= 1:
  [pose10d(inv(T_t) @ T_(t+k)), g_(t+k)]
```

Training data still stores absolute pose10d. The new processor makes the relative representation after temporal sampling so padding and the first valid observation are handled correctly. Image-only mode zeros the fully normalized 10D state only after real state and target labels have been constructed.

## Normalization And Validation

- Train-only q01/q99 statistics are calculated independently for relative state and every requested action horizon.
- Quantile scaling applies to relative translation `xyz` and absolute gripper only. `rot6d` is already bounded geometry and is passed through; applying component-wise quantiles would corrupt the rotation representation.
- State noise, when enabled, applies only during training and only to relative translation and gripper. Image-only state remains exactly zero.
- At inference and for physical validation, prediction and target are unnormalized, relative transforms are projected to SO(3), and action transforms are composed with the true current absolute anchor.
- This workflow reports the existing normalized loss plus gripper loss and gripper physical MSE/RMSE. It deliberately does not use component-wise pose MSE as a selection metric.

## Isolation

New converter, stats, processors, tests, launch scripts, data roots, and output roots use `end_effector_pose` / `pose10d` names. Existing 7D relative-joint behavior remains unchanged.
