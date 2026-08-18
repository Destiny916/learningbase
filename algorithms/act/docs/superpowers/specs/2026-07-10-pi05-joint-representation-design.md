# pi05 Joint Representation Design

Date: 2026-07-10

## Goal

Train the local LeRobot pi05 policy on
`/home/wengyikun/workplace/joint_songling/dataset/0704_video_224_crop_no_top_grasp_close20`
with two selectable joint representations:

- `absolute`: current absolute joint/gripper state in, future absolute joint/gripper targets out.
- `relative`: current arm-joint delta state in, future arm-joint deltas out, while gripper stays absolute.

The selection must be controlled by pi05 configuration parameters so training and future inference use the same behavior.

## Dataset Facts

The dataset is LeRobot v3.0 with:

- `observation.images.left_wrist` and `observation.images.right_wrist`, both 224x224 video.
- `observation.state.shape == [14]`.
- `action.shape == [14]`.
- one task: `catchpi dual arm grasp`.

Important naming rule:

```text
left_joint_0..left_joint_5   = left Piper arm joints
left_joint_6                 = left Pika gripper width
right_joint_0..right_joint_5 = right Piper arm joints
right_joint_6                = right Pika gripper width
```

So the gripper dimensions are indices `6` and `13`, even though the dataset names them `joint_6`.

## Hardware Limits

The Piper arm limits come from the real-robot URDF and SDK docs under:

- `/home/kw/workspace/piper_ros/src/piper_description/urdf/piper_description.xacro`
- `/home/kw/workspace/piper_sdk/piper_sdk/piper_msgs/msg_v2/transmit/arm_motor_angle_limit_max_spd_config.py`

Per arm:

```text
joint_0: [-2.618,   2.618]
joint_1: [ 0.0,     3.14]
joint_2: [-2.967,   0.0]
joint_3: [-1.745,   1.745]
joint_4: [-1.22,    1.22]
joint_5: [-2.0944,  2.0944]
gripper: [ 0.0,     0.10] meters
```

The gripper limit intentionally uses `0.10m` instead of the Pika SDK's typical `0.09m` so the current dataset's observed values around `0.096m` remain inside the training range.

For 14D bimanual state/action:

```text
absolute_min = [
  -2.618, 0.0, -2.967, -1.745, -1.22, -2.0944, 0.0,
  -2.618, 0.0, -2.967, -1.745, -1.22, -2.0944, 0.0,
]

absolute_max = [
   2.618, 3.14, 0.0, 1.745, 1.22, 2.0944, 0.10,
   2.618, 3.14, 0.0, 1.745, 1.22, 2.0944, 0.10,
]
```

For relative arm-joint values, use the symmetric range `[-joint_range, +joint_range]`, where
`joint_range = absolute_max - absolute_min`. Gripper values stay absolute in both modes.

## New Configuration

Add pi05 config fields:

```text
joint_representation: "absolute" | "relative" = "absolute"
joint_chunk_size: int = 10
joint_limit_profile: "piper_pika_14d" = "piper_pika_14d"
joint_limit_path: str | None = None
joint_gripper_indices: list[int] = [6, 13]
```

`joint_chunk_size` should drive both `chunk_size` and `n_action_steps` for this workflow. The final CLI should allow:

```bash
--policy.joint_representation=absolute --policy.chunk_size=10 --policy.n_action_steps=10
--policy.joint_representation=relative --policy.chunk_size=10 --policy.n_action_steps=10
```

The implementation may keep `chunk_size` and `n_action_steps` as the actual LeRobot fields, but the design requires all examples and launch scripts for this dataset to set them to 10 explicitly.

## Absolute Mode

Training input:

```text
observation.state = q_t absolute, including gripper absolute
images = left_wrist, right_wrist
```

Training target:

```text
action[k] = q_{t+k} absolute, k = 1..10
```

Normalization:

- Arm joints use the hardware `absolute_min` and `absolute_max`.
- Grippers use `[0.0, 0.10]`.
- Images remain `IDENTITY`, following pi05 defaults.

Inference:

```text
cmd[k] = model_action[k]
cmd[k] = clip(cmd[k], absolute_min, absolute_max)
```

## Relative Mode

Training input:

```text
arm_state_t = q_t - q_{t-1}
gripper_state_t = g_t absolute
images = left_wrist, right_wrist
```

For the first frame of each episode:

```text
arm_state_t = 0
gripper_state_t = g_t absolute
```

The first frame remains trainable.

Training target:

```text
arm_action[k] = q_{t+k} - q_t, k = 1..10
gripper_action[k] = g_{t+k} absolute
```

This is relative to the current action-chunk start state `q_t`, not accumulated step-by-step deltas.

Normalization:

- Arm state deltas and arm action deltas use `[-joint_range, +joint_range]`.
- Gripper state and gripper action use `[0.0, 0.10]`.
- Gripper indices are excluded from relative subtraction.

Inference:

```text
arm_cmd[k] = current_arm_q + predicted_arm_delta[k]
gripper_cmd[k] = predicted_gripper_abs[k]
cmd[k] = clip([arm_cmd[k], gripper_cmd[k]], absolute_min, absolute_max)
```

If inference starts without a previous frame, the relative state arm dimensions are zero, matching training's first-frame behavior.

## Implementation Shape

Use LeRobot's existing pi05 structure rather than creating a separate policy type.

Main pieces:

- Extend `PI05Config` with the representation and limit parameters.
- Reuse the existing relative action processor behavior for `action -= state`, but ensure gripper dims `[6, 13]` remain absolute.
- Add a pi05 joint representation processor for observation state:
  - `absolute`: pass through.
  - `relative`: replace arm dims with `q_t - q_{t-1}`, keep gripper dims absolute.
- Add hardware-limit based min-max stats for both modes.
- Ensure saved preprocessors/postprocessors carry enough config for inference.

The implementation must avoid decoding or depending on top-camera data. Only the two wrist image features are part of this workflow.

## Testing

Add tests before implementation.

Required behavior tests:

- The gripper mask treats indices `6` and `13` as absolute even though their names are `left_joint_6` and `right_joint_6`.
- Absolute mode leaves state/action values in absolute representation.
- Relative mode converts only arm dims in `observation.state` to `q_t - q_{t-1}` and sets first-frame arm deltas to zero.
- Relative mode converts only arm dims in action chunks to `q_{t+k} - q_t`; gripper chunk values stay absolute.
- Hardware-limit normalization maps absolute arm/gripper limits to the expected model range.
- Relative inference postprocessing reconstructs absolute arm commands and clips every dimension to hardware limits.

Smoke checks:

- Instantiate the local dataset with `HF_HOME`/`HF_DATASETS_CACHE` pointed to a writable location.
- Load one batch in absolute mode and one batch in relative mode.
- Confirm images are only left/right wrist, state/action shape stays 14, and action horizon is 10.

## Out Of Scope

- Changing robot execution safety gates.
- Running real-robot motion.
- Adding top-camera support.
- Rewriting the pi05 model architecture.
- Training from scratch during implementation.

