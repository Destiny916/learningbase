# ACT-DINOv3 W1 EE/FK Auxiliary Loss Design

## Goal

Add opt-in W1 bimanual end-effector auxiliary losses to
`algorithms/act_dinov3` while preserving the existing relative-joint ACT-DINOv3
training and inference contracts.

The feature must support the current Popcorn 19D workflow:

- left arm joints and right arm joints use relative state/action values;
- waist, neck, and grippers remain absolute;
- state and action use separate q01/q99 statistics;
- `clip_quantiles=true` remains valid for the normal ACT L1 path;
- FK targets use the original, unclipped absolute action labels;
- inference continues to return only a 19D action chunk.

## Non-Goals

- Do not alter the running ACT-DINOv3 training process.
- Do not change the default ACT-DINOv3 model structure or loss.
- Do not replace DINOv3, its normalization, or its gradient-checkpointing path.
- Do not add an end-effector control output to runtime inference.
- Do not silently support action layouts other than the exact W1 19D contract.

## W1 19D Contract

The required state/action order is:

```text
0      WAIST
1..7   LEFT_J1..LEFT_J7
8      NECK1
9      NECK2
10..16 RIGHT_J1..RIGHT_J7
17     LEFT_GRIPPER
18     RIGHT_GRIPPER
```

Relative dimensions are `1..7` and `10..16`. Absolute dimensions are
`0`, `8`, `9`, `17`, and `18`.

For an observation at time `t` and action-chunk offset `k`:

```text
model state arm dimensions  = q_t - q_(t-1)
model action arm dimensions = q_(t+k+1) - q_t
absolute dimensions         = their physical values at the corresponding time
```

The source dataset contract remains `action[t] == observation.state[t+1]` for
non-tail frames. Padded future action steps are excluded from every loss.

## URDF Contract

Use the W1 v024 BrainCo Revo1-R URDF:

```text
source:
w1_act-ljl-act_train/w1_simulation/urdf/
dexforce_w1_v024_brainco_revo1_r/robot_with_ee.urdf

ACT-DINOv3 asset destination:
algorithms/act_dinov3/assets/w1/v024/robot_with_ee.urdf

SHA256:
9eeaa3748253ffe43671bba9c5cea2569a83fc354c1f322cb4a575af65f6e987
```

The required kinematic chains are:

```text
reference: buttock
left:  WAIST + LEFT_J1..LEFT_J7 -> left_hand_base_link
right: WAIST + RIGHT_J1..RIGHT_J7 -> right_hand_base_link
```

Neck and gripper dimensions remain supervised by joint-space L1 but do not
enter either arm's FK chain.

## Configuration

Add the following fields to `ACTDINOv3Config`:

```text
ee_pose_loss_weight: float = 0.0
fk_loss_weight: float = 0.0
kinematics_urdf_path: str | None = None
kinematics_urdf_sha256: str | None = None
ee_reference_link: str = "buttock"
ee_left_link: str = "left_hand_base_link"
ee_right_link: str = "right_hand_base_link"
ee_position_scale_m: float = 0.1
ee_rotation_loss_weight: float = 0.25
```

Recommended initial experiment values are:

```text
ee_pose_loss_weight=0.05
fk_loss_weight=0.10
ee_position_scale_m=0.1
ee_rotation_loss_weight=0.25
```

Both auxiliary weights default to zero. Zero weights preserve the existing
model structure, preprocessor outputs, state-dict keys, forward return values,
loss, and inference behavior.

Configuration validation must reject:

- negative auxiliary weights;
- non-positive position scale;
- negative rotation weight;
- enabled auxiliary losses without a URDF path and SHA256;
- a SHA256 value that does not match the selected file;
- action/state names or dimensions that differ from the exact W1 19D contract.

Unlike the source W1 implementation, this design explicitly supports
quantile-normalized relative actions and must not require `MEAN_STD` action
normalization.

## Training-Only Absolute Data

When either auxiliary weight is positive, the relative-joint preprocessor must
preserve two tensors before relative conversion and quantile normalization:

```text
absolute_anchor_state:  [B, 19]
absolute_target_action: [B, chunk_size, 19]
```

`absolute_anchor_state` is the original current `state_t` frame.
`absolute_target_action` is the original absolute future action chunk from the
dataset. These tensors are moved to the policy device but are not normalized,
not passed into the ACT encoder/VAE encoder, and not exposed during inference.

The tensors are not created when auxiliary losses are disabled.

## Predicted Absolute Action Reconstruction

The predicted action used by FK is reconstructed as follows:

1. Start from the normalized predicted `[B, chunk_size, 19]` action.
2. Apply the action q01/q99 inverse transform without clipping.
3. Add `absolute_anchor_state` to dimensions `1..7` and `10..16`.
4. Leave dimensions `0`, `8`, `9`, `17`, and `18` unchanged.
5. Pass the resulting absolute 19D action to differentiable FK.

The true FK target is computed directly from `absolute_target_action`. It must
not be reconstructed from the clipped normalized label because quantile
clipping loses values outside q01/q99.

Target FK runs under `torch.no_grad()`. Predicted FK remains differentiable so
its gradients reach the action head, ACT decoder/encoder, and trainable DINOv3
backbone.

## Auxiliary Losses

All kinematics and pose-loss calculations run in FP32 with autocast disabled.

The FK branch computes:

```text
target absolute action    -> FK (no_grad)      -> target position/rotation
predicted absolute action -> differentiable FK -> predicted position/rotation
```

The EE-head branch adds an 18D projection to every ACT decoder token:

```text
2 arms * (3D position + 6D rotation representation) = 18D
```

The 6D rotation representation is converted to a 3x3 rotation matrix before
loss computation.

For EE-head and FK branches:

```text
position_loss = masked SmoothL1(pred_position / position_scale,
                                target_position / position_scale)
rotation_loss = masked mean((pred_rotation - target_rotation)^2)
pose_loss     = position_loss + rotation_weight * rotation_loss
```

The final loss is:

```text
total_loss =
    valid_only_normalized_relative_action_l1
    + kl_weight * kld_loss
    + ee_pose_loss_weight * ee_pose_loss
    + fk_loss_weight * fk_loss
```

The existing valid-only L1 denominator is retained. The source W1 behavior
that includes padded elements in the L1 denominator is not copied.

## Model and Checkpoint Compatibility

The EE head is created only when `ee_pose_loss_weight > 0`. FK-only training
does not add trainable parameters.

Loading rules:

- old checkpoint + both weights zero: strict loading remains unchanged;
- old checkpoint + EE weight positive: the new EE-head keys are expected to be
  missing and must be explicitly initialized and reported;
- EE-enabled checkpoint + EE weight positive: strict head loading is required;
- EE-enabled checkpoint + EE weight zero: unexpected EE-head keys must not be
  silently ignored by a normal strict load.

Inference does not run FK and does not return the EE-head prediction. The
postprocessor still converts the policy's relative arm action back to an
absolute 19D action chunk exactly as before.

## Observability

When auxiliary losses are enabled, console logs must expose:

```text
train/l1_loss
train/kld_loss
train/ee_pose_loss
train/ee_position_loss
train/ee_rotation_loss
train/fk_loss
train/fk_position_loss
train/fk_rotation_loss
train/auxiliary_loss
train/loss
```

The same scalar values may be forwarded to W&B when enabled. Logging must not
retain computation graphs or add a distributed synchronization per metric.

## Code Boundaries

The implementation is limited to:

- `ACTDINOv3Config` for opt-in parameters and validation;
- a W1-specific kinematics module under `policies/act_dinov3`;
- `ACTDINOv3Policy` for auxiliary-loss calculation;
- the ACT forward API only as needed to optionally expose decoder outputs;
- the relative-joint preprocessor only as needed to preserve training-only
  absolute tensors;
- the trainer's scalar logging path for auxiliary metrics;
- tests, the copied v024 URDF asset, and one separate smoke-test launcher.

No existing launcher is modified to enable the new loss.

## Error Handling

Training must fail before the first optimizer update when:

- the URDF file is missing or has the wrong hash;
- required reference/end-effector links are missing;
- the FK chain includes unsupported or unobserved actuated joints;
- action names, state names, or index sets do not match the W1 contract;
- required absolute training tensors are absent or have incorrect shapes;
- reconstructed or target absolute actions contain non-finite values;
- any auxiliary loss or auxiliary gradient becomes non-finite.

## Verification

Tests are written before production changes and must cover:

1. Disabled auxiliary losses preserve existing ACT-DINOv3 outputs, loss, and
   state-dict keys.
2. Config validation accepts the v024 contract and rejects invalid values.
3. URDF parsing produces the expected left and right actuated chains.
4. URDF SHA256 mismatches fail closed.
5. Relative predicted actions reconstruct the expected absolute 19D actions.
6. Raw absolute FK targets remain unchanged when normalized labels would be
   clipped by q01/q99.
7. Padded chunk positions contribute zero to EE and FK losses and denominators.
8. FK gradients reach predicted actions and upstream trainable parameters.
9. EE-head output has shape `[B, chunk_size, 2, 9]` and finite gradients.
10. Auxiliary metrics are finite and sum to the reported total loss.
11. EE-enabled checkpoints save and reload with the expected head keys.
12. The existing ACT-DINOv3 test suite remains green.

After unit tests pass, run a separate one-step GPU smoke test with a unique
output directory. The smoke test must verify the v024 hash, finite L1/KL/EE/FK
losses, finite gradients, and nonzero gradient flow through the action head and
DINOv3. It must not stop, reuse, overwrite, or modify the current formal
training container or output directory.

## Acceptance Criteria

The feature is complete when:

- both auxiliary weights zero reproduce the old behavior;
- the recommended EE+FK configuration completes a forward/backward optimizer
  step with finite component losses;
- predicted FK uses reconstructed absolute arm joints;
- target FK uses original unclipped absolute actions;
- the exact v024 URDF and hash are recorded in the checkpoint config;
- inference output remains a normal 19D action chunk;
- all focused and existing regression tests pass.
