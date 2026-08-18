# ACT Stereo-Top RGB-D Design

## Goal

Evaluate a StereoPolicy-style fixed top stereo encoder for the dual-arm 14D
relative-joint ACT workflow, without modifying existing datasets, launchers,
or running training jobs.

## Scope

The source is `/data/joint_songling/0729`. Build a new LeRobot v3 dataset
from a copy of the source frames and use only these inclusive, zero-based,
post-alignment frame windows:

```text
episode0: 179-343   episode1: 163-317   episode2: 131-305
episode3: 132-262   episode4: 138-233   episode5: 161-343
episode6: 196-363   episode7: 166-330   episode8: 158-345
episode9: 167-352   episode10: 196-325  episode11: 161-342
episode12: 165-321  episode13: 162-306  episode14: 152-311
episode15: 170-287  episode16: 186-320  episode17: 150-278
episode18: 150-276  episode19: 157-327  episode20: 129-243
episode21: 146-283  episode22: 148-327  episode23: 163-292
episode24: 171-317  episode25: 209-368  episode26: 208-388
episode27: 152-302  episode28: 140-297  episode29: 146-267
episode30: 142-262  episode31: 114-215  episode32: 132-233
episode33: 144-231  episode34: 117-219  episode35: 131-211
episode36: 134-265  episode37: 117-276  episode38: 163-272
```

The windows contain 5,547 frames. Each output episode resets its
`frame_index` to zero. Actions are regenerated inside each window such that
`action[t] == state[t+1]`; the terminal action repeats the terminal state.

## Dataset

The new dataset root is:

```text
/data/joint_songling/0729_dualarm14d_stereo_top_rgbd_subtask_v30
```

It contains six visual keys:

```text
observation.images.top_left
observation.images.top_right
observation.images.gripper_left
observation.images.gripper_right
observation.images.gripper_left_depth
observation.images.gripper_right_depth
```

`top_left` and `top_right` come from the rigid external
`stereoLeft`/`stereoRight` cameras. `stereoRight` is the timing anchor, and
all six camera/state modalities must match within 10 ms. The original and all
existing LeRobot datasets remain unchanged.

Depth is stored in metres. Before encoding, left depth is clipped to
`[0.07, 0.90] m` and right depth to `[0.07, 0.60] m`. Both are encoded with a
common storage quantization range `[0.07, 0.90] m`; right frames are clipped
to `0.60 m` first. Runtime preprocessing applies the per-camera ranges again
and maps depth to `[0, 1]`.

No train/test split is made. Every episode is training data. State and
chunk-16 action q01/q99 statistics are separately computed from all 5,547
frames. State/action semantics remain the established dual-arm 14D relative
joint representation: 12 joints relative and grippers `[6, 13]` absolute.

## Models

Keep the existing ACT RGB and ACT RGB-D runs as controls. Add two new models:

| Name | Top input | Wrist input |
| --- | --- | --- |
| `ACT-StereoTop-RGB` | `top_left` + `top_right` | left/right RGB |
| `ACT-StereoTop-RGBD` | `top_left` + `top_right` | left/right RGB plus depth |

The external top stereo path includes a frozen `dinov2_vits14` feature
encoder, matching StereoPolicy's use of DINOv2 only for external views. The
ablation remains controlled because DINOv2 is included in both new
StereoTop-RGB and StereoTop-RGBD models, is never trainable, and is absent
from both existing non-stereo controls. Wrist views do not use DINOv2.

Both top images use the fixed existing geometry transform independently:

```text
405x720 -> vertical pad to 720x720 -> resize to 224x224
```

The RGB-D model uses the same top stereo path. Each wrist depth image is
converted from metres to `[0, 1]`, repeated to three channels, encoded by a
separate pretrained ResNet18, and fused with its matching RGB ResNet18 feature
map through a 1x1 projection. RGB and depth branches are distinct.

## Stereo ACT Encoder

The current ACT ResNet18 feature map is `(B, 512, 7, 7)`. For the top pair,
the RGB ResNet18 is shared between left and right images. Frozen DINOv2
`vits14` features are independently extracted from the same left/right
224x224 frames, spatially projected, resized to the ResNet token grid, and
concatenated with each ResNet feature map before a learned projection. A
two-layer, eight-head `StereoTopFusion` then performs, per layer:

1. independent left and right pixel-token self-attention;
2. left-to-right and right-to-left cross-attention;
3. 2D RoPE on cross-attention query/key projections;
4. residual MLP.

The final left/right tokens are concatenated per spatial location and
projected back to a fused `(B, 512, 7, 7)` top feature map. DINOv2 parameters
have `requires_grad=False` and run under `torch.no_grad()`; the ResNet,
ResNet/DINO projection, Stereo Transformer, wrist fusion, and ACT remain
trainable. The ACT encoder therefore receives exactly three camera feature
maps in both new variants:

```text
fused_top, fused_left_wrist, fused_right_wrist
```

The original ACT VAE, transformer, action chunk, 14D output head, relative
processor, loss, and inference action postprocessor are not changed.

## Training and Evaluation

Use the same full-dataset q01/q99, relative action chunk size 16, state noise,
batch/effective batch size, and training duration for the two new stereo runs.
Save all checkpoints. Do not configure an evaluation dataset.

Training loss and per-action training metrics can be logged, but there is no
held-out `valid/*` metric and no claim of offline generalization. The final
comparison is matched real-robot evaluation: same task and randomized initial
conditions, 20 trials per model, with grasp success, placement success, and
failure categories recorded.

## Isolation and Commits

Create a separate converter, model components, launchers, tests, output paths,
and remote dataset copy. Store the verified DINOv2 checkpoint under
`/data/wengyikun/models/`. Do not modify the existing three-camera converter
or existing outputs. Commit and push each completed task to the repository
main branch, staging only files created or changed for this feature.
