# ACT with a Shared DINOv3 Vision Backbone

## Goal

Add an independent `act_dinov3` policy that replaces ACT's ResNet18 image encoder with one shared,
trainable DINOv3 ViT-L/16 while preserving ACT's existing state/action processing, relative-joint
semantics, action chunking, VAE objective, validation metrics, and inference behavior.

The implementation must not change the behavior, checkpoint schema, or launchers of `policy.type=act`.
No training is started as part of this design or its implementation.

## Weight Source

The initial DINOv3 weights are loaded from:

`/data/wengyikun/models/turbovla_joint_songling/dinov3-vitl16-pretrain-lvd1689m`

The verified model is DINOv3 ViT-L/16 with:

- 303,129,600 parameters
- 24 transformer blocks
- hidden size 1024
- patch size 16
- input size 224 by 224
- four register tokens
- FP32 source parameters

The path is only an initialization source. Saved `act_dinov3` checkpoints contain the complete
DINOv3 state dict and are self-contained for resume and inference.

## Isolation

Register a new policy type, `act_dinov3`, with its own configuration and model modules. Reuse ACT
processor construction rather than modifying its normalization and relative-joint processors.

The factory gains explicit `act_dinov3` branches. Existing `act` branches remain unchanged. The new
policy has a distinct class and checkpoint config, preventing an ACT checkpoint from being silently
loaded as ACT-DINOv3 or vice versa.

## Image Pipeline

Camera order is fixed by `input_features` and, for the current dual-arm dataset, is:

1. `observation.images.top`
2. `observation.images.gripper_left`
3. `observation.images.gripper_right`

Each image follows the existing ACT geometric preprocessing:

- top 405 by 720: pad vertically with black pixels to 720 by 720, then resize to 224 by 224
- wrist 480 by 640: apply the existing camera crop/resize rule to 224 by 224

The dataset processor already applies ImageNet mean/std normalization because visual normalization is
`MEAN_STD` and dataset ImageNet statistics are enabled. The DINOv3 adapter therefore accepts already
normalized tensors and must not normalize them a second time.

## Shared DINOv3 Adapter

All cameras use the same DINOv3 instance. For each camera tensor:

1. Run DINOv3 with `pixel_values=image`.
2. Read `last_hidden_state`.
3. Remove the CLS token and four register tokens.
4. Require exactly 196 remaining patch tokens.
5. Reshape `B x 196 x 1024` to `B x 1024 x 14 x 14`.
6. Return the spatial feature map to ACT.

The existing ACT image projection becomes a 1 by 1 convolution from 1024 to `dim_model=512`. ACT's
existing two-dimensional sinusoidal position embedding is applied to each 14 by 14 feature map.
Camera tokens are appended in the fixed camera order and may interact through the existing ACT
transformer encoder self-attention.

CLS and register tokens are intentionally excluded from the first implementation. Multi-layer feature
fusion and camera-specific DINOv3 copies are out of scope.

## Training And Dtype

DINOv3 participates fully in backpropagation. No DINOv3 parameters are frozen.

- source and master DINOv3 parameters: FP32
- DINOv3 forward compute: BF16 autocast on CUDA
- DINOv3 gradient checkpointing: enabled
- DINOv3 output: cast to the ACT projection input dtype when needed
- ACT mixed precision: follows the existing training launcher
- optimizer state: standard FP32 AdamW state

The shared DINOv3 module is executed once per camera with the same parameters. Gradients from all three
cameras accumulate into that one parameter set.

## Optimizer Groups

Use two explicit optimizer groups:

- DINOv3 parameters: learning rate `1e-6`
- ACT transformer, VAE, image projection, position embeddings, and action head: learning rate `1e-5`

Both groups use the configured AdamW weight decay, initially `1e-4`. Parameter grouping is based on an
exact DINOv3 module prefix and is tested for complete, non-overlapping coverage.

## ACT Semantics Preserved

The following behavior is inherited unchanged from ACT:

- state and action feature definitions
- relative arm-joint state/action conversion
- absolute indices such as endpoint XYZ and grippers
- separate state and action q01/q99 statistics
- quantile clipping configuration
- action chunk size and execution horizon
- state noise configuration
- VAE encoder and latent sampling
- reconstruction loss, KL loss, and `kl_weight`
- gripper loss and per-dimension validation metrics
- action unnormalization and conversion back to absolute commands

Replacing the visual encoder does not change targets or loss units.

## Configuration

`ACTDINOv3Config` extends ACT configuration with:

- `dinov3_pretrained_path`
- `dinov3_learning_rate=1e-6`
- `dinov3_gradient_checkpointing=true`
- `dinov3_autocast_dtype=bfloat16`
- `dinov3_num_register_tokens=4`
- `dinov3_patch_size=16`

Validation rejects missing local weights for a fresh model, non-224 visual input after policy
preprocessing, unsupported patch layouts, and incompatible pretrained ACT-DINOv3 checkpoints.

## Memory Strategy

ViT-L/16 is substantially larger than ResNet18. Initial validation uses:

- physical batch size 1
- gradient accumulation to obtain the required effective batch
- gradient checkpointing enabled
- three cameras processed sequentially through the shared encoder
- no model compilation

Batch size is increased only after a measured forward/backward memory test. The implementation must not
assume that current ACT batch sizes of 32 or 64 fit.

## Checkpoint Behavior

`save_pretrained` stores the complete ACT-DINOv3 state dict, including DINOv3, projection, ACT, VAE, and
action head parameters. `from_pretrained` reconstructs from checkpoint config and does not reload or
overwrite DINOv3 parameters from `dinov3_pretrained_path` when checkpoint model weights are present.

Fresh initialization requires the external DINOv3 path. Resume and inference require only the saved
ACT-DINOv3 checkpoint and its processors.

## Tests

Tests must cover:

- registration and factory construction for `act_dinov3`
- unchanged construction of `act`
- camera order preservation
- one shared DINOv3 instance for all cameras
- removal of one CLS and four register tokens
- exact `196 -> 14 x 14` reshape
- no second ImageNet normalization in the adapter
- output feature shape and dtype
- finite three-camera forward and backward passes using a small test double
- nonzero DINOv3 gradients
- gradient checkpointing enablement
- complete, disjoint optimizer parameter groups and their learning rates
- save/load equality without access to the original initialization path
- unchanged ACT loss and processor behavior for relative 20D state and 14D action

A real-weight smoke test may be provided separately, but it must not be run automatically on the active
training server without explicit approval.

## Success Criteria

The implementation is ready for a later smoke test when:

1. Existing ACT tests remain green.
2. ACT-DINOv3 unit and checkpoint tests pass.
3. A synthetic three-camera batch produces correctly ordered 14 by 14 patch maps.
4. Backward propagation produces finite, nonzero gradients in DINOv3 and ACT parameters.
5. No existing training process, output directory, or remote mounted source tree was modified.

