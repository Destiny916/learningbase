# TurboVLA 0806 Swap Stage-2 Training Design

## Goal

Continue optimization from the completed retry8 non-EMA checkpoint without overwriting the first-stage run.

## Training contract

- Initial weights: `retry8/.../checkpoints/steps_200000_model.safetensors`.
- Dataset: `/data/wengyikun/datasets/joint_songling/0806swap_gripper_fixed_pi052_task_en`.
- Preserve the existing 20D state, 14D action, 50-step action horizon, q01/q99 normalization, relative arm joints, absolute endpoint xyz and grippers, task text, and three-view image preprocessing.
- Train on GPU6 with batch size 16 for 100,000 additional optimizer steps.
- Use AdamW, cosine decay, learning rate `1e-5`, 2,000 warmup steps, EMA 0.999, and save every 20,000 steps.
- Store outputs in a new stage-2 run root. The retry8 directory remains read-only training input.

## Implementation

Create a stage-2 YAML derived from the validated first-stage YAML and a dedicated GPU6 launcher derived from the validated first-stage launcher. The launcher requires an explicit stage-1 checkpoint, refuses an existing output directory, constructs a fresh metadata overlay, and starts one isolated GPU process.

## Verification

Add a configuration test that checks the checkpoint environment variable and all changed stage-2 hyperparameters while ensuring the original dataset and representation contract remains intact. Before launch, verify the source checkpoint and dataset exist. After launch, require a live container, GPU6 memory allocation, generated dataset statistics with 20D state and 14D action q01/q99 arrays, and at least one reported optimizer step without traceback or CUDA OOM.
