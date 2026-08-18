from pathlib import Path


def test_0806swap_state_contract_keeps_xyz_and_grippers_absolute():
    registry = Path("experiments/joint_songling/data_registry/data_config.py").read_text()

    assert "class JointSonglingSwapEndpoint20DataConfig" in registry
    assert '"state.left_endpoint"' in registry
    assert '"state.left_gripper"' in registry
    assert '"state.right_endpoint"' in registry
    assert '"state.right_gripper"' in registry
    assert '"joint_songling_0806swap_endpoint20"' in registry


def test_0806swap_config_uses_50_step_relative_joint_targets():
    config = Path("experiments/joint_songling/configs/0806swap_3view.yaml").read_text()

    assert "action_dim: 14" in config
    assert "state_dim: 20" in config
    assert "horizon: 50" in config
    assert "gradient_checkpointing: true" in config
    assert "data_mix: joint_songling_0806swap_endpoint20" in config
    assert "state_mode: delta" in config
    assert "state_mode_apply_keys: [left_joints, right_joints]" in config
    assert "action_mode: rel" in config
    assert "action_mode_apply_keys: [left_joints, right_joints]" in config
    assert "per_device_batch_size: 16" in config
    assert "max_train_steps: 200000" in config
    assert "num_warmup_steps: 10000" in config
    assert "save_interval: 20000" in config
    assert "lr_scheduler_type: cosine" in config
    assert "scheduler_specific_kwargs: {}" in config
    assert "gradient_accumulation_steps: 1" in config


def test_0806swap_launcher_uses_gpu6_right_eye_and_fixed_task_text():
    script = Path("scripts/joint_songling/train_0806swap_3view_gpu6.sh").read_text()

    assert 'CUDA_VISIBLE_DEVICES:-6' in script
    assert "observation.images.top" in script
    assert "observation.images.gripper_left" in script
    assert "observation.images.gripper_right" in script
    assert "observation.images.top_left" not in script
    assert "observation.images.top_right" not in script
    assert '-e WANDB_MODE="${WANDB_MODE:-disabled}"' in script
    assert '-e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"' in script
    assert (
        "Pick up the bread with the right gripper, transfer it to the left gripper, and place it in the bowl."
        in script
    )


def test_0806swap_stage2_config_uses_reduced_lr_and_extended_schedule():
    config = Path("experiments/joint_songling/configs/0806swap_3view_stage2.yaml").read_text()

    assert "pretrained_checkpoint: ${oc.env:TURBOVLA_STAGE1_CKPT}" in config
    assert "action_dim: 14" in config
    assert "state_dim: 20" in config
    assert "horizon: 50" in config
    assert "per_device_batch_size: 16" in config
    assert "max_train_steps: 100000" in config
    assert "num_warmup_steps: 2000" in config
    assert "save_interval: 20000" in config
    assert config.count("5.0e-05") == 0
    assert config.count("1.0e-05") == 6
    assert "lr_scheduler_type: cosine" in config
    assert "gradient_accumulation_steps: 1" in config


def test_0806swap_stage2_launcher_isolated_gpu6_and_preserves_data_contract():
    script = Path("scripts/joint_songling/train_0806swap_3view_stage2_gpu6.sh").read_text()

    assert 'CUDA_VISIBLE_DEVICES:-6' in script
    assert "TURBOVLA_STAGE1_CKPT" in script
    assert '[[ ! -e "$RUN_ROOT_DIR" ]]' in script
    assert "experiments/joint_songling/configs/0806swap_3view_stage2.yaml" in script
    assert "observation.images.top" in script
    assert "observation.images.gripper_left" in script
    assert "observation.images.gripper_right" in script
    assert (
        "Pick up the bread with the right gripper, transfer it to the left gripper, and place it in the bowl."
        in script
    )
