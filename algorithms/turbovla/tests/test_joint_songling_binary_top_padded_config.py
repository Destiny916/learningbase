from pathlib import Path

import yaml


CONFIG_PATH = Path("experiments/joint_songling/configs/0806swap_binary_top_padded_3view.yaml")
LAUNCHER_PATH = Path("scripts/joint_songling/train_0806swap_binary_top_padded_gpu6.sh")


def test_binary_top_padded_config_preserves_requested_training_contract():
    config = yaml.safe_load(CONFIG_PATH.read_text())

    assert config["run_id"] == "turbovla_0806swap_binary_gripper_top_padded_3view_gpu6"
    assert config["framework"]["action"] == {
        "action_dim": 14,
        "state_dim": 20,
        "horizon": 50,
        "num_layers": 3,
        "dropout": 0.1,
        "num_state_tokens": 2,
        "state_hidden_dim": 256,
        "mlp_hidden_dim": 512,
        "loss_type": "l1",
    }
    data = config["datasets"]["vla_data"]
    assert data["image_layout"] == "joint_songling_top_padded"
    assert data["per_device_batch_size"] == 16
    assert data["state_mode"] == "delta"
    assert data["state_mode_apply_keys"] == ["left_joints", "right_joints"]
    assert data["action_mode"] == "rel"
    assert data["action_mode_apply_keys"] == ["left_joints", "right_joints"]
    trainer = config["trainer"]
    assert trainer["pretrained_checkpoint"] == "${oc.env:TURBOVLA_INITIAL_CKPT}"
    assert trainer["max_train_steps"] == 200000
    assert trainer["num_warmup_steps"] == 10000
    assert trainer["save_interval"] == 20000
    assert trainer["lr_scheduler_type"] == "cosine"
    assert trainer["gradient_accumulation_steps"] == 1
    assert set(trainer["learning_rate"].values()) == {5e-5}


def test_binary_top_padded_launcher_filters_data_and_maps_wrist_source_keys():
    script = LAUNCHER_PATH.read_text()

    assert 'CUDA_VISIBLE_DEVICES:-6' in script
    assert 'EXCLUDED_EPISODES = {16, 17}' in script
    assert 'allowed_episode_ids' in script
    assert '"relative_action_anchor": "current"' in script
    assert 'total_episodes"] = 71' in script
    assert 'total_frames"] = 49430' in script
    assert 'observation.images.wrist_left' in script
    assert 'observation.images.wrist_right' in script
    assert '"gripper_left": {"original_key": "observation.images.wrist_left"}' in script
    assert '"gripper_right": {"original_key": "observation.images.wrist_right"}' in script
    assert 'TURBOVLA_INITIAL_CKPT' in script
    assert 'PREPARE_ONLY' in script
    assert 'SMOKE_TEST' in script
    assert 'trainer["max_train_steps"] = 1' in script
    assert 'trainer["logging_frequency"] = 1' in script
    assert '[[ ! -e "$RUN_ROOT_DIR" ]]' in script
    assert (
        "Pick up the bread with the right gripper, transfer it to the left gripper, and place it in the bowl."
        in script
    )
