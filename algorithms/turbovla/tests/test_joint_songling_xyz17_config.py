from pathlib import Path

from omegaconf import OmegaConf


def test_xyz17_config_preserves_state_xyz_and_three_requested_views():
    config = OmegaConf.to_container(
        OmegaConf.load(Path("experiments/joint_songling/configs/xyz17_3view.yaml")),
        resolve=False,
    )

    assert config["framework"]["vision"]["num_views"] == 3
    assert config["framework"]["action"]["state_dim"] == 17
    assert config["framework"]["action"]["action_dim"] == 14
    assert config["datasets"]["vla_data"]["data_mix"] == "joint_songling_xyz17_full37"
    assert config["datasets"]["vla_data"]["state_mode_apply_keys"] == [
        "left_joints",
        "right_joints",
    ]
    assert config["datasets"]["vla_data"]["action_mode_apply_keys"] == [
        "left_joints",
        "right_joints",
    ]


def test_xyz17_config_uses_fresh_checkpoint_and_single_gpu_training_defaults():
    config = OmegaConf.to_container(
        OmegaConf.load(Path("experiments/joint_songling/configs/xyz17_3view.yaml")),
        resolve=False,
    )
    assert config["trainer"]["pretrained_checkpoint"] == "${oc.env:TURBOVLA_RELEASE_CKPT}"
    assert config["trainer"]["max_train_steps"] == 100000

