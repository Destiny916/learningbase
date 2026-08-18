from pathlib import Path

from omegaconf import OmegaConf


def test_endpoint20_config_preserves_full_state_and_requested_views():
    config = OmegaConf.to_container(
        OmegaConf.load(Path("experiments/joint_songling/configs/endpoint20_3view.yaml")),
        resolve=False,
    )

    assert config["framework"]["vision"]["num_views"] == 3
    assert config["framework"]["action"]["state_dim"] == 20
    assert config["framework"]["action"]["action_dim"] == 14
    assert config["datasets"]["vla_data"]["data_mix"] == "joint_songling_730_endpoint20"
    assert config["datasets"]["vla_data"]["state_mode_apply_keys"] == [
        "left_joints",
        "right_joints",
    ]
    assert config["datasets"]["vla_data"]["action_mode_apply_keys"] == [
        "left_joints",
        "right_joints",
    ]
