from pathlib import Path

from omegaconf import OmegaConf


def test_config_uses_relative_actions_three_views_and_release_checkpoint():
    config_path = Path("experiments/joint_songling/configs/relative_3view.yaml")
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False)

    assert config["is_debug"] is False
    assert config["framework"]["vision"]["num_views"] == 3
    assert config["datasets"]["vla_data"]["image_layout"] == "joint_songling"
    assert config["framework"]["text"]["freeze_text_encoder"] is True
    assert config["framework"]["action"]["action_dim"] == 14
    assert config["datasets"]["vla_data"]["state_mode"] == "delta"
    assert config["datasets"]["vla_data"]["state_mode_apply_keys"] == [
        "left_joints",
        "right_joints",
    ]
    assert config["datasets"]["vla_data"]["action_mode"] == "rel"
    assert config["datasets"]["vla_data"]["action_mode_apply_keys"] == [
        "left_joints",
        "right_joints",
    ]
    assert config["trainer"]["pretrained_checkpoint"] == "${oc.env:TURBOVLA_RELEASE_CKPT}"
    assert config["trainer"]["max_train_steps"] == 100000
    assert config["trainer"]["num_warmup_steps"] == 5000
    assert config["trainer"]["save_interval"] == 10000
    assert config["trainer"]["eval_interval"] == 10000
