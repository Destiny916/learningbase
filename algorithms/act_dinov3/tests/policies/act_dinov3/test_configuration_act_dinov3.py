from pathlib import Path

import pytest


def test_act_dinov3_config_registers_independently_from_act(tmp_path: Path):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act_dinov3.configuration_act_dinov3 import ACTDINOv3Config

    weights = tmp_path / "dinov3"
    weights.mkdir()
    config = ACTDINOv3Config(dinov3_pretrained_path=str(weights))

    assert config.type == "act_dinov3"
    assert config.chunk_size == 100
    assert config.pretrained_backbone_weights is None
    assert config.dinov3_learning_rate == 1e-6
    assert config.dinov3_gradient_checkpointing is True
    assert config.dinov3_autocast_dtype == "bfloat16"
    assert config.dinov3_num_register_tokens == 4
    assert config.dinov3_patch_size == 16
    assert PreTrainedConfig.get_choice_class("act_dinov3") is ACTDINOv3Config


def test_factory_constructs_act_dinov3_config(tmp_path: Path):
    from lerobot.policies.act_dinov3.configuration_act_dinov3 import ACTDINOv3Config
    from lerobot.policies.factory import make_policy_config

    weights = tmp_path / "dinov3"
    weights.mkdir()

    config = make_policy_config("act_dinov3", dinov3_pretrained_path=str(weights))

    assert isinstance(config, ACTDINOv3Config)


def test_act_dinov3_config_requires_an_initialization_path():
    from lerobot.policies.act_dinov3.configuration_act_dinov3 import ACTDINOv3Config

    with pytest.raises(ValueError, match="DINOv3 initialization path"):
        ACTDINOv3Config(dinov3_pretrained_path="")
