import shutil

import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_STATE


def _tiny_dinov3():
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.for_model(
        "dinov3_vit",
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        image_size=224,
        patch_size=16,
        num_register_tokens=4,
    )
    return AutoModel.from_config(config)


def _config(weights):
    from lerobot.policies.act_dinov3.configuration_act_dinov3 import ACTDINOv3Config

    return ACTDINOv3Config(
        dinov3_pretrained_path=str(weights),
        chunk_size=2,
        n_action_steps=2,
        dim_model=16,
        n_heads=4,
        dim_feedforward=32,
        n_encoder_layers=1,
        n_decoder_layers=1,
        use_vae=False,
        latent_dim=4,
        input_features={
            "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
    )


def test_checkpoint_loads_without_original_dinov3_initialization_directory(tmp_path):
    from lerobot.policies.act_dinov3.modeling_act_dinov3 import ACTDINOv3Policy

    weights = tmp_path / "initial_weights"
    weights.mkdir()
    policy = ACTDINOv3Policy(_config(weights), dinov3_model=_tiny_dinov3())
    policy.eval()
    batch = {
        "observation.images.top": torch.randn(1, 3, 224, 224),
        OBS_STATE: torch.randn(1, 4),
    }
    with torch.no_grad():
        expected = policy.predict_action_chunk(batch)

    checkpoint = tmp_path / "checkpoint"
    policy.save_pretrained(checkpoint)
    shutil.rmtree(weights)

    restored = ACTDINOv3Policy.from_pretrained(checkpoint, local_files_only=True)
    with torch.no_grad():
        actual = restored.predict_action_chunk(batch)

    assert restored.config.type == "act_dinov3"
    assert not weights.exists()
    assert policy.state_dict().keys() == restored.state_dict().keys()
    for name, expected_parameter in policy.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], expected_parameter)
    torch.testing.assert_close(actual, expected)
