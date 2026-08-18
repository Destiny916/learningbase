from types import SimpleNamespace

import torch

from turbovla.models.configuration import VisionEncoderConfig
from turbovla.models.vision_encoder import DINOv3VisionEncoder


class _Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4, patch_size=2, num_register_tokens=0)
        self.embeddings = SimpleNamespace(mask_token=torch.nn.Parameter(torch.zeros(1)))
        self.gradient_checkpointing_calls = 0

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_calls += 1


def test_trainable_dinov3_enables_gradient_checkpointing(monkeypatch):
    backbone = _Backbone()
    monkeypatch.setattr(
        "turbovla.models.vision_encoder._load_pretrained_model",
        lambda _config: backbone,
    )

    DINOv3VisionEncoder(
        VisionEncoderConfig(
            model_name_or_path="unused",
            image_size=224,
            gradient_checkpointing=True,
        )
    )

    assert backbone.gradient_checkpointing_calls == 1
