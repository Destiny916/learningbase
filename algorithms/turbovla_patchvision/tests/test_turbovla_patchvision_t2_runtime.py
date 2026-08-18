from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from turbovla.models.configuration import VisionEncoderConfig
from turbovla.models.turbovla import TurboVLA, VisionProjection
from turbovla.models.vision_encoder import DINOv3VisionEncoder


class _TinyDINO(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4, patch_size=2, num_register_tokens=0)
        self.embeddings = SimpleNamespace(mask_token=torch.nn.Parameter(torch.zeros(1)))
        self.weight = torch.nn.Parameter(torch.ones(4))

    def forward(self, pixel_values, output_hidden_states):
        batch_size = pixel_values.shape[0]
        patches = pixel_values.mean(dim=(1, 2, 3)).unsqueeze(1).unsqueeze(-1).expand(batch_size, 1, 4)
        prefixes = self.weight.view(1, 1, -1).expand(batch_size, 1, -1)
        tokens = torch.cat([prefixes, patches], dim=1)
        return SimpleNamespace(hidden_states=[tokens], last_hidden_state=tokens)


def test_frozen_dino_encodes_two_frames_and_three_views_without_backprop(monkeypatch):
    backbone = _TinyDINO()
    monkeypatch.setattr("turbovla.models.vision_encoder._load_pretrained_model", lambda _config: backbone)
    encoder = DINOv3VisionEncoder(
        VisionEncoderConfig(
            model_name_or_path="unused",
            image_size=2,
            num_views=3,
            temporal_window_size=2,
            frozen=True,
        )
    )

    output = encoder(torch.ones(2, 2, 3, 3, 2, 2, requires_grad=True))

    assert output.shape == (2, 2, 3, 1, 4)
    assert not output.requires_grad
    assert all(not parameter.requires_grad for parameter in backbone.parameters())


def test_time_embedding_expands_two_frame_visual_context_to_six_patch_grids():
    model = object.__new__(TurboVLA)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(vision=SimpleNamespace(position_embedding="learned_patch"))
    model.temporal_window_size = 2
    model.num_views = 3
    model.vision_encoder = torch.nn.Identity()
    model.vision_projection = VisionProjection(in_dim=4, out_dim=6, hidden_dim=8, dropout=0.0)
    model.view_embedding = torch.nn.Parameter(torch.zeros(1, 3, 1, 6))
    model.patch_position_embedding = torch.nn.Parameter(torch.zeros(1, 3, 1, 6))
    model.patch_position_scale = torch.nn.Parameter(torch.ones(1, 3, 1, 1))
    model.time_embedding = torch.nn.Parameter(torch.zeros(1, 2, 1, 1, 6))

    tokens = model.encode_vision(torch.ones(2, 2, 3, 1, 4))
    tokens.square().mean().backward()

    assert tokens.shape == (2, 2 * 3 * 1, 6)
    assert model.time_embedding.grad is not None
