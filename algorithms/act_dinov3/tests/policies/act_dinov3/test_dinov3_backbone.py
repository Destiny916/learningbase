from types import SimpleNamespace

import pytest
import torch
from torch import nn


class TinyDINOv3(nn.Module):
    def __init__(self, hidden_size: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            image_size=224,
            patch_size=16,
            num_register_tokens=4,
        )
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.gradient_checkpointing_enabled = False
        self.seen_inputs: list[torch.Tensor] = []

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True

    def forward(self, pixel_values: torch.Tensor):
        self.seen_inputs.append(pixel_values.detach().clone())
        batch_size = pixel_values.shape[0]
        special = torch.full(
            (batch_size, 5, self.hidden_size),
            -100.0,
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        )
        patch_values = torch.arange(
            196 * self.hidden_size,
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        ).reshape(1, 196, self.hidden_size)
        patches = patch_values.expand(batch_size, -1, -1) * self.scale
        return SimpleNamespace(last_hidden_state=torch.cat([special, patches], dim=1))


def test_shared_backbone_removes_special_tokens_and_reshapes_patches():
    from lerobot.policies.act_dinov3.dinov3_backbone import DINOv3SpatialBackbone

    model = TinyDINOv3()
    backbone = DINOv3SpatialBackbone(
        model=model,
        hidden_size=8,
        num_register_tokens=4,
        patch_size=16,
        gradient_checkpointing=True,
        autocast_dtype="float32",
    )
    image = torch.ones(2, 3, 224, 224)

    feature_map = backbone(image)

    expected = torch.arange(196 * 8, dtype=image.dtype).reshape(1, 196, 8)
    expected = expected.expand(2, -1, -1).transpose(1, 2).reshape(2, 8, 14, 14)
    torch.testing.assert_close(feature_map, expected)
    assert model.gradient_checkpointing_enabled is True
    assert feature_map.dtype == image.dtype


def test_shared_backbone_preserves_camera_order_and_accumulates_gradients():
    from lerobot.policies.act_dinov3.dinov3_backbone import DINOv3SpatialBackbone

    model = TinyDINOv3()
    backbone = DINOv3SpatialBackbone(
        model=model,
        hidden_size=8,
        num_register_tokens=4,
        patch_size=16,
        gradient_checkpointing=False,
        autocast_dtype="float32",
        apply_image_normalization=False,
    )
    cameras = [torch.full((1, 3, 224, 224), value) for value in (1.0, 2.0, 3.0)]

    outputs = backbone.forward_cameras(cameras)
    sum(feature.sum() for feature in outputs).backward()

    assert len(outputs) == 3
    assert len({id(backbone.model) for _ in cameras}) == 1
    assert [seen[0, 0, 0, 0].item() for seen in model.seen_inputs] == [1.0, 2.0, 3.0]
    assert model.scale.grad is not None
    assert torch.isfinite(model.scale.grad)
    assert model.scale.grad.abs() > 0


def test_shared_backbone_applies_dinov3_imagenet_normalization():
    from lerobot.policies.act_dinov3.dinov3_backbone import DINOv3SpatialBackbone

    model = TinyDINOv3()
    backbone = DINOv3SpatialBackbone(
        model=model,
        hidden_size=8,
        num_register_tokens=4,
        patch_size=16,
        gradient_checkpointing=False,
        autocast_dtype="float32",
        apply_image_normalization=True,
    )

    backbone(torch.full((1, 3, 224, 224), 0.5))

    expected = torch.tensor([(0.5 - 0.485) / 0.229, (0.5 - 0.456) / 0.224, (0.5 - 0.406) / 0.225])
    torch.testing.assert_close(model.seen_inputs[-1][0, :, 0, 0], expected)


def test_shared_backbone_rejects_invalid_patch_token_count():
    from lerobot.policies.act_dinov3.dinov3_backbone import DINOv3SpatialBackbone

    model = TinyDINOv3()
    model.forward = lambda pixel_values: SimpleNamespace(
        last_hidden_state=torch.zeros(pixel_values.shape[0], 200, model.hidden_size)
    )
    backbone = DINOv3SpatialBackbone(
        model=model,
        hidden_size=8,
        num_register_tokens=4,
        patch_size=16,
        gradient_checkpointing=False,
        autocast_dtype="float32",
    )

    with pytest.raises(ValueError, match="196 patch tokens"):
        backbone(torch.ones(1, 3, 224, 224))


def test_shared_backbone_rejects_model_layout_mismatch():
    from lerobot.policies.act_dinov3.dinov3_backbone import DINOv3SpatialBackbone

    model = TinyDINOv3()
    model.config.patch_size = 14

    with pytest.raises(ValueError, match="patch size"):
        DINOv3SpatialBackbone(
            model=model,
            hidden_size=8,
            num_register_tokens=4,
            patch_size=16,
            gradient_checkpointing=False,
            autocast_dtype="float32",
        )
