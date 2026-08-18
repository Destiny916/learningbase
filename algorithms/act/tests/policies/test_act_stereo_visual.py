import torch
from torch import nn

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig


class FakeDinoV2(nn.Module):
    """Small DINO-shaped module to test the frozen adapter without network access."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, 32, kernel_size=14, stride=14, bias=False)

    def forward_features(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = self.proj(image).flatten(2).transpose(1, 2)
        return {"x_norm_patchtokens": tokens}


def test_five_camera_rgbd_emits_one_feature_map_per_input() -> None:
    from lerobot.policies.act.stereo_visual import StereoACTVisual

    visual = StereoACTVisual(mode="five_camera_rgbd", pretrained_backbone_weights=None)
    images = [torch.rand(1, 3, 64, 64) for _ in range(3)]
    images.extend([torch.full((1, 1, 64, 64), 0.20), torch.full((1, 1, 64, 64), 0.40)])
    features = visual(
        images,
        [
            "observation.images.top",
            "observation.images.gripper_left",
            "observation.images.gripper_right",
            "observation.images.gripper_left_depth",
            "observation.images.gripper_right_depth",
        ],
    )

    assert [feature.shape for feature in features] == [(1, 512, 2, 2)] * 5


def test_five_camera_rgbd_requires_exactly_three_rgb_and_two_depth_inputs() -> None:
    config = ACTConfig(stereo_visual_mode="five_camera_rgbd")
    config.input_features = {
        "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 405, 720)),
        "observation.images.gripper_left": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        "observation.images.gripper_right": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        "observation.images.gripper_left_depth": PolicyFeature(type=FeatureType.VISUAL, shape=(1, 480, 640)),
        "observation.images.gripper_right_depth": PolicyFeature(type=FeatureType.VISUAL, shape=(1, 480, 640)),
    }

    config.validate_features()


def test_stereo_top_rgb_uses_frozen_dino_and_emits_one_fused_top_map() -> None:
    from lerobot.policies.act.stereo_visual import StereoACTVisual

    dino = FakeDinoV2()
    visual = StereoACTVisual(mode="stereo_top_rgb", pretrained_backbone_weights=None, dino_model=dino)
    images = [torch.rand(1, 3, 64, 64, requires_grad=True) for _ in range(4)]
    features = visual(
        images,
        [
            "observation.images.top_left",
            "observation.images.top_right",
            "observation.images.gripper_left",
            "observation.images.gripper_right",
        ],
    )

    assert [feature.shape for feature in features] == [(1, 512, 2, 2)] * 3
    assert not any(parameter.requires_grad for parameter in visual.dino.parameters())
    assert visual.dino.training is False
    sum(feature.mean() for feature in features).backward()
    assert dino.proj.weight.grad is None
    assert visual.top_backbone.conv1.weight.grad is not None


def test_stereo_top_rgbd_fuses_depth_only_for_wrist_views() -> None:
    from lerobot.policies.act.stereo_visual import StereoACTVisual

    visual = StereoACTVisual(mode="stereo_top_rgbd", pretrained_backbone_weights=None, dino_model=FakeDinoV2())
    images = [torch.rand(1, 3, 64, 64) for _ in range(4)]
    images.extend([torch.full((1, 1, 64, 64), 0.07), torch.full((1, 1, 64, 64), 0.60)])
    features = visual(
        images,
        [
            "observation.images.top_left",
            "observation.images.top_right",
            "observation.images.gripper_left",
            "observation.images.gripper_right",
            "observation.images.gripper_left_depth",
            "observation.images.gripper_right_depth",
        ],
    )

    assert [feature.shape for feature in features] == [(1, 512, 2, 2)] * 3
    assert visual.wrist_depth_backbone is not None
    assert visual.wrist_depth_fusion is not None
