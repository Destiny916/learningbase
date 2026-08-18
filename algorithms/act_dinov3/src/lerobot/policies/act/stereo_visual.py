"""StereoPolicy-style visual encoder used only by opt-in ACT configurations.

The top cameras are a calibrated stereo pair. DINOv2 contributes frozen
external-view priors, while the ResNet/FPN-style feature paths, stereo fusion,
and ACT policy remain trainable.
"""

from __future__ import annotations

from collections.abc import Sequence
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d


TOP_LEFT_KEY = "observation.images.top_left"
TOP_RIGHT_KEY = "observation.images.top_right"
TOP_KEY = "observation.images.top"
WRIST_LEFT_KEY = "observation.images.gripper_left"
WRIST_RIGHT_KEY = "observation.images.gripper_right"
WRIST_LEFT_DEPTH_KEY = "observation.images.gripper_left_depth"
WRIST_RIGHT_DEPTH_KEY = "observation.images.gripper_right_depth"

_RGB_KEYS = (TOP_LEFT_KEY, TOP_RIGHT_KEY, WRIST_LEFT_KEY, WRIST_RIGHT_KEY)
_RGBD_KEYS = (*_RGB_KEYS, WRIST_LEFT_DEPTH_KEY, WRIST_RIGHT_DEPTH_KEY)
_FIVE_CAMERA_RGBD_KEYS = (TOP_KEY, WRIST_LEFT_KEY, WRIST_RIGHT_KEY, WRIST_LEFT_DEPTH_KEY, WRIST_RIGHT_DEPTH_KEY)


def _resnet18_feature_map(weights: str | None) -> IntermediateLayerGetter:
    model = torchvision.models.resnet18(weights=weights, norm_layer=FrozenBatchNorm2d)
    return IntermediateLayerGetter(model, return_layers={"layer4": "feature_map"})


class FrozenDinoV2(nn.Module):
    """Expose DINOv2 patch features while making accidental finetuning impossible."""

    def __init__(self, model: nn.Module | None = None) -> None:
        super().__init__()
        if model is None:
            source = os.environ.get("DINO_V2_REPO", "/data/wengyikun/models/dinov2")
            model = torch.hub.load(
                source if Path(source).is_dir() else "facebookresearch/dinov2",
                "dinov2_vits14",
                pretrained=True,
                source="local" if Path(source).is_dir() else "github",
            )
        self.model = model
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> FrozenDinoV2:
        # DINOv2 uses no train-time state in this policy and must remain frozen.
        super().train(False)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, image: Tensor) -> Tensor:
        mean = image.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = image.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        with torch.autocast(device_type=image.device.type, enabled=False):
            output = self.model.forward_features(((image.float() - mean.float()) / std.float()))
        tokens = output["x_norm_patchtokens"]
        patch_count = tokens.shape[1]
        grid = int(patch_count**0.5)
        if grid * grid != patch_count:
            raise ValueError(f"DINOv2 patch tokens must form a square grid, got {patch_count}")
        return tokens.transpose(1, 2).reshape(image.shape[0], tokens.shape[-1], grid, grid)


class _StereoTransformerLayer(nn.Module):
    """One official-style self-attention, bidirectional cross-attention and MLP block."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.left_self = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.right_self = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.left_cross = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.right_cross = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.left_norm1 = nn.LayerNorm(dim)
        self.right_norm1 = nn.LayerNorm(dim)
        self.left_norm2 = nn.LayerNorm(dim)
        self.right_norm2 = nn.LayerNorm(dim)
        self.left_norm3 = nn.LayerNorm(dim)
        self.right_norm3 = nn.LayerNorm(dim)
        self.left_mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))
        self.right_mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))

    @staticmethod
    def _rope(tokens: Tensor, positions: Tensor) -> Tensor:
        """2D RoPE over the cross-attention query/key channels."""
        _, _, dim = tokens.shape
        if dim % 4:
            raise ValueError(f"Stereo Transformer dim must be divisible by four, got {dim}")
        quarter = dim // 4
        frequencies = torch.arange(quarter // 2, device=tokens.device, dtype=tokens.dtype)
        frequencies = 1.0 / (10000 ** (frequencies / max(quarter // 2, 1)))

        def rotate(part: Tensor, coordinate: Tensor) -> Tensor:
            angles = coordinate[:, None] * frequencies[None, :]
            sine, cosine = angles.sin()[None], angles.cos()[None]
            even, odd = part[..., 0::2], part[..., 1::2]
            return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1).flatten(-2)

        x1, x2, y1, y2 = tokens.split(quarter, dim=-1)
        return torch.cat((rotate(x1, positions[:, 0]), rotate(x2, positions[:, 0]), rotate(y1, positions[:, 1]), rotate(y2, positions[:, 1])), dim=-1)

    def forward(self, left: Tensor, right: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        left = self.left_norm1(left + self.left_self(left, left, left, need_weights=False)[0])
        right = self.right_norm1(right + self.right_self(right, right, right, need_weights=False)[0])
        left_q, right_q = self._rope(left, positions), self._rope(right, positions)
        left_cross = self.left_cross(left_q, right_q, right, need_weights=False)[0]
        right_cross = self.right_cross(right_q, left_q, left, need_weights=False)[0]
        left = self.left_norm2(left + left_cross)
        right = self.right_norm2(right + right_cross)
        return self.left_norm3(left + self.left_mlp(left)), self.right_norm3(right + self.right_mlp(right))


class StereoACTVisual(nn.Module):
    """Trainable ACT feature maps with a frozen-DINO top stereo pair.

    Returns exactly three maps: fused top stereo, left wrist, right wrist.
    Depth, when requested, is fused only into the corresponding wrist map.
    """

    feature_dim = 512

    def __init__(
        self,
        *,
        mode: str,
        pretrained_backbone_weights: str | None,
        dino_model: nn.Module | None = None,
        stereo_dim: int = 256,
    ) -> None:
        super().__init__()
        if mode not in {"stereo_top_rgb", "stereo_top_rgbd", "five_camera_rgbd"}:
            raise ValueError(f"unsupported stereo ACT mode: {mode}")
        self.mode = mode
        if mode == "five_camera_rgbd":
            self.rgb_backbone = _resnet18_feature_map(pretrained_backbone_weights)
            self.depth_backbone = _resnet18_feature_map(pretrained_backbone_weights)
            return
        self.top_backbone = _resnet18_feature_map(pretrained_backbone_weights)
        self.wrist_rgb_backbone = _resnet18_feature_map(pretrained_backbone_weights)
        self.dino = FrozenDinoV2(dino_model)
        dino_dim = self._dino_feature_dim()
        self.dino_downsample = nn.Sequential(
            nn.Conv2d(dino_dim, stereo_dim, kernel_size=4, stride=4), nn.GELU()
        )
        self.top_hybrid_project = nn.Sequential(nn.Conv2d(512 + stereo_dim, stereo_dim, kernel_size=1), nn.GELU())
        self.stereo_layers = nn.ModuleList([_StereoTransformerLayer(stereo_dim, heads=8, dropout=0.1) for _ in range(2)])
        self.top_output = nn.Conv2d(stereo_dim * 2, self.feature_dim, kernel_size=1)
        self.wrist_depth_backbone: IntermediateLayerGetter | None = None
        self.wrist_depth_fusion: nn.Module | None = None
        if mode == "stereo_top_rgbd":
            self.wrist_depth_backbone = _resnet18_feature_map(pretrained_backbone_weights)
            self.wrist_depth_fusion = nn.Sequential(nn.Conv2d(1024, self.feature_dim, kernel_size=1), nn.GELU())

    def _dino_feature_dim(self) -> int:
        with torch.no_grad():
            return self.dino(torch.zeros(1, 3, 224, 224)).shape[1]

    @staticmethod
    def _as_mapping(images: Sequence[Tensor], image_keys: Sequence[str]) -> dict[str, Tensor]:
        if len(images) != len(image_keys):
            raise ValueError(f"received {len(images)} images for {len(image_keys)} configured keys")
        mapping = dict(zip(image_keys, images, strict=True))
        return mapping

    @staticmethod
    def _positions(height: int, width: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        y, x = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype), torch.arange(width, device=device, dtype=dtype), indexing="ij"
        )
        return torch.stack((x, y), dim=-1).reshape(-1, 2)

    def _top_features(self, left_image: Tensor, right_image: Tensor) -> Tensor:
        left_cnn = self.top_backbone(left_image)["feature_map"]
        right_cnn = self.top_backbone(right_image)["feature_map"]
        left_dino, right_dino = self.dino(left_image), self.dino(right_image)
        # DINO's 16x16 patch map becomes a 4x4 stride-4 map, then aligns with the ACT 7x7 CNN map.
        left_dino = F.interpolate(self.dino_downsample(left_dino), size=left_cnn.shape[-2:], mode="bilinear", align_corners=False)
        right_dino = F.interpolate(self.dino_downsample(right_dino), size=right_cnn.shape[-2:], mode="bilinear", align_corners=False)
        left = self.top_hybrid_project(torch.cat((left_cnn, left_dino.to(dtype=left_cnn.dtype)), dim=1))
        right = self.top_hybrid_project(torch.cat((right_cnn, right_dino.to(dtype=right_cnn.dtype)), dim=1))
        height, width = left.shape[-2:]
        left_tokens = left.flatten(2).transpose(1, 2)
        right_tokens = right.flatten(2).transpose(1, 2)
        positions = self._positions(height, width, device=left.device, dtype=left.dtype)
        for layer in self.stereo_layers:
            left_tokens, right_tokens = layer(left_tokens, right_tokens, positions)
        fused = torch.cat((left_tokens, right_tokens), dim=-1).transpose(1, 2).reshape(left.shape[0], -1, height, width)
        return self.top_output(fused)

    @staticmethod
    def _depth_to_rgb(depth: Tensor, *, upper_m: float) -> Tensor:
        if depth.shape[1] != 1:
            raise ValueError(f"wrist depth must have one channel, got {depth.shape}")
        return (depth.clamp(0.07, upper_m) - 0.07).div(upper_m - 0.07).repeat(1, 3, 1, 1)

    def forward(self, images: Sequence[Tensor], image_keys: Sequence[str]) -> list[Tensor]:
        if self.mode == "five_camera_rgbd":
            if set(image_keys) != set(_FIVE_CAMERA_RGBD_KEYS):
                raise ValueError(f"{self.mode} requires exactly {list(_FIVE_CAMERA_RGBD_KEYS)}, got {list(image_keys)}")
            image_by_key = self._as_mapping(images, image_keys)
            rgb_features = [
                self.rgb_backbone(image_by_key[key])["feature_map"]
                for key in (TOP_KEY, WRIST_LEFT_KEY, WRIST_RIGHT_KEY)
            ]
            depth_features = [
                self.depth_backbone(self._depth_to_rgb(image_by_key[WRIST_LEFT_DEPTH_KEY], upper_m=0.90))["feature_map"],
                self.depth_backbone(self._depth_to_rgb(image_by_key[WRIST_RIGHT_DEPTH_KEY], upper_m=0.60))["feature_map"],
            ]
            return [*rgb_features, *depth_features]

        expected = _RGBD_KEYS if self.mode == "stereo_top_rgbd" else _RGB_KEYS
        if set(image_keys) != set(expected):
            raise ValueError(f"{self.mode} requires exactly {list(expected)}, got {list(image_keys)}")
        image_by_key = self._as_mapping(images, image_keys)
        top = self._top_features(image_by_key[TOP_LEFT_KEY], image_by_key[TOP_RIGHT_KEY])
        left = self.wrist_rgb_backbone(image_by_key[WRIST_LEFT_KEY])["feature_map"]
        right = self.wrist_rgb_backbone(image_by_key[WRIST_RIGHT_KEY])["feature_map"]
        if self.mode == "stereo_top_rgbd":
            assert self.wrist_depth_backbone is not None and self.wrist_depth_fusion is not None
            left_depth = self.wrist_depth_backbone(self._depth_to_rgb(image_by_key[WRIST_LEFT_DEPTH_KEY], upper_m=0.90))["feature_map"]
            right_depth = self.wrist_depth_backbone(self._depth_to_rgb(image_by_key[WRIST_RIGHT_DEPTH_KEY], upper_m=0.60))["feature_map"]
            left = self.wrist_depth_fusion(torch.cat((left, left_depth.to(dtype=left.dtype)), dim=1))
            right = self.wrist_depth_fusion(torch.cat((right, right_depth.to(dtype=right.dtype)), dim=1))
        return [top, left, right]
