"""Deterministic image preprocessing shared by the PI05 and ACT policies."""

import torch
import torch.nn.functional as F


TOP_CAMERA_KEY = "observation.images.top"
TOP_STEREO_CAMERA_KEYS = {
    TOP_CAMERA_KEY,
    "observation.images.top_left",
    "observation.images.top_right",
}


def center_crop_resize_torch(
    images: torch.Tensor,
    *,
    height: int,
    width: int,
    channels_last: bool = False,
) -> torch.Tensor:
    """Center-crop every image to a square, then resize it without padding."""
    if images.ndim != 4:
        raise ValueError(f"images must have four dimensions, got {tuple(images.shape)}")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")

    channels = images.shape[-1] if channels_last else images.shape[1]
    if channels not in {1, 3}:
        raise ValueError(f"expected one or three image channels, got {channels}")

    channels_first = images.movedim(-1, 1) if channels_last else images
    source_height, source_width = channels_first.shape[-2:]
    square_size = min(source_height, source_width)
    top = (source_height - square_size) // 2
    left = (source_width - square_size) // 2
    cropped = channels_first[..., top : top + square_size, left : left + square_size]

    if cropped.shape[-2:] == (height, width):
        resized = cropped
    else:
        original_dtype = cropped.dtype
        resized = F.interpolate(cropped.float(), size=(height, width), mode="bilinear", align_corners=False)
        if original_dtype == torch.uint8:
            resized = resized.round().clamp(0, 255).to(torch.uint8)
        else:
            resized = resized.to(dtype=original_dtype)

    return resized.movedim(1, -1) if channels_last else resized


def camera_crop_resize_torch(
    images: torch.Tensor,
    *,
    height: int,
    width: int,
    camera_key: str,
    channels_last: bool = False,
) -> torch.Tensor:
    """Apply the fixed per-camera crop rule used by dual-arm training and inference."""
    if camera_key not in TOP_STEREO_CAMERA_KEYS:
        return center_crop_resize_torch(images, height=height, width=width, channels_last=channels_last)
    if images.ndim != 4:
        raise ValueError(f"images must have four dimensions, got {tuple(images.shape)}")

    channels = images.shape[-1] if channels_last else images.shape[1]
    if channels not in {1, 3}:
        raise ValueError(f"expected one or three image channels, got {channels}")

    channels_first = images.movedim(-1, 1) if channels_last else images
    source_height, source_width = channels_first.shape[-2:]
    square_size = max(source_height, source_width)
    pad_top = (square_size - source_height) // 2
    pad_bottom = square_size - source_height - pad_top
    pad_left = (square_size - source_width) // 2
    pad_right = square_size - source_width - pad_left
    padded = F.pad(channels_first, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)

    original_dtype = padded.dtype
    resized = F.interpolate(padded.float(), size=(height, width), mode="bilinear", align_corners=False)
    if original_dtype == torch.uint8:
        resized = resized.round().clamp(0, 255).to(torch.uint8)
    else:
        resized = resized.to(dtype=original_dtype)
    return resized.movedim(1, -1) if channels_last else resized
