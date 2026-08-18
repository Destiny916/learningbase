#!/usr/bin/env python

import torch

from lerobot.policies.image_preprocessing import camera_crop_resize_torch, center_crop_resize_torch


def test_center_crop_resize_uses_middle_480_columns_before_resizing():
    image = torch.arange(480 * 640, dtype=torch.float32).reshape(1, 1, 480, 640)

    result = center_crop_resize_torch(image, height=224, width=224)

    assert result.shape == (1, 1, 224, 224)
    expected = torch.nn.functional.interpolate(
        image[:, :, :, 80:560], size=(224, 224), mode="bilinear", align_corners=False
    )
    torch.testing.assert_close(result, expected)


def test_center_crop_resize_preserves_uint8_dtype_and_channels_last_layout():
    image = torch.zeros((1, 480, 640, 3), dtype=torch.uint8)
    image[:, :, 80:560, :] = 255

    result = center_crop_resize_torch(image, height=224, width=224, channels_last=True)

    assert result.shape == (1, 224, 224, 3)
    assert result.dtype is torch.uint8
    assert torch.equal(result, torch.full_like(result, 255))


def test_top_camera_padding_preserves_full_width_before_resizing():
    image = torch.arange(405 * 720, dtype=torch.float32).reshape(1, 1, 405, 720)

    result = camera_crop_resize_torch(
        image, height=224, width=224, camera_key="observation.images.top"
    )

    expected = torch.nn.functional.interpolate(
        torch.nn.functional.pad(image, (0, 0, 157, 158)),
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )
    torch.testing.assert_close(result, expected)


def test_stereo_top_camera_padding_preserves_full_width_before_resizing():
    image = torch.arange(405 * 720, dtype=torch.float32).reshape(1, 1, 405, 720)

    result = camera_crop_resize_torch(
        image, height=224, width=224, camera_key="observation.images.top_right"
    )

    expected = torch.nn.functional.interpolate(
        torch.nn.functional.pad(image, (0, 0, 157, 158)),
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )
    torch.testing.assert_close(result, expected)
