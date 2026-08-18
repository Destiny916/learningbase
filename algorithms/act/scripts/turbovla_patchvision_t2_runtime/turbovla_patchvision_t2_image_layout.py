"""Image geometry for the PatchVision T2 Joint Songling checkpoint."""

from __future__ import annotations

import numpy as np


def preprocess_patchvision_t2_views(images: list[np.ndarray]) -> list[np.ndarray]:
    """Match training-time joint_songling geometry before model preprocessing."""
    if len(images) != 3:
        raise ValueError(f"expected top, left, right images, got {len(images)}")
    top, left, right = images
    if top.shape != (405, 720, 3) or top.dtype != np.uint8:
        raise ValueError(f"top must be uint8 [405,720,3], got {top.shape} {top.dtype}")
    for name, image in (("gripper_left", left), ("gripper_right", right)):
        if image.shape != (480, 640, 3) or image.dtype != np.uint8:
            raise ValueError(f"{name} must be uint8 [480,640,3], got {image.shape} {image.dtype}")
    return [
        np.pad(top, ((157, 158), (0, 0), (0, 0)), mode="constant"),
        left[:, 80:560, :],
        right[:, 80:560, :],
    ]
