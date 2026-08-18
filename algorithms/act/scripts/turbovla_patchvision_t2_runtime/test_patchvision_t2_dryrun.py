from __future__ import annotations

from argparse import Namespace

import numpy as np
import pytest

from dual_turbovla_patchvision_t2_dryrun import (
    temporal_pair_to_model_images,
    validate_dry_run_args,
)
from temporal_image_cache import TemporalImageCache


def raw_image(shape: tuple[int, int, int], marker: int) -> np.ndarray:
    return np.full(shape, marker, dtype=np.uint8)


def add_raw_triplet(cache: TemporalImageCache, timestamp: float, marker: int) -> None:
    cache.add_frame("top", raw_image((405, 720, 3), marker), timestamp)
    cache.add_frame("gripper_left", raw_image((480, 640, 3), marker + 1), timestamp + 0.001)
    cache.add_frame("gripper_right", raw_image((480, 640, 3), marker + 2), timestamp + 0.002)


def test_temporal_pair_to_model_images_preserves_time_and_view_order() -> None:
    cache = TemporalImageCache(max_camera_skew_s=0.01)
    add_raw_triplet(cache, 1.000, marker=10)
    add_raw_triplet(cache, 1.033, marker=20)

    model_images = temporal_pair_to_model_images(cache.latest_pair(timeout_s=0.01))

    assert [[list(image.shape) for image in step] for step in model_images] == [
        [[720, 720, 3], [480, 480, 3], [480, 480, 3]],
        [[720, 720, 3], [480, 480, 3], [480, 480, 3]],
    ]
    assert [int(image[240, 240, 0]) for image in model_images[0]] == [10, 11, 12]
    assert [int(image[240, 240, 0]) for image in model_images[1]] == [20, 21, 22]


@pytest.mark.parametrize("flag", ["enable_arms", "enable_grippers", "execute_robot_actions"])
def test_dry_run_rejects_hardware_control_flags(flag: str) -> None:
    args = Namespace(enable_arms=False, enable_grippers=False, execute_robot_actions=False)
    setattr(args, flag, True)

    with pytest.raises(SystemExit, match="forbids all hardware enable/action flags"):
        validate_dry_run_args(args)
