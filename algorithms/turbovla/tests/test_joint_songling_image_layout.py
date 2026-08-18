import numpy as np
from PIL import Image

from starVLA.model.framework.VLM4A.TurboVLA import (
    TurboVLAFramework,
)
from starVLA.dataloader.gr00t_lerobot.datasets import (
    preprocess_joint_songling_frame,
    preprocess_joint_songling_top_padded_frame,
)


def test_joint_songling_layout_pads_top_and_center_crops_wrist_views():
    top = np.full((405, 720, 3), 255, dtype=np.uint8)
    left = np.full((480, 640, 3), (255, 0, 0), dtype=np.uint8)
    right = np.full((480, 640, 3), (0, 0, 255), dtype=np.uint8)

    processed = [
        preprocess_joint_songling_frame(top, 0),
        preprocess_joint_songling_frame(left, 1),
        preprocess_joint_songling_frame(right, 2),
    ]

    assert [image.shape for image in processed] == [(720, 720, 3), (480, 480, 3), (480, 480, 3)]
    assert processed[0][0, 0].tolist() == [0, 0, 0]
    assert processed[0][360, 360].tolist() == [255, 255, 255]
    assert processed[1][0, 0].tolist() == [255, 0, 0]
    assert processed[2][0, 0].tolist() == [0, 0, 255]


def test_joint_songling_top_padded_layout_only_changes_top_geometry():
    top = np.arange(405 * 720 * 3, dtype=np.uint32).reshape(405, 720, 3).astype(np.uint8)
    left = np.full((480, 640, 3), (255, 0, 0), dtype=np.uint8)
    right = np.full((480, 640, 3), (0, 0, 255), dtype=np.uint8)

    padded_top = preprocess_joint_songling_top_padded_frame(top, 0)
    processed_left = preprocess_joint_songling_top_padded_frame(left, 1)
    processed_right = preprocess_joint_songling_top_padded_frame(right, 2)

    assert padded_top.shape == (720, 720, 3)
    assert np.count_nonzero(padded_top[:157]) == 0
    assert np.count_nonzero(padded_top[562:]) == 0
    np.testing.assert_array_equal(padded_top[157:562], top)
    assert processed_left is left
    assert processed_right is right
    assert processed_left.shape == (480, 640, 3)
    assert processed_right.shape == (480, 640, 3)


def test_native_layout_repeats_the_last_view_when_a_view_is_missing():
    framework = object.__new__(TurboVLAFramework)
    framework.num_views = 3
    framework.input_layout = "native"

    views = framework._as_view_list([Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))])

    assert len(views) == 3
    assert views[2] is views[1]
