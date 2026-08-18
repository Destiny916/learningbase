from starVLA.dataloader.gr00t_lerobot.datasets import _video_resolution_from_feature


def test_video_resolution_uses_lerobot_v3_hwc_shape_without_dimension_names():
    assert _video_resolution_from_feature({"shape": [405, 720, 3], "names": None}) == (720, 405, 3)
