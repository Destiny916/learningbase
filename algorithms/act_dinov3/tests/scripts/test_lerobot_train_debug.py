import torch

from lerobot.scripts.lerobot_train import _format_debug_batch_metadata


def test_format_debug_batch_metadata_reports_dataset_identifiers():
    batch = {
        "index": torch.tensor([41, 42]),
        "episode_index": torch.tensor([3, 3]),
        "frame_index": torch.tensor([8, 9]),
        "observation.state": torch.zeros(2, 14),
    }

    assert _format_debug_batch_metadata(batch) == {
        "index": [41, 42],
        "episode_index": [3, 3],
        "frame_index": [8, 9],
    }
