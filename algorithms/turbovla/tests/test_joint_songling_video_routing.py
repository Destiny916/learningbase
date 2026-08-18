from pathlib import Path

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
from starVLA.dataloader.gr00t_lerobot.schema import LeRobotModalityMetadata


def test_lerobot_v3_video_path_uses_per_camera_file_index():
    dataset = object.__new__(LeRobotSingleDataset)
    dataset._lerobot_version = "v3.0"
    dataset._dataset_path = Path("/dataset")
    dataset._chunk_size = 1000
    dataset._video_path_pattern = (
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    )
    dataset._lerobot_modality_meta = LeRobotModalityMetadata.model_validate(
        {
            "state": {},
            "action": {},
            "video": {"top": {"original_key": "observation.images.top"}},
        }
    )
    dataset.trajectory_ids_to_metadata = {
        42: {
            "data/chunk_index": 0,
            "data/file_index": 0,
            "videos/file_indices": {
                "observation.images.top": {"chunk_index": 0, "file_index": 1}
            },
        }
    }

    path = dataset.get_video_path(42, "top")

    assert path == Path(
        "/dataset/videos/observation.images.top/chunk-000/file-001.mp4"
    )
