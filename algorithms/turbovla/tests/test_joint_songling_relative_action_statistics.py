import numpy as np
import pandas as pd

from starVLA.dataloader.gr00t_lerobot.datasets import (
    LeRobotSingleDataset,
    calculate_rel_action_statistics,
)
from starVLA.dataloader.gr00t_lerobot.schema import LeRobotModalityMetadata


def test_relative_action_statistics_never_cross_episode_boundaries(tmp_path):
    parquet_path = tmp_path / "file-000.parquet"
    pd.DataFrame(
        {
            "episode_index": [0, 0, 1, 1],
            "observation.state": [
                np.array([0.0], dtype=np.float32),
                np.array([0.0], dtype=np.float32),
                np.array([100.0], dtype=np.float32),
                np.array([100.0], dtype=np.float32),
            ],
            "action": [
                np.array([0.0], dtype=np.float32),
                np.array([0.0], dtype=np.float32),
                np.array([100.0], dtype=np.float32),
                np.array([100.0], dtype=np.float32),
            ],
        }
    ).to_parquet(parquet_path)
    metadata = LeRobotModalityMetadata.model_validate(
        {
            "state": {
                "joints": {"start": 0, "end": 1, "original_key": "observation.state"}
            },
            "action": {"joints": {"start": 0, "end": 1, "original_key": "action"}},
            "video": {},
        }
    )

    stats = calculate_rel_action_statistics(
        parquet_paths=[parquet_path],
        lerobot_modality_meta=metadata,
        action_keys_full=["action.joints"],
        state_keys_full=["state.joints"],
        action_indices=[0, 1],
        state_indices=[0],
        action_mode_apply_keys=["action.joints"],
    )

    np.testing.assert_allclose(stats["action"]["min"], [0.0])
    np.testing.assert_allclose(stats["action"]["max"], [0.0])
    np.testing.assert_allclose(stats["action"]["q01"], [0.0])
    np.testing.assert_allclose(stats["action"]["q99"], [0.0])


def test_relative_action_statistics_anchor_to_current_state(tmp_path):
    parquet_path = tmp_path / "file-000.parquet"
    pd.DataFrame(
        {
            "episode_index": [0, 0],
            "observation.state": [
                np.array([0.0], dtype=np.float32),
                np.array([10.0], dtype=np.float32),
            ],
            "action": [
                np.array([0.0], dtype=np.float32),
                np.array([10.0], dtype=np.float32),
            ],
        }
    ).to_parquet(parquet_path)
    metadata = LeRobotModalityMetadata.model_validate(
        {
            "state": {
                "joints": {"start": 0, "end": 1, "original_key": "observation.state"}
            },
            "action": {"joints": {"start": 0, "end": 1, "original_key": "action"}},
            "video": {},
        }
    )

    stats = calculate_rel_action_statistics(
        parquet_paths=[parquet_path],
        lerobot_modality_meta=metadata,
        action_keys_full=["action.joints"],
        state_keys_full=["state.joints"],
        action_indices=[0],
        state_indices=[-1, 0],
        action_mode_apply_keys=["action.joints"],
    )

    np.testing.assert_allclose(stats["action"]["min"], [0.0])
    np.testing.assert_allclose(stats["action"]["max"], [0.0])
    np.testing.assert_allclose(stats["action"]["q01"], [0.0])
    np.testing.assert_allclose(stats["action"]["q99"], [0.0])


def test_runtime_relative_action_anchor_is_current_state():
    dataset = object.__new__(LeRobotSingleDataset)
    dataset._action_mode = "rel"
    dataset._action_mode_apply_keys = ["action.joints"]
    dataset._action_mode_state_map = {}
    dataset._modality_keys = {
        "action": ["action.joints"],
        "state": ["state.joints"],
    }
    data = {
        "action.joints": np.array([[10.0], [11.0]], dtype=np.float32),
        "state.joints": np.array([[2.0], [10.0]], dtype=np.float32),
    }

    transformed = dataset._apply_action_mode(data)

    np.testing.assert_allclose(transformed["action.joints"], [[0.0], [1.0]])
