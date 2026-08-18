"""Train-only relative pose10d quantile statistics tests."""

from __future__ import annotations

import numpy as np
import pytest

from lerobot.datasets.end_effector_pose_stats import (
    POSE_FEATURE_NAMES,
    SCALED_POSE_INDICES,
    compute_relative_pose_stats_from_episodes,
    load_relative_pose_stats_paths,
    save_relative_pose_stats,
)
from lerobot.scripts.convert_right_end_effector_pose_to_lerobot_v30 import pose10d_from_end_pose


def _pose(x: float, gripper: float) -> np.ndarray:
    return pose10d_from_end_pose(
        {"x": x, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}, gripper
    )


def test_relative_pose_stats_use_se3_values_and_only_scale_xyz_gripper() -> None:
    episodes = [
        np.stack((_pose(0.0, 0.01), _pose(0.1, 0.02), _pose(0.3, 0.03))),
        np.stack((_pose(1.0, 0.04), _pose(1.2, 0.05))),
    ]

    bundle = compute_relative_pose_stats_from_episodes(episodes, horizons=[2])

    assert bundle.feature_names == POSE_FEATURE_NAMES
    assert bundle.scaled_indices == SCALED_POSE_INDICES
    assert bundle.state.count == 5
    assert bundle.actions[2].count == 4
    assert bundle.state.q01.shape == (10,)
    assert bundle.actions[2].q99.shape == (10,)
    expected_gripper_q01 = np.quantile([0.01, 0.02, 0.03, 0.04, 0.05], 0.01)
    assert bundle.state.q01[9] == pytest.approx(expected_gripper_q01)
    assert bundle.state.q01[3] == 1.0
    assert bundle.state.q99[7] == 1.0


def test_relative_pose_stats_round_trip_through_persistent_files(tmp_path) -> None:
    bundle = compute_relative_pose_stats_from_episodes(
        [np.stack((_pose(0.0, 0.01), _pose(0.1, 0.02), _pose(0.2, 0.03)))], horizons=[2]
    )

    save_relative_pose_stats(bundle, tmp_path)
    loaded = load_relative_pose_stats_paths(
        tmp_path / "relative_pose_state_q01_q99.json",
        tmp_path / "relative_pose_action_chunk2_q01_q99.json",
        expected_horizon=2,
    )

    np.testing.assert_allclose(loaded.state.q01, bundle.state.q01)
    np.testing.assert_allclose(loaded.actions[2].q99, bundle.actions[2].q99)
    assert loaded.scaled_indices == SCALED_POSE_INDICES
