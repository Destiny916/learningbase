"""Tests for train-only absolute state and future-action statistics."""

from __future__ import annotations

import numpy as np

from lerobot.datasets.absolute_action_stats import compute_absolute_action_stats_from_episodes


def test_absolute_stats_keep_state_and_future_action_quantiles_separate() -> None:
    episodes = [
        np.asarray(
            [
                [0.0, 10.0],
                [1.0, 11.0],
                [2.0, 12.0],
                [3.0, 13.0],
            ],
            dtype=np.float32,
        )
    ]

    bundle = compute_absolute_action_stats_from_episodes(
        episodes,
        horizons=[2],
        feature_names=["joint_0", "gripper"],
        scaled_indices=[0, 1],
    )

    expected_state = np.quantile(episodes[0], [0.01, 0.99], axis=0)
    expected_action_values = np.concatenate((episodes[0][1:], episodes[0][2:]), axis=0)
    expected_action = np.quantile(expected_action_values, [0.01, 0.99], axis=0)
    assert np.allclose(bundle.state.q01, expected_state[0])
    assert np.allclose(bundle.state.q99, expected_state[1])
    assert np.allclose(bundle.actions[2].q01, expected_action[0])
    assert np.allclose(bundle.actions[2].q99, expected_action[1])
    assert not np.allclose(bundle.state.q01, bundle.actions[2].q01)
    assert bundle.state.count == 4
    assert bundle.actions[2].count == 5
