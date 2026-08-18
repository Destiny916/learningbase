from __future__ import annotations

import numpy as np

from lerobot.scripts.preprocess_relative_chunk_dataset import (
    make_state_with_object_centers_features,
    make_preprocessing_summary,
    make_relative_action_stats,
    make_relative_output_features,
    make_relative_state_and_action_chunks,
)


def test_object_center_state_features_append_six_metric_coordinates_without_changing_action() -> None:
    joints = [f"left_joint_{index}" for index in range(6)] + ["left_gripper"]
    joints += [f"right_joint_{index}" for index in range(6)] + ["right_gripper"]
    source_features = {
        "observation.state": {"dtype": "float32", "shape": [14], "names": joints},
        "action": {"dtype": "float32", "shape": [14], "names": joints},
    }

    output_features = make_state_with_object_centers_features(source_features)

    assert output_features["observation.state"]["shape"] == [20]
    assert output_features["observation.state"]["names"][-6:] == [
        "bread_x_m",
        "bread_y_m",
        "bread_z_m",
        "bowl_x_m",
        "bowl_y_m",
        "bowl_z_m",
    ]
    assert output_features["action"] == source_features["action"]
    assert source_features["observation.state"]["shape"] == [14]


def test_relative_output_features_preserve_joint_names_and_make_action_a_chunk() -> None:
    names = [f"left_joint_{index}" for index in range(6)] + ["left_gripper"]
    names += [f"right_joint_{index}" for index in range(6)] + ["right_gripper"]
    source_features = {
        "observation.state": {"dtype": "float32", "shape": [14], "names": names},
        "action": {"dtype": "float32", "shape": [14], "names": names},
    }

    output_features = make_relative_output_features(source_features, chunk_size=20)

    assert output_features["observation.state"]["shape"] == [14]
    assert output_features["action"]["shape"] == [20, 14]
    assert output_features["action"]["names"] == names
    assert source_features["action"]["shape"] == [14]


def test_relative_action_stats_flatten_action_horizon_but_keep_joint_dimensions() -> None:
    chunks = np.arange(2 * 3 * 14, dtype=np.float32).reshape(2, 3, 14)

    stats = make_relative_action_stats(chunks)

    assert stats["count"].tolist() == [6]
    assert stats["min"].shape == (14,)
    assert stats["max"].shape == (14,)
    np.testing.assert_array_equal(stats["min"], np.arange(14, dtype=np.float32))
    np.testing.assert_array_equal(stats["max"], np.arange(70, 84, dtype=np.float32))


def test_preprocessing_summary_records_exact_source_and_output_paths() -> None:
    source = "/data/joint_songling/source"
    output = "/data/joint_songling/output"

    summary = make_preprocessing_summary(
        source=source,
        output=output,
        chunk_size=20,
        total_episodes=2,
        total_frames=3,
        inherited_preprocessing={"mode": "previous"},
    )

    assert summary["source"] == source
    assert summary["output"] == output
    assert summary["chunk_size"] == 20
    assert summary["inherited_preprocessing"] == {"mode": "previous"}


def test_relative_state_zeros_first_arm_frame_and_keeps_grippers_absolute() -> None:
    states = np.array(
        [
            [1.0] * 6 + [0.02] + [2.0] * 6 + [0.03],
            [1.5] * 6 + [0.04] + [2.5] * 6 + [0.05],
            [2.0] * 6 + [0.06] + [3.0] * 6 + [0.07],
            [4.0] * 6 + [0.08] + [5.0] * 6 + [0.09],
        ],
        dtype=np.float32,
    )

    relative_state, _ = make_relative_state_and_action_chunks(states, [3, 1], chunk_size=2)

    np.testing.assert_array_equal(relative_state[0, :6], np.zeros(6))
    np.testing.assert_array_equal(relative_state[0, 7:13], np.zeros(6))
    np.testing.assert_array_equal(relative_state[0, [6, 13]], states[0, [6, 13]])
    np.testing.assert_array_equal(relative_state[1, :6], np.full(6, 0.5))
    np.testing.assert_array_equal(relative_state[1, 7:13], np.full(6, 0.5))
    np.testing.assert_array_equal(relative_state[3, :6], np.zeros(6))
    np.testing.assert_array_equal(relative_state[3, 7:13], np.zeros(6))
    np.testing.assert_array_equal(relative_state[3, [6, 13]], states[3, [6, 13]])


def test_action_chunk_is_anchored_to_current_state_and_pads_episode_end() -> None:
    states = np.array(
        [
            [1.0] * 6 + [0.02] + [2.0] * 6 + [0.03],
            [1.5] * 6 + [0.04] + [2.5] * 6 + [0.05],
            [2.0] * 6 + [0.06] + [3.0] * 6 + [0.07],
        ],
        dtype=np.float32,
    )

    _, chunks = make_relative_state_and_action_chunks(states, [3], chunk_size=3)

    np.testing.assert_array_equal(chunks[0, 0, :6], np.full(6, 0.5))
    np.testing.assert_array_equal(chunks[0, 1, :6], np.full(6, 1.0))
    np.testing.assert_array_equal(chunks[0, 2, :6], np.full(6, 1.0))
    np.testing.assert_array_equal(chunks[0, :, 7:13], np.array([[0.5] * 6, [1.0] * 6, [1.0] * 6]))
    np.testing.assert_array_equal(chunks[0][:, [6, 13]], states[[1, 2, 2]][:, [6, 13]])
    np.testing.assert_array_equal(chunks[2, :, :6], np.zeros((3, 6)))
    np.testing.assert_array_equal(chunks[2, :, 7:13], np.zeros((3, 6)))
    np.testing.assert_array_equal(chunks[2][:, [6, 13]], np.repeat(states[2, [6, 13]][None, :], 3, axis=0))
