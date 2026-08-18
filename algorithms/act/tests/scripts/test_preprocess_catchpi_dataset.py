from __future__ import annotations

import numpy as np

from lerobot.scripts.preprocess_catchpi_dataset import (
    make_output_features,
    make_terminal_action,
    trim_alignment_frames,
    trimmed_episode_length,
)


def test_trimmed_episode_length_uses_floor() -> None:
    assert trimmed_episode_length(100) == 66
    assert trimmed_episode_length(111) == 74
    assert trimmed_episode_length(223) == 148


def test_make_output_features_removes_top_and_renames_grippers() -> None:
    joint_names = [f"left_joint_{i}" for i in range(7)] + [f"right_joint_{i}" for i in range(7)]
    features = {
        "observation.state": {"dtype": "float32", "shape": [14], "names": joint_names},
        "action": {"dtype": "float32", "shape": [14], "names": joint_names},
        "observation.images.top": {"dtype": "video", "shape": [480, 640, 3]},
        "observation.images.left_wrist": {"dtype": "video", "shape": [480, 640, 3]},
        "observation.images.right_wrist": {"dtype": "video", "shape": [480, 640, 3]},
        "timestamp": {"dtype": "float32", "shape": [1]},
    }

    output = make_output_features(features)

    assert "observation.images.top" not in output
    assert "timestamp" not in output
    assert output["observation.state"]["names"][6] == "left_gripper"
    assert output["observation.state"]["names"][13] == "right_gripper"
    assert output["action"]["names"][6] == "left_gripper"
    assert output["action"]["names"][13] == "right_gripper"
    assert features["observation.state"]["names"][6] == "left_joint_6"


def test_make_terminal_action_returns_final_state_copy() -> None:
    state = np.arange(14, dtype=np.float32)
    source_action = state + 1

    terminal_action = make_terminal_action(state, source_action, is_terminal=True)
    normal_action = make_terminal_action(state, source_action, is_terminal=False)

    np.testing.assert_array_equal(terminal_action, state)
    np.testing.assert_array_equal(normal_action, source_action)
    assert terminal_action is not state
    assert normal_action is not source_action


def test_trim_alignment_frames_removes_top_and_reterminates_action() -> None:
    frames = [
        {
            "frame_index": index,
            "source_timestamps": {
                "observation.images.top": float(index),
                "observation.images.left_wrist": float(index),
            },
            "images": {
                "observation.images.top": f"top-{index}.jpg",
                "observation.images.left_wrist": f"left-{index}.jpg",
            },
            "state": [float(index)] * 14,
            "action": [float(index + 1)] * 14,
        }
        for index in range(6)
    ]

    output = trim_alignment_frames(frames, keep_length=4)

    assert len(output) == 4
    assert "observation.images.top" not in output[-1]["source_timestamps"]
    assert "observation.images.top" not in output[-1]["images"]
    assert output[-1]["action"] == output[-1]["state"]
    assert frames[-1]["images"]["observation.images.top"] == "top-5.jpg"
