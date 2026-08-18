from __future__ import annotations

import pytest

from lerobot.scripts.realign_pose_split_to_joint7d import build_pose_split_indices


def _joint_manifest() -> dict:
    return {
        "splits": {
            "train": {"source_episode_indices": [0, 2]},
            "test": {"source_episode_indices": [1]},
        }
    }


def _joint_summary() -> dict:
    return {
        "episodes": [
            {"output_episode_index": 0, "source_episode": "/raw/normal/episode0"},
            {"output_episode_index": 1, "source_episode": "/raw/normal/episode2"},
            {"output_episode_index": 2, "source_episode": "/raw/wrong/episode0"},
        ]
    }


def test_build_pose_split_indices_uses_joint_episode_membership_not_pose_split_membership() -> None:
    pose_summaries = [
        {
            "episodes": [
                {"output_episode_index": 0, "source_episode": "/raw/normal/episode2"},
                {"output_episode_index": 1, "source_episode": "/raw/wrong/episode0"},
            ]
        },
        {"episodes": [{"output_episode_index": 0, "source_episode": "/raw/normal/episode0"}]},
    ]

    assert build_pose_split_indices(_joint_manifest(), _joint_summary(), pose_summaries) == {
        "train": [1, 2],
        "test": [0],
    }


def test_build_pose_split_indices_rejects_missing_joint_episode() -> None:
    with pytest.raises(ValueError, match="missing from pose datasets"):
        build_pose_split_indices(
            _joint_manifest(),
            _joint_summary(),
            [{"episodes": [{"output_episode_index": 0, "source_episode": "/raw/normal/episode0"}]}],
        )
