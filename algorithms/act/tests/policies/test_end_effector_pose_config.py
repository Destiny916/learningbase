"""Explicit pose10d configuration contracts for ACT and PI05."""

from __future__ import annotations

import pytest

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.utils.constants import ACTION, OBS_STATE


def _pose_features():
    return {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,)),
    }, {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(10,)),
    }


def test_act_pose_relative_mode_accepts_pose10d_contract() -> None:
    inputs, outputs = _pose_features()
    config = ACTConfig(
        input_features=inputs,
        output_features=outputs,
        end_effector_pose_representation="relative",
        pose_state_stats_path="/tmp/relative_pose_state_q01_q99.json",
        pose_action_stats_path="/tmp/relative_pose_action_chunk16_q01_q99.json",
        chunk_size=16,
        n_action_steps=16,
    )

    assert config.observation_delta_indices == [-1, 0]


def test_pi05_pose_relative_mode_accepts_pose10d_contract() -> None:
    inputs, outputs = _pose_features()
    config = PI05Config(
        input_features=inputs,
        output_features=outputs,
        end_effector_pose_representation="relative",
        pose_state_stats_path="/tmp/relative_pose_state_q01_q99.json",
        pose_action_stats_path="/tmp/relative_pose_action_chunk50_q01_q99.json",
        joint_gripper_indices=[9],
        chunk_size=50,
        n_action_steps=50,
    )

    config.validate_features()
    assert config.observation_delta_indices == [-1, 0]


def test_pi05_pose_relative_mode_requires_gripper_at_index_nine() -> None:
    inputs, outputs = _pose_features()
    config = PI05Config(
        input_features=inputs,
        output_features=outputs,
        end_effector_pose_representation="relative",
        pose_state_stats_path="/tmp/relative_pose_state_q01_q99.json",
        pose_action_stats_path="/tmp/relative_pose_action_chunk50_q01_q99.json",
        chunk_size=50,
        n_action_steps=50,
    )

    with pytest.raises(ValueError, match=r"joint_gripper_indices must be \[9\]"):
        config.validate_features()
