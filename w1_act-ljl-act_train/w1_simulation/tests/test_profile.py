from __future__ import annotations

from w1_simulation.robot.joints import ACT_STATE_JOINTS, BODY_JOINTS, HAND_POSITION_JOINTS
from w1_simulation.w1_profile import DEFAULT_PROFILE


def test_default_profile_is_the_complete_w1_popcorn_contract() -> None:
    profile = DEFAULT_PROFILE

    assert profile.name == "w1_popcorn_v1"
    assert profile.checkpoint.name == "pretrained_model"
    assert tuple(profile.act["state_action_order"]) == ACT_STATE_JOINTS
    assert profile.body_command_names == BODY_JOINTS
    assert tuple(profile.commands["hand_order"]) == HAND_POSITION_JOINTS
    assert profile.endpoints.body == "/control/joint_position"
    assert profile.endpoints.left_hand == "/control/hand/left"
    assert profile.endpoints.right_hand == "/control/hand/right"
    assert profile.simulation["bridge"]["replan_threshold"] == 0.5
    assert profile.simulation["bridge"]["lipo_blend_policy_points"] == 5


def test_profile_records_intentional_training_runtime_urdf_difference() -> None:
    profile = DEFAULT_PROFILE

    assert profile.hashes["training_kinematics_urdf_sha256"] != profile.hashes["runtime_urdf_sha256"]
    assert profile.payload["compatibility"]["allow_training_runtime_urdf_hash_mismatch"] is True
