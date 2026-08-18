from experiments.joint_songling.data_registry.data_config import JointSonglingDataConfig


def test_registry_preserves_joint_songling_camera_and_action_order():
    config = JointSonglingDataConfig()
    modalities = config.modality_config()

    assert modalities["video"].modality_keys == [
        "video.top",
        "video.gripper_left",
        "video.gripper_right",
    ]
    assert modalities["action"].modality_keys == [
        "action.left_joints",
        "action.left_gripper",
        "action.right_joints",
        "action.right_gripper",
    ]


def test_registry_uses_separate_q99_state_and_action_transforms():
    config = JointSonglingDataConfig()
    transforms = config.transform().transforms

    assert transforms[1].normalization_modes == {key: "q99" for key in config.state_keys}
    assert transforms[3].normalization_modes == {key: "q99" for key in config.action_keys}
