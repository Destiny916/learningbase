from pathlib import Path


def test_endpoint20_launcher_uses_right_eye_and_wrist_rgb_only():
    script = Path("scripts/joint_songling/train_endpoint20_3view.sh").read_text()
    assert 'CUDA_VISIBLE_DEVICES:-7' in script
    assert "observation.images.top_right" in script
    assert "observation.images.gripper_left" in script
    assert "observation.images.gripper_right" in script
    assert "observation.images.top_left" not in script
    assert "gripper_left_depth" not in script
    assert "gripper_right_depth" not in script
