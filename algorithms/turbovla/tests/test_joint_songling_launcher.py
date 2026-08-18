from pathlib import Path


def test_launcher_selects_gpu_2_and_requires_training_assets():
    script = Path("scripts/joint_songling/train_relative_3view.sh").read_text()

    assert 'selected_gpu="${CUDA_VISIBLE_DEVICES:-2}"' in script
    assert '--gpus "device=${selected_gpu}"' in script
    assert "-e CUDA_VISIBLE_DEVICES=0" in script
    assert "JOINT_SONGLING_DATA_ROOT" in script
    assert "TURBOVLA_RELEASE_CKPT" in script
    assert "top,gripper_left,gripper_right" in script
    assert "Pick up the bread with the right arm" in script
    assert 'tasks.parquet' in script
    assert 'to_parquet' in script
