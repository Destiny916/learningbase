from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]
WARM_CONFIG = ROOT / "experiments/joint_songling/configs/0812_closed_gripper_top_padded_warm.yaml"
FRESH_CONFIG = ROOT / "experiments/joint_songling/configs/0812_closed_gripper_top_padded_fresh.yaml"
LAUNCHER = ROOT / "scripts/joint_songling/train_0812_closed_gripper_top_padded.sh"
TASK = (
    "first pick up the bread with the right hand, then hand it to the left hand "
    "at the middle point, then place the bread in the bowl with the left hand."
)


def test_0812_configs_share_requested_data_and_schedule():
    warm = yaml.safe_load(WARM_CONFIG.read_text())
    fresh = yaml.safe_load(FRESH_CONFIG.read_text())
    for config in (warm, fresh):
        data = config["datasets"]["vla_data"]
        trainer = config["trainer"]
        assert data["per_device_batch_size"] == 16
        assert data["state_mode"] == "delta"
        assert data["state_mode_apply_keys"] == ["left_joints", "right_joints"]
        assert data["action_mode"] == "rel"
        assert data["action_mode_apply_keys"] == ["left_joints", "right_joints"]
        assert data["image_layout"] == "joint_songling_top_padded"
        assert data["task"] == TASK
        assert data["num_workers"] == 8
        assert data["prefetch_factor"] == 2
        assert data["persistent_workers"] is True
        assert config["framework"]["action"]["state_dim"] == 20
        assert config["framework"]["action"]["action_dim"] == 14
        assert config["framework"]["action"]["horizon"] == 50
        assert trainer["max_train_steps"] == 500000
        assert trainer["num_warmup_steps"] == 25000
        assert trainer["save_interval"] == 20000
        assert trainer["lr_scheduler_type"] == "cosine"
        assert trainer["scheduler_specific_kwargs"] == {}
    assert warm["trainer"]["pretrained_checkpoint"] == "${oc.env:TURBOVLA_INITIAL_CKPT}"
    assert fresh["trainer"]["pretrained_checkpoint"] is None


def test_0812_launcher_isolates_gpu_and_preserves_top_padding_contract():
    script = LAUNCHER.read_text()
    assert 'CUDA_VISIBLE_DEVICES:-0' in script
    assert (
        '/data/wengyikun/datasets/joint_songling/0812_closed_gripper_zero_without_ep173_174'
        in script
    )
    assert 'relative_action_anchor": "current"' in script
    assert 'gripper_mode": "binary_absolute_closed_zero"' in script
    assert 'total_episodes": 202' in script
    assert 'total_frames": 125558' in script
    assert 'top 405x720 pad(157,158)->720x720->224x224' in script
    assert 'gripper_left": {"original_key": "observation.images.gripper_left"}' in script
    assert 'gripper_right": {"original_key": "observation.images.gripper_right"}' in script
    assert 'SMOKE_TEST' in script
    assert 'trainer["max_train_steps"] = 1' in script
    assert '[[ ! -e "$RUN_ROOT_DIR" ]]' in script


def test_0812_patchvision_uses_sixteen_workers_and_single_thread_environment():
    config = (ROOT / "experiments/joint_songling/configs/0812_closed_patchvision_t2_gpu7.yaml").read_text()
    script = (ROOT / "scripts/joint_songling/train_0812_closed_patchvision_t2_gpu7.sh").read_text()
    assert "num_workers: 16" in config
    assert "video_backend: pyav" in config
    assert 'OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"' in script
    assert 'MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"' in script
    assert 'stream.codec_context.thread_count = 1' in (
        ROOT / "third_party/starvla_runtime/starVLA/dataloader/gr00t_lerobot/video.py"
    ).read_text()
