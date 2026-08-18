from pathlib import Path


CONFIG = Path("experiments/joint_songling/configs/0806swap_patchvision_t2_3view.yaml")
LAUNCHER = Path("scripts/joint_songling/train_0806swap_patchvision_t2_3view.sh")
BASE_LAUNCHER = Path("scripts/joint_songling/train_0806swap_3view_gpu6.sh")


def test_patchvision_t2_recipe_freezes_dino_and_keeps_batch_16():
    config = CONFIG.read_text()

    assert "temporal_window_size: 2" in config
    assert "freeze_vision_encoder: true" in config
    assert "gradient_checkpointing: false" in config
    assert "data_mix: joint_songling_0806swap_endpoint20_t2" in config
    assert "per_device_batch_size: 16" in config
    assert "max_train_steps: 200000" in config
    assert "num_warmup_steps: 10000" in config
    assert "save_interval: 20000" in config
    assert "lr_scheduler_type: cosine" in config


def test_patchvision_launcher_requires_an_explicit_gpu_and_selects_new_recipe():
    launcher = LAUNCHER.read_text()
    base_launcher = BASE_LAUNCHER.read_text()

    assert '"${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to one idle host GPU.}"' in launcher
    assert "TURBOVLA_CONFIG_YAML=experiments/joint_songling/configs/0806swap_patchvision_t2_3view.yaml" in launcher
    assert "TURBOVLA_CONFIG_YAML" in base_launcher
    assert '--config_yaml "${TURBOVLA_CONFIG_YAML:-experiments/joint_songling/configs/0806swap_3view.yaml}"' in base_launcher
