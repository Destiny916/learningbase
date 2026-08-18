from pathlib import Path


RUN_SCRIPTS = Path(__file__).parents[2] / "run_scripts"


def test_pi05_0806_relative_state_launcher_contract() -> None:
    script = (RUN_SCRIPTS / "launch_pi05_0806_relative_state_relative_action_gpu3.sh").read_text(encoding="utf-8")

    assert "0806swap_gripper_fixed_pi052_task_en_v2" in script
    assert "normalization_relative_pi05_action_aligned_v2" in script
    assert 'PALIGEMMA_TOKENIZER="${PALIGEMMA_TOKENIZER:-/data/jianan/weight/paligemma-3b-pt-224}"' in script
    assert "--policy.type=pi05" in script
    assert "--policy.pretrained_path=\"$PI05_BASE\"" in script
    assert "--policy.tokenizer_name=\"$PALIGEMMA_TOKENIZER\"" in script
    assert "--policy.freeze_language_model=true" in script
    assert "--policy.freeze_vision_encoder=false" in script
    assert "--policy.empty_cameras=0" in script
    assert '"observation.images.top","observation.images.gripper_left","observation.images.gripper_right"' in script
    assert "--policy.joint_representation=relative" in script
    assert "--policy.state_absolute_indices='[6,7,8,9,16,17,18,19]'" in script
    assert "--policy.relative_state_stats_path" in script
    assert "--policy.relative_action_stats_path" in script
    assert "--policy.clip_quantiles=false" in script
    assert "--policy.state_noise_std_rad=0" in script
    assert "--policy.gripper_noise_std_m=0" in script
    assert 'BATCH_SIZE="${BATCH_SIZE:-16}"' in script
    assert 'STEPS="${STEPS:-400000}"' in script
    assert 'SAVE_FREQ="${SAVE_FREQ:-40000}"' in script
    assert "--policy.scheduler_warmup_steps=20000" in script
    assert "--policy.scheduler_decay_steps=400000" in script
    assert "--eval_steps=0" in script
