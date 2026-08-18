from pathlib import Path

import pytest


RUN_SCRIPTS = Path(__file__).parents[2] / "run_scripts"
LAUNCHERS = {
    "launch_pi052_0806_subtask_mem_relative_gpu3.sh": "relative",
    "launch_pi052_0806_subtask_mem_absolute_gpu4.sh": "absolute",
}


@pytest.mark.parametrize(("name", "state_representation"), LAUNCHERS.items())
def test_pi052_0806_subtask_memory_launcher_contract(name: str, state_representation: str) -> None:
    script = (RUN_SCRIPTS / name).read_text(encoding="utf-8")

    assert "0806swap_gripper_fixed_pi052_task_en_subtask_mem" in script
    assert "--policy.type=pi052" in script
    assert "--policy.recipe_path=recipes/subtask_mem.yaml" in script
    assert "--policy.knowledge_insulation=true" in script
    assert "--policy.freeze_language_model=true" in script
    assert "--policy.unfreeze_lm_head=false" in script
    assert "--policy.freeze_vision_encoder=false" in script
    assert "--policy.empty_cameras=0" in script
    assert '"observation.images.top","observation.images.gripper_left","observation.images.gripper_right"' in script
    assert "--policy.state_noise_std_rad=0" in script
    assert "--policy.gripper_noise_std_m=0" in script
    assert "--policy.scheduler_warmup_steps=20000" in script
    assert "--policy.scheduler_decay_steps=400000" in script
    assert 'STEPS="${STEPS:-400000}"' in script
    assert 'SAVE_FREQ="${SAVE_FREQ:-40000}"' in script
    assert 'BATCH_SIZE="${BATCH_SIZE:-1}"' in script
    assert "--policy.joint_representation=" + state_representation in script
    assert "--policy.relative_action_stats_path" in script


def test_relative_launcher_keeps_endpoint_and_gripper_state_absolute() -> None:
    script = (RUN_SCRIPTS / "launch_pi052_0806_subtask_mem_relative_gpu3.sh").read_text(encoding="utf-8")

    assert "--policy.state_absolute_indices='[6,7,8,9,16,17,18,19]'" in script
    assert "--policy.relative_state_stats_path" in script


def test_absolute_launcher_uses_absolute_state_with_relative_joint_actions() -> None:
    script = (RUN_SCRIPTS / "launch_pi052_0806_subtask_mem_absolute_gpu4.sh").read_text(encoding="utf-8")

    assert "--policy.use_relative_actions=true" in script
    assert "--policy.relative_exclude_joints='[\"gripper\"]'" in script
    assert "--policy.absolute_state_stats_path" in script
