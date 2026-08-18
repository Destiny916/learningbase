from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_SCRIPTS = REPO_ROOT / "run_scripts"


def test_v044_reference_launcher_uses_official_checkpoint_and_protocol():
    script = (RUN_SCRIPTS / "eval_pi05_libero_reference_v044.sh").read_text()

    assert 'POLICY_PATH="lerobot/pi05_libero_finetuned_v044"' in script
    assert '--env.type=libero' in script
    assert '--env.task=libero_spatial,libero_object,libero_goal,libero_10' in script
    assert '--eval.n_episodes=10' in script
    assert '--policy.n_action_steps=10' in script
    assert '--env.control_mode=relative' in script
