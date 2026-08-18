from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_SCRIPTS = REPO_ROOT / "run_scripts"
SMOKE_ROOT = "/data/wengyikun/pi05_official_libero_smoke"


def _read_script(name: str) -> str:
    return (RUN_SCRIPTS / name).read_text()


def test_official_libero_pi05_train_launcher_contract():
    script = _read_script("launch_pi05_official_libero_smoke.sh")

    assert script.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert 'OUTPUT_DIR="${OUTPUT_DIR:-' + SMOKE_ROOT + '/train_out}"' in script
    assert "MUJOCO_GL=egl" in script
    assert "-m lerobot.scripts.lerobot_train" in script
    assert 'DATASET_REPO_ID="HuggingFaceVLA/libero"' in script
    assert '--dataset.repo_id="$DATASET_REPO_ID"' in script
    assert "--policy.type=pi05" in script
    assert 'PRETRAINED_PATH="lerobot/pi05_libero"' in script
    assert '--policy.pretrained_path="$PRETRAINED_PATH"' in script
    assert "--env.type=libero" in script
    assert "--env.task=libero_spatial" in script
    assert "--env.task_ids='[0]'" in script
    assert "--policy.dtype=bfloat16" in script
    assert "--policy.chunk_size=50" in script
    assert "--policy.n_action_steps=10" in script
    assert "--policy.joint_gripper_indices='[6]'" in script
    assert "--policy.freeze_language_model=true" in script
    assert "--policy.freeze_vision_encoder=false" in script
    assert "--policy.train_expert_only=false" in script
    assert "--policy.push_to_hub=false" in script


def test_frozen_language_libero_pi05_train_launcher_contract():
    script = _read_script("launch_pi05_libero_frozen_language.sh")

    assert script.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert 'DATASET_REPO_ID="HuggingFaceVLA/libero"' in script
    assert 'PRETRAINED_PATH="lerobot/pi05_libero"' in script
    assert 'OUTPUT_DIR="${OUTPUT_DIR:-/data/wengyikun/pi05_libero_frozen_language/train_out}"' in script
    assert 'STEPS="${STEPS:-250000}"' in script
    assert 'SAVE_FREQ="${SAVE_FREQ:-50000}"' in script
    assert 'BATCH_SIZE="${BATCH_SIZE:-16}"' in script
    assert "--dataset.episodes" not in script
    assert "--policy.visual_pretrained_path" not in script
    assert "--policy.freeze_language_model=true" in script
    assert "--policy.freeze_vision_encoder=false" in script
    assert "--policy.train_expert_only=false" in script
    assert "--policy.gradient_checkpointing=true" in script
    assert "--policy.scheduler_decay_steps=250000" in script
    assert "--policy.chunk_size=50" in script
    assert "--policy.n_action_steps=50" in script
    assert "--policy.dtype=bfloat16" in script
    assert '--steps="$STEPS"' in script
    assert '--save_freq="$SAVE_FREQ"' in script
    assert '--batch_size="$BATCH_SIZE"' in script


def test_official_libero_pi05_eval_launcher_contract():
    script = _read_script("eval_pi05_official_libero_smoke.sh")

    assert script.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert 'TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-' + SMOKE_ROOT + '/train_out}"' in script
    assert 'CHECKPOINT_STEP="${CHECKPOINT_STEP:-000001}"' in script
    assert 'PRETRAINED_MODEL_DIR="${TRAIN_OUTPUT_DIR}/checkpoints/${CHECKPOINT_STEP}/pretrained_model"' in script
    assert 'OUTPUT_DIR="${OUTPUT_DIR:-' + SMOKE_ROOT + '/eval_out}"' in script
    assert "MUJOCO_GL=egl" in script
    assert "-m lerobot.scripts.lerobot_eval" in script
    assert '--policy.path="$PRETRAINED_MODEL_DIR"' in script
    assert "--env.type=libero" in script
    assert "--env.task=libero_spatial" in script
    assert "--env.task_ids='[0]'" in script
    assert "--env.control_mode=relative" in script
    assert "--eval.batch_size=1" in script
    assert "--eval.n_episodes=1" in script
