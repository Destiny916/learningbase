#!/usr/bin/env python

import os
import re
import subprocess
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.common.offline_validation import evaluate_offline, make_action_unnormalizer
from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.datasets.relative_joint_stats import (
    compute_relative_joint_stats_from_episodes,
    save_relative_joint_stats,
)
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
from lerobot.scripts.lerobot_train import audit_pi05_parameters, format_named_metrics
from lerobot.utils.constants import ACTION, OBS_STATE

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_SCRIPTS = REPO_ROOT / "run_scripts"
DATA_ROOT = "/data/wengyikun/datasets/joint_songling/0714_gripper_bread_combined_split_seed42"
STATS_ROOT = f"{DATA_ROOT}/normalization"

LAUNCHERS = {
    "launch_pi05_relative_state_0714.sh": ("pi05", "true", "50", "16"),
    "launch_pi05_image_only_0714.sh": ("pi05", "false", "50", "16"),
    "launch_act_relative_state_0714.sh": ("act", "true", "16", "32"),
    "launch_act_image_only_0714.sh": ("act", "false", "16", "32"),
}


def _read_script(name: str) -> str:
    return (RUN_SCRIPTS / name).read_text()


@pytest.mark.parametrize("name,contract", LAUNCHERS.items())
def test_relative_policy_launcher_contract(name: str, contract: tuple[str, str, str, str]):
    policy, condition_on_state, chunk_size, default_batch_size = contract
    script = _read_script(name)

    assert script.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert f'DATA_ROOT="{DATA_ROOT}"' in script
    assert '--dataset.root="$TRAIN_ROOT"' in script
    assert '--validation_dataset.root="$TEST_ROOT"' in script
    assert f"--policy.type={policy}" in script
    assert "--policy.joint_representation=relative" in script
    assert "--policy.gripper_indices='[6]'" in script or "--policy.joint_gripper_indices='[6]'" in script
    assert f"--policy.condition_on_state={condition_on_state}" in script
    assert f"--policy.chunk_size={chunk_size}" in script
    assert f"--policy.n_action_steps={chunk_size}" in script
    assert f'BATCH_SIZE="${{BATCH_SIZE:-{default_batch_size}}}"' in script
    assert 'GRAD_ACC="${GRAD_ACC:-1}"' in script
    assert 'STEPS="${STEPS:-10000}"' in script
    assert 'SAVE_FREQ="${SAVE_FREQ:-2000}"' in script
    assert 'EVAL_STEPS="${EVAL_STEPS:-1000}"' in script
    assert 'MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}"' in script
    assert 'LOG_FREQ="${LOG_FREQ:-10}"' in script
    assert "--policy.push_to_hub=false" in script
    assert "--save_checkpoint=true" in script
    assert "best_validation" not in script

    for stats_name in (
        "relative_state_q01_q99.json",
        "relative_action_chunk16_q01_q99.json",
        "relative_action_chunk50_q01_q99.json",
    ):
        assert stats_name in script
    assert 'STATS_ROOT="${DATA_ROOT}/normalization"' in script
    assert "sha256sum" in script
    assert "effective batch" in script
    assert "NUM_PROCESSES" in script
    assert "condition_on_state" in script
    assert "freeze_language_model" in script
    assert "output_dir" in script
    assert '--max_eval_samples="$MAX_EVAL_SAMPLES"' in script


def test_pi05_launchers_preserve_trainable_visual_path():
    for name in ("launch_pi05_relative_state_0714.sh", "launch_pi05_image_only_0714.sh"):
        script = _read_script(name)
        assert "--policy.pretrained_path=/data/wengyikun/openpi/lerobot_pi05_base" in script
        assert (
            "--policy.visual_pretrained_path="
            "/data/wengyikun/models/TeleEmbodied_VISTA/pretrained_model/model.safetensors"
        ) in script
        assert "--policy.visual_pretrained_include_projector=true" in script
        assert "--policy.freeze_language_model=true" in script
        assert "--policy.freeze_vision_encoder=false" in script
        assert "--policy.train_expert_only=false" in script
        assert "--policy.dtype=bfloat16" in script
        assert "--policy.gradient_checkpointing=true" in script


def test_dual_arm_pi05_launcher_disables_torchdynamo_compile():
    script = _read_script("launch_pi05_relative_dualarm14d_0724.sh")

    assert "export TORCHDYNAMO_DISABLE=1" in script


STATE_ABLATION_PI05_LAUNCHERS = {
    "launch_pi05_relative_state_dualarm14d_0724_0727_full99_nonoise_noclip.sh": "relative",
    "launch_pi05_absolute_state_relative_action_dualarm14d_0724_0727_full99_nonoise_noclip.sh": "absolute",
}

STATE_ABLATION_PI052_LAUNCHERS = {
    "launch_pi052_relative_state_dualarm14d_0724_0727_full99_nonoise_noclip_b8.sh": "relative",
    "launch_pi052_absolute_state_relative_action_dualarm14d_0724_0727_full99_nonoise_noclip_b8.sh": "absolute",
}


@pytest.mark.parametrize("name,state_representation", STATE_ABLATION_PI05_LAUNCHERS.items())
def test_dual_arm_pi05_state_ablation_launchers_share_training_contract(
    name: str, state_representation: str
) -> None:
    script = _read_script(name)

    assert script.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert "0724_0727_doublefripper_top_grippebread_combined_full_99episodes_task_en" in script
    assert (
        '--policy.image_feature_order=\'["observation.images.top","observation.images.gripper_left",'
        '"observation.images.gripper_right"]\'' in script
    )
    assert "--policy.pretrained_path=/data/wengyikun/openpi/lerobot_pi05_base" in script
    assert "--policy.dtype=bfloat16" in script
    assert "--policy.vision_encoder_dtype" not in script
    assert "--policy.freeze_language_model=true" in script
    assert "--policy.freeze_vision_encoder=false" in script
    assert "--policy.train_expert_only=false" in script
    assert "--policy.gradient_checkpointing=true" in script
    assert "--policy.chunk_size=50" in script
    assert "--policy.n_action_steps=50" in script
    assert "--policy.empty_cameras=0" in script
    assert "--policy.joint_gripper_indices='[6,13]'" in script
    assert "relative_action_chunk50_q01_q99.json" in script
    assert "--policy.clip_quantiles=false" in script
    assert "--policy.state_noise_std_rad=0" in script
    assert "--policy.gripper_noise_std_m=0" in script
    assert "--policy.scheduler_warmup_steps=5000" in script
    assert "--policy.scheduler_decay_steps=100000" in script
    assert "--batch_size=16" in script
    assert "--gradient_accumulation_steps=1" in script
    assert "--num_workers=8" in script
    assert "--steps=100000" in script
    assert "--save_freq=10000" in script
    assert "--eval_steps=0" in script
    assert "--dataset.image_transforms.enable=false" in script
    assert "--validation_dataset" not in script
    assert '[[ ! -e "$OUTPUT_DIR" ]]' in script
    assert f"--policy.joint_representation={state_representation}" in script


def test_dual_arm_pi05_state_ablation_launchers_differ_only_at_state_boundary() -> None:
    relative_script = _read_script("launch_pi05_relative_state_dualarm14d_0724_0727_full99_nonoise_noclip.sh")
    absolute_script = _read_script(
        "launch_pi05_absolute_state_relative_action_dualarm14d_0724_0727_full99_nonoise_noclip.sh"
    )

    assert "relative_state_q01_q99.json" in relative_script
    assert "--policy.use_relative_actions" not in relative_script
    assert "--policy.absolute_state_stats_path" not in relative_script
    assert "absolute_state_q01_q99.json" in absolute_script
    assert "--policy.use_relative_actions=true" in absolute_script
    assert "--policy.relative_exclude_joints='[\"gripper\"]'" in absolute_script
    assert "--policy.relative_state_stats_path" not in absolute_script
    assert "--policy.absolute_action_stats_path" not in absolute_script


@pytest.mark.parametrize("name,state_representation", STATE_ABLATION_PI052_LAUNCHERS.items())
def test_dual_arm_pi052_state_ablation_launcher_contract(name: str, state_representation: str) -> None:
    script = _read_script(name)

    assert "0724_0727_doublefripper_top_grippebread_combined_full_99episodes_task_en" in script
    assert "--policy.type=pi052" in script
    assert 'PI052_BASE="${PI052_BASE:-/data/wengyikun/openpi/lerobot_pi052_base}"' in script
    assert '--policy.pretrained_path="$PI052_BASE"' in script
    assert '--policy.action_tokenizer_name="$PI052_BASE/action_tokenizer"' in script
    assert "--policy.auto_fit_fast_tokenizer=false" in script
    assert "--policy.recipe_path=recipes/subtask_mem.yaml" in script
    assert "--policy.enable_fast_action_loss=true" in script
    assert "--policy.flow_loss_weight=10" in script
    assert "--policy.fast_action_loss_weight=1" in script
    assert "--policy.text_loss_weight=1" in script
    assert "--policy.knowledge_insulation=true" in script
    assert "--policy.flow_num_repeats=5" in script
    assert "--policy.dtype=bfloat16" in script
    assert "--policy.vision_encoder_dtype" not in script
    assert "--policy.freeze_language_model=true" in script
    assert "--policy.freeze_vision_encoder=false" in script
    assert "--policy.gradient_checkpointing=true" in script
    assert (
        '--policy.image_feature_order=\'["observation.images.top","observation.images.gripper_left",'
        '"observation.images.gripper_right"]\'' in script
    )
    assert "--policy.empty_cameras=0" in script
    assert f"--policy.joint_representation={state_representation}" in script
    assert "--policy.joint_gripper_indices='[6,13]'" in script
    assert "--policy.relative_action_stats_path=" in script
    assert "--policy.clip_quantiles=false" in script
    assert "--policy.state_noise_std_rad=0" in script
    assert "--policy.gripper_noise_std_m=0" in script
    assert "--batch_size=8" in script
    assert "--num_workers=8" in script
    assert "--steps=100000" in script
    assert "--save_freq=10000" in script
    assert "--eval_steps=0" in script
    assert "--policy.scheduler_warmup_steps=5000" in script
    assert "--policy.scheduler_decay_steps=100000" in script
    assert '[[ ! -e "$OUTPUT_DIR" ]]' in script

    if state_representation == "relative":
        assert "--policy.relative_state_stats_path=" in script
        assert "--policy.absolute_state_stats_path" not in script
        assert "--policy.use_relative_actions" not in script
    else:
        assert "--policy.absolute_state_stats_path=" in script
        assert "--policy.relative_state_stats_path" not in script
        assert "--policy.use_relative_actions=true" in script
        assert "--policy.relative_exclude_joints='[\"gripper\"]'" in script


@pytest.mark.parametrize(
    "name,data_root,representation,gripper_index,state_stats,action_stats",
    [
        (
            "launch_pi05_relative_joint7d_combined_empty2_0714.sh",
            "/data/wengyikun/datasets/joint_songling/0714_gripper_bread_combined_split_seed42",
            "--policy.joint_representation=relative",
            "[6]",
            "relative_state_q01_q99.json",
            "relative_action_chunk50_q01_q99.json",
        ),
        (
            "launch_pi05_relative_endpose10d_combined_empty2_0714.sh",
            "/data/wengyikun/datasets/joint_songling/0714_gripper_bread_single_teleop_normal_differentplace_wrongplace_right_fisheye_endpose_pose10d_combined_v30_joint7d_split_seed42",
            "--policy.end_effector_pose_representation=relative",
            "[9]",
            "relative_pose_state_q01_q99.json",
            "relative_pose_action_chunk50_q01_q99.json",
        ),
    ],
)
def test_combined_relative_pi05_empty_camera_launchers(
    name, data_root, representation, gripper_index, state_stats, action_stats
):
    script = _read_script(name)

    assert data_root in script
    assert representation in script
    assert f"--policy.joint_gripper_indices='{gripper_index}'" in script
    assert state_stats in script
    assert action_stats in script
    assert "--policy.empty_cameras=2" in script
    assert 'BATCH_SIZE="${BATCH_SIZE:-16}"' in script
    assert 'GRAD_ACC="${GRAD_ACC:-1}"' in script
    assert 'STEPS="${STEPS:-250000}"' in script
    assert 'SAVE_FREQ="${SAVE_FREQ:-50000}"' in script
    assert 'EVAL_STEPS="${EVAL_STEPS:-1000}"' in script
    assert "--policy.chunk_size=50" in script
    assert "--policy.n_action_steps=50" in script
    assert "--policy.freeze_language_model=true" in script
    assert "--policy.freeze_vision_encoder=false" in script
    assert "--policy.train_expert_only=false" in script
    assert "--dataset.image_transforms.enable=false" in script
    assert "--validation_dataset.image_transforms.enable=false" in script


@pytest.mark.parametrize(
    "name,data_root,gripper_indices,representation",
    [
        (
            "launch_pi05_absolute_joint7d_0714.sh",
            "/data/wengyikun/datasets/joint_songling/0714_gripper_bread_normal_even_wrongplace_joint7d_v30_split_seed42",
            "[6]",
            "--policy.joint_representation=absolute",
        ),
        (
            "launch_pi05_absolute_end_effector_pose_0714.sh",
            "/data/wengyikun/datasets/joint_songling/0714_gripper_bread_normal_even_wrongplace_endpose10d_v30_split_seed42",
            "[9]",
            "--policy.end_effector_pose_representation=absolute",
        ),
    ],
)
def test_absolute_pi05_launcher_contract(name, data_root, gripper_indices, representation):
    script = _read_script(name)

    assert script.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert data_root in script
    assert 'STATS_ROOT="${DATA_ROOT}/normalization_absolute"' in script
    assert "absolute_state_q01_q99.json" in script
    assert "absolute_action_chunk50_q01_q99.json" in script
    assert representation in script
    assert f"--policy.joint_gripper_indices='{gripper_indices}'" in script
    assert "--policy.relative_state_stats_path" not in script
    assert "--policy.relative_action_stats_path" not in script
    assert "--policy.pose_state_stats_path" not in script
    assert "--policy.pose_action_stats_path" not in script
    assert "--policy.chunk_size=50" in script
    assert "--policy.n_action_steps=50" in script
    assert 'BATCH_SIZE="${BATCH_SIZE:-16}"' in script
    assert 'STEPS="${STEPS:-32000}"' in script
    assert 'SAVE_FREQ="${SAVE_FREQ:-4000}"' in script
    assert 'EVAL_STEPS="${EVAL_STEPS:-1000}"' in script
    assert "--policy.freeze_language_model=true" in script
    assert "--policy.freeze_vision_encoder=false" in script
    assert "--policy.visual_pretrained_include_projector=true" in script
    assert "--policy.push_to_hub=false" in script
    assert '--eval_steps="$EVAL_STEPS"' in script
    assert "absolute" in script


def test_act_launchers_use_quantile_normalization():
    for name in ("launch_act_relative_state_0714.sh", "launch_act_image_only_0714.sh"):
        script = _read_script(name)
        assert "--policy.normalization_mapping.STATE" not in script
        assert "--policy.normalization_mapping.ACTION" not in script


@pytest.mark.parametrize(
    "name,mode",
    [
        ("launch_act_stereo_top_rgb_0729_subtask.sh", "stereo_top_rgb"),
        ("launch_act_stereo_top_rgbd_0729_subtask.sh", "stereo_top_rgbd"),
    ],
)
def test_stereo_act_launchers_use_full_dataset_500k_relative_contract(name: str, mode: str):
    script = _read_script(name)

    assert script.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert "0729_dualarm14d_stereo_top_rgbd_subtask_v30" in script
    assert 'STEPS="${STEPS:-500000}"' in script
    assert 'SAVE_FREQ="${SAVE_FREQ:-50000}"' in script
    assert "--eval_steps=0" in script
    assert f"--policy.stereo_visual_mode={mode}" in script
    assert "--policy.joint_representation=relative" in script
    assert "--policy.gripper_indices='[6,13]'" in script
    assert "--policy.relative_state_stats_path" in script
    assert "relative_state_q01_q99.json" in script
    assert "relative_action_chunk16_q01_q99.json" in script
    assert "--policy.state_noise_std_rad=0.003" in script
    assert "--policy.gripper_noise_std_m=0.001" in script
    assert "--dataset.depth_output_unit=m" in script
    assert 'BATCH_SIZE="${BATCH_SIZE:-32}"' in script
    assert "--num_workers=8" in script
    assert "DINO_V2_REPO" in script
    assert "TORCH_HOME" in script


def test_five_camera_730_launcher_uses_rgbd_mode_and_no_validation() -> None:
    script = _read_script("launch_act_five_camera_rgbd_730_subtask.sh")

    assert script.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert "730_subtask_with_depth" in script
    assert "--policy.stereo_visual_mode=five_camera_rgbd" in script
    assert "--dataset.depth_output_unit=m" in script
    assert 'STEPS="${STEPS:-500000}"' in script
    assert 'SAVE_FREQ="${SAVE_FREQ:-50000}"' in script
    assert 'BATCH_SIZE="${BATCH_SIZE:-32}"' in script
    assert "--policy.joint_representation=relative" in script
    assert "--policy.gripper_indices='[6,13]'" in script
    assert "relative_state_q01_q99.json" in script
    assert "relative_action_chunk16_q01_q99.json" in script
    assert "--policy.state_noise_std_rad=0.003" in script
    assert "--policy.gripper_noise_std_m=0.001" in script
    assert "--eval_steps=0" in script


@pytest.mark.parametrize("name", LAUNCHERS)
def test_launcher_uses_multi_gpu_flag_only_for_multiple_processes(name: str):
    script = _read_script(name)
    assert re.search(r'if \[\[ "\$NUM_PROCESSES" -gt 1 \]\]; then', script)
    assert "ACCELERATE_ARGS+=(--multi_gpu)" in script
    assert '--num_processes="$NUM_PROCESSES"' in script


def test_remote_container_requires_explicit_gpu_and_current_user():
    script = _read_script("remote_lerobot_container.sh")
    assert script.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert ': "${LEROBOT_GPUS:?Set LEROBOT_GPUS to device=<gpu-id>' in script
    assert '--gpus "${LEROBOT_GPUS}"' in script
    assert '--user "$(id -u):$(id -g)"' in script
    assert "--gpus all" not in script
    assert "-e HOME=/home/wengyikun" in script
    assert "-e USER=wengyikun" in script
    assert "-e LOGNAME=wengyikun" in script
    assert "-e PYTHONPATH=/data/wengyikun/lerobot_py311/site-packages:/workspace/lerobot/src" in script
    assert "-e TRITON_CACHE_DIR=/data/wengyikun/.cache/triton" in script
    assert "-e TORCHINDUCTOR_CACHE_DIR=/data/wengyikun/.cache/torchinductor" in script
    assert "-v /data/wengyikun:/data/wengyikun" in script


def test_remote_container_rejects_all_gpu_visibility():
    result = subprocess.run(
        ["bash", str(RUN_SCRIPTS / "remote_lerobot_container.sh"), "true"],
        env={**os.environ, "LEROBOT_GPUS": "all"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "device=<gpu-id>" in result.stderr


class _TinyPI05(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = nn.Linear(2, 2)
        self.vision_tower = nn.Linear(2, 2)
        self.multi_modal_projector = nn.Linear(2, 2)
        self.gemma_expert = nn.Linear(2, 2)
        self.other = nn.Parameter(torch.ones(1))
        self.language_model.requires_grad_(False)


def test_pi05_parameter_audit_reports_frozen_language_and_trainable_visual_path():
    audit = audit_pi05_parameters(_TinyPI05())

    assert set(audit) == {
        "language_model",
        "vision_encoder",
        "multimodal_projector",
        "action_expert_other",
    }
    assert audit["language_model"]["total"] > 0
    assert audit["language_model"]["trainable"] == 0
    assert audit["vision_encoder"]["trainable"] == audit["vision_encoder"]["total"]
    assert audit["multimodal_projector"]["trainable"] == audit["multimodal_projector"]["total"]
    assert audit["action_expert_other"]["trainable"] > 0


def test_named_metric_log_preserves_six_exact_training_and_validation_keys():
    line = format_named_metrics(
        200,
        {
            "train/loss": 1.0,
            "train/gripper_loss": 2.0,
            "valid/loss": 3.0,
            "valid/gripper_loss": 4.0,
            "valid/action_mse": 5.0,
            "valid/gripper_mse": 6.0,
        },
    )
    assert line == (
        "step=200 train/loss=1 train/gripper_loss=2 valid/loss=3 "
        "valid/gripper_loss=4 valid/action_mse=5 valid/gripper_mse=6"
    )


class _ContractAccelerator:
    @contextmanager
    def autocast(self):
        yield

    def gather_for_metrics(self, tensor):
        return tensor

    def reduce(self, tensor, reduction):
        assert reduction == "sum"
        return tensor

    def unwrap_model(self, policy):
        return policy


class _FlowBoundary:
    @staticmethod
    def sample_time(batch_size, device):
        return torch.full((batch_size,), 0.5, device=device)


class _RelativePolicyBoundary(nn.Module):
    def __init__(self, config: PI05Config | ACTConfig, predicted: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.config = config
        self.model = _FlowBoundary()
        self.predicted = predicted
        self.seen_states: list[torch.Tensor] = []

    def forward(self, batch, reduction="mean", *, return_action_chunk=False, **_kwargs):
        self.seen_states.append(batch[OBS_STATE].detach().clone())
        if return_action_chunk:
            return self.predicted
        batch_size = batch[ACTION].shape[0]
        if reduction == "mean":
            return self.weight.square(), {"gripper_loss": self.weight.square().detach().item()}
        horizon = batch[ACTION].shape[1]
        return torch.ones(batch_size), {
            "loss_sum_per_sample": torch.full((batch_size,), float(horizon * 7)),
            "loss_count_per_sample": torch.full((batch_size,), horizon * 7),
            "gripper_loss_sum_per_sample": torch.full((batch_size,), float(horizon)),
            "gripper_loss_count_per_sample": torch.full((batch_size,), horizon),
        }


class _FakeTokenizer:
    def __call__(self, text, *, max_length, **_kwargs):
        batch_size = len(text)
        return {
            "input_ids": torch.zeros(batch_size, max_length, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, max_length, dtype=torch.long),
        }


def _make_relative_stats(tmp_path: Path) -> tuple[Path, dict[int, Path]]:
    increments = torch.linspace(0.001, 0.02, 60).unsqueeze(-1) * torch.arange(1, 8)
    episode = torch.cumsum(increments, dim=0)
    episode[:, 6] = torch.linspace(0.0, 0.1, 60)
    bundle = compute_relative_joint_stats_from_episodes(
        [episode],
        gripper_indices=[6],
        horizons=[16, 50],
        feature_names=[f"joint_{index}" for index in range(6)] + ["gripper"],
        source_manifest_sha256="a" * 64,
    )
    save_relative_joint_stats(bundle, tmp_path)
    return (
        tmp_path / "relative_state_q01_q99.json",
        {
            16: tmp_path / "relative_action_chunk16_q01_q99.json",
            50: tmp_path / "relative_action_chunk50_q01_q99.json",
        },
    )


def _make_real_relative_pipeline(
    policy_type: str,
    condition_on_state: bool,
    horizon: int,
    stats_root: Path,
):
    state_stats, action_stats = _make_relative_stats(stats_root)
    common = {
        "joint_representation": "relative",
        "condition_on_state": condition_on_state,
        "chunk_size": horizon,
        "n_action_steps": horizon,
        "relative_state_stats_path": str(state_stats),
        "relative_action_stats_path": str(action_stats[horizon]),
        "input_features": {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            "observation.images.camera": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)),
        },
        "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        "action_feature_names": [f"joint_{index}" for index in range(6)] + ["gripper"],
        "device": "cpu",
    }
    if policy_type == "pi05":
        config = PI05Config(joint_gripper_indices=[6], **common)
        config.validate_features()
        preprocessor, postprocessor = make_pi05_pre_post_processors(config)
    else:
        config = ACTConfig(gripper_indices=[6], **common)
        config.validate_features()
        assert config.normalization_mapping["STATE"] is NormalizationMode.QUANTILES
        assert config.normalization_mapping["ACTION"] is NormalizationMode.QUANTILES
        preprocessor, postprocessor = make_act_pre_post_processors(config)
    return config, preprocessor, postprocessor


@pytest.mark.parametrize(
    "policy_type,condition_on_state,horizon",
    [
        ("pi05", True, 50),
        ("pi05", False, 50),
        ("act", True, 16),
        ("act", False, 16),
    ],
)
def test_four_relative_policy_single_batch_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    policy_type: str,
    condition_on_state: bool,
    horizon: int,
):
    tokenizer_module = pytest.importorskip("lerobot.processor.tokenizer_processor")
    monkeypatch.setattr(tokenizer_module, "_transformers_available", True)
    monkeypatch.setattr(
        tokenizer_module,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: _FakeTokenizer()),
    )
    config, preprocessor, postprocessor = _make_real_relative_pipeline(
        policy_type,
        condition_on_state,
        horizon,
        tmp_path / policy_type,
    )
    previous = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.02]])
    current = torch.tensor([[1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 0.04]])
    raw_state = torch.stack((previous, current), dim=1)
    offsets = torch.linspace(0.0, 0.2, horizon).reshape(1, horizon, 1)
    absolute_action = current[:, None, :].expand(-1, horizon, -1).clone()
    absolute_action[..., :6] += offsets
    absolute_action[..., 6] = torch.linspace(0.01, 0.09, horizon)
    batch = {
        OBS_STATE: raw_state,
        ACTION: absolute_action,
        "task": ["grasp bread"],
        "action_is_pad": torch.zeros(1, horizon, dtype=torch.bool),
        "episode_index": torch.tensor([0]),
        "frame_index": torch.tensor([1]),
    }
    processed = preprocessor(batch)
    policy = _RelativePolicyBoundary(config, processed[ACTION].detach().clone())
    train_loss, train_output = policy(processed)

    assert processed[ACTION].shape == (1, horizon, 7)
    assert processed[OBS_STATE].shape == (1, 7)
    if condition_on_state:
        assert torch.count_nonzero(processed[OBS_STATE]) > 0
    else:
        torch.testing.assert_close(processed[OBS_STATE], torch.zeros(1, 7))
    assert torch.isfinite(train_loss)
    assert torch.isfinite(torch.tensor(train_output["gripper_loss"]))

    metrics = evaluate_offline(
        policy,
        [batch],
        preprocessor=preprocessor,
        accelerator=_ContractAccelerator(),
        action_unnormalizer=make_action_unnormalizer(postprocessor),
        seed=42,
    )
    assert set(metrics) == {
        "valid/loss",
        "valid/gripper_loss",
        "valid/action_mse",
        "valid/gripper_mse",
        "valid/joint_0_mse_rad2",
        "valid/joint_0_rmse_rad",
        "valid/joint_0_rmse_deg",
        "valid/joint_1_mse_rad2",
        "valid/joint_1_rmse_rad",
        "valid/joint_1_rmse_deg",
        "valid/joint_2_mse_rad2",
        "valid/joint_2_rmse_rad",
        "valid/joint_2_rmse_deg",
        "valid/joint_3_mse_rad2",
        "valid/joint_3_rmse_rad",
        "valid/joint_3_rmse_deg",
        "valid/joint_4_mse_rad2",
        "valid/joint_4_rmse_rad",
        "valid/joint_4_rmse_deg",
        "valid/joint_5_mse_rad2",
        "valid/joint_5_rmse_rad",
        "valid/joint_5_rmse_deg",
        "valid/gripper_rmse_m",
        "valid/gripper_rmse_mm",
    }
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert metrics["valid/action_mse"] == pytest.approx(0.0)
    assert metrics["valid/gripper_mse"] == pytest.approx(0.0)
    if not condition_on_state:
        assert all(torch.count_nonzero(state) == 0 for state in policy.seen_states[1:])
