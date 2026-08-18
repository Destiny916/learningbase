#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Multi-GPU Training Tests

This module tests multi-GPU training functionality with accelerate.
These tests are designed to run on machines with 2+ GPUs and are executed
in the nightly CI workflow.

The tests automatically generate accelerate configs and launch training
with subprocess to properly test the distributed training environment.
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path

import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.datasets.lerobot_dataset import LeRobotDataset


RELATIVE_VALIDATION_KEYS = {
    "train/loss",
    "train/gripper_loss",
    "valid/loss",
    "valid/gripper_loss",
    "valid/action_mse",
    "valid/gripper_mse",
}


def _parse_relative_validation_events(output: str) -> dict[int, dict[str, float]]:
    events: dict[int, dict[str, float]] = {}
    for line in output.splitlines():
        match = re.search(r"step[ =](\d+).*?((?:train|valid)/[^\n]+)", line)
        if match is None:
            continue
        step = int(match.group(1))
        metrics = events.setdefault(step, {})
        for key, value in re.findall(r"((?:train|valid)/[a-z_]+)=([-+0-9.eE]+)", line):
            metrics[key] = float(value)
    return events


def _count_complete_relative_validation_events(output: str) -> dict[int, int]:
    counts: dict[int, int] = {}
    for line in output.splitlines():
        step_match = re.search(r"step[ =](\d+)", line)
        if step_match is None:
            continue
        keys = {key for key, _value in re.findall(r"((?:train|valid)/[a-z_]+)=([-+0-9.eE]+)", line)}
        if RELATIVE_VALIDATION_KEYS.issubset(keys):
            step = int(step_match.group(1))
            counts[step] = counts.get(step, 0) + 1
    return counts


def test_parse_relative_validation_events_requires_all_six_metric_keys():
    output = """
step=2 train/loss=0.2 train/gripper_loss=0.3 valid/loss=0.4 valid/gripper_loss=0.5 valid/action_mse=0.6 valid/gripper_mse=0.7
step=4 train/loss=0.1 train/gripper_loss=0.2 valid/loss=0.3 valid/gripper_loss=0.4 valid/action_mse=0.5 valid/gripper_mse=0.6
"""
    events = _parse_relative_validation_events(output)
    assert set(events) == {2, 4}
    assert set(events[2]) == RELATIVE_VALIDATION_KEYS
    assert set(events[4]) == RELATIVE_VALIDATION_KEYS


def test_count_complete_validation_events_does_not_hide_duplicate_logs():
    output = """
step=2 train/loss=0.2 train/gripper_loss=0.3 valid/loss=0.4 valid/gripper_loss=0.5 valid/action_mse=0.6 valid/gripper_mse=0.7
step=2 train/loss=0.2 train/gripper_loss=0.3 valid/loss=0.4 valid/gripper_loss=0.5 valid/action_mse=0.6 valid/gripper_mse=0.7
step=4 train/loss=0.1 train/gripper_loss=0.2 valid/loss=0.3 valid/gripper_loss=0.4 valid/action_mse=0.5 valid/gripper_mse=0.6
"""
    assert _count_complete_relative_validation_events(output) == {2: 2, 4: 1}


def get_num_available_gpus():
    """Returns the number of available GPUs."""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def download_dataset(repo_id, episodes):
    """
    Pre-download dataset to avoid race conditions in multi-GPU training.

    Args:
        repo_id: HuggingFace dataset repository ID
        episodes: List of episode indices to download
    """
    # Simply instantiating the dataset will download it
    _ = LeRobotDataset(repo_id, episodes=episodes)
    print(f"Dataset {repo_id} downloaded successfully")


def _write_multi_gpu_config(f, num_processes):
    f.write("compute_environment: LOCAL_MACHINE\n")
    f.write("distributed_type: MULTI_GPU\n")
    f.write("mixed_precision: 'no'\n")
    f.write(f"num_processes: {num_processes}\n")
    f.write("use_cpu: false\n")
    f.write("gpu_ids: all\n")
    f.write("downcast_bf16: 'no'\n")
    f.write("machine_rank: 0\n")
    f.write("main_training_function: main\n")
    f.write("num_machines: 1\n")
    f.write("rdzv_backend: static\n")
    f.write("same_network: true\n")


def _write_fsdp_config(f, num_processes):
    # FSDP1 with FULL_SHARD (ZeRO-3-equivalent) and FULL_STATE_DICT, matching
    # docs/source/multi_gpu_training.mdx. ACT's repeated transformer blocks are the wrap units;
    # fsdp_use_orig_params is required because LeRobot builds the optimizer before prepare().
    f.write("compute_environment: LOCAL_MACHINE\n")
    f.write("distributed_type: FSDP\n")
    f.write("mixed_precision: 'no'\n")
    f.write(f"num_processes: {num_processes}\n")
    f.write("use_cpu: false\n")
    f.write("gpu_ids: all\n")
    f.write("machine_rank: 0\n")
    f.write("main_training_function: main\n")
    f.write("num_machines: 1\n")
    f.write("rdzv_backend: static\n")
    f.write("same_network: true\n")
    f.write("fsdp_config:\n")
    f.write("  fsdp_version: 1\n")
    f.write("  fsdp_sharding_strategy: FULL_SHARD\n")
    f.write("  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP\n")
    f.write("  fsdp_transformer_layer_cls_to_wrap: ACTEncoderLayer,ACTDecoderLayer\n")
    f.write("  fsdp_use_orig_params: true\n")
    f.write("  fsdp_state_dict_type: FULL_STATE_DICT\n")


def run_accelerate_training(config_args, num_processes=4, temp_dir=None, distributed_type="MULTI_GPU"):
    """
    Helper function to run training with accelerate launch.

    Args:
        config_args: List of config arguments to pass to lerobot_train.py
        num_processes: Number of processes (GPUs) to use
        temp_dir: Temporary directory for outputs
        distributed_type: "MULTI_GPU" (DDP) or "FSDP" — selects the generated accelerate config.

    Returns:
        subprocess.CompletedProcess result
    """

    config_path = Path(temp_dir) / "accelerate_config.yaml"

    # Write YAML config
    with open(config_path, "w") as f:
        if distributed_type == "FSDP":
            _write_fsdp_config(f, num_processes)
        else:
            _write_multi_gpu_config(f, num_processes)

    cmd = [
        "accelerate",
        "launch",
        "--config_file",
        str(config_path),
        "-m",
        "lerobot.scripts.lerobot_train",
    ] + config_args

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, range(num_processes)))},
    )

    return result


@pytest.mark.skipif(
    get_num_available_gpus() < 2,
    reason="Multi-GPU tests require at least 2 GPUs",
)
class TestMultiGPUTraining:
    """Test suite for multi-GPU training functionality."""

    def test_basic_multi_gpu_training(self):
        """
        Test that basic multi-GPU training runs successfully.
        Verifies that the training completes without errors.
        """
        # Pre-download dataset to avoid race conditions
        download_dataset("lerobot/pusht", episodes=[0])

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "outputs"

            config_args = [
                "--dataset.repo_id=lerobot/pusht",
                "--dataset.episodes=[0]",
                "--policy.type=act",
                "--policy.device=cuda",
                "--policy.push_to_hub=false",
                f"--output_dir={output_dir}",
                "--batch_size=4",
                "--steps=10",
                "--env_eval_freq=-1",
                "--log_freq=5",
                "--save_freq=10",
                "--seed=42",
                "--num_workers=0",
            ]

            result = run_accelerate_training(config_args, num_processes=4, temp_dir=temp_dir)

            # Check that training completed successfully
            assert result.returncode == 0, (
                f"Multi-GPU training failed with return code {result.returncode}\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )

            # Verify checkpoint was saved
            checkpoints_dir = output_dir / "checkpoints"
            assert checkpoints_dir.exists(), "Checkpoints directory was not created"

            # Verify that training completed
            assert "End of training" in result.stdout or "End of training" in result.stderr

    def test_checkpoint_saving_multi_gpu(self):
        """
        Test that checkpoints are correctly saved during multi-GPU training.
        Only the main process (rank 0) should save checkpoints.
        """
        # Pre-download dataset to avoid race conditions
        download_dataset("lerobot/pusht", episodes=[0])

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "outputs"

            config_args = [
                "--dataset.repo_id=lerobot/pusht",
                "--dataset.episodes=[0]",
                "--policy.type=act",
                "--policy.device=cuda",
                "--policy.push_to_hub=false",
                f"--output_dir={output_dir}",
                "--batch_size=4",
                "--steps=20",
                "--env_eval_freq=-1",
                "--log_freq=5",
                "--save_freq=10",
                "--seed=42",
                "--num_workers=0",
            ]

            result = run_accelerate_training(config_args, num_processes=2, temp_dir=temp_dir)

            assert result.returncode == 0, (
                f"Training failed:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
            )

            # Verify checkpoint directory exists
            checkpoints_dir = output_dir / "checkpoints"
            assert checkpoints_dir.exists(), "Checkpoints directory not created"

            # Count checkpoint directories (should have checkpoint at step 10 and 20)
            checkpoint_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir()]
            assert len(checkpoint_dirs) >= 1, f"Expected at least 1 checkpoint, found {len(checkpoint_dirs)}"

            # Verify checkpoint contents
            for checkpoint_dir in checkpoint_dirs:
                # Check for model files
                model_files = list(checkpoint_dir.rglob("*.safetensors"))
                assert len(model_files) > 0, f"No model files in checkpoint {checkpoint_dir}"

                # Check for training state
                training_state_dir = checkpoint_dir / "training_state"
                assert training_state_dir.exists(), f"No training state in checkpoint {checkpoint_dir}"

                # Verify optimizer state exists
                optimizer_state = training_state_dir / "optimizer_state.safetensors"
                assert optimizer_state.exists(), f"No optimizer state in checkpoint {checkpoint_dir}"

    def test_fsdp_optimizer_save_and_resume(self):
        """
        Test that FSDP saves the (gathered) optimizer state and can resume from it.

        Trains a few steps under FSDP, verifies the gathered optimizer state is written next to the
        rest of the training state, then resumes from the checkpoint for more steps and checks it
        completes without shape/key errors in the FSDP optimizer load path.
        """
        # Pre-download dataset to avoid race conditions
        download_dataset("lerobot/pusht", episodes=[0])

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "outputs"

            config_args = [
                "--dataset.repo_id=lerobot/pusht",
                "--dataset.episodes=[0]",
                "--policy.type=act",
                "--policy.device=cuda",
                "--policy.push_to_hub=false",
                f"--output_dir={output_dir}",
                "--batch_size=4",
                "--steps=10",
                "--env_eval_freq=-1",
                "--log_freq=5",
                "--save_freq=10",
                "--seed=42",
                "--num_workers=0",
            ]

            result = run_accelerate_training(
                config_args, num_processes=2, temp_dir=temp_dir, distributed_type="FSDP"
            )
            assert result.returncode == 0, (
                f"FSDP training failed:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
            )

            # The gathered optimizer state must be written under FSDP (proves the save collective ran),
            # in the same safetensors format as single-GPU training.
            training_state_dir = output_dir / "checkpoints" / "last" / "training_state"
            optimizer_state = training_state_dir / "optimizer_state.safetensors"
            optimizer_param_groups = training_state_dir / "optimizer_param_groups.json"
            assert optimizer_state.exists(), f"FSDP optimizer state not saved in {training_state_dir}"
            assert optimizer_param_groups.exists(), (
                f"FSDP optimizer param groups not saved in {training_state_dir}"
            )

            # Resume from the checkpoint for more steps. A successful run proves load_fsdp_optimizer
            # accepts the saved state and reshards it without shape/key errors.
            resume_config = output_dir / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
            resume_args = [
                f"--config_path={resume_config}",
                "--resume=true",
                "--steps=20",
            ]
            resume_result = run_accelerate_training(
                resume_args, num_processes=2, temp_dir=temp_dir, distributed_type="FSDP"
            )
            assert resume_result.returncode == 0, (
                f"FSDP resume failed:\nSTDOUT:\n{resume_result.stdout}\n\nSTDERR:\n{resume_result.stderr}"
            )
            assert "End of training" in resume_result.stdout or "End of training" in resume_result.stderr


@pytest.mark.skipif(
    os.environ.get("LEROBOT_RUN_RELATIVE_VALIDATION_MULTI_GPU") != "1",
    reason="Nightly-only 2-GPU relative validation smoke",
)
def test_relative_validation_four_step_single_and_multi_gpu_aggregation(tmp_path):
    """Run 4 optimizer steps and compare exact validation reductions at steps 2 and 4."""
    if get_num_available_gpus() < 2:
        pytest.skip("requires at least two visible CUDA devices")

    repo_root = Path(__file__).resolve().parents[2]
    launcher = repo_root / "run_scripts" / os.environ.get(
        "LEROBOT_RELATIVE_VALIDATION_LAUNCHER", "launch_pi05_relative_state_0714.sh"
    )
    common_env = {
        **os.environ,
        "STEPS": "4",
        "EVAL_STEPS": "2",
        "LOG_FREQ": "2",
        "SAVE_FREQ": "2000",
        "NUM_WORKERS": "0",
        "WANDB_ENABLE": "false",
    }

    def run(world_size: int, batch_size: int, visible_devices: str, output_dir: Path):
        env = {
            **common_env,
            "NUM_PROCESSES": str(world_size),
            "BATCH_SIZE": str(batch_size),
            "CUDA_VISIBLE_DEVICES": visible_devices,
            "OUTPUT_DIR": str(output_dir),
            "JOB_NAME": f"relative_validation_smoke_{world_size}gpu",
        }
        return subprocess.run(
            ["bash", str(launcher)],
            cwd=repo_root,
            env=env,
            text=True,
            capture_output=True,
            timeout=1800,
        )

    single = run(1, 2, "0", tmp_path / "single")
    assert single.returncode == 0, f"single-GPU smoke failed:\n{single.stdout}\n{single.stderr}"
    multi = run(2, 1, "0,1", tmp_path / "multi")
    assert multi.returncode == 0, f"two-GPU smoke failed:\n{multi.stdout}\n{multi.stderr}"

    single_events = _parse_relative_validation_events(single.stdout + single.stderr)
    multi_events = _parse_relative_validation_events(multi.stdout + multi.stderr)
    assert _count_complete_relative_validation_events(single.stdout + single.stderr) == {2: 1, 4: 1}
    assert _count_complete_relative_validation_events(multi.stdout + multi.stderr) == {2: 1, 4: 1}
    for step in (2, 4):
        assert set(single_events[step]) == RELATIVE_VALIDATION_KEYS
        assert set(multi_events[step]) == RELATIVE_VALIDATION_KEYS
        for key in (
            "valid/loss",
            "valid/gripper_loss",
            "valid/action_mse",
            "valid/gripper_mse",
        ):
            assert multi_events[step][key] == pytest.approx(single_events[step][key], rel=1e-5, abs=1e-7)
