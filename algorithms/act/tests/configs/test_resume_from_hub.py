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

import pytest

import lerobot.configs.train as tc
from lerobot.common.train_utils import load_training_gradient_accumulation_steps, save_training_step
from lerobot.configs.train import TrainPipelineConfig
from lerobot.utils.constants import TRAINING_STATE_DIR


class _FakeHTTPError(tc.HfHubHTTPError):
    """HfHubHTTPError that can be raised without a real HTTP response object."""

    def __init__(self):
        pass


def test_from_pretrained_falls_back_to_latest_checkpoint_config(tmp_path, monkeypatch):
    """A Hub repo with no root train_config.json (an interrupted run that only pushed
    checkpoints/) resolves via the latest checkpoint's config."""
    # A real train_config.json written by save_pretrained, to be returned by the fallback.
    parsed = tc.draccus.parse(TrainPipelineConfig, args=["--dataset.repo_id", "u/d"])
    cfg_file = tmp_path / "train_config.json"
    parsed._save_pretrained(tmp_path)
    assert cfg_file.is_file()

    calls = []

    def fake_hf_hub_download(filename=None, **kwargs):
        calls.append(filename)
        if filename == "train_config.json":
            raise _FakeHTTPError()  # no root config
        if filename == "checkpoints/000010/pretrained_model/train_config.json":
            return str(cfg_file)
        raise AssertionError(f"unexpected filename {filename}")

    monkeypatch.setattr(tc, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(
        tc, "find_latest_hub_checkpoint", lambda repo_id, token=None, revision=None: "checkpoints/000010"
    )

    loaded = TrainPipelineConfig.from_pretrained("user/interrupted-run")
    assert loaded.dataset.repo_id == "u/d"
    # Tried the root config first, then fell back to the latest checkpoint's config.
    assert calls == ["train_config.json", "checkpoints/000010/pretrained_model/train_config.json"]


def test_from_pretrained_raises_when_no_root_config_and_no_checkpoints(monkeypatch):
    """No root config AND no checkpoints → a clear FileNotFoundError, not the raw HTTP error."""

    def fake_hf_hub_download(filename=None, **kwargs):
        raise _FakeHTTPError()

    monkeypatch.setattr(tc, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(tc, "find_latest_hub_checkpoint", lambda repo_id, token=None, revision=None: None)

    with pytest.raises(FileNotFoundError, match="train_config.json not found"):
        TrainPipelineConfig.from_pretrained("user/empty-repo")


def test_train_config_accepts_gradient_accumulation_steps():
    parsed = tc.draccus.parse(
        TrainPipelineConfig,
        args=["--dataset.repo_id", "u/d", "--gradient_accumulation_steps", "2"],
    )

    assert parsed.gradient_accumulation_steps == 2


def test_training_step_records_gradient_accumulation_steps(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    training_state_dir = checkpoint_dir / TRAINING_STATE_DIR
    training_state_dir.mkdir(parents=True)
    save_training_step(7, training_state_dir, num_processes=2, batch_size=4, gradient_accumulation_steps=2)

    assert load_training_gradient_accumulation_steps(checkpoint_dir) == 2
