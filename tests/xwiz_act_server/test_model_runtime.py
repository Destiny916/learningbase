from pathlib import Path

import numpy as np
import pytest

from xwiz_act_server.model_runtime import (
    CheckpointError,
    load_policy_config,
    validate_action_chunk,
    validate_checkpoint,
)


REQUIRED_FILES = {
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
}


def test_validate_checkpoint_requires_all_runtime_files(tmp_path):
    for name in REQUIRED_FILES - {"model.safetensors"}:
        (tmp_path / name).touch()
    with pytest.raises(CheckpointError, match="model.safetensors"):
        validate_checkpoint(tmp_path)


def test_validate_checkpoint_accepts_complete_directory(tmp_path):
    for name in REQUIRED_FILES:
        (tmp_path / name).touch()
    assert validate_checkpoint(tmp_path) == Path(tmp_path).resolve()


def test_validate_checkpoint_resolves_training_checkpoint_root(tmp_path):
    model_dir = tmp_path / "pretrained_model"
    model_dir.mkdir()
    for name in REQUIRED_FILES:
        (model_dir / name).touch()
    assert validate_checkpoint(tmp_path) == model_dir.resolve()


def test_validate_action_chunk_accepts_batched_model_output():
    actions = np.zeros((1, 100, 19), dtype=np.float32)
    validated = validate_action_chunk(actions)
    assert validated.shape == (100, 19)
    assert validated.dtype == np.float32


def test_load_policy_config_decodes_config_json_without_training_stack(tmp_path):
    (tmp_path / "config.json").write_text('{"chunk_size": 100}')

    class FakeConfig:
        device = "cpu"

    class FakePolicy:
        config_class = FakeConfig

    def decoder(config_class, raw):
        assert config_class is FakeConfig
        assert raw == {"chunk_size": 100}
        return FakeConfig()

    config = load_policy_config(FakePolicy, tmp_path, "cuda", decoder=decoder)
    assert isinstance(config, FakeConfig)
    assert config.device == "cuda"


@pytest.mark.parametrize(
    "actions, message",
    [
        (np.zeros((32, 19), dtype=np.float32), "100, 19"),
        (np.zeros((1, 100, 18), dtype=np.float32), "100, 19"),
        (np.full((100, 19), np.inf, dtype=np.float32), "finite"),
    ],
)
def test_validate_action_chunk_rejects_invalid_output(actions, message):
    with pytest.raises(CheckpointError, match=message):
        validate_action_chunk(actions)
