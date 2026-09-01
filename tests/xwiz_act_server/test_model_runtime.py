from pathlib import Path

import numpy as np
import pytest
import xwiz_act_server.model_runtime as model_runtime

from xwiz_act_server.model_runtime import (
    CheckpointError,
    load_policy_config,
    validate_action_chunk,
    validate_checkpoint,
    resample_runtime_action_chunk,
    adapt_observation_to_policy,
    preprocess_observation_image,
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
    actions = np.zeros((1, 16, 19), dtype=np.float32)
    validated = validate_action_chunk(actions)
    assert validated.shape == (16, 19)
    assert validated.dtype == np.float32


def test_load_policy_config_decodes_config_json_without_training_stack(tmp_path):
    (tmp_path / "config.json").write_text('{"chunk_size": 16}')

    class FakeConfig:
        device = "cpu"

    class FakePolicy:
        config_class = FakeConfig

    def decoder(config_class, raw):
        assert config_class is FakeConfig
        assert raw == {"chunk_size": 16}
        return FakeConfig()

    config = load_policy_config(FakePolicy, tmp_path, "cuda", decoder=decoder)
    assert isinstance(config, FakeConfig)
    assert config.device == "cuda"


def test_normalize_observation_batches_state_and_converts_hwc_uint8_images_to_chw():
    assert hasattr(model_runtime, "normalize_observation")
    observation = {
        "observation.state": np.array([3.0, 5.0], dtype=np.float32),
        "observation.images.cam_high_left": np.array(
            [[[255, 0, 128], [0, 255, 64]]], dtype=np.uint8
        ),
    }
    stats = {
        "observation.state": {
            "mean": np.array([1.0, 1.0], dtype=np.float32),
            "std": np.array([2.0, 4.0], dtype=np.float32),
        },
        "observation.images.cam_high_left": {
            "mean": np.zeros((3, 1, 1), dtype=np.float32),
            "std": np.ones((3, 1, 1), dtype=np.float32),
        },
    }

    batch = model_runtime.normalize_observation(observation, stats)

    np.testing.assert_allclose(batch["observation.state"], [[1.0, 1.0]])
    assert batch["observation.images.cam_high_left"].shape == (1, 3, 1, 2)
    np.testing.assert_allclose(
        batch["observation.images.cam_high_left"][0, :, 0, 0],
        [1.0, 0.0, 128.0 / 255.0],
    )


def test_unnormalize_action_chunk_uses_checkpoint_action_mean_and_std():
    assert hasattr(model_runtime, "unnormalize_action_chunk")
    normalized = np.zeros((1, 16, 19), dtype=np.float32)
    mean = np.arange(19, dtype=np.float32)
    std = np.full(19, 2.0, dtype=np.float32)

    actions = model_runtime.unnormalize_action_chunk(
        normalized,
        {"action": {"mean": mean, "std": std}},
    )

    assert actions.shape == (16, 19)
    np.testing.assert_allclose(actions[0], mean)


def test_runtime_resampling_helper_remains_explicit_only():
    actions = np.stack([np.zeros(19), np.ones(19)], axis=0)
    converted = resample_runtime_action_chunk(actions, source_horizon=2, target_horizon=5)
    assert converted.shape == (5, 19)
    np.testing.assert_allclose(converted[0], 0.0)
    np.testing.assert_allclose(converted[-1], 1.0)


def test_runtime_adapts_wire_head_camera_name_to_checkpoint_feature_name():
    observation = {"observation.images.cam_high_left": np.zeros((2, 2, 3), dtype=np.uint8)}
    adapted = adapt_observation_to_policy(
        observation,
        {"observation.images.cam_high_right": {"shape": [3, 2, 2]}},
    )
    assert "observation.images.cam_high_right" in adapted
    assert "observation.images.cam_high_left" not in adapted


def test_head_image_preprocessing_matches_popcorn_dataset_letterbox_contract():
    image = np.full((360, 640, 3), 255, dtype=np.uint8)
    processed = preprocess_observation_image(
        image, "observation.images.cam_high_right", (3, 224, 224)
    )
    assert processed.shape == (224, 224, 3)
    assert processed.dtype == np.uint8
    np.testing.assert_array_equal(processed[0, 0], 0)
    np.testing.assert_array_equal(processed[112, 112], 255)


def test_wrist_image_is_stretched_to_square_before_resize():
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    image[:, :140] = (255, 0, 0)
    image[:, 500:] = (0, 0, 255)
    image[:, 140:500] = (0, 255, 0)

    processed = preprocess_observation_image(
        image, "observation.images.cam_hand_left", (3, 224, 224)
    )

    assert processed.shape == (224, 224, 3)
    np.testing.assert_array_equal(processed[112, 0], (255, 0, 0))


@pytest.mark.parametrize(
    "actions, message",
    [
        (np.zeros((32, 19), dtype=np.float32), "16, 19"),
        (np.zeros((1, 16, 18), dtype=np.float32), "16, 19"),
        (np.full((16, 19), np.inf, dtype=np.float32), "finite"),
    ],
)
def test_validate_action_chunk_rejects_invalid_output(actions, message):
    with pytest.raises(CheckpointError, match=message):
        validate_action_chunk(actions)
