import json

import numpy as np
import pytest

from xwiz_act_server.checkpoint_converter import (
    CheckpointConversionError,
    convert_checkpoint,
    resample_action_chunk,
)


def test_resample_action_chunk_preserves_endpoints_and_target_shape():
    source = np.stack([np.zeros(19), np.ones(19)], axis=0)

    converted = resample_action_chunk(source, target_horizon=5)

    assert converted.shape == (5, 19)
    np.testing.assert_allclose(converted[0], 0.0)
    np.testing.assert_allclose(converted[-1], 1.0)


def test_convert_checkpoint_writes_sidecar_without_mutating_source(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "type": "act",
                "chunk_size": 16,
                "n_action_steps": 16,
                "input_features": {
                    "observation.state": {"shape": [19]},
                    "observation.images.cam_high_right": {"shape": [3, 224, 224]},
                    "observation.images.cam_hand_left": {"shape": [3, 224, 224]},
                    "observation.images.cam_hand_right": {"shape": [3, 224, 224]},
                },
                "output_features": {"action": {"shape": [19]}},
            }
        )
    )
    for name in (
        "model.safetensors",
        "policy_postprocessor.json",
        "policy_preprocessor_step_3_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    ):
        (source / name).write_bytes(b"fixture")
    (source / "policy_preprocessor.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "registry_name": "normalizer_processor",
                        "config": {
                            "norm_map": {
                                "VISUAL": "MEAN_STD",
                                "STATE": "MEAN_STD",
                                "ACTION": "MEAN_STD",
                            }
                        },
                    }
                ]
            }
        )
    )

    result = convert_checkpoint(source, target, target_horizon=100)

    assert result == target
    manifest = json.loads((target / "xwiz_conversion.json").read_text())
    assert manifest["source_horizon"] == 16
    assert manifest["target_horizon"] == 100
    assert manifest["model_type"] == "act"
    assert manifest["action_semantics"] == "absolute"
    assert manifest["hand_contract"]["joint_order"] == [
        "T_MCP", "T_CMC_YAW", "IF_MCP_PITCH", "MF_MCP_PITCH", "RF_MCP_PITCH", "LF_MCP_PITCH",
    ]
    assert manifest["hand_contract"]["default_pose"] == [0.0, 70.0, 0.0, 0.0, 0.0, 0.0]
    assert manifest["normalization"]["mode"] == "MEAN_STD"
    assert json.loads((source / "config.json").read_text())["chunk_size"] == 16
    converted_config = json.loads((target / "config.json").read_text())
    assert "type" not in converted_config
    assert "camera_keys" not in converted_config


def test_convert_checkpoint_rejects_dinov3(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"type": "act_dinov3", "chunk_size": 16, "n_action_steps": 16})
    )

    with pytest.raises(CheckpointConversionError, match="act_dinov3"):
        convert_checkpoint(source, tmp_path / "target")
