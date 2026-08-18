from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from lerobot.configs.train import TrainPipelineConfig
from lerobot.utils.constants import ACTION
from w1_simulation.inference.contract import resolve_execution_horizon as _resolve_execution_horizon
from w1_simulation.inference.direct import ActPolicyRuntime, _contract_from_config
from w1_simulation.replay.origin import OriginReplay
from w1_simulation.w1_profile import ACT_IMAGE_KEYS

pytestmark = pytest.mark.integration


def test_checkpoint_contract_accepts_prediction_horizon_larger_than_execution_horizon() -> None:
    feature = SimpleNamespace(shape=(19,))
    image = SimpleNamespace(shape=(3, 360, 640))
    config = SimpleNamespace(
        type="act",
        image_features={"observation.images.camera": image},
        input_features={"observation.state": feature, "observation.images.camera": image},
        output_features={ACTION: feature},
        chunk_size=100,
        n_action_steps=30,
    )

    contract = _contract_from_config(config)

    assert contract.prediction_horizon == 100
    assert contract.execution_horizon == 30
    assert contract.image_shapes == {"observation.images.camera": (360, 640, 3)}


@pytest.mark.parametrize("prediction_horizon,execution_horizon", [(0, 0), (30, 31)])
def test_checkpoint_contract_rejects_invalid_horizons(
    prediction_horizon: int, execution_horizon: int
) -> None:
    feature = SimpleNamespace(shape=(19,))
    image = SimpleNamespace(shape=(3, 360, 640))
    config = SimpleNamespace(
        type="act",
        image_features={"observation.images.camera": image},
        input_features={"observation.state": feature, "observation.images.camera": image},
        output_features={ACTION: feature},
        chunk_size=prediction_horizon,
        n_action_steps=execution_horizon,
    )

    with pytest.raises(ValueError, match="horizon"):
        _contract_from_config(config)


@pytest.mark.parametrize(
    "runtime_horizon,expected_horizon,expected_source",
    [(0, 30, "checkpoint_config"), (100, 100, "runtime_override")],
)
def test_runtime_execution_horizon_resolves_checkpoint_and_override_values(
    runtime_horizon: int,
    expected_horizon: int,
    expected_source: str,
) -> None:
    assert _resolve_execution_horizon(100, 30, runtime_horizon) == (
        expected_horizon,
        expected_source,
    )


@pytest.mark.parametrize("runtime_horizon", [-1, 101])
def test_runtime_execution_horizon_rejects_values_outside_prediction_chunk(
    runtime_horizon: int,
) -> None:
    with pytest.raises(ValueError, match="execution_horizon"):
        _resolve_execution_horizon(100, 30, runtime_horizon)


def test_real_checkpoint_declares_act_19d_three_image_100_step_contract(checkpoint_root) -> None:
    config = TrainPipelineConfig.from_pretrained(checkpoint_root, local_files_only=True).policy

    assert config.type == "act"
    assert tuple(config.input_features) == ("observation.state", *ACT_IMAGE_KEYS)
    assert config.input_features["observation.state"].shape == (19,)
    assert all(config.input_features[key].shape == (3, 360, 640) for key in ACT_IMAGE_KEYS)
    assert config.output_features[ACTION].shape == (19,)
    assert (config.chunk_size, config.n_action_steps) == (100, 100)


def test_real_checkpoint_cuda_forward_accepts_three_origin_images(
    checkpoint_root, origin_root, cuda_device
) -> None:
    runtime = ActPolicyRuntime(checkpoint_root, device=cuda_device)
    images, _, _ = OriginReplay(origin_root).frames[0].load_images()

    chunk, latency_ms = runtime.predict_chunk(np.zeros(19, dtype=np.float32), images)

    assert chunk.shape == (100, 19)
    assert np.isfinite(chunk).all()
    assert latency_ms > 0.0
    assert runtime.last_model_ms > 0.0


def test_checkpoint_processor_files_are_local_and_well_formed(checkpoint_root) -> None:
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        payload = json.loads((checkpoint_root / name).read_text(encoding="utf-8"))
        assert payload
