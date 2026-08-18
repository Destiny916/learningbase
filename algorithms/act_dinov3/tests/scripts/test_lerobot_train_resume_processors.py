import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.configs import FeatureType, NormalizationMode
from lerobot.scripts.lerobot_train import _normalization_processor_overrides


def test_resume_does_not_override_checkpoint_normalization_stats():
    preprocessor, postprocessor = _normalization_processor_overrides(
        resume=True,
        stats={"observation.state": {"q01": [1.0], "q99": [2.0]}},
        input_features={},
        output_features={},
        norm_map={FeatureType.STATE: NormalizationMode.MIN_MAX},
    )

    assert preprocessor == {}
    assert postprocessor == {}


def test_new_training_overrides_pretrained_processor_with_dataset_stats():
    stats = {"observation.state": {"q01": [1.0], "q99": [2.0]}}
    input_features = {"observation.state": object()}
    output_features = {"action": object()}
    norm_map = {FeatureType.STATE: NormalizationMode.MIN_MAX}

    preprocessor, postprocessor = _normalization_processor_overrides(
        resume=False,
        stats=stats,
        input_features=input_features,
        output_features=output_features,
        norm_map=norm_map,
    )

    assert preprocessor == {
        "normalizer_processor": {
            "stats": stats,
            "features": {**input_features, **output_features},
            "norm_map": norm_map,
        }
    }
    assert postprocessor == {
        "unnormalizer_processor": {
            "stats": stats,
            "features": output_features,
            "norm_map": norm_map,
        }
    }
