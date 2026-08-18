from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import get_args, get_type_hints
from unittest.mock import Mock

import draccus
import pytest

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets import factory
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.transforms import ImageTransformsConfig
from lerobot.utils.feature_utils import dataset_to_policy_features


def _policy(*, observation_delta_indices=None, action_delta_indices=None):
    policy = Mock()
    policy.type = "test"
    policy.pretrained_path = None
    policy.observation_delta_indices = observation_delta_indices
    policy.action_delta_indices = action_delta_indices
    policy.reward_delta_indices = None
    policy.get_optimizer_preset.return_value = Mock()
    policy.get_scheduler_preset.return_value = Mock()
    return policy


def _config(tmp_path: Path, **kwargs) -> TrainPipelineConfig:
    defaults = {
        "dataset": DatasetConfig(repo_id="local/train", root=str(tmp_path / "train")),
        "policy": _policy(),
        "output_dir": tmp_path / "output",
    }
    defaults.update(kwargs)
    return TrainPipelineConfig(**defaults)


def _features(
    *,
    fps: int = 30,
    state_shape=(2,),
    state_names=None,
    camera_shape=(8, 12, 3),
    camera_names=None,
    camera_key="observation.images.main",
    camera_depth=False,
):
    del fps
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": state_shape,
            "names": state_names or ["s0", "s1"],
        },
        "action": {"dtype": "float32", "shape": (2,), "names": ["a0", "a1"]},
        camera_key: {
            "dtype": "image",
            "shape": camera_shape,
            "names": camera_names or ["height", "width", "channels"],
            "info": {"is_depth_map": camera_depth},
        },
        "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
        "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
    }


def _dataset(
    root: Path,
    *,
    fps=30,
    features=None,
    episodes=None,
    stats=None,
    transforms=None,
    depth_output_unit="mm",
):
    features = deepcopy(features or _features())
    camera_keys = [key for key, value in features.items() if value["dtype"] in ("image", "video")]
    depth_keys = [
        key for key in camera_keys if features[key].get("info", {}).get("is_depth_map", False)
    ]
    meta = SimpleNamespace(
        root=root,
        fps=fps,
        features=features,
        camera_keys=camera_keys,
        depth_keys=depth_keys,
        stats=stats if stats is not None else {"sentinel": str(root)},
        episodes={"tasks": [["task"] for _ in range(200)]},
    )
    return SimpleNamespace(
        root=root,
        meta=meta,
        episodes=episodes,
        num_episodes=200 if episodes is None else len(episodes),
        image_transforms=transforms,
        depth_output_unit=depth_output_unit,
    )


def test_make_dataset_return_annotation_includes_streaming_dataset():
    return_type = get_type_hints(factory.make_dataset)["return"]

    assert StreamingLeRobotDataset in get_args(return_type)
    assert "StreamingLeRobotDataset" in factory.make_dataset.__doc__


@pytest.mark.parametrize("dtype", ["image", "video"])
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_dataset_to_policy_features_infers_hwc_when_visual_names_are_none(dtype, channels):
    features = {
        "observation.images.main": {
            "dtype": dtype,
            "shape": (48, 64, channels),
            "names": None,
        }
    }

    assert dataset_to_policy_features(features) == {
        "observation.images.main": PolicyFeature(
            type=FeatureType.VISUAL, shape=(channels, 48, 64)
        )
    }


@pytest.mark.parametrize("dtype", ["image", "video"])
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_dataset_to_policy_features_preserves_chw_when_visual_names_are_none(dtype, channels):
    features = {
        "observation.images.main": {
            "dtype": dtype,
            "shape": (channels, 48, 64),
            "names": None,
        }
    }

    assert dataset_to_policy_features(features) == {
        "observation.images.main": PolicyFeature(
            type=FeatureType.VISUAL, shape=(channels, 48, 64)
        )
    }


@pytest.mark.parametrize("shape", [(3, 48, 3), (8, 48, 64)])
def test_dataset_to_policy_features_rejects_ambiguous_visual_shape_without_names(shape):
    features = {
        "observation.images.main": {
            "dtype": "video",
            "shape": shape,
            "names": None,
        }
    }

    with pytest.raises(ValueError, match="observation.images.main.*names=None.*shape=.*ambiguous"):
        dataset_to_policy_features(features)


@pytest.mark.parametrize(("field", "value"), [("eval_steps", -1), ("max_eval_samples", -1), ("validation_seed", -1)])
def test_validation_numeric_fields_must_be_non_negative(tmp_path, field, value):
    cfg = _config(tmp_path, **{field: value})

    with pytest.raises(ValueError, match=field):
        cfg.validate()


def test_validation_dataset_and_internal_split_are_mutually_exclusive(tmp_path):
    cfg = _config(
        tmp_path,
        dataset=DatasetConfig(repo_id="local/train", root=str(tmp_path / "train"), eval_split=0.1),
        validation_dataset=DatasetConfig(repo_id="local/test", root=str(tmp_path / "test")),
    )

    with pytest.raises(ValueError, match="validation_dataset.*eval_split"):
        cfg.validate()


def test_eval_steps_requires_validation_source(tmp_path):
    cfg = _config(tmp_path, eval_steps=1)

    with pytest.raises(ValueError, match="eval_steps.*validation_dataset.*eval_split"):
        cfg.validate()


@pytest.mark.parametrize(
    ("validation_overrides", "message"),
    [({"eval_split": 0.1}, "validation_dataset.eval_split"), ({"streaming": True}, "streaming")],
)
def test_validation_dataset_rejects_split_and_streaming(tmp_path, validation_overrides, message):
    cfg = _config(
        tmp_path,
        validation_dataset=DatasetConfig(
            repo_id="local/test", root=str(tmp_path / "test"), **validation_overrides
        ),
    )

    with pytest.raises((ValueError, NotImplementedError), match=message):
        cfg.validate()


def test_validation_dataset_draccus_json_and_cli_roundtrip(tmp_path):
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="local/train", root=str(tmp_path / "train")),
        validation_dataset=DatasetConfig(
            repo_id="local/test",
            root=str(tmp_path / "test"),
            episodes=[1, 4],
            revision="rev-test",
            video_backend="pyav",
        ),
        validation_seed=17,
    )
    payload = cfg.to_dict()

    with draccus.config_type("json"):
        decoded = draccus.decode(TrainPipelineConfig, payload)
        parsed = draccus.parse(
            TrainPipelineConfig,
            args=[
                "--dataset.repo_id",
                "local/train",
                "--validation_dataset.repo_id",
                "local/test",
                "--validation_dataset.root",
                str(tmp_path / "test"),
                "--validation_seed",
                "23",
            ],
        )

    assert decoded.validation_dataset.repo_id == "local/test"
    assert decoded.validation_dataset.root == str(tmp_path / "test")
    assert decoded.validation_dataset.episodes == [1, 4]
    assert decoded.validation_dataset.revision == "rev-test"
    assert decoded.validation_dataset.video_backend == "pyav"
    assert decoded.validation_seed == 17
    assert parsed.validation_dataset.repo_id == "local/test"
    assert parsed.validation_dataset.root == str(tmp_path / "test")
    assert parsed.validation_seed == 23


def test_legacy_json_defaults_validation_fields(tmp_path):
    config_path = tmp_path / "train_config.json"
    config_path.write_text('{"dataset": {"repo_id": "local/train"}}')

    loaded = TrainPipelineConfig.from_pretrained(config_path)

    assert loaded.validation_dataset is None
    assert loaded.validation_seed == 42


def test_external_validation_uses_full_config_and_train_deltas(monkeypatch, tmp_path):
    train = _dataset(tmp_path / "train-resolved", fps=10, episodes=list(range(143)), transforms=Mock())
    test = _dataset(tmp_path / "test-resolved", fps=10, episodes=list(range(16)))
    calls = []

    monkeypatch.setattr(factory, "make_dataset", lambda cfg: train)
    monkeypatch.setattr(factory, "LeRobotDatasetMetadata", lambda *args, **kwargs: test.meta)

    def fake_dataset(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        test.image_transforms = kwargs["image_transforms"]
        test.delta_timestamps = kwargs["delta_timestamps"]
        return test

    monkeypatch.setattr(factory, "LeRobotDataset", fake_dataset)
    cfg = _config(
        tmp_path,
        policy=_policy(observation_delta_indices=[-1, 0], action_delta_indices=[0, 1]),
        dataset=DatasetConfig(
            repo_id="local/train",
            root=str(tmp_path / "train"),
            image_transforms=ImageTransformsConfig(enable=True),
        ),
        validation_dataset=DatasetConfig(
            repo_id="local/test",
            root=str(tmp_path / "test"),
            episodes=list(range(16)),
            revision="test-revision",
            video_backend="pyav",
            depth_output_unit="mm",
        ),
    )

    actual_train, actual_test = factory.make_train_eval_datasets(cfg)

    assert actual_train is train
    assert actual_test is test
    assert train.episodes == list(range(143))
    assert test.episodes == list(range(16))
    assert train.image_transforms is not None
    assert test.image_transforms is None
    assert calls == [
        (
            "local/test",
            {
                "root": str(tmp_path / "test"),
                "episodes": list(range(16)),
                "delta_timestamps": {
                    "observation.state": [-0.1, 0.0],
                    "action": [0.0, 0.1],
                    "observation.images.main": [-0.1, 0.0],
                },
                "image_transforms": None,
                "revision": "test-revision",
                "video_backend": "pyav",
                "return_uint8": True,
                "depth_output_unit": "mm",
                "tolerance_s": cfg.tolerance_s,
            },
        )
    ]


def test_external_validation_preserves_each_dataset_stats(monkeypatch, tmp_path):
    train_stats = {"action": {"mean": "train"}}
    test_stats = {"action": {"mean": "test"}}
    train = _dataset(tmp_path / "train", stats=train_stats)
    test = _dataset(tmp_path / "test", stats=test_stats)
    monkeypatch.setattr(factory, "make_dataset", lambda cfg: train)
    monkeypatch.setattr(factory, "LeRobotDatasetMetadata", lambda *args, **kwargs: test.meta)
    monkeypatch.setattr(factory, "LeRobotDataset", lambda *args, **kwargs: test)
    cfg = _config(
        tmp_path,
        dataset=DatasetConfig(repo_id="local/train", root=str(tmp_path / "train"), use_imagenet_stats=False),
        validation_dataset=DatasetConfig(
            repo_id="local/test", root=str(tmp_path / "test"), use_imagenet_stats=False
        ),
    )

    actual_train, actual_test = factory.make_train_eval_datasets(cfg)

    assert actual_train.meta.stats is train_stats
    assert actual_test.meta.stats is test_stats


@pytest.mark.parametrize("use_symlink", [False, True], ids=["same-root", "symlink"])
def test_external_validation_rejects_same_resolved_root(monkeypatch, tmp_path, use_symlink):
    root = tmp_path / "dataset"
    root.mkdir()
    alias = root
    if use_symlink:
        alias = tmp_path / "alias"
        alias.symlink_to(root, target_is_directory=True)
    train = _dataset(root)
    test = _dataset(alias)
    monkeypatch.setattr(factory, "make_dataset", lambda cfg: train)
    metadata_constructor = Mock(return_value=test.meta)
    dataset_constructor = Mock(return_value=test)
    monkeypatch.setattr(factory, "LeRobotDatasetMetadata", metadata_constructor)
    monkeypatch.setattr(factory, "LeRobotDataset", dataset_constructor)
    cfg = _config(
        tmp_path,
        dataset=DatasetConfig(repo_id="local/train", root=str(root)),
        validation_dataset=DatasetConfig(repo_id="local/test", root=str(alias)),
    )

    with pytest.raises(ValueError, match="resolved root.*train=.*test="):
        factory.make_train_eval_datasets(cfg)

    metadata_constructor.assert_not_called()
    dataset_constructor.assert_not_called()


def _assert_contract_rejected(monkeypatch, tmp_path, train, test, message):
    monkeypatch.setattr(factory, "make_dataset", lambda cfg: train)
    monkeypatch.setattr(factory, "LeRobotDatasetMetadata", lambda *args, **kwargs: test.meta)
    dataset_constructor = Mock(return_value=test)
    monkeypatch.setattr(factory, "LeRobotDataset", dataset_constructor)
    cfg = _config(
        tmp_path,
        dataset=DatasetConfig(repo_id="local/train", root=str(tmp_path / "train")),
        validation_dataset=DatasetConfig(
            repo_id="local/test",
            root=str(tmp_path / "test"),
            depth_output_unit=test.depth_output_unit,
        ),
    )

    with pytest.raises(ValueError, match=message) as exc_info:
        factory.make_train_eval_datasets(cfg)

    assert "train=" in str(exc_info.value)
    assert "test=" in str(exc_info.value)
    dataset_constructor.assert_not_called()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda ds: setattr(ds.meta, "fps", 15), "fps"),
        (lambda ds: ds.meta.features.pop("episode_index"), "episode_index"),
        (lambda ds: ds.meta.features.pop("frame_index"), "frame_index"),
        (lambda ds: ds.meta.features.pop("action"), "action"),
        (lambda ds: ds.meta.features.pop("observation.state"), "observation.state"),
        (
            lambda ds: ds.meta.features["action"].update(dtype="float64"),
            "action.dtype",
        ),
        (
            lambda ds: ds.meta.features["observation.state"].update(shape=(3,)),
            "observation.state.shape",
        ),
        (
            lambda ds: ds.meta.features["observation.state"].update(names=["other0", "other1"]),
            "observation.state.names",
        ),
        (
            lambda ds: ds.meta.features["episode_index"].update(dtype="int32"),
            "episode_index.dtype",
        ),
        (
            lambda ds: ds.meta.features["episode_index"].update(shape=(2,)),
            "episode_index.shape",
        ),
        (
            lambda ds: ds.meta.features["episode_index"].update(names=["episode"]),
            "episode_index.names",
        ),
        (
            lambda ds: ds.meta.features["frame_index"].update(dtype="int32"),
            "frame_index.dtype",
        ),
        (
            lambda ds: ds.meta.features["frame_index"].update(shape=(2,)),
            "frame_index.shape",
        ),
        (
            lambda ds: ds.meta.features["frame_index"].update(names=["frame"]),
            "frame_index.names",
        ),
        (
            lambda ds: (
                ds.meta.features.__setitem__(
                    "observation.images.other", ds.meta.features.pop("observation.images.main")
                ),
                setattr(ds.meta, "camera_keys", ["observation.images.other"]),
            ),
            "camera_keys",
        ),
        (
            lambda ds: ds.meta.features["observation.images.main"].update(shape=(8, 12, 1)),
            "policy features",
        ),
        (
            lambda ds: (
                ds.meta.features["observation.images.main"]["info"].update(is_depth_map=True),
                ds.meta.depth_keys.append("observation.images.main"),
            ),
            "observation.images.main.is_depth_map",
        ),
        (
            lambda ds: ds.meta.features.__setitem__(
                "language", {"dtype": "string", "shape": (1,), "names": None}
            ),
            "feature keys",
        ),
        (
            lambda ds: ds.meta.features["observation.environment_state"].update(shape=(3,)),
            "policy features",
        ),
    ],
    ids=[
        "fps",
        "episode-index",
        "frame-index",
        "action-missing",
        "state-missing",
        "action-dtype",
        "state-shape",
        "state-names",
        "episode-index-dtype",
        "episode-index-shape",
        "episode-index-names",
        "frame-index-dtype",
        "frame-index-shape",
        "frame-index-names",
        "camera-keys",
        "camera-shape",
        "camera-depth",
        "feature-keys",
        "policy-visible",
    ],
)
def test_external_validation_contract_mismatches_fail_fast(monkeypatch, tmp_path, mutation, message):
    features = _features()
    features["observation.environment_state"] = {
        "dtype": "float32",
        "shape": (2,),
        "names": ["e0", "e1"],
    }
    train = _dataset(tmp_path / "train", features=features)
    test = _dataset(tmp_path / "test", features=features)
    mutation(test)

    _assert_contract_rejected(monkeypatch, tmp_path, train, test, message)


def test_external_validation_rejects_different_depth_output_units(monkeypatch, tmp_path):
    features = _features(camera_shape=(8, 12, 1), camera_depth=True)
    train = _dataset(tmp_path / "train", features=features, depth_output_unit="mm")
    test = _dataset(tmp_path / "test", features=features, depth_output_unit="m")

    _assert_contract_rejected(monkeypatch, tmp_path, train, test, "depth_output_unit")


def test_external_validation_allows_different_depth_output_units_without_depth(monkeypatch, tmp_path):
    train = _dataset(tmp_path / "train", depth_output_unit="mm")
    test = _dataset(tmp_path / "test", depth_output_unit="m")
    monkeypatch.setattr(factory, "make_dataset", lambda cfg: train)
    monkeypatch.setattr(factory, "LeRobotDatasetMetadata", lambda *args, **kwargs: test.meta)
    monkeypatch.setattr(factory, "LeRobotDataset", lambda *args, **kwargs: test)
    cfg = _config(
        tmp_path,
        dataset=DatasetConfig(repo_id="local/train", root=str(tmp_path / "train")),
        validation_dataset=DatasetConfig(repo_id="local/test", root=str(tmp_path / "test")),
    )

    assert factory.make_train_eval_datasets(cfg) == (train, test)


def test_external_validation_allows_equivalent_camera_storage_and_metadata(monkeypatch, tmp_path):
    train_features = _features()
    test_features = deepcopy(train_features)
    test_features["observation.images.main"].update(dtype="video", names=None)
    train = _dataset(tmp_path / "train", features=train_features)
    test = _dataset(tmp_path / "test", features=test_features)
    monkeypatch.setattr(factory, "make_dataset", lambda cfg: train)
    monkeypatch.setattr(factory, "LeRobotDatasetMetadata", lambda *args, **kwargs: test.meta)
    monkeypatch.setattr(factory, "LeRobotDataset", lambda *args, **kwargs: test)
    cfg = _config(
        tmp_path,
        dataset=DatasetConfig(repo_id="local/train", root=str(tmp_path / "train")),
        validation_dataset=DatasetConfig(repo_id="local/test", root=str(tmp_path / "test")),
    )

    assert factory.make_train_eval_datasets(cfg) == (train, test)


def test_external_validation_root_none_uses_metadata_resolved_root(monkeypatch, tmp_path):
    train = _dataset(tmp_path / "train-resolved")
    test = _dataset(tmp_path / "test-resolved")
    monkeypatch.setattr(factory, "make_dataset", lambda cfg: train)
    metadata_constructor = Mock(return_value=test.meta)
    dataset_constructor = Mock(return_value=test)
    monkeypatch.setattr(factory, "LeRobotDatasetMetadata", metadata_constructor)
    monkeypatch.setattr(factory, "LeRobotDataset", dataset_constructor)
    cfg = _config(
        tmp_path,
        dataset=DatasetConfig(repo_id="local/train", root=None),
        validation_dataset=DatasetConfig(repo_id="local/test", root=None),
    )

    assert factory.make_train_eval_datasets(cfg) == (train, test)
    metadata_constructor.assert_called_once_with("local/test", root=None, revision=None)
    dataset_constructor.assert_called_once()


def test_internal_eval_split_keeps_depth_unit_and_skips_depth_imagenet_stats(monkeypatch, tmp_path):
    features = _features()
    features["observation.images.depth"] = {
        "dtype": "image",
        "shape": (8, 12, 1),
        "names": ["height", "width", "channels"],
        "info": {"is_depth_map": True},
    }
    original_depth_stats = {"mean": "depth-sentinel"}
    full = _dataset(tmp_path / "train", features=features, episodes=[0, 1, 2, 3])
    full.meta.episodes = {"tasks": [["task"], ["task"], ["task"], ["task"]]}
    calls = []

    def fake_dataset(repo_id, **kwargs):
        calls.append(kwargs)
        stats = {
            "observation.images.main": {},
            "observation.images.depth": deepcopy(original_depth_stats),
        }
        return _dataset(
            tmp_path / "train",
            features=features,
            episodes=kwargs["episodes"],
            stats=stats,
            transforms=kwargs["image_transforms"],
        )

    monkeypatch.setattr(factory, "make_dataset", lambda cfg: full)
    monkeypatch.setattr(factory, "LeRobotDataset", fake_dataset)
    cfg = _config(
        tmp_path,
        dataset=DatasetConfig(
            repo_id="local/train",
            root=str(tmp_path / "train"),
            eval_split=0.25,
            depth_output_unit="mm",
        ),
    )

    train, test = factory.make_train_eval_datasets(cfg)

    assert train.episodes == [0, 1, 2]
    assert test.episodes == [3]
    assert [call["depth_output_unit"] for call in calls] == ["mm", "mm"]
    assert train.meta.stats["observation.images.depth"] == original_depth_stats
    assert test.meta.stats["observation.images.depth"] == original_depth_stats
    assert train.image_transforms is None
    assert test.image_transforms is None


def test_external_validation_loads_distinct_local_v3_roots_without_resplitting(
    tmp_path, lerobot_dataset_factory, info_factory
):
    train_root = tmp_path / "train-v3"
    test_root = tmp_path / "test-v3"
    motor_features = {
        "action": {"dtype": "float32", "shape": (2,), "names": ["a0", "a1"]},
        "observation.state": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["s0", "s1"],
        },
    }
    camera_features = {
        "observation.images.main": {
            "shape": (8, 12, 3),
            "names": None,
            "info": {"is_depth_map": False},
        }
    }
    lerobot_dataset_factory(
        root=train_root,
        repo_id="local/train-v3",
        total_episodes=3,
        total_frames=12,
        use_videos=False,
        info=info_factory(
            total_episodes=3,
            total_frames=12,
            total_tasks=1,
            use_videos=False,
            motor_features=motor_features,
            camera_features=camera_features,
        ),
    )
    lerobot_dataset_factory(
        root=test_root,
        repo_id="local/test-v3",
        total_episodes=2,
        total_frames=8,
        use_videos=False,
        info=info_factory(
            total_episodes=2,
            total_frames=8,
            total_tasks=1,
            use_videos=False,
            motor_features=motor_features,
            camera_features=camera_features,
        ),
    )
    cfg = _config(
        tmp_path,
        dataset=DatasetConfig(
            repo_id="local/train-v3", root=str(train_root), use_imagenet_stats=False
        ),
        validation_dataset=DatasetConfig(
            repo_id="local/test-v3", root=str(test_root), use_imagenet_stats=False
        ),
    )

    train, test = factory.make_train_eval_datasets(cfg)

    assert train.num_episodes == 3
    assert test.num_episodes == 2
    assert train.episodes is None
    assert test.episodes is None
    assert Path(train.root).resolve() == train_root.resolve()
    assert Path(test.root).resolve() == test_root.resolve()
