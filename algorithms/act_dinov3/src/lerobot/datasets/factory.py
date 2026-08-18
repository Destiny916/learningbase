#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import logging
import math
from pathlib import Path
from pprint import pformat

import torch

from lerobot.configs import PreTrainedConfig
from lerobot.configs.rewards import RewardModelConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.transforms import ImageTransforms
from lerobot.utils.constants import ACTION, IMAGENET_STATS, OBS_PREFIX, OBS_STATE, REWARD
from lerobot.utils.feature_utils import dataset_to_policy_features

from .dataset_metadata import LeRobotDatasetMetadata
from .lerobot_dataset import LeRobotDataset
from .multi_dataset import MultiLeRobotDataset
from .streaming_dataset import StreamingLeRobotDataset


def resolve_delta_timestamps(
    cfg: PreTrainedConfig | RewardModelConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the config.

    Args:
        cfg (PreTrainedConfig | RewardModelConfig): The config to read delta_indices from. Both
            ``PreTrainedConfig`` and concrete ``RewardModelConfig`` subclasses expose the
            ``{observation,action,reward}_delta_indices`` properties used below.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if (
            key.startswith(OBS_PREFIX)
            and cfg.observation_delta_indices is not None
            and (
                not (
                    getattr(cfg, "type", None) in {"pi05", "pi052"}
                    and getattr(cfg, "joint_representation", None) == "relative"
                )
                or key == OBS_STATE
            )
        ):
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def _resolve_episodes(
    episodes: list[int] | None, exclude_episodes: list[int] | None, total_episodes: int
) -> list[int] | None:
    """Apply an episode exclusion list on top of an optional allowlist."""
    if not exclude_episodes:
        return episodes
    base = episodes if episodes is not None else list(range(total_episodes))
    excluded = set(exclude_episodes)
    return [episode for episode in base if episode not in excluded]


def make_dataset(
    cfg: TrainPipelineConfig,
) -> LeRobotDataset | StreamingLeRobotDataset | MultiLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | StreamingLeRobotDataset | MultiLeRobotDataset
    """
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    if isinstance(cfg.dataset.repo_id, str):
        ds_meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
        )
        delta_timestamps = resolve_delta_timestamps(cfg.trainable_config, ds_meta)
        episodes = _resolve_episodes(
            cfg.dataset.episodes,
            getattr(cfg.dataset, "exclude_episodes", None),
            ds_meta.total_episodes,
        )
        if not cfg.dataset.streaming:
            dataset = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                video_backend=cfg.dataset.video_backend,
                return_uint8=True,
                depth_output_unit=cfg.dataset.depth_output_unit,
                tolerance_s=cfg.tolerance_s,
            )
        else:
            dataset = StreamingLeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                max_num_shards=cfg.num_workers,
                tolerance_s=cfg.tolerance_s,
                return_uint8=True,
            )
    else:
        raise NotImplementedError("The MultiLeRobotDataset isn't supported for now.")
        dataset = MultiLeRobotDataset(
            cfg.dataset.repo_id,
            # TODO(aliberts): add proper support for multi dataset
            # delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=cfg.dataset.video_backend,
        )
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(dataset.repo_id_to_index, indent=2)}"
        )

    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            if key in dataset.meta.depth_keys:
                continue  # Exclude depth keys from ImageNet stats
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset


def make_train_eval_datasets(
    cfg: TrainPipelineConfig,
) -> tuple[
    LeRobotDataset | StreamingLeRobotDataset | MultiLeRobotDataset,
    LeRobotDataset | StreamingLeRobotDataset | None,
]:
    """Create train and optional eval datasets from an independent root or an episode split.

    The last ceil(n_episodes * eval_split) episodes per task are held out for evaluation.
    If eval_split == 0.0, returns (full_dataset, None).
    """
    full_dataset = make_dataset(cfg)

    if cfg.validation_dataset is not None:
        validation_cfg = cfg.validation_dataset
        if validation_cfg.root is not None:
            _ensure_distinct_roots(full_dataset.root, validation_cfg.root)

        eval_meta = LeRobotDatasetMetadata(
            validation_cfg.repo_id, root=validation_cfg.root, revision=validation_cfg.revision
        )
        _ensure_distinct_roots(full_dataset.root, eval_meta.root)
        _validate_train_eval_contract(
            full_dataset, eval_meta, test_depth_output_unit=validation_cfg.depth_output_unit
        )

        delta_timestamps = resolve_delta_timestamps(cfg.trainable_config, full_dataset.meta)
        eval_dataset = LeRobotDataset(
            validation_cfg.repo_id,
            root=validation_cfg.root,
            episodes=validation_cfg.episodes,
            delta_timestamps=delta_timestamps,
            image_transforms=None,
            revision=validation_cfg.revision,
            video_backend=validation_cfg.video_backend,
            return_uint8=True,
            depth_output_unit=validation_cfg.depth_output_unit,
            tolerance_s=cfg.tolerance_s,
        )
        return full_dataset, eval_dataset

    if cfg.dataset.eval_split == 0.0:
        return full_dataset, None

    base_episodes = (
        full_dataset.episodes if full_dataset.episodes is not None else list(range(full_dataset.num_episodes))
    )

    episode_tasks = full_dataset.meta.episodes["tasks"]
    task_to_episodes: dict[str, list[int]] = {}
    for ep_idx in base_episodes:
        task_key = episode_tasks[ep_idx][0] if episode_tasks[ep_idx] else ""
        task_to_episodes.setdefault(task_key, []).append(ep_idx)

    train_episodes, eval_episodes = [], []
    for eps in task_to_episodes.values():
        n_eval = math.ceil(len(eps) * cfg.dataset.eval_split)
        train_episodes.extend(eps[: len(eps) - n_eval])
        eval_episodes.extend(eps[len(eps) - n_eval :])

    if not train_episodes:
        raise ValueError(
            f"eval_split={cfg.dataset.eval_split} leaves 0 training episodes from {len(base_episodes)} total."
        )

    logging.info(
        f"Train/eval split: {len(train_episodes)} train, {len(eval_episodes)} eval "
        f"(eval_split={cfg.dataset.eval_split}, {len(task_to_episodes)} tasks)"
    )

    delta_timestamps = resolve_delta_timestamps(cfg.trainable_config, full_dataset.meta)

    train_image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    train_dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=train_episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=train_image_transforms,
        revision=cfg.dataset.revision,
        video_backend=cfg.dataset.video_backend,
        return_uint8=True,
        depth_output_unit=cfg.dataset.depth_output_unit,
        tolerance_s=cfg.tolerance_s,
    )

    eval_dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=eval_episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=None,
        revision=cfg.dataset.revision,
        video_backend=cfg.dataset.video_backend,
        return_uint8=True,
        depth_output_unit=cfg.dataset.depth_output_unit,
        tolerance_s=cfg.tolerance_s,
    )

    if cfg.dataset.use_imagenet_stats:
        for ds in (train_dataset, eval_dataset):
            for key in ds.meta.camera_keys:
                if key in ds.meta.depth_keys:
                    continue
                for stats_type, stats in IMAGENET_STATS.items():
                    ds.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return train_dataset, eval_dataset


def _contract_error(field: str, train_value, test_value) -> ValueError:
    return ValueError(
        f"Validation dataset contract mismatch for {field}: train={train_value!r}, test={test_value!r}."
    )


def _ensure_distinct_roots(train_root: str | Path, test_root: str | Path) -> None:
    resolved_train_root = Path(train_root).resolve()
    resolved_test_root = Path(test_root).resolve()
    if resolved_train_root == resolved_test_root:
        raise ValueError(
            "Train and validation datasets must have different resolved roots: "
            f"train={resolved_train_root}, test={resolved_test_root}."
        )


def _validate_train_eval_contract(
    train_dataset: LeRobotDataset | StreamingLeRobotDataset | MultiLeRobotDataset,
    test_meta: LeRobotDatasetMetadata,
    test_depth_output_unit: str,
) -> None:
    train_meta = train_dataset.meta
    if train_meta.fps != test_meta.fps:
        raise _contract_error("fps", train_meta.fps, test_meta.fps)

    train_features = train_meta.features
    test_features = test_meta.features
    for key in ("episode_index", "frame_index", ACTION, OBS_STATE):
        if key not in train_features or key not in test_features:
            raise _contract_error(f"feature {key}", key in train_features, key in test_features)

    train_cameras = set(train_meta.camera_keys)
    test_cameras = set(test_meta.camera_keys)
    if train_cameras != test_cameras:
        raise _contract_error("camera_keys", sorted(train_cameras), sorted(test_cameras))

    train_keys = set(train_features)
    test_keys = set(test_features)
    if train_keys != test_keys:
        raise _contract_error("feature keys", sorted(train_keys), sorted(test_keys))

    for key in (ACTION, OBS_STATE, "episode_index", "frame_index"):
        for attribute in ("dtype", "shape", "names"):
            train_value = train_features[key].get(attribute)
            test_value = test_features[key].get(attribute)
            if train_value != test_value:
                raise _contract_error(f"{key}.{attribute}", train_value, test_value)

    train_depth_keys = set(train_meta.depth_keys)
    test_depth_keys = set(test_meta.depth_keys)
    for key in sorted(train_cameras):
        train_is_depth = key in train_depth_keys
        test_is_depth = key in test_depth_keys
        if train_is_depth != test_is_depth:
            raise _contract_error(f"{key}.is_depth_map", train_is_depth, test_is_depth)

    if train_depth_keys and train_dataset.depth_output_unit != test_depth_output_unit:
        raise _contract_error(
            "depth_output_unit", train_dataset.depth_output_unit, test_depth_output_unit
        )

    train_policy_features = dataset_to_policy_features(train_features)
    test_policy_features = dataset_to_policy_features(test_features)
    if train_policy_features != test_policy_features:
        raise _contract_error("policy features", train_policy_features, test_policy_features)
