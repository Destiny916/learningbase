#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import hashlib
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from lerobot.datasets.relative_joint_stats import (
    QuantileStats,
    compute_relative_joint_stats_from_episodes,
    load_relative_joint_stats,
    save_relative_joint_stats,
)


FEATURE_NAMES = [f"joint_{index}" for index in range(6)] + ["gripper"]
MANIFEST_SHA256 = "a" * 64
STATE_17_NAMES = [f"left_joint_{index}" for index in range(6)] + ["left_gripper"] + [
    f"right_joint_{index}" for index in range(6)
] + ["right_gripper", "bread_x", "bread_y", "bread_z"]
ACTION_14_NAMES = STATE_17_NAMES[:14]


def _episodes() -> list[torch.Tensor]:
    return [
        torch.tensor(
            [
                [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 0.01],
                [11.0, 18.0, 33.0, 36.0, 55.0, 54.0, 0.04],
                [18.0, 10.0, 42.0, 26.0, 66.0, 42.0, 0.09],
            ]
        ),
        torch.tensor(
            [
                [-5.0, -10.0, 5.0, 15.0, -20.0, 25.0, 0.02],
                [-7.0, -6.0, -1.0, 23.0, -30.0, 37.0, 0.07],
            ]
        ),
    ]


def _expected_relative_states(episodes: list[torch.Tensor]) -> np.ndarray:
    relative_episodes = []
    for episode in episodes:
        values = episode.numpy().astype(np.float64)
        relative = values.copy()
        relative[0, :6] = 0.0
        relative[1:, :6] = values[1:, :6] - values[:-1, :6]
        relative_episodes.append(relative)
    return np.concatenate(relative_episodes)


def _expected_relative_actions(episodes: list[torch.Tensor], horizon: int) -> np.ndarray:
    targets = []
    for episode in episodes:
        values = episode.numpy().astype(np.float64)
        for t in range(len(values)):
            for k in range(1, min(horizon, len(values) - 1 - t) + 1):
                target = values[t + k].copy()
                target[:6] -= values[t, :6]
                targets.append(target)
    return np.stack(targets)


def test_compute_relative_joint_stats_separates_state_and_valid_horizon_targets() -> None:
    episodes = _episodes()

    bundle = compute_relative_joint_stats_from_episodes(
        episodes,
        gripper_indices=[6],
        horizons=[2, 3],
        feature_names=FEATURE_NAMES,
        source_manifest_sha256=MANIFEST_SHA256,
    )

    expected_state = _expected_relative_states(episodes)
    expected_action = _expected_relative_actions(episodes, horizon=2)
    assert bundle.state.count == 5
    assert bundle.actions[2].count == 4
    assert bundle.actions[3].count == 4
    assert bundle.state.q01.shape == (7,)
    assert bundle.state.q99.shape == (7,)
    assert bundle.actions[2].q01.shape == (7,)
    assert bundle.actions[2].q99.shape == (7,)
    np.testing.assert_array_equal(bundle.state.q01, np.quantile(expected_state, 0.01, axis=0))
    np.testing.assert_array_equal(bundle.state.q99, np.quantile(expected_state, 0.99, axis=0))
    np.testing.assert_array_equal(bundle.actions[2].q01, np.quantile(expected_action, 0.01, axis=0))
    np.testing.assert_array_equal(bundle.actions[2].q99, np.quantile(expected_action, 0.99, axis=0))
    np.testing.assert_array_equal(bundle.actions[3].q01, bundle.actions[2].q01)
    np.testing.assert_array_equal(bundle.actions[3].q99, bundle.actions[2].q99)

    # Every dimension has its own deliberately different source range and column-wise quantile.
    assert len({tuple(expected_state[:, index]) for index in range(7)}) == 7
    assert not np.all(bundle.state.q01 == bundle.state.q01[0])
    assert not np.all(bundle.state.q99 == bundle.state.q99[0])
    for index in range(7):
        assert bundle.state.q01[index] == np.quantile(expected_state[:, index], 0.01)
        assert bundle.state.q99[index] == np.quantile(expected_state[:, index], 0.99)


def test_first_state_arm_delta_is_zero_and_gripper_stays_absolute() -> None:
    episodes = _episodes()
    expected_state = _expected_relative_states(episodes)

    np.testing.assert_array_equal(expected_state[[0, 3], :6], np.zeros((2, 6)))
    np.testing.assert_array_equal(
        expected_state[:, 6], np.concatenate([episode[:, 6].numpy() for episode in episodes])
    )
    np.testing.assert_array_equal(expected_state[1, :6], np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0]))


def test_compute_relative_joint_stats_supports_independent_17d_state_and_14d_action() -> None:
    states = [
        torch.tensor(
            [
                [0.0] * 14 + [0.10, 0.20, 0.30],
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.11, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 0.22, 0.12, 0.25, 0.35],
                [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 0.12, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 0.24, 0.14, 0.30, 0.40],
            ]
        )
    ]
    actions = [
        torch.tensor(
            [
                [-1.0] * 14,
                states[0][1, :14].tolist(),
                states[0][2, :14].tolist(),
            ]
        )
    ]
    bundle = compute_relative_joint_stats_from_episodes(
        states,
        action_episodes=actions,
        state_gripper_indices=[6, 13],
        action_gripper_indices=[6, 13],
        action_state_indices=list(range(14)),
        horizons=[2],
        state_feature_names=STATE_17_NAMES,
        action_feature_names=ACTION_14_NAMES,
        source_manifest_sha256=MANIFEST_SHA256,
    )

    assert bundle.state.q01.shape == (17,)
    assert bundle.actions[2].q99.shape == (14,)
    assert bundle.state_feature_names == STATE_17_NAMES
    assert bundle.action_feature_names == ACTION_14_NAMES
    assert bundle.state_gripper_indices == [6, 13]
    assert bundle.action_gripper_indices == [6, 13]


def test_compute_relative_joint_stats_keeps_explicit_state_absolute_indices() -> None:
    state_names = ACTION_14_NAMES + ["right_endpoint_x", "right_endpoint_y"]
    states = [
        torch.tensor(
            [
                [0.0] * 14 + [1.0, 10.0],
                [1.0] * 6 + [0.1] + [2.0] * 6 + [0.2] + [3.0, 20.0],
                [2.0] * 6 + [0.2] + [4.0] * 6 + [0.3] + [5.0, 30.0],
            ]
        )
    ]
    actions = [torch.tensor([[-1.0] * 14, states[0][1, :14].tolist(), states[0][2, :14].tolist()])]

    bundle = compute_relative_joint_stats_from_episodes(
        states,
        action_episodes=actions,
        state_feature_names=state_names,
        action_feature_names=ACTION_14_NAMES,
        state_gripper_indices=[6, 13],
        action_gripper_indices=[6, 13],
        action_state_indices=list(range(14)),
        state_absolute_indices=[14, 15],
        horizons=[2],
        source_manifest_sha256=MANIFEST_SHA256,
    )

    assert bundle.state_absolute_indices == [14, 15]
    expected = np.asarray(
        [[0.0] * 14 + [1.0, 10.0], [1.0] * 6 + [0.1] + [2.0] * 6 + [0.2] + [3.0, 20.0], [1.0] * 6 + [0.2] + [2.0] * 6 + [0.3] + [5.0, 30.0]],
        dtype=np.float64,
    )
    np.testing.assert_allclose(bundle.state.q01, np.quantile(expected, 0.01, axis=0))
    np.testing.assert_allclose(bundle.state.q99, np.quantile(expected, 0.99, axis=0))


def test_quantile_stats_is_frozen() -> None:
    stats = QuantileStats(q01=np.zeros(7), q99=np.ones(7), count=1)

    with pytest.raises(FrozenInstanceError):
        stats.count = 2


@pytest.mark.parametrize(
    ("episodes", "horizons", "gripper_indices", "match"),
    [
        ([], [2], [6], "episode"),
        ([torch.ones(3, 6), torch.ones(3, 7)], [2], [5], "shape"),
        ([torch.ones(3, 7)], [], [6], "horizon"),
        ([torch.ones(3, 7)], [0], [6], "positive"),
        ([torch.ones(3, 7)], [2], [], "gripper"),
        ([torch.ones(3, 7)], [2], [7], "gripper"),
        ([torch.ones(1, 7)], [2], [6], "action"),
    ],
)
def test_compute_relative_joint_stats_validates_inputs(
    episodes: list[torch.Tensor], horizons: list[int], gripper_indices: list[int], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        compute_relative_joint_stats_from_episodes(
            episodes,
            gripper_indices=gripper_indices,
            horizons=horizons,
        )


def test_save_and_load_relative_joint_stats_uses_fixed_files_and_validates_provenance(tmp_path: Path) -> None:
    source_dataset_root = str((tmp_path / "train").resolve())
    bundle = compute_relative_joint_stats_from_episodes(
        _episodes(),
        gripper_indices=[6],
        horizons=[50, 16],
        feature_names=FEATURE_NAMES,
        source_manifest_sha256=MANIFEST_SHA256,
        source_dataset_root=source_dataset_root,
    )

    generation_command = (
        "python -m lerobot.scripts.compute_relative_joint_stats --dataset-root=train "
        "--split-manifest=split_manifest.json --output-dir=normalization "
        "--horizons='[50,16]' --gripper-indices='[6]'"
    )
    save_relative_joint_stats(bundle, tmp_path, generation_command=generation_command)

    assert {path.name for path in tmp_path.iterdir()} == {
        "relative_state_q01_q99.json",
        "relative_action_chunk50_q01_q99.json",
        "relative_action_chunk16_q01_q99.json",
        "relative_stats_manifest.json",
    }
    manifest = json.loads((tmp_path / "relative_stats_manifest.json").read_text(encoding="utf-8"))
    assert manifest["formula_version"] == "relative_joint_v1"
    assert manifest["generation_command"] == generation_command
    assert manifest["horizons"] == [16, 50]
    assert manifest["feature_names"] == FEATURE_NAMES
    assert manifest["gripper_indices"] == [6]
    assert manifest["source_manifest_sha256"] == MANIFEST_SHA256
    assert manifest["source_dataset_root"] == source_dataset_root
    state_payload = json.loads((tmp_path / "relative_state_q01_q99.json").read_text(encoding="utf-8"))
    action_payload = json.loads(
        (tmp_path / "relative_action_chunk16_q01_q99.json").read_text(encoding="utf-8")
    )
    assert state_payload["count"] == 5
    assert action_payload["horizon"] == 16
    assert action_payload["count"] == 4
    assert np.asarray(state_payload["q01"]).shape == (7,)
    assert np.asarray(state_payload["q99"]).shape == (7,)
    assert np.asarray(action_payload["q01"]).shape == (7,)
    assert np.asarray(action_payload["q99"]).shape == (7,)
    assert bundle.actions[50].q01.shape == (7,)
    assert bundle.actions[50].q99.shape == (7,)
    loaded = load_relative_joint_stats(
        tmp_path,
        requested_horizon=50,
        expected_feature_names=FEATURE_NAMES,
        expected_gripper_indices=[6],
        expected_source_manifest_sha256=MANIFEST_SHA256,
        expected_source_dataset_root=source_dataset_root,
    )
    np.testing.assert_array_equal(loaded.state.q01, bundle.state.q01)
    np.testing.assert_array_equal(loaded.actions[50].q99, bundle.actions[50].q99)
    assert loaded.actions[50].count == bundle.actions[50].count
    assert loaded.source_dataset_root == source_dataset_root

    with pytest.raises(ValueError, match="feature names"):
        load_relative_joint_stats(
            tmp_path,
            requested_horizon=50,
            expected_feature_names=[*FEATURE_NAMES[:-1], "other_gripper"],
            expected_gripper_indices=[6],
            expected_source_manifest_sha256=MANIFEST_SHA256,
        )
    with pytest.raises(ValueError, match="gripper indices"):
        load_relative_joint_stats(
            tmp_path,
            requested_horizon=50,
            expected_feature_names=FEATURE_NAMES,
            expected_gripper_indices=[5],
            expected_source_manifest_sha256=MANIFEST_SHA256,
        )
    with pytest.raises(ValueError, match="SHA256"):
        load_relative_joint_stats(
            tmp_path,
            requested_horizon=50,
            expected_feature_names=FEATURE_NAMES,
            expected_gripper_indices=[6],
            expected_source_manifest_sha256="b" * 64,
        )
    with pytest.raises(ValueError, match="horizon"):
        load_relative_joint_stats(
            tmp_path,
            requested_horizon=8,
            expected_feature_names=FEATURE_NAMES,
            expected_gripper_indices=[6],
            expected_source_manifest_sha256=MANIFEST_SHA256,
        )
    with pytest.raises(ValueError, match="source dataset root"):
        load_relative_joint_stats(
            tmp_path,
            requested_horizon=50,
            expected_feature_names=FEATURE_NAMES,
            expected_gripper_indices=[6],
            expected_source_manifest_sha256=MANIFEST_SHA256,
            expected_source_dataset_root=tmp_path / "other_train",
        )


def test_load_relative_joint_stats_rejects_formula_and_action_horizon_tampering(tmp_path: Path) -> None:
    bundle = compute_relative_joint_stats_from_episodes(
        _episodes(),
        gripper_indices=[6],
        horizons=[16, 50],
        feature_names=FEATURE_NAMES,
        source_manifest_sha256=MANIFEST_SHA256,
    )
    save_relative_joint_stats(bundle, tmp_path)

    manifest_path = tmp_path / "relative_stats_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["formula_version"] = "tampered_formula"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="formula_version"):
        load_relative_joint_stats(
            tmp_path,
            requested_horizon=50,
            expected_feature_names=FEATURE_NAMES,
            expected_gripper_indices=[6],
            expected_source_manifest_sha256=MANIFEST_SHA256,
        )

    save_relative_joint_stats(bundle, tmp_path)
    action_path = tmp_path / "relative_action_chunk50_q01_q99.json"
    action_payload = json.loads(action_path.read_text(encoding="utf-8"))
    action_payload["horizon"] = 16
    action_path.write_text(json.dumps(action_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="horizon"):
        load_relative_joint_stats(
            tmp_path,
            requested_horizon=50,
            expected_feature_names=FEATURE_NAMES,
            expected_gripper_indices=[6],
            expected_source_manifest_sha256=MANIFEST_SHA256,
        )


def _write_v30_dataset(
    root: Path,
    episodes: list[torch.Tensor],
    *,
    break_action_contract: bool = False,
    feature_names: list[str] | None = None,
) -> None:
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    names = FEATURE_NAMES if feature_names is None else feature_names
    info = {
        "codebase_version": "v3.0",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [7], "names": names},
            "action": {"dtype": "float32", "shape": [7], "names": names},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    states = [row.tolist() for episode in episodes for row in episode]
    actions = []
    for episode_index, episode in enumerate(episodes):
        for frame_index in range(len(episode)):
            action = episode[min(frame_index + 1, len(episode) - 1)].clone()
            if break_action_contract and episode_index == 0 and frame_index == 0:
                action[0] += 1
            actions.append(action.tolist())
    episode_indices = [index for index, episode in enumerate(episodes) for _ in episode]
    frame_indices = [frame for episode in episodes for frame in range(len(episode))]
    table = pa.table(
        {
            "observation.state": pa.array(states, type=pa.list_(pa.float32(), 7)),
            "action": pa.array(actions, type=pa.list_(pa.float32(), 7)),
            "episode_index": pa.array(episode_indices, type=pa.int64()),
            "frame_index": pa.array(frame_indices, type=pa.int64()),
        }
    )
    pq.write_table(table, root / "data" / "chunk-000" / "file-000.parquet")


def test_cli_reads_only_manifest_train_parquet_and_hashes_manifest(tmp_path: Path) -> None:
    train_root = tmp_path / "train"
    held_out_root = tmp_path / "held_out"
    output_dir = tmp_path / "stats"
    _write_v30_dataset(train_root, _episodes())
    held_out_root.mkdir()
    (held_out_root / "must_not_be_read.parquet").write_bytes(b"not parquet")
    manifest = {
        "format_version": 1,
        "splits": {"train": {"root": str(train_root)}, "test": {"root": str(held_out_root)}},
    }
    manifest_path = tmp_path / "split_manifest.json"
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode()
    manifest_path.write_bytes(manifest_bytes)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lerobot.scripts.compute_relative_joint_stats",
            f"--dataset-root={train_root}",
            f"--split-manifest={manifest_path}",
            f"--output-dir={output_dir}",
            "--horizons=[16,50]",
            "--gripper-indices=[6]",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    loaded = load_relative_joint_stats(
        output_dir,
        requested_horizon=16,
        expected_feature_names=FEATURE_NAMES,
        expected_gripper_indices=[6],
        expected_source_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        expected_source_dataset_root=train_root,
    )
    assert loaded.state.count == 5
    assert loaded.actions[16].count == 4
    assert {path.name for path in output_dir.iterdir()} == {
        "relative_state_q01_q99.json",
        "relative_action_chunk16_q01_q99.json",
        "relative_action_chunk50_q01_q99.json",
        "relative_stats_manifest.json",
    }
    manifest_payload = json.loads((output_dir / "relative_stats_manifest.json").read_text(encoding="utf-8"))
    assert manifest_payload["formula_version"] == "relative_joint_v1"
    assert "lerobot.scripts.compute_relative_joint_stats" in manifest_payload["generation_command"]
    assert manifest_payload["source_dataset_root"] == str(train_root.resolve())


@pytest.mark.parametrize(
    ("horizons", "gripper_indices", "match"),
    [
        ("[16]", "[6]", "horizons"),
        ("[16,50]", "[5]", "gripper indices"),
    ],
)
def test_cli_rejects_noncanonical_horizons_and_gripper_indices(
    tmp_path: Path,
    horizons: str,
    gripper_indices: str,
    match: str,
) -> None:
    train_root = tmp_path / "train"
    test_root = tmp_path / "test"
    _write_v30_dataset(train_root, _episodes())
    test_root.mkdir()
    manifest_path = tmp_path / "split_manifest.json"
    manifest_path.write_text(
        json.dumps({"splits": {"train": {"root": str(train_root)}, "test": {"root": str(test_root)}}}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lerobot.scripts.compute_relative_joint_stats",
            f"--dataset-root={train_root}",
            f"--split-manifest={manifest_path}",
            f"--output-dir={tmp_path / 'stats'}",
            f"--horizons={horizons}",
            f"--gripper-indices={gripper_indices}",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode != 0
    assert match in result.stderr.lower()


def test_cli_rejects_noncanonical_feature_names(tmp_path: Path) -> None:
    train_root = tmp_path / "train"
    test_root = tmp_path / "test"
    _write_v30_dataset(train_root, _episodes(), feature_names=[f"axis_{index}" for index in range(7)])
    test_root.mkdir()
    manifest_path = tmp_path / "split_manifest.json"
    manifest_path.write_text(
        json.dumps({"splits": {"train": {"root": str(train_root)}, "test": {"root": str(test_root)}}}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lerobot.scripts.compute_relative_joint_stats",
            f"--dataset-root={train_root}",
            f"--split-manifest={manifest_path}",
            f"--output-dir={tmp_path / 'stats'}",
            "--horizons=[16,50]",
            "--gripper-indices=[6]",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode != 0
    assert "feature names" in result.stderr.lower()


def test_cli_rejects_action_state_contract_mismatch(tmp_path: Path) -> None:
    train_root = tmp_path / "train"
    test_root = tmp_path / "test"
    _write_v30_dataset(train_root, _episodes(), break_action_contract=True)
    test_root.mkdir()
    manifest_path = tmp_path / "split_manifest.json"
    manifest_path.write_text(
        json.dumps({"splits": {"train": {"root": str(train_root)}, "test": {"root": str(test_root)}}}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lerobot.scripts.compute_relative_joint_stats",
            f"--dataset-root={train_root}",
            f"--split-manifest={manifest_path}",
            f"--output-dir={tmp_path / 'stats'}",
            "--horizons=[16,50]",
            "--gripper-indices=[6]",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode != 0
    assert "action contract" in result.stderr.lower()


def test_cli_rejects_dataset_root_not_declared_as_train(tmp_path: Path) -> None:
    train_root = tmp_path / "train"
    test_root = tmp_path / "test"
    other_root = tmp_path / "other"
    train_root.mkdir()
    test_root.mkdir()
    other_root.mkdir()
    manifest_path = tmp_path / "split_manifest.json"
    manifest_path.write_text(
        json.dumps({"splits": {"train": {"root": str(train_root)}, "test": {"root": str(test_root)}}}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lerobot.scripts.compute_relative_joint_stats",
            f"--dataset-root={other_root}",
            f"--split-manifest={manifest_path}",
            f"--output-dir={tmp_path / 'stats'}",
            "--horizons=[16,50]",
            "--gripper-indices=[6]",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode != 0
    assert "manifest train root" in result.stderr.lower()


def test_cli_rejects_manifest_test_root_even_through_symlink(tmp_path: Path) -> None:
    train_root = tmp_path / "train"
    held_out_root = tmp_path / "held_out"
    alias_root = tmp_path / "candidate"
    train_root.mkdir()
    held_out_root.mkdir()
    alias_root.symlink_to(held_out_root, target_is_directory=True)
    manifest_path = tmp_path / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {"splits": {"train": {"root": str(train_root)}, "test": {"root": str(held_out_root)}}}
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lerobot.scripts.compute_relative_joint_stats",
            f"--dataset-root={alias_root}",
            f"--split-manifest={manifest_path}",
            f"--output-dir={tmp_path / 'stats'}",
            "--horizons=[16,50]",
            "--gripper-indices=[6]",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode != 0
    assert "test" in result.stderr.lower()
