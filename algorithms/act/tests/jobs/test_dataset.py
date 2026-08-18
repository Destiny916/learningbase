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

from unittest.mock import MagicMock

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.jobs.dataset import ensure_dataset_available


def _api_with_dataset(exists: bool):
    api = MagicMock()
    api.repo_exists.return_value = exists
    return api


def _make_local_cache(tmp_path, repo_id: str) -> None:
    """Create the minimal local-cache layout that ensure_dataset_available checks."""
    info = tmp_path / repo_id / "meta" / "info.json"
    info.parent.mkdir(parents=True)
    info.write_text("{}")


# Branch 1: dataset already on Hub → no push, no error (pod downloads by repo_id).
def test_dataset_already_on_hub_is_noop():
    api = _api_with_dataset(True)
    assert ensure_dataset_available("user/ds", api=api) is None
    api.repo_exists.assert_called_once_with("user/ds", repo_type="dataset")


def test_dataset_already_on_hub_ignores_custom_root(tmp_path, monkeypatch):
    api = _api_with_dataset(True)
    mock_ds_cls = MagicMock()
    monkeypatch.setattr("lerobot.jobs.dataset.LeRobotDataset", mock_ds_cls)

    assert ensure_dataset_available("user/ds", api=api, root=tmp_path / "missing") is None

    mock_ds_cls.assert_not_called()


# Branch 2: not on Hub but present locally → always push privately.
def test_dataset_local_only_uploads_privately(tmp_path, monkeypatch):
    monkeypatch.setattr("lerobot.jobs.dataset.HF_LEROBOT_HOME", tmp_path)
    _make_local_cache(tmp_path, "user/ds")

    api = _api_with_dataset(False)
    mock_ds_cls = MagicMock()
    monkeypatch.setattr("lerobot.jobs.dataset.LeRobotDataset", mock_ds_cls)

    assert ensure_dataset_available("user/ds", api=api, tags=["lerobot", "lelab"]) is None

    mock_ds_cls.assert_called_once_with("user/ds")
    mock_ds_cls.return_value.push_to_hub.assert_called_once_with(private=True, tags=["lerobot", "lelab"])


@pytest.mark.parametrize("as_string", [False, True], ids=["path", "string"])
def test_dataset_custom_root_uploads_from_that_root(tmp_path, monkeypatch, as_string):
    root = tmp_path / "custom-dataset"
    info = root / "meta" / "info.json"
    info.parent.mkdir(parents=True)
    info.write_text("{}")
    api = _api_with_dataset(False)
    mock_ds_cls = MagicMock()
    monkeypatch.setattr("lerobot.jobs.dataset.LeRobotDataset", mock_ds_cls)
    root_arg = str(root) if as_string else root

    ensure_dataset_available("user/ds", api=api, root=root_arg, tags=["lerobot"])

    mock_ds_cls.assert_called_once_with("user/ds", root=root_arg)
    mock_ds_cls.return_value.push_to_hub.assert_called_once_with(private=True, tags=["lerobot"])


# Branch 3: not on Hub, NOT in local cache → RuntimeError.
def test_dataset_neither_on_hub_nor_local_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("lerobot.jobs.dataset.HF_LEROBOT_HOME", tmp_path)
    # tmp_path is empty — no local cache.

    api = _api_with_dataset(False)
    with pytest.raises(RuntimeError, match="not in the local cache"):
        ensure_dataset_available("user/ds", api=api)


def test_dataset_missing_custom_root_reports_checked_path(tmp_path):
    root = tmp_path / "custom-dataset"
    checked_path = root / "meta" / "info.json"
    api = _api_with_dataset(False)

    with pytest.raises(RuntimeError, match=str(checked_path)):
        ensure_dataset_available("user/ds", api=api, root=root)
