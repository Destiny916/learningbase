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
import sys
import types
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.scripts.lerobot_dataset_viz import (
    action_series_for_visualization,
    build_blueprint_from_dataset,
    visualize_dataset,
)
from lerobot.utils.constants import ACTION


def test_action_series_for_visualization_splits_action_chunk_by_horizon():
    action = torch.arange(20 * 14, dtype=torch.float32).reshape(20, 14)

    series = action_series_for_visualization(action)

    assert [path for path, _ in series] == [f"action/chunk/{step:02d}" for step in range(20)]
    assert all(values.shape == (14,) for _, values in series)
    torch.testing.assert_close(torch.from_numpy(series[0][1]), action[0])
    torch.testing.assert_close(torch.from_numpy(series[-1][1]), action[-1])


def test_action_series_for_visualization_preserves_flat_action_path():
    action = torch.arange(14, dtype=torch.float32)

    series = action_series_for_visualization(action)

    assert len(series) == 1
    assert series[0][0] == "action"
    torch.testing.assert_close(torch.from_numpy(series[0][1]), action)


def test_action_chunk_blueprint_uses_one_view_with_all_horizons(monkeypatch):
    class FakeTimeSeriesView:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeGrid:
        def __init__(self, *views):
            self.views = views

    class FakeBlueprint:
        def __init__(self, grid):
            self.grid = grid

    fake_rerun = types.ModuleType("rerun")
    fake_rerun.SeriesLines = lambda **kwargs: kwargs
    fake_blueprint = types.ModuleType("rerun.blueprint")
    fake_blueprint.TimeSeriesView = FakeTimeSeriesView
    fake_blueprint.Grid = FakeGrid
    fake_blueprint.Blueprint = FakeBlueprint
    monkeypatch.setitem(sys.modules, "rerun", fake_rerun)
    monkeypatch.setitem(sys.modules, "rerun.blueprint", fake_blueprint)
    dataset = SimpleNamespace(
        meta=SimpleNamespace(camera_keys=[]),
        features={ACTION: {"shape": [20, 14], "names": [f"joint_{index}" for index in range(14)]}},
    )

    blueprint = build_blueprint_from_dataset(dataset)

    assert len(blueprint.grid.views) == 1
    view = blueprint.grid.views[0].kwargs
    assert view["name"] == "action"
    assert view["contents"] == [f"action/chunk/{step:02d}" for step in range(20)]


@pytest.mark.skip("TODO: add dummy videos")
def test_visualize_local_dataset(tmp_path, lerobot_dataset_factory):
    root = tmp_path / "dataset"
    output_dir = tmp_path / "outputs"
    dataset = lerobot_dataset_factory(root=root)
    rrd_path = visualize_dataset(
        dataset,
        episode_index=0,
        batch_size=32,
        save=True,
        output_dir=output_dir,
    )
    assert rrd_path.exists()
