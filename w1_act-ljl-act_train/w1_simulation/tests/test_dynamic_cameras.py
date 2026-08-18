from __future__ import annotations

import sys

import pytest
import w1_simulation.execution.rollout as run_module
from w1_simulation.replay.origin import OriginReplay

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "camera_sources",
    [
        {
            "observation.images.overhead": "head_left",
            "observation.images.wrist": "hand_right",
        },
        {
            "observation.images.head_left": "head_left",
            "observation.images.head_right": "head_right",
            "observation.images.hand_left": "hand_left",
            "observation.images.hand_right": "hand_right",
        },
    ],
    ids=("two-camera-policy", "four-camera-policy"),
)
def test_origin_replay_supports_arbitrary_policy_camera_sources(origin_root, camera_sources) -> None:
    replay = OriginReplay(origin_root, camera_sources=camera_sources)

    assert replay.camera_sources == camera_sources
    assert len(replay.frames) == 1391
    assert all(tuple(frame.records) == tuple(camera_sources) for frame in replay.frames)
    assert all(frame.camera_skew_ms <= 50.0 for frame in replay.frames)


def test_run_cli_parses_repeated_key_source_camera_arguments(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_run_act_simulation(**kwargs):
        captured.update(kwargs)
        return tmp_path / "summary.json"

    monkeypatch.setattr(run_module, "run_act_simulation", fake_run_act_simulation)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "w1-simulation-sim",
            "--run-name",
            "camera_contract",
            "--camera-source",
            "observation.images.overhead=head_left",
            "--camera-source",
            "observation.images.wrist=hand/right",
        ],
    )

    run_module.main()

    assert captured["camera_sources"] == {
        "observation.images.overhead": "head_left",
        "observation.images.wrist": "hand/right",
    }
    assert captured["execution_horizon"] == 0


@pytest.mark.parametrize(
    "argument",
    (
        "missing-separator",
        "=head_left",
        "observation.images.overhead=",
    ),
)
def test_camera_source_parser_rejects_malformed_argument(argument: str) -> None:
    with pytest.raises(ValueError, match="MODEL_INPUT=SOURCE"):
        run_module._parse_camera_sources([argument])


def test_camera_source_parser_rejects_duplicate_policy_input() -> None:
    with pytest.raises(ValueError, match="Duplicate model image input"):
        run_module._parse_camera_sources(
            [
                "observation.images.overhead=head_left",
                "observation.images.overhead=head_right",
            ]
        )
