from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import w1_simulation.launch as launch_module
from w1_simulation.w1_profile import DEFAULT_PROFILE


def test_resolve_port_zero_skips_an_excluded_available_port(monkeypatch: pytest.MonkeyPatch) -> None:
    selected_ports = iter((41000, 41000, 41001))

    class FakeSocket:
        def __enter__(self) -> FakeSocket:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def bind(self, _: tuple[str, int]) -> None:
            return None

        def getsockname(self) -> tuple[str, int]:
            return ("127.0.0.1", next(selected_ports))

    monkeypatch.setattr(launch_module.socket, "socket", lambda *_: FakeSocket())
    rerun_port = launch_module._resolve_port(0)
    tensorboard_port = launch_module._resolve_port(0, {rerun_port})

    assert rerun_port == 41000
    assert tensorboard_port == 41001


def test_visualization_service_uses_ten_gb_rerun_memory_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    logs = tmp_path / "logs"
    tensorboard = tmp_path / "tensorboard"
    logs.mkdir()
    tensorboard.mkdir()
    commands: list[list[str]] = []

    class FakeProcess:
        pid = 12345

        def poll(self) -> None:
            return None

    def fake_popen(command: list[str], **_: object) -> FakeProcess:
        commands.append(command)
        return FakeProcess()

    monkeypatch.setattr(
        launch_module,
        "ensure_simulation_artifact_dirs",
        lambda _: {"logs": logs, "tensorboard": tensorboard},
    )
    monkeypatch.setattr(launch_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(launch_module, "_wait_for_port", lambda *_: None)

    _, handles = launch_module._start_visualization_services(tmp_path, 41000, 41001)
    for handle in handles:
        handle.close()

    rerun_command = commands[0]
    memory_limit_index = rerun_command.index("--memory-limit")
    assert rerun_command[memory_limit_index + 1] == "10GB"


def _launch_kwargs(tmp_path: Path) -> dict[str, object]:
    return {
        "checkpoint": tmp_path / "checkpoint",
        "origin": tmp_path / "origin",
        "artifacts": tmp_path / "artifacts",
        "run_name": "test_run",
        "start_frame": 0,
        "max_frames": 1,
        "device": "cuda:0",
        "policy_backend": "script",
        "policy_script": tmp_path / "policy.py",
        "bridge_script": tmp_path / "bridge.py",
        "whole_script": tmp_path / "whole.sh",
        "camera_sources": {"observation.images.camera": "camera"},
        "profile": DEFAULT_PROFILE,
        "rerun_port": 0,
        "tensorboard_port": 0,
        "save_artifacts": True,
        "realtime": True,
        "keep_open": False,
        "control_mode": "kinematic",
        "action_pipeline": "bridge",
        "execution_horizon": 0,
        "replan_interval": 10,
        "bridge_simulated_inference_ms": 200.0,
        "bridge_inference_budget_ms": 300.0,
        "bridge_policy_hz": 20.0,
        "bridge_replan_threshold": 0.5,
        "bridge_lipo_blend_policy_points": 5,
        "bridge_replan_margin_policy_points": 2,
        "bridge_sample_factor": 2,
        "image_replay_mode": "state",
        "image_search_ahead_frames": 15,
        "image_max_advance_frames": 2,
        "image_match_threshold": 0.18,
        "image_similarity_slack": 0.005,
        "quality_metrics": ("pose", "end_effector", "motion_direction", "amplitude"),
        "score_smoothness": True,
        "score_realtime": True,
        "rerun_view_mode": "eye",
        "eye_camera_width": 1280,
        "eye_camera_height": 720,
        "eye_camera_fps": 30.0,
        "eye_camera_fovy": 70.0,
        "eye_camera_scene": "grid",
    }


def test_non_strict_verification_warning_does_not_discard_completed_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    summary_path = tmp_path / "summary.json"
    process = SimpleNamespace(pid=12345)
    monkeypatch.setattr(launch_module, "_resolve_port", lambda port, *_: port or 12345)
    monkeypatch.setattr(launch_module, "_start_visualization_services", lambda *_: ([process], []))
    monkeypatch.setattr(launch_module, "_stop_processes", lambda *_: None)
    monkeypatch.setattr(launch_module, "run_act_simulation", lambda **_: summary_path)
    monkeypatch.setattr(
        launch_module,
        "verify_run",
        lambda *_: (_ for _ in ()).throw(AssertionError("p95 cycle exceeded")),
    )

    result = launch_module.launch(**_launch_kwargs(tmp_path), strict_verification=False)

    assert result == summary_path
    output = capsys.readouterr().out
    assert "ACT_SIM_VERIFICATION_WARNING=p95 cycle exceeded" in output
    assert "ACT_SIM_VERIFICATION=not-passed" in output


def test_strict_verification_preserves_failure_exit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    process = SimpleNamespace(pid=12345)
    monkeypatch.setattr(launch_module, "_resolve_port", lambda port, *_: port or 12345)
    monkeypatch.setattr(launch_module, "_start_visualization_services", lambda *_: ([process], []))
    monkeypatch.setattr(launch_module, "_stop_processes", lambda *_: None)
    monkeypatch.setattr(launch_module, "run_act_simulation", lambda **_: tmp_path / "summary.json")
    monkeypatch.setattr(
        launch_module,
        "verify_run",
        lambda *_: (_ for _ in ()).throw(AssertionError("p95 cycle exceeded")),
    )

    with pytest.raises(AssertionError, match="p95 cycle exceeded"):
        launch_module.launch(**_launch_kwargs(tmp_path), strict_verification=True)


def test_disabled_artifacts_are_ephemeral_and_skip_verification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    summary_path = tmp_path / "summary.json"
    captured: dict[str, object] = {}
    process = SimpleNamespace(pid=12345)

    def fake_run(**kwargs: object) -> Path:
        captured.update(kwargs)
        return summary_path

    monkeypatch.setattr(launch_module, "_resolve_port", lambda port, *_: port or 12345)
    monkeypatch.setattr(launch_module, "_start_visualization_services", lambda *_: ([process], []))
    monkeypatch.setattr(launch_module, "_stop_processes", lambda *_: None)
    monkeypatch.setattr(launch_module, "run_act_simulation", fake_run)
    monkeypatch.setattr(launch_module, "verify_run", lambda *_: pytest.fail("verification must be skipped"))
    kwargs = _launch_kwargs(tmp_path)
    kwargs["save_artifacts"] = False

    result = launch_module.launch(**kwargs, strict_verification=False)

    assert result is None
    assert captured["save_artifacts"] is False
    assert not Path(captured["artifact_root"]).exists()
    assert not Path(kwargs["artifacts"]).exists()
    assert "ACT_SIM_VERIFICATION=skipped-artifacts-disabled" in capsys.readouterr().out


def test_strict_verification_requires_artifacts(tmp_path: Path) -> None:
    kwargs = _launch_kwargs(tmp_path)
    kwargs["save_artifacts"] = False

    with pytest.raises(ValueError, match="SAVE_ARTIFACTS=true"):
        launch_module.launch(**kwargs, strict_verification=True)


def test_launch_forwards_bridge_timing_to_runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    captured: dict[str, object] = {}

    def fake_run(**kwargs: object) -> Path:
        captured.update(kwargs)
        return summary_path

    process = SimpleNamespace(pid=12345)
    monkeypatch.setattr(launch_module, "_resolve_port", lambda port, *_: port or 12345)
    monkeypatch.setattr(launch_module, "_start_visualization_services", lambda *_: ([process], []))
    monkeypatch.setattr(launch_module, "_stop_processes", lambda *_: None)
    monkeypatch.setattr(launch_module, "run_act_simulation", fake_run)
    monkeypatch.setattr(launch_module, "verify_run", lambda *_: tmp_path / "verification.json")

    launch_module.launch(**_launch_kwargs(tmp_path), strict_verification=True)

    assert captured["bridge_simulated_inference_ms"] == 200.0
    assert captured["bridge_inference_budget_ms"] == 300.0
    assert captured["bridge_policy_hz"] == 20.0
    assert captured["bridge_replan_threshold"] == 0.5
    assert captured["bridge_lipo_blend_policy_points"] == 5
    assert captured["bridge_replan_margin_policy_points"] == 2
    assert captured["bridge_sample_factor"] == 2
    assert captured["execution_horizon"] == 0
    assert captured["image_replay_mode"] == "state"
    assert captured["image_search_ahead_frames"] == 15
    assert captured["image_max_advance_frames"] == 2
    assert captured["image_match_threshold"] == 0.18
    assert captured["image_similarity_slack"] == 0.005
    assert captured["quality_metrics"] == ("pose", "end_effector", "motion_direction", "amplitude")
    assert captured["score_smoothness"] is True
    assert captured["score_realtime"] is True
    assert captured["save_artifacts"] is True
    run_directory = Path(captured["artifact_root"])
    assert run_directory.parent == (tmp_path / "artifacts" / "runs").resolve()
    assert run_directory.name.endswith("_test_run")
