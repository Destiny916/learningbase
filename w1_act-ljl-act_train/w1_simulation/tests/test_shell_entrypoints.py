from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_SCRIPT = REPO_ROOT / "w1_simulation" / "run_act_sim.sh"
BRIDGE_SCRIPT = REPO_ROOT / "w1_simulation" / "run_act_sim_bridge.sh"
LEGACY_EE_FK_SCRIPT = REPO_ROOT / "w1_simulation" / "run_act_ee_fk.sh"


def _captured_args(
    script: Path,
    tmp_path: Path,
    *arguments: str,
    environment: dict[str, str] | None = None,
) -> list[str]:
    fake_python = tmp_path / "capture_python.sh"
    fake_python.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n', encoding="utf-8")
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": str(fake_python),
            "RUN_NAME": "entrypoint_test",
            "KEEP_OPEN": "0",
            "REALTIME": "0",
            "STRICT_VERIFICATION": "0",
        }
    )
    env.update(environment or {})
    result = subprocess.run(
        ["bash", str(script), *arguments],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()


def _values(arguments: list[str], option: str) -> list[str]:
    return [arguments[index + 1] for index, value in enumerate(arguments[:-1]) if value == option]


def test_all_simulation_entrypoints_live_under_w1_simulation() -> None:
    assert not (REPO_ROOT / "run_act_sim").exists()
    assert not (REPO_ROOT / "run_act_sim.sh").exists()
    assert RAW_SCRIPT.is_file() and os.access(RAW_SCRIPT, os.X_OK)
    assert BRIDGE_SCRIPT.is_file() and os.access(BRIDGE_SCRIPT, os.X_OK)
    assert not LEGACY_EE_FK_SCRIPT.exists()


def test_policy_entrypoints_prefer_bundled_lerobot_source() -> None:
    expected = 'PROJECT_PYTHONPATH="$WORKSPACE_ROOT/w1_lerobot/src:$WORKSPACE_ROOT"'

    for script in (RAW_SCRIPT, BRIDGE_SCRIPT):
        source = script.read_text(encoding="utf-8")
        assert expected in source
        assert 'exec env "PYTHONPATH=$PROJECT_PYTHONPATH"' in source


def test_raw_script_cannot_be_switched_to_bridge(tmp_path: Path) -> None:
    arguments = _captured_args(
        RAW_SCRIPT,
        tmp_path,
        "--action-pipeline",
        "bridge",
        environment={"ACTION_PIPELINE": "bridge"},
    )

    assert _values(arguments, "--action-pipeline")[-1] == "raw"
    assert "ACTION_PIPELINE" not in RAW_SCRIPT.read_text(encoding="utf-8")


def test_raw_script_defaults_to_full_checkpoint_chunk_and_allows_override(tmp_path: Path) -> None:
    defaults = _captured_args(RAW_SCRIPT, tmp_path)
    checkpoint_default = _captured_args(
        RAW_SCRIPT,
        tmp_path,
        environment={"RAW_EXECUTION_HORIZON": "0"},
    )

    assert _values(defaults, "--profile") == [str(REPO_ROOT / "w1_simulation/configs/w1_popcorn_v1.json")]
    assert _values(defaults, "--execution-horizon") == []
    assert _values(checkpoint_default, "--execution-horizon") == ["0"]
    assert _values(defaults, "--replan-interval") == []


def test_bridge_script_cannot_be_switched_to_raw_and_forwards_parameters(tmp_path: Path) -> None:
    arguments = _captured_args(
        BRIDGE_SCRIPT,
        tmp_path,
        "--action-pipeline",
        "raw",
        "--bridge-replan-hz",
        "1.5",
        "--bridge-sample-factor",
        "2",
        "--bridge-simulated-inference-ms",
        "225",
        environment={"ACTION_PIPELINE": "raw"},
    )

    assert _values(arguments, "--action-pipeline")[-1] == "bridge"
    assert _values(arguments, "--bridge-replan-hz")[-1] == "1.5"
    assert _values(arguments, "--bridge-sample-factor")[-1] == "2"
    assert _values(arguments, "--bridge-simulated-inference-ms")[-1] == "225"
    assert "ACTION_PIPELINE" not in BRIDGE_SCRIPT.read_text(encoding="utf-8")


def test_bridge_script_defaults_to_hundred_policy_points_and_allows_override(tmp_path: Path) -> None:
    defaults = _captured_args(BRIDGE_SCRIPT, tmp_path)
    checkpoint_default = _captured_args(
        BRIDGE_SCRIPT,
        tmp_path,
        environment={"BRIDGE_EXECUTION_HORIZON": "0"},
    )

    assert _values(defaults, "--profile") == [str(REPO_ROOT / "w1_simulation/configs/w1_popcorn_v1.json")]
    assert _values(defaults, "--checkpoint") == []
    assert _values(defaults, "--execution-horizon") == []
    assert _values(checkpoint_default, "--execution-horizon") == ["0"]
    assert _values(defaults, "--bridge-replan-threshold") == []
    assert _values(defaults, "--bridge-lipo-blend-policy-points") == []
    assert _values(defaults, "--bridge-inference-budget-ms") == []
    assert _values(defaults, "--bridge-replan-margin-policy-points") == []
    assert _values(defaults, "--bridge-sample-factor") == []


def test_all_simulation_entrypoints_default_to_eye_view_with_dataset_cameras(tmp_path: Path) -> None:
    for script in (RAW_SCRIPT, BRIDGE_SCRIPT):
        defaults = _captured_args(script, tmp_path)
        assert _values(defaults, "--rerun-view-mode") == []
        assert _values(defaults, "--eye-camera-width") == []
        assert _values(defaults, "--eye-camera-height") == []
        assert _values(defaults, "--eye-camera-fps") == []
        assert _values(defaults, "--eye-camera-fovy") == []
        assert _values(defaults, "--eye-camera-scene") == []

    standard = _captured_args(
        RAW_SCRIPT,
        tmp_path,
        environment={"RERUN_VIEW_MODE": "standard"},
    )
    assert _values(standard, "--rerun-view-mode") == ["standard"]


def test_all_artifacts_are_opt_in_for_all_simulation_entrypoints(tmp_path: Path) -> None:
    for script in (RAW_SCRIPT, BRIDGE_SCRIPT):
        defaults = _captured_args(script, tmp_path)
        enabled = _captured_args(
            script,
            tmp_path,
            environment={"SAVE_ARTIFACTS": "true"},
        )

        assert "--no-save-artifacts" in defaults
        assert "--save-artifacts" not in defaults
        assert "--save-artifacts" in enabled
        assert "--no-save-artifacts" not in enabled


def test_entrypoints_separate_external_assets_from_project_runtime(tmp_path: Path) -> None:
    asset_root = tmp_path / "external_assets"
    arguments = _captured_args(
        BRIDGE_SCRIPT,
        tmp_path,
        environment={"W1_SIMULATION_ASSET_ROOT": str(asset_root)},
    )

    assert _values(arguments, "--profile") == [str(REPO_ROOT / "w1_simulation/configs/w1_popcorn_v1.json")]
    assert _values(arguments, "--checkpoint") == []
    assert _values(arguments, "--origin") == []
    assert _values(arguments, "--policy-script") == []


def test_entrypoints_can_place_runtime_artifacts_outside_the_source_tree(tmp_path: Path) -> None:
    shared_root = tmp_path / "runs" / "w1_simulation"
    arguments = _captured_args(
        RAW_SCRIPT,
        tmp_path,
        environment={"W1_SIMULATION_ARTIFACT_ROOT": str(shared_root)},
    )
    overridden = _captured_args(
        RAW_SCRIPT,
        tmp_path,
        environment={
            "W1_SIMULATION_ARTIFACT_ROOT": str(shared_root),
            "ARTIFACT_ROOT": str(tmp_path / "one_run"),
        },
    )

    assert _values(arguments, "--artifacts") == []
    assert _values(overridden, "--artifacts") == [str(tmp_path / "one_run")]


def test_bridge_script_defaults_to_time_based_image_replay(tmp_path: Path) -> None:
    defaults = _captured_args(BRIDGE_SCRIPT, tmp_path)
    experimental = _captured_args(
        BRIDGE_SCRIPT,
        tmp_path,
        environment={"IMAGE_REPLAY_MODE": "state"},
    )

    assert _values(defaults, "--image-replay-mode") == ["time"]
    assert _values(experimental, "--image-replay-mode") == ["state"]


def test_quality_metrics_default_on_and_can_be_disabled_individually(tmp_path: Path) -> None:
    defaults = _captured_args(RAW_SCRIPT, tmp_path)
    assert _values(defaults, "--quality-metric") == [
        "pose",
        "end_effector",
        "motion_direction",
        "amplitude",
    ]

    selected = _captured_args(
        BRIDGE_SCRIPT,
        tmp_path,
        environment={
            "QUALITY_POSE": "1",
            "QUALITY_END_EFFECTOR": "0",
            "QUALITY_MOTION_DIRECTION": "1",
            "QUALITY_AMPLITUDE": "0",
        },
    )
    assert _values(selected, "--quality-metric") == ["pose", "motion_direction"]


def test_run_score_defaults_on_and_components_can_be_disabled(tmp_path: Path) -> None:
    defaults = _captured_args(BRIDGE_SCRIPT, tmp_path)
    assert "--score-smoothness" in defaults
    assert "--score-realtime" in defaults

    disabled = _captured_args(
        RAW_SCRIPT,
        tmp_path,
        environment={"SCORE_SMOOTHNESS": "0", "SCORE_REALTIME": "0"},
    )
    assert "--no-score-smoothness" in disabled
    assert "--no-score-realtime" in disabled


def test_custom_camera_arguments_replace_script_defaults(tmp_path: Path) -> None:
    custom_sources = (
        "observation.images.front=custom/front",
        "observation.images.wrist=custom/wrist",
    )
    arguments = _captured_args(
        BRIDGE_SCRIPT,
        tmp_path,
        "--camera-source",
        custom_sources[0],
        f"--camera-source={custom_sources[1]}",
    )

    captured_sources = _values(arguments, "--camera-source")
    captured_sources.extend(
        argument.removeprefix("--camera-source=")
        for argument in arguments
        if argument.startswith("--camera-source=")
    )
    assert captured_sources == list(custom_sources)


def test_hardware_simulation_branch_defaults_to_bridge_entrypoint() -> None:
    launcher = (REPO_ROOT / "w1_simulation/runtime/start_infer_lipo.sh").read_text(encoding="utf-8")

    assert 'BRIDGE_MODE="async"' in launcher
    assert "-m w1_simulation.runtime.bridge" in launcher
    assert "w1_popcorn_v1.json" in launcher
