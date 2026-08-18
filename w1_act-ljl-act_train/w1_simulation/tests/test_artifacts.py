from __future__ import annotations

import re
from pathlib import Path

import pytest
from w1_simulation.artifacts import (
    ensure_simulation_artifact_dirs,
    simulation_artifact_paths,
    simulation_run_directory,
)


def test_ephemeral_run_directory_is_removed_without_creating_artifact_root(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"

    with simulation_run_directory(artifact_root, "act_bridge", False) as run_directory:
        paths = ensure_simulation_artifact_dirs(run_directory)
        paths["summary"].write_text("temporary", encoding="utf-8")
        captured_run_directory = run_directory

        assert run_directory.parent == Path("/dev/shm")
        assert paths["summary"].is_file()
        assert not artifact_root.exists()

    assert not captured_run_directory.exists()
    assert not artifact_root.exists()


def test_saved_run_uses_one_timestamp_directory(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"

    with simulation_run_directory(artifact_root, "act_raw", True) as run_directory:
        paths = ensure_simulation_artifact_dirs(run_directory)
        paths["summary"].write_text("saved", encoding="utf-8")

    assert run_directory.parent == artifact_root.resolve() / "runs"
    assert re.fullmatch(r"\d{8}_\d{6}_\d{6}_act_raw", run_directory.name)
    assert paths["summary"].is_file()
    assert all(path == run_directory or run_directory in path.parents for path in paths.values())


def test_simulation_artifact_paths_have_fixed_per_run_names(tmp_path: Path) -> None:
    paths = simulation_artifact_paths(tmp_path)

    assert paths["summary"] == tmp_path / "summary.json"
    assert paths["trajectory"] == tmp_path / "trajectory.npz"
    assert paths["rerun"] == tmp_path / "recording.rrd"
    assert paths["verification"] == tmp_path / "verification.json"


def test_run_directory_rejects_unsafe_run_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="run_name"):
        with simulation_run_directory(tmp_path, "../escape", True):
            pass
