from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


def simulation_artifact_paths(run_directory: Path) -> dict[str, Path]:
    return {
        "root": run_directory,
        "generated": run_directory / "generated",
        "tensorboard": run_directory / "tensorboard",
        "logs": run_directory / "logs",
        "summary": run_directory / "summary.json",
        "trajectory": run_directory / "trajectory.npz",
        "rerun": run_directory / "recording.rrd",
        "verification": run_directory / "verification.json",
    }


def ensure_simulation_artifact_dirs(run_directory: Path) -> dict[str, Path]:
    paths = simulation_artifact_paths(run_directory)
    for key in ("root", "generated", "tensorboard", "logs"):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


@contextmanager
def simulation_run_directory(
    artifact_root: Path,
    run_name: str,
    save_artifacts: bool,
) -> Iterator[Path]:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("run_name may contain only letters, digits, underscores, and hyphens")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if save_artifacts:
        run_directory = artifact_root.resolve() / "runs" / f"{timestamp}_{run_name}"
        run_directory.mkdir(parents=True, exist_ok=False)
        yield run_directory
        return
    shared_memory = Path("/dev/shm")
    temporary_parent = (
        shared_memory if shared_memory.is_dir() and os.access(shared_memory, os.W_OK) else None
    )
    with tempfile.TemporaryDirectory(
        prefix=f"w1_simulation_{timestamp}_{run_name}_",
        dir=temporary_parent,
    ) as directory:
        yield Path(directory)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: npt.ArrayLike) -> str:
    contiguous = np.ascontiguousarray(array)
    return sha256_bytes(contiguous.tobytes())


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
