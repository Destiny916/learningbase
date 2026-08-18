from __future__ import annotations

from pathlib import Path

import pytest
import torch
from w1_simulation.w1_profile import DEFAULT_CHECKPOINT, DEFAULT_ORIGIN


def pytest_configure(config) -> None:
    config.addinivalue_line("markers", "integration: exercises repository data, checkpoints, or CUDA")


@pytest.fixture(scope="session")
def origin_root() -> Path:
    if not DEFAULT_ORIGIN.is_dir():
        pytest.skip(f"origin recording is unavailable: {DEFAULT_ORIGIN}")
    return DEFAULT_ORIGIN


@pytest.fixture(scope="session")
def checkpoint_root() -> Path:
    if not (DEFAULT_CHECKPOINT / "model.safetensors").is_file():
        pytest.skip(f"ACT checkpoint is unavailable: {DEFAULT_CHECKPOINT}")
    return DEFAULT_CHECKPOINT


@pytest.fixture(scope="session")
def cuda_device() -> str:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the real ACT inference integration tests")
    return "cuda:0"
