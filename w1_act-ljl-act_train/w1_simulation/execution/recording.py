from __future__ import annotations

from pathlib import Path

import numpy as np


def save_trajectory(path: Path, **arrays: np.ndarray) -> None:
    np.savez_compressed(path, **arrays)
