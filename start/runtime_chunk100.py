"""Contracts for the isolated ACT 200000 full 100-step deployment."""

from __future__ import annotations

import numpy as np


ACTION_HORIZON_100 = 100
ACTION_DIM = 19


def validate_action_chunk_100(actions: object) -> np.ndarray:
    array = np.asarray(actions, dtype=np.float32)
    if array.shape != (ACTION_HORIZON_100, ACTION_DIM):
        raise ValueError(
            f"action chunk must have shape ({ACTION_HORIZON_100}, {ACTION_DIM}), "
            f"got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("action chunk must contain only finite values")
    return array

