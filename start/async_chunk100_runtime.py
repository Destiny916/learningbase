"""Pure timing helpers for the isolated ACT 100-point async runtime."""

from __future__ import annotations

import numpy as np

POLICY_HORIZON = 100
SAMPLE_FACTOR = 2
CONTROL_HORIZON = POLICY_HORIZON * SAMPLE_FACTOR
REPLAN_REMAINING_POLICY_POINTS = 15
REPLAN_REMAINING_CONTROL_POINTS = REPLAN_REMAINING_POLICY_POINTS * SAMPLE_FACTOR
BLEND_POLICY_POINTS = 15
BLEND_CONTROL_POINTS = BLEND_POLICY_POINTS * SAMPLE_FACTOR
CHUNK_SIZE_THRESHOLD = (
    CONTROL_HORIZON - REPLAN_REMAINING_CONTROL_POINTS
) / CONTROL_HORIZON


def expand_policy_chunk(actions: object, sample_factor: int = SAMPLE_FACTOR) -> np.ndarray:
    array = np.asarray(actions, dtype=np.float32)
    if array.shape != (POLICY_HORIZON, 19):
        raise ValueError(f"expected (100,19), got {array.shape}")
    if sample_factor < 1 or not np.isfinite(array).all():
        raise ValueError("invalid sample_factor or non-finite action")
    if sample_factor == 1:
        return array.copy()
    x = np.arange(POLICY_HORIZON, dtype=np.float32)
    target = np.linspace(0, POLICY_HORIZON - 1, CONTROL_HORIZON, dtype=np.float32)
    return np.stack([np.interp(target, x, array[:, i]) for i in range(19)], axis=1).astype(np.float32)


def should_prefetch(queue_size: int, remaining: int = REPLAN_REMAINING_CONTROL_POINTS) -> bool:
    return int(queue_size) <= int(remaining)


def align_and_blend(
    *, old_queue: dict[int, np.ndarray], new_actions: np.ndarray,
    chunk_start_timestep: int, latest_executed_timestep: int,
) -> tuple[np.ndarray, int, int]:
    new_array = np.asarray(new_actions, dtype=np.float32)
    if new_array.shape != (CONTROL_HORIZON, 19) or not np.isfinite(new_array).all():
        raise ValueError(f"expected new actions (200,19), got {new_array.shape}")
    first_live = max(0, int(latest_executed_timestep) - int(chunk_start_timestep) + 1)
    if first_live >= CONTROL_HORIZON:
        return np.empty((0, 19), dtype=np.float32), first_live, 0
    live = new_array[first_live:].copy()
    blend_len = min(BLEND_CONTROL_POINTS, len(live))
    if blend_len and old_queue:
        old_values = []
        for offset in range(blend_len):
            timestep = int(chunk_start_timestep) + first_live + offset
            if timestep not in old_queue:
                blend_len = offset
                break
            old_values.append(np.asarray(old_queue[timestep], dtype=np.float32))
        if blend_len == 0:
            return live, first_live, 0
        old_values = old_values[:blend_len]
        old_array = np.stack(old_values)
        weights = np.arange(1, blend_len + 1, dtype=np.float32) / float(BLEND_CONTROL_POINTS)
        live[:blend_len] = old_array * (1.0 - weights[:, None]) + live[:blend_len] * weights[:, None]
    return live, first_live, blend_len
