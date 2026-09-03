"""Timing helpers for isolated 100-policy-point asynchronous execution."""
from __future__ import annotations
import numpy as np

POLICY_HORIZON = 100
SAMPLE_FACTOR = 2
CONTROL_HORIZON = 200
BLEND_CONTROL_POINTS = 30


def execution_parameters(*, sample_factor: int = 2, replan_remaining: int = 15,
                         blend_points: int = 15) -> dict[str, int]:
    """Return validated async-100 timing parameters for a launcher profile."""
    factor = int(sample_factor)
    remaining = int(replan_remaining)
    blend = int(blend_points)
    if factor < 1 or remaining < 1 or blend < 0:
        raise ValueError("async parameters must be positive (blend_points may be zero)")
    return {
        "sample_factor": factor,
        "control_horizon": POLICY_HORIZON * factor,
        "replan_remaining": remaining,
        "blend_points": blend,
    }

def expand_policy_chunk(actions, sample_factor=2):
    a=np.asarray(actions,dtype=np.float32)
    factor = int(sample_factor)
    if a.shape != (100,19) or factor < 1 or not np.isfinite(a).all():
        raise ValueError(f"expected finite (100,19), got {a.shape}")
    if factor == 1:
        return a.copy()
    x=np.arange(100,dtype=np.float32); target=np.linspace(0,99,100 * factor,dtype=np.float32)
    return np.stack([np.interp(target,x,a[:,i]) for i in range(19)],axis=1).astype(np.float32)
