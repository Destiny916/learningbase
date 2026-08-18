from __future__ import annotations


def resolve_execution_horizon(
    prediction_horizon: int,
    checkpoint_execution_horizon: int,
    runtime_execution_horizon: int,
) -> tuple[int, str]:
    if isinstance(runtime_execution_horizon, bool) or not isinstance(runtime_execution_horizon, int):
        raise ValueError("execution_horizon must be an integer")
    execution_horizon = (
        checkpoint_execution_horizon if runtime_execution_horizon == 0 else runtime_execution_horizon
    )
    if not 1 <= execution_horizon <= prediction_horizon:
        raise ValueError(
            "execution_horizon must be zero or within the checkpoint prediction horizon: "
            f"prediction={prediction_horizon}, execution={runtime_execution_horizon}"
        )
    source = "checkpoint_config" if runtime_execution_horizon == 0 else "runtime_override"
    return execution_horizon, source
