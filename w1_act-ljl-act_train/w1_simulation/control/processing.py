from __future__ import annotations

import inspect
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from w1_simulation.simulation.telemetry import sha256_file

STAGE_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]*")


@dataclass(frozen=True)
class ActionChunkTrace:
    raw: np.ndarray
    processed: np.ndarray
    stages: Mapping[str, np.ndarray]


@runtime_checkable
class ActionChunkProcessor(Protocol):
    name: str

    def reset(self) -> None: ...

    def process_chunk(
        self,
        raw_chunk: np.ndarray,
        previous_action: np.ndarray | None,
    ) -> ActionChunkTrace: ...

    def process_action(self, action: np.ndarray) -> np.ndarray: ...


def _validate_chunk(chunk: np.ndarray, expected_shape: tuple[int, int] | None = None) -> np.ndarray:
    array = np.asarray(chunk, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"Action chunk must be a finite 2D array, got {array.shape}")
    if expected_shape is not None and array.shape != expected_shape:
        raise ValueError(f"Action processor changed chunk shape: {array.shape}, expected {expected_shape}")
    return array


def validate_trace(trace: ActionChunkTrace, raw_chunk: np.ndarray) -> ActionChunkTrace:
    raw = _validate_chunk(trace.raw, raw_chunk.shape).copy()
    np.testing.assert_array_equal(raw, np.asarray(raw_chunk, dtype=np.float32))
    processed = _validate_chunk(trace.processed).copy()
    if processed.shape[0] == 0 or processed.shape[1] != raw.shape[1]:
        raise ValueError(
            "Action processor must preserve the action dimension and emit at least one step: "
            f"raw={raw.shape}, processed={processed.shape}"
        )
    stages: dict[str, np.ndarray] = {}
    for name, values in trace.stages.items():
        if not STAGE_NAME_PATTERN.fullmatch(name):
            raise ValueError(f"Invalid action processor stage name: {name!r}")
        expected_shape = raw.shape if name == "raw" else processed.shape
        stages[name] = _validate_chunk(values, expected_shape).copy()
    if "raw" not in stages or "processed" not in stages:
        raise ValueError("Action processor trace must expose raw and processed stages")
    np.testing.assert_array_equal(stages["raw"], raw)
    np.testing.assert_array_equal(stages["processed"], processed)
    return ActionChunkTrace(raw=raw, processed=processed, stages=stages)


class IdentityActionChunkProcessor:
    name = "raw"

    def reset(self) -> None:
        return None

    def process_chunk(
        self,
        raw_chunk: np.ndarray,
        previous_action: np.ndarray | None,
    ) -> ActionChunkTrace:
        del previous_action
        raw = _validate_chunk(raw_chunk).copy()
        return ActionChunkTrace(
            raw=raw,
            processed=raw.copy(),
            stages={"raw": raw.copy(), "processed": raw.copy()},
        )

    def process_action(self, action: np.ndarray) -> np.ndarray:
        values = np.asarray(action, dtype=np.float32)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError(f"Action must be a finite vector, got {values.shape}")
        return values.copy()


class BridgeActionChunkProcessor:
    name = "bridge_lipo_interpolation"

    def __init__(self, sample_factor: int, full_dim: int = 19) -> None:
        if isinstance(sample_factor, bool) or not isinstance(sample_factor, int) or sample_factor < 1:
            raise ValueError("sample_factor must be a positive integer")
        self.sample_factor = sample_factor
        self.full_dim = full_dim

    def reset(self) -> None:
        return None

    def process_chunk(
        self,
        raw_chunk: np.ndarray,
        previous_action: np.ndarray | None,
    ) -> ActionChunkTrace:
        del previous_action
        raw = _validate_chunk(raw_chunk).copy()
        if raw.shape[1] != self.full_dim:
            raise ValueError(f"Expected {self.full_dim} action dimensions, got {raw.shape[1]}")
        if self.sample_factor == 1:
            interpolated = raw.copy()
        else:
            source = np.linspace(0.0, 1.0, raw.shape[0], dtype=np.float64)
            target = np.linspace(0.0, 1.0, raw.shape[0] * self.sample_factor, dtype=np.float64)
            interpolated = np.stack(
                [np.interp(target, source, raw[:, index]) for index in range(raw.shape[1])],
                axis=1,
            ).astype(np.float32)
        return ActionChunkTrace(
            raw=raw,
            processed=interpolated,
            stages={"raw": raw, "interpolated": interpolated, "processed": interpolated},
        )

    def process_action(self, action: np.ndarray) -> np.ndarray:
        values = np.asarray(action, dtype=np.float32).copy()
        if values.shape != (self.full_dim,) or not np.isfinite(values).all():
            raise ValueError(f"Action must have shape ({self.full_dim},), got {values.shape}")
        return values


def build_action_processor(action_pipeline: str, sample_factor: int) -> ActionChunkProcessor:
    if action_pipeline == "raw":
        return IdentityActionChunkProcessor()
    if action_pipeline == "bridge":
        return BridgeActionChunkProcessor(sample_factor)
    raise ValueError(f"Unknown action pipeline: {action_pipeline}")


def processor_manifest(processor: ActionChunkProcessor) -> dict[str, object]:
    source = inspect.getsourcefile(type(processor))
    if source is None:
        raise ValueError(f"Could not resolve action processor source: {type(processor).__name__}")
    source_path = Path(source).resolve()
    return {
        "name": processor.name,
        "selection": "fixed_by_action_pipeline",
        "class": f"{type(processor).__module__}.{type(processor).__qualname__}",
        "source": str(source_path),
        "source_sha256": sha256_file(source_path),
    }
