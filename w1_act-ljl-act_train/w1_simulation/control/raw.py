from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class ChunkPolicy(Protocol):
    def predict_chunk(self, state: np.ndarray, images: dict[str, np.ndarray]) -> tuple[np.ndarray, float]: ...


@dataclass(frozen=True)
class ControllerStep:
    action: np.ndarray
    action_index: int
    chunk_origin_step: int
    chunk_install_step: int
    replan_submitted: bool
    replan_installed: bool
    policy_latency_ms: float


@dataclass(frozen=True)
class InferenceRecord:
    submit_step: int
    install_step: int
    chunk: np.ndarray
    latency_ms: float


class RecedingHorizonController:
    def __init__(self, policy: ChunkPolicy, replan_interval: int, asynchronous: bool = True) -> None:
        if replan_interval <= 0:
            raise ValueError("replan_interval must be positive")
        self.policy = policy
        self.replan_interval = replan_interval
        self.asynchronous = asynchronous
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="act-inference")
        self.active_chunk: np.ndarray | None = None
        self.active_submit_step = 0
        self.active_install_step = 0
        self.pending: Future[tuple[np.ndarray, float]] | None = None
        self.pending_submit_step = 0
        self.last_latency_ms = 0.0
        self.replan_count = 0
        self.inference_records: list[InferenceRecord] = []

    @staticmethod
    def _copy_images(images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {key: np.asarray(value).copy() for key, value in images.items()}

    def reset(self, state: np.ndarray, images: dict[str, np.ndarray]) -> None:
        if self.pending is not None:
            self.pending.cancel()
            self.pending = None
        chunk, latency_ms = self.policy.predict_chunk(np.asarray(state).copy(), self._copy_images(images))
        self.active_chunk = self._validated_chunk(chunk)
        self.active_submit_step = 0
        self.active_install_step = 0
        self.last_latency_ms = float(latency_ms)
        self.replan_count = 1
        self.inference_records = [
            InferenceRecord(
                submit_step=0,
                install_step=0,
                chunk=self.active_chunk.copy(),
                latency_ms=self.last_latency_ms,
            )
        ]

    @staticmethod
    def _validated_chunk(chunk: np.ndarray) -> np.ndarray:
        candidate = np.asarray(chunk, dtype=np.float32)
        if candidate.ndim != 2:
            raise ValueError(f"Expected 2D action chunk, got {candidate.shape}")
        return candidate

    def _activate_chunk(
        self,
        chunk: np.ndarray,
        latency_ms: float,
        submit_step: int,
        install_step: int,
    ) -> None:
        candidate = self._validated_chunk(chunk)
        self.active_chunk = candidate
        self.active_submit_step = submit_step
        self.active_install_step = install_step
        self.last_latency_ms = float(latency_ms)
        self.inference_records.append(
            InferenceRecord(
                submit_step=submit_step,
                install_step=install_step,
                chunk=candidate.copy(),
                latency_ms=self.last_latency_ms,
            )
        )

    def _install_ready(self, step: int) -> bool:
        if self.pending is None or not self.pending.done():
            return False
        chunk, latency_ms = self.pending.result()
        self._activate_chunk(chunk, latency_ms, self.pending_submit_step, step)
        self.pending = None
        return True

    def step(
        self,
        step: int,
        state: np.ndarray,
        images: dict[str, np.ndarray],
    ) -> ControllerStep:
        if self.active_chunk is None:
            self.reset(state, images)
        installed = self._install_ready(step)
        submitted = False
        if step > 0 and step % self.replan_interval == 0 and self.pending is None:
            self.replan_count += 1
            submitted = True
            if self.asynchronous:
                self.pending_submit_step = step
                self.pending = self.executor.submit(
                    self.policy.predict_chunk,
                    np.asarray(state).copy(),
                    self._copy_images(images),
                )
            else:
                chunk, latency_ms = self.policy.predict_chunk(
                    np.asarray(state).copy(), self._copy_images(images)
                )
                self._activate_chunk(chunk, latency_ms, step, step)
                installed = True
        if self.active_chunk is None:
            raise RuntimeError("ACT controller has no active chunk")
        action_index = step - self.active_install_step
        action_index = min(max(action_index, 0), self.active_chunk.shape[0] - 1)
        return ControllerStep(
            action=self.active_chunk[action_index].copy(),
            action_index=action_index,
            chunk_origin_step=self.active_submit_step,
            chunk_install_step=self.active_install_step,
            replan_submitted=submitted,
            replan_installed=installed,
            policy_latency_ms=self.last_latency_ms,
        )

    def close(self) -> None:
        if self.pending is not None:
            chunk, latency_ms = self.pending.result()
            candidate = self._validated_chunk(chunk)
            self.inference_records.append(
                InferenceRecord(
                    submit_step=self.pending_submit_step,
                    install_step=-1,
                    chunk=candidate.copy(),
                    latency_ms=float(latency_ms),
                )
            )
            self.pending = None
        self.executor.shutdown(wait=True, cancel_futures=False)
