from __future__ import annotations

import hashlib
import math
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from w1_simulation.control.processing import ActionChunkProcessor, ActionChunkTrace, validate_trace
from w1_simulation.control.raw import ChunkPolicy


@dataclass(frozen=True)
class ActionInferenceRecord:
    submit_step: int
    install_step: int
    latency_ms: float
    seed_state: np.ndarray
    image_sha256: tuple[str, ...]
    trace: ActionChunkTrace


@dataclass(frozen=True)
class ActionControllerStep:
    action: np.ndarray
    queue_action: np.ndarray
    record_index: int
    action_index: int
    replan_submitted: bool
    replan_installed: bool
    held_last_command: bool
    queue_size: int
    policy_latency_ms: float
    candidate_count: int
    observation_age_steps: int
    target_step_error: int
    discarded_prefix_steps: int
    blend_active: bool = False
    blend_alpha: float = 1.0
    old_record_index: int = -1
    new_record_index: int = -1


@dataclass(frozen=True)
class LipoControllerConfig:
    simulated_inference_ms: float = 200.0
    inference_budget_ms: float = 300.0
    replan_threshold: float = 0.5
    lipo_blend_policy_points: int = 5
    replan_margin_policy_points: int = 2
    policy_hz: float = 30.0
    sample_factor: int = 2
    body_dimensions: int = 17
    execution_horizon: int = 100

    def __post_init__(self) -> None:
        if not math.isfinite(self.simulated_inference_ms) or self.simulated_inference_ms < 0.0:
            raise ValueError("simulated_inference_ms must be finite and non-negative")
        if not math.isfinite(self.inference_budget_ms) or self.inference_budget_ms < 0.0:
            raise ValueError("inference_budget_ms must be finite and non-negative")
        if not math.isfinite(self.policy_hz) or self.policy_hz <= 0.0:
            raise ValueError("policy_hz must be finite and positive")
        if not math.isfinite(self.replan_threshold) or not 0.0 < self.replan_threshold < 1.0:
            raise ValueError("replan_threshold must be finite and in (0, 1)")
        if (
            isinstance(self.sample_factor, bool)
            or not isinstance(self.sample_factor, int)
            or self.sample_factor < 1
        ):
            raise ValueError("sample_factor must be a positive integer")
        if (
            isinstance(self.execution_horizon, bool)
            or not isinstance(self.execution_horizon, int)
            or self.execution_horizon <= 0
        ):
            raise ValueError("execution_horizon must be a positive integer")
        if not 1 <= self.trigger_policy_points < self.execution_horizon:
            raise ValueError(
                "replan_threshold resolves outside the execution horizon: "
                f"threshold={self.replan_threshold}, trigger={self.trigger_policy_points}, "
                f"horizon={self.execution_horizon}"
            )
        if (
            isinstance(self.lipo_blend_policy_points, bool)
            or not isinstance(self.lipo_blend_policy_points, int)
            or not 1 <= self.lipo_blend_policy_points <= self.trigger_policy_points
        ):
            raise ValueError("lipo_blend_policy_points must be in [1, trigger_policy_points]")
        if (
            isinstance(self.replan_margin_policy_points, bool)
            or not isinstance(self.replan_margin_policy_points, int)
            or self.replan_margin_policy_points < 0
        ):
            raise ValueError("replan_margin_policy_points must be a non-negative integer")
        if self.required_policy_points > self.trigger_policy_points:
            raise ValueError(
                "Replan threshold does not leave enough policy points for the inference budget, "
                "LIPO blend, and safety margin: "
                f"required={self.required_policy_points}, trigger={self.trigger_policy_points}"
            )
        if self.body_dimensions <= 0:
            raise ValueError("body_dimensions must be positive")

    @property
    def trigger_policy_points(self) -> int:
        return math.ceil(self.execution_horizon * self.replan_threshold)

    @property
    def trigger_control_points(self) -> int:
        return self.trigger_policy_points * self.sample_factor

    @property
    def lipo_blend_control_points(self) -> int:
        return self.lipo_blend_policy_points * self.sample_factor

    @property
    def inference_budget_policy_points(self) -> int:
        return math.ceil(self.inference_budget_ms / 1000.0 * self.policy_hz)

    @property
    def required_policy_points(self) -> int:
        return (
            self.inference_budget_policy_points
            + self.lipo_blend_policy_points
            + self.replan_margin_policy_points
        )

    @property
    def available_policy_points(self) -> int:
        return self.trigger_policy_points - self.required_policy_points


@dataclass(frozen=True)
class _PendingRequest:
    submit_step: int
    seed_state: np.ndarray
    image_sha256: tuple[str, ...]
    future: Future[tuple[np.ndarray, float]]


@dataclass(frozen=True)
class _ActiveTrajectory:
    record_index: int
    origin_step: int


@dataclass(frozen=True)
class _LipoTransition:
    old_trajectory: _ActiveTrajectory | None
    new_trajectory: _ActiveTrajectory
    start_step: int
    length: int
    hold_action: np.ndarray | None


class ActionChunkController:
    """Synchronous ACT controller used only by the raw-policy validation path."""

    def __init__(
        self,
        policy: ChunkPolicy,
        processor: ActionChunkProcessor,
        *,
        low_watermark: int | None = None,
        replan_interval: int | None = None,
        asynchronous: bool = False,
    ) -> None:
        if (low_watermark is None) == (replan_interval is None):
            raise ValueError("Select exactly one raw ACT schedule")
        if low_watermark is not None and low_watermark < 0:
            raise ValueError("low_watermark must be non-negative")
        if replan_interval is not None and replan_interval <= 0:
            raise ValueError("replan_interval must be positive")
        self.policy = policy
        self.processor = processor
        self.low_watermark = low_watermark
        self.replan_interval = replan_interval
        self.asynchronous = bool(asynchronous)
        self.schedule_mode = (
            "receding_horizon_replace" if replan_interval is not None else "fifo_low_watermark"
        )
        self.inference_records: list[ActionInferenceRecord] = []
        self.plan_queue: deque[np.ndarray] = deque()
        self._provenance: deque[tuple[int, int]] = deque()
        self.last_command: np.ndarray | None = None
        self.last_latency_ms = 0.0
        self.replan_count = 0

    @staticmethod
    def _copy_images(images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {key: np.asarray(value).copy() for key, value in images.items()}

    @staticmethod
    def _image_hashes(images: dict[str, np.ndarray]) -> tuple[str, ...]:
        return tuple(
            hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest() for image in images.values()
        )

    def observation_state(self, feedback_state: np.ndarray) -> np.ndarray:
        return np.asarray(feedback_state, dtype=np.float32).copy()

    def _predict(self, state: np.ndarray, images: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
        started = time.monotonic()
        chunk, reported_latency_ms = self.policy.predict_chunk(state, images)
        elapsed_ms = (time.monotonic() - started) * 1000.0
        return chunk, max(float(reported_latency_ms), elapsed_ms)

    def _predict_sync(self, step: int, state: np.ndarray, images: dict[str, np.ndarray]) -> None:
        seed_state = np.asarray(state, dtype=np.float32).copy()
        images_copy = self._copy_images(images)
        chunk, latency_ms = self._predict(seed_state.copy(), images_copy)
        trace = validate_trace(self.processor.process_chunk(chunk, self.last_command), chunk)
        if self.replan_interval is not None:
            self.plan_queue.clear()
            self._provenance.clear()
        record_index = len(self.inference_records)
        self.inference_records.append(
            ActionInferenceRecord(
                submit_step=step,
                install_step=step,
                latency_ms=latency_ms,
                seed_state=seed_state,
                image_sha256=self._image_hashes(images_copy),
                trace=trace,
            )
        )
        for index, action in enumerate(trace.processed):
            self.plan_queue.append(action.copy())
            self._provenance.append((record_index, index))
        self.last_latency_ms = latency_ms
        self.replan_count += 1

    def reset(self, state: np.ndarray, images: dict[str, np.ndarray]) -> None:
        self.processor.reset()
        self.plan_queue.clear()
        self._provenance.clear()
        self.last_command = None
        self.inference_records.clear()
        self.last_latency_ms = 0.0
        self.replan_count = 0
        self._predict_sync(0, state, images)

    def step(self, step: int, state: np.ndarray, images: dict[str, np.ndarray]) -> ActionControllerStep:
        submitted = False
        installed = False
        if self.replan_interval is not None and step > 0 and step % self.replan_interval == 0:
            self._predict_sync(step, state, images)
            submitted = installed = True
        if not self.plan_queue:
            self._predict_sync(step, state, images)
            submitted = installed = True
        queue_action = self.plan_queue.popleft()
        record_index, action_index = self._provenance.popleft()
        self.last_command = queue_action.copy()
        action = np.asarray(self.processor.process_action(queue_action), dtype=np.float32)
        if self.low_watermark is not None and len(self.plan_queue) <= self.low_watermark:
            self._predict_sync(step, state, images)
            submitted = installed = True
        return ActionControllerStep(
            action=action,
            queue_action=queue_action.copy(),
            record_index=record_index,
            action_index=action_index,
            replan_submitted=submitted,
            replan_installed=installed,
            held_last_command=False,
            queue_size=len(self.plan_queue),
            policy_latency_ms=self.last_latency_ms,
            candidate_count=1,
            observation_age_steps=step - self.inference_records[record_index].submit_step,
            target_step_error=0,
            discarded_prefix_steps=0,
            new_record_index=record_index,
        )

    def close(self) -> None:
        return None


class LipoActionChunkController:
    """Asynchronous single-trajectory LIPO controller used by the simulator."""

    def __init__(
        self,
        policy: ChunkPolicy,
        processor: ActionChunkProcessor,
        *,
        config: LipoControllerConfig,
        asynchronous: bool = True,
        published_body_indices: np.ndarray | None = None,
    ) -> None:
        self.policy = policy
        self.processor = processor
        self.config = config
        self.asynchronous = bool(asynchronous)
        self.schedule_mode = "remaining_ratio_absolute_step_lipo"
        self.low_watermark = None
        self.replan_interval = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="act-lipo-inference")
        self.pending: _PendingRequest | None = None
        self.inference_records: list[ActionInferenceRecord] = []
        self.active: _ActiveTrajectory | None = None
        self.transition: _LipoTransition | None = None
        self.last_command: np.ndarray | None = None
        self.last_latency_ms = 0.0
        self.replan_count = 0
        selected = (
            np.arange(self.config.body_dimensions, dtype=np.int64)
            if published_body_indices is None
            else np.asarray(published_body_indices, dtype=np.int64)
        )
        if selected.ndim != 1 or len(np.unique(selected)) != len(selected):
            raise ValueError("published_body_indices must be a unique 1D sequence")
        if np.any(selected < 0) or np.any(selected >= self.config.body_dimensions):
            raise ValueError("published_body_indices exceed the ACT body dimensions")
        self.published_body_indices = selected.copy()
        self.initial_state: np.ndarray | None = None

    @staticmethod
    def _copy_images(images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {key: np.asarray(value).copy() for key, value in images.items()}

    @staticmethod
    def _image_hashes(images: dict[str, np.ndarray]) -> tuple[str, ...]:
        return tuple(
            hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest() for image in images.values()
        )

    def observation_state(self, feedback_state: np.ndarray) -> np.ndarray:
        if self.last_command is None:
            return np.asarray(feedback_state, dtype=np.float32).copy()
        if self.initial_state is None:
            raise RuntimeError("LIPO controller has no session initial state")
        state = self.initial_state.copy()
        state[self.published_body_indices] = self.last_command[self.published_body_indices]
        state[self.config.body_dimensions :] = self.last_command[self.config.body_dimensions :]
        return state

    def _predict(self, state: np.ndarray, images: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
        started = time.monotonic()
        chunk, reported_latency_ms = self.policy.predict_chunk(state, images)
        remaining = self.config.simulated_inference_ms / 1000.0 - (time.monotonic() - started)
        if remaining > 0.0:
            time.sleep(remaining)
        elapsed_ms = (time.monotonic() - started) * 1000.0
        return chunk, max(float(reported_latency_ms), elapsed_ms)

    def _record(
        self,
        chunk: np.ndarray,
        latency_ms: float,
        *,
        submit_step: int,
        install_step: int,
        seed_state: np.ndarray,
        image_sha256: tuple[str, ...],
    ) -> int:
        trace = validate_trace(self.processor.process_chunk(chunk, None), chunk)
        record_index = len(self.inference_records)
        self.inference_records.append(
            ActionInferenceRecord(
                submit_step=submit_step,
                install_step=install_step,
                latency_ms=float(latency_ms),
                seed_state=np.asarray(seed_state, dtype=np.float32).copy(),
                image_sha256=tuple(image_sha256),
                trace=trace,
            )
        )
        self.last_latency_ms = float(latency_ms)
        return record_index

    def _predict_sync(self, state: np.ndarray, images: dict[str, np.ndarray]) -> int:
        seed_state = np.asarray(state, dtype=np.float32).copy()
        images_copy = self._copy_images(images)
        chunk, latency_ms = self._predict(seed_state.copy(), images_copy)
        self.replan_count += 1
        return self._record(
            chunk,
            latency_ms,
            submit_step=0,
            install_step=0,
            seed_state=seed_state,
            image_sha256=self._image_hashes(images_copy),
        )

    def reset(self, state: np.ndarray, images: dict[str, np.ndarray]) -> None:
        if self.pending is not None:
            self.pending.future.cancel()
        self.pending = None
        self.processor.reset()
        self.inference_records.clear()
        self.active = None
        self.transition = None
        self.last_command = None
        self.last_latency_ms = 0.0
        self.replan_count = 0
        self.initial_state = np.asarray(state, dtype=np.float32).copy()
        record_index = self._predict_sync(self.initial_state, images)
        self.active = _ActiveTrajectory(record_index, 0)

    def _trajectory_action(self, trajectory: _ActiveTrajectory, step: int) -> np.ndarray | None:
        actions = self.inference_records[trajectory.record_index].trace.processed
        index = step - trajectory.origin_step
        if not 0 <= index < len(actions):
            return None
        return actions[index]

    def _trajectory_end(self, trajectory: _ActiveTrajectory) -> int:
        return (
            trajectory.origin_step + len(self.inference_records[trajectory.record_index].trace.processed) - 1
        )

    def _submit(self, step: int, state: np.ndarray, images: dict[str, np.ndarray]) -> bool:
        if self.pending is not None:
            return False
        seed_state = self.observation_state(state)
        images_copy = self._copy_images(images)
        image_hashes = self._image_hashes(images_copy)
        self.replan_count += 1
        if self.asynchronous:
            self.pending = _PendingRequest(
                submit_step=step,
                seed_state=seed_state,
                image_sha256=image_hashes,
                future=self.executor.submit(self._predict, seed_state.copy(), images_copy),
            )
            return True
        chunk, latency_ms = self._predict(seed_state.copy(), images_copy)
        future: Future[tuple[np.ndarray, float]] = Future()
        future.set_result((chunk, latency_ms))
        self.pending = _PendingRequest(step, seed_state, image_hashes, future)
        return True

    def _install_ready(self, step: int) -> bool:
        if self.pending is None or not self.pending.future.done():
            return False
        request = self.pending
        chunk, latency_ms = request.future.result()
        self.pending = None
        record_index = self._record(
            chunk,
            latency_ms,
            submit_step=request.submit_step,
            install_step=step,
            seed_state=request.seed_state,
            image_sha256=request.image_sha256,
        )
        new_trajectory = _ActiveTrajectory(record_index, request.submit_step)
        if step > self._trajectory_end(new_trajectory):
            return True
        old_action = self._trajectory_action(self.active, step) if self.active is not None else None
        available_new = self._trajectory_end(new_trajectory) - step + 1
        blend_length = min(self.config.lipo_blend_control_points, max(0, available_new))
        if old_action is None and self.last_command is None:
            self.active = new_trajectory
            self.transition = None
        elif blend_length > 0:
            self.transition = _LipoTransition(
                old_trajectory=self.active if old_action is not None else None,
                new_trajectory=new_trajectory,
                start_step=step,
                length=blend_length,
                hold_action=(
                    self.last_command.copy()
                    if self.last_command is not None
                    else np.asarray(old_action, dtype=np.float32).copy()
                ),
            )
        return True

    def _planning_trajectory(self) -> _ActiveTrajectory | None:
        return self.transition.new_trajectory if self.transition is not None else self.active

    def _maybe_submit(self, step: int, state: np.ndarray, images: dict[str, np.ndarray]) -> bool:
        if self.pending is not None:
            return False
        trajectory = self._planning_trajectory()
        if trajectory is None:
            return self._submit(step, state, images)
        remaining = self._trajectory_end(trajectory) - step + 1
        if remaining <= self.config.trigger_control_points:
            return self._submit(step, state, images)
        return False

    def _command(self, step: int) -> tuple[np.ndarray | None, int, int, bool, float, int, int]:
        if self.transition is not None:
            transition = self.transition
            offset = step - transition.start_step
            new_action = self._trajectory_action(transition.new_trajectory, step)
            old_action = (
                self._trajectory_action(transition.old_trajectory, step)
                if transition.old_trajectory is not None
                else None
            )
            if old_action is None:
                old_action = transition.hold_action
            if new_action is None or not 0 <= offset < transition.length:
                raise RuntimeError("LIPO transition lost its absolute-step action")
            alpha = float(offset + 1) / float(transition.length)
            action = new_action.copy()
            body = slice(0, self.config.body_dimensions)
            action[body] = old_action[body] * (1.0 - alpha) + new_action[body] * alpha
            old_record_index = (
                transition.old_trajectory.record_index if transition.old_trajectory is not None else -1
            )
            new_record_index = transition.new_trajectory.record_index
            action_index = step - transition.new_trajectory.origin_step
            if offset + 1 == transition.length:
                self.active = transition.new_trajectory
                self.transition = None
            return action, new_record_index, action_index, True, alpha, old_record_index, new_record_index
        if self.active is not None:
            action = self._trajectory_action(self.active, step)
            if action is not None:
                return (
                    action.copy(),
                    self.active.record_index,
                    step - self.active.origin_step,
                    False,
                    1.0,
                    -1,
                    self.active.record_index,
                )
        return None, -1, -1, False, 0.0, -1, -1

    def step(self, step: int, state: np.ndarray, images: dict[str, np.ndarray]) -> ActionControllerStep:
        installed = self._install_ready(step)
        submitted = self._maybe_submit(step, state, images)
        action, record_index, action_index, blending, alpha, old_index, new_index = self._command(step)
        held = action is None
        if held:
            if self.last_command is None:
                raise RuntimeError("LIPO controller has neither a trajectory nor a previous command")
            action = self.last_command.copy()
        else:
            self.last_command = action.copy()
        effective_action = np.asarray(self.processor.process_action(action), dtype=np.float32)
        if effective_action.shape != action.shape or not np.isfinite(effective_action).all():
            raise ValueError("Action processor returned an invalid emitted action")
        if record_index >= 0:
            record = self.inference_records[record_index]
            observation_age = step - record.submit_step
            target_error = step - (record.submit_step + action_index)
            discarded = max(record.install_step - record.submit_step, 0)
        else:
            observation_age = target_error = discarded = -1
        return ActionControllerStep(
            action=effective_action,
            queue_action=action.copy(),
            record_index=record_index,
            action_index=action_index,
            replan_submitted=submitted,
            replan_installed=installed,
            held_last_command=held,
            queue_size=int(self.pending is not None),
            policy_latency_ms=self.last_latency_ms,
            candidate_count=int(record_index >= 0),
            observation_age_steps=observation_age,
            target_step_error=target_error,
            discarded_prefix_steps=discarded,
            blend_active=blending,
            blend_alpha=alpha,
            old_record_index=old_index,
            new_record_index=new_index,
        )

    def close(self) -> None:
        if self.pending is not None:
            request = self.pending
            chunk, latency_ms = request.future.result()
            self._record(
                chunk,
                latency_ms,
                submit_step=request.submit_step,
                install_step=-1,
                seed_state=request.seed_state,
                image_sha256=request.image_sha256,
            )
            self.pending = None
        self.executor.shutdown(wait=True, cancel_futures=False)


BridgeFaithfulController = LipoActionChunkController
BridgeInferenceRecord = ActionInferenceRecord
BridgeControllerStep = ActionControllerStep
