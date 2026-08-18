#!/usr/bin/env python3

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import policy_bridge_act as blocking


@dataclass(frozen=True)
class TrajectoryBlock:
    actions: np.ndarray
    origin_step: int
    session_id: int
    block_id: int
    source_frame: float
    state_source: str
    inference_ms: float

    @property
    def end_step(self) -> int:
        return self.origin_step + len(self.actions) - 1

    def action_at(self, control_step: int) -> np.ndarray | None:
        index = control_step - self.origin_step
        if index < 0 or index >= len(self.actions):
            return None
        return self.actions[index]


@dataclass(frozen=True)
class InferenceResult:
    submit_step: int
    session_id: int
    source_frame: float
    state_source: str
    actions: np.ndarray | None
    inference_ms: float
    error: str | None


@dataclass
class LipoTransition:
    old_block: TrajectoryBlock | None
    new_block: TrajectoryBlock
    start_step: int
    length: int
    hold_action: np.ndarray | None


def scaled_control_points(policy_points: int, sample_factor: int) -> int:
    if isinstance(policy_points, bool) or not isinstance(policy_points, int) or policy_points < 1:
        raise ValueError("policy_points must be a positive integer")
    if isinstance(sample_factor, bool) or not isinstance(sample_factor, int) or sample_factor < 1:
        raise ValueError("sample_factor must be a positive integer")
    return policy_points * sample_factor


def lipo_body_action(
    old_action: np.ndarray,
    new_action: np.ndarray,
    alpha: float,
    body_indices: np.ndarray,
) -> np.ndarray:
    old_values = np.asarray(old_action, dtype=np.float32)
    new_values = np.asarray(new_action, dtype=np.float32)
    if old_values.shape != new_values.shape or old_values.ndim != 1:
        raise ValueError("old_action and new_action must be same-shaped 1D arrays")
    if not np.isfinite(old_values).all() or not np.isfinite(new_values).all():
        raise ValueError("LIPO actions must be finite")
    weight = float(alpha)
    if not 0.0 < weight <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    result = new_values.copy()
    result[body_indices] = old_values[body_indices] * (1.0 - weight) + new_values[body_indices] * weight
    return result


class _LoggerFilter:
    def __init__(self, logger: object) -> None:
        self._logger = logger

    def info(self, message: str, *args: object, **kwargs: object) -> object | None:
        if str(message).startswith("Blocking Policy Bridge ready:"):
            return None
        return self._logger.info(message, *args, **kwargs)

    def __getattr__(self, name: str) -> object:
        return getattr(self._logger, name)


class LipoPolicyBridgeNode(blocking.PolicyBridgeNode):
    def __init__(self) -> None:
        self._lipo_initialized = threading.Event()
        self._suppress_blocking_ready_log = True
        super().__init__()
        self._suppress_blocking_ready_log = False

        self.declare_parameter("lipo_trigger_points", 15)
        self.declare_parameter("lipo_blend_points", 6)
        trigger_points = self.get_parameter("lipo_trigger_points").value
        blend_points = self.get_parameter("lipo_blend_points").value
        if isinstance(trigger_points, bool) or not isinstance(trigger_points, int):
            raise ValueError("lipo_trigger_points must be a positive integer")
        if isinstance(blend_points, bool) or not isinstance(blend_points, int):
            raise ValueError("lipo_blend_points must be a positive integer")
        if not 1 <= trigger_points < blocking.CHUNK_SIZE:
            raise ValueError(f"lipo_trigger_points must be in [1, {blocking.CHUNK_SIZE - 1}]")
        if not 1 <= blend_points <= trigger_points:
            raise ValueError("lipo_blend_points must be in [1, lipo_trigger_points]")

        self.lipo_trigger_policy_points = trigger_points
        self.lipo_blend_policy_points = blend_points
        self.lipo_trigger_control_points = scaled_control_points(trigger_points, self.sample_factor)
        self.lipo_blend_control_points = scaled_control_points(blend_points, self.sample_factor)
        self.trajectory_horizon = blocking.CHUNK_SIZE * self.sample_factor

        gripper_indices = {
            index for index in (self.left_gripper_index, self.right_gripper_index) if index is not None
        }
        self.body_lipo_indices = np.asarray(
            [index for index in range(self.full_dim) if index not in gripper_indices],
            dtype=np.int64,
        )

        self.lipo_lock = threading.RLock()
        self.inference_pending = False
        self.inference_results: deque[InferenceResult] = deque()
        self.active_block: TrajectoryBlock | None = None
        self.transition: LipoTransition | None = None
        self.control_step = 0
        self.lipo_session = 0
        self.hold_logged = False
        self.replan_wait_logged = False

        self.get_logger().info(
            f"Asynchronous LIPO Policy Bridge ready: dim={self.full_dim} "
            f"policy={self.policy_hz:.1f}Hz control={self.control_hz:.1f}Hz "
            f"sample_factor={self.sample_factor} "
            f"trigger={self.lipo_trigger_control_points}control_points "
            f"blend={self.lipo_blend_control_points}control_points "
            f"shadow={self.shadow_mode}"
        )
        self._lipo_initialized.set()

    def get_logger(self) -> object:
        logger = super().get_logger()
        if getattr(self, "_suppress_blocking_ready_log", False):
            return _LoggerFilter(logger)
        return logger

    def _reset_lipo_state(self, session_id: int) -> None:
        with self.lipo_lock:
            self.inference_results.clear()
            self.active_block = None
            self.transition = None
            self.control_step = 0
            self.lipo_session = session_id
            self.hold_logged = False
            self.replan_wait_logged = False
        self.get_logger().info(f"LIPO state reset for subscriber session={session_id}")

    def _replan_busy(self) -> bool:
        with self.lipo_lock:
            return self.inference_pending or bool(self.inference_results)

    def _inference_worker(
        self,
        observation: dict[str, np.ndarray],
        snapshot: blocking.ObservationSnapshot,
        session_id: int,
        submit_step: int,
    ) -> None:
        actions = None
        inference_ms = 0.0
        error = None
        try:
            raw_actions, inference_ms = self.ipc.infer_chunk(observation)
            actions = blocking.interpolate_actions(raw_actions, self.sample_factor)
            expected_shape = (self.trajectory_horizon, self.full_dim)
            if actions.shape != expected_shape:
                raise ValueError(f"expected action shape {expected_shape}, got {actions.shape}")
        except Exception as exc:
            error = str(exc)
        result = InferenceResult(
            submit_step=submit_step,
            session_id=session_id,
            source_frame=snapshot.anchor_time,
            state_source=snapshot.state_source,
            actions=actions,
            inference_ms=inference_ms,
            error=error,
        )
        with self.lipo_lock:
            self.inference_pending = False
            self.inference_results.append(result)

    def _submit_replan(self, session_id: int, submit_step: int) -> bool:
        with self.lipo_lock:
            if self.inference_pending or self.inference_results:
                return False
        if not self.ipc.connected:
            return False
        snapshot = self._latest_snapshot()
        if snapshot is None:
            if not self.replan_wait_logged:
                self.get_logger().warning(
                    "LIPO replan waiting for fresh latest images and joint feedback; "
                    "the active trajectory remains valid"
                )
                self.replan_wait_logged = True
            return False
        self.replan_wait_logged = False
        observation = self._build_observation(snapshot)
        with self.lipo_lock:
            if self.inference_pending or self.inference_results:
                return False
            self.inference_pending = True
        threading.Thread(
            target=self._inference_worker,
            args=(observation, snapshot, session_id, submit_step),
            daemon=True,
        ).start()
        self.get_logger().info(
            f"LIPO replan submitted: session={session_id} submit={submit_step} "
            f"source_frame={snapshot.anchor_time:.6f} state_source={snapshot.state_source}"
        )
        return True

    def _install_ready_result(self, session_id: int) -> None:
        with self.lipo_lock:
            if not self.inference_results:
                return
            result = self.inference_results.popleft()
        if result.session_id != session_id:
            self.get_logger().info(
                f"LIPO result discarded across subscriber sessions: "
                f"result={result.session_id} current={session_id}"
            )
            return
        if result.error is not None or result.actions is None:
            self.get_logger().error(f"Asynchronous LIPO inference failed: {result.error}")
            return

        self.block_count += 1
        new_block = TrajectoryBlock(
            actions=result.actions,
            origin_step=result.submit_step,
            session_id=result.session_id,
            block_id=self.block_count,
            source_frame=result.source_frame,
            state_source=result.state_source,
            inference_ms=result.inference_ms,
        )
        discarded_prefix = max(0, self.control_step - new_block.origin_step)
        if self.control_step > new_block.end_step:
            self.get_logger().warning(
                f"LIPO result expired and was discarded: block={new_block.block_id} "
                f"submit={new_block.origin_step} current={self.control_step} "
                f"latency={new_block.inference_ms:.2f}ms"
            )
            return

        old_block = self.active_block
        old_action = old_block.action_at(self.control_step) if old_block is not None else None
        if old_action is None and self.last_command_state is None:
            self.active_block = new_block
            self.transition = None
            blend_length = 0
        else:
            overlap_end = new_block.end_step
            if old_block is not None:
                overlap_end = min(overlap_end, old_block.end_step)
            available = overlap_end - self.control_step + 1
            if old_action is None:
                available = new_block.end_step - self.control_step + 1
            blend_length = min(self.lipo_blend_control_points, max(0, available))
            if blend_length <= 0:
                self.get_logger().warning(
                    f"LIPO result has no valid transition interval: block={new_block.block_id}"
                )
                return
            self.transition = LipoTransition(
                old_block=old_block if old_action is not None else None,
                new_block=new_block,
                start_step=self.control_step,
                length=blend_length,
                hold_action=(
                    self.last_command_state.copy()
                    if old_action is None and self.last_command_state is not None
                    else None
                ),
            )

        self.get_logger().info(
            f"LIPO block installed: block={new_block.block_id} submit={new_block.origin_step} "
            f"install={self.control_step} discarded={discarded_prefix} "
            f"blend={blend_length} latency={new_block.inference_ms:.2f}ms"
        )

    def _planning_block(self) -> TrajectoryBlock | None:
        if self.transition is not None:
            return self.transition.new_block
        return self.active_block

    def _command_for_step(self) -> tuple[np.ndarray | None, bool]:
        if self.transition is not None:
            transition = self.transition
            offset = self.control_step - transition.start_step
            if offset < 0 or offset >= transition.length:
                raise RuntimeError("invalid LIPO transition offset")
            new_action = transition.new_block.action_at(self.control_step)
            if new_action is None:
                raise RuntimeError("new LIPO trajectory has no action for current step")
            old_action = (
                transition.old_block.action_at(self.control_step)
                if transition.old_block is not None
                else transition.hold_action
            )
            if old_action is None:
                raise RuntimeError("old LIPO trajectory has no action for current step")
            alpha = float(offset + 1) / float(transition.length)
            command = lipo_body_action(old_action, new_action, alpha, self.body_lipo_indices)
            if offset + 1 == transition.length:
                self.active_block = transition.new_block
                self.transition = None
            return command, True

        if self.active_block is not None:
            action = self.active_block.action_at(self.control_step)
            if action is not None:
                return action.copy(), False
        if self.last_command_state is not None:
            return self.last_command_state.copy(), False
        return None, False

    def _maybe_submit_replan(self, session_id: int) -> None:
        if self._replan_busy():
            return
        planning_block = self._planning_block()
        if planning_block is None:
            self._submit_replan(session_id, self.control_step)
            return
        remaining = planning_block.end_step - self.control_step + 1
        if remaining <= self.lipo_trigger_control_points:
            self._submit_replan(session_id, self.control_step)

    def _execution_loop(self) -> None:
        self._lipo_initialized.wait()
        period_s = 1.0 / self.control_hz
        deadline = time.monotonic()
        while not self.stop_event.is_set() and blocking.rclpy.ok():
            subscriber_present, session_id = self._subscriber_state()
            if self._consume_subscriber_reset():
                self._reset_lipo_state(session_id)
            if not subscriber_present:
                if not self.waiting_for_subscriber_logged:
                    self.get_logger().info("Waiting for an action subscriber before LIPO inference")
                    self.waiting_for_subscriber_logged = True
                deadline = time.monotonic()
                self.stop_event.wait(0.05)
                continue
            self.waiting_for_subscriber_logged = False
            if self.lipo_session != session_id:
                self._reset_lipo_state(session_id)

            self._install_ready_result(session_id)
            command, blending = self._command_for_step()
            if command is None:
                self._maybe_submit_replan(session_id)
                deadline = time.monotonic()
                self.stop_event.wait(0.01)
                continue

            self._publish_action(command)
            if self.active_block is not None and self.control_step > self.active_block.end_step:
                if not self.hold_logged:
                    self.get_logger().warning(
                        "LIPO trajectory exhausted before the next result; holding the last command"
                    )
                    self.hold_logged = True
            else:
                self.hold_logged = False

            if blending:
                transition = self.transition
                remaining_blend = (
                    0
                    if transition is None
                    else transition.length - (self.control_step - transition.start_step + 1)
                )
                if remaining_blend == 0:
                    self.get_logger().info(f"LIPO transition completed at step={self.control_step}")

            self.control_step += 1
            self._maybe_submit_replan(session_id)
            deadline += period_s
            now = time.monotonic()
            if deadline < now - period_s:
                deadline = now
            self.stop_event.wait(max(0.0, deadline - now))

    def close(self) -> None:
        self._lipo_initialized.set()
        super().close()


def main() -> None:
    if blocking.ROS_IMPORT_ERROR is not None:
        raise RuntimeError(
            "ROS2 Python dependencies are required to run policy_bridge_act_lipo.py"
        ) from blocking.ROS_IMPORT_ERROR
    blocking.rclpy.init()
    node = LipoPolicyBridgeNode()
    try:
        blocking.rclpy.spin(node)
    except (KeyboardInterrupt, blocking.ExternalShutdownException):
        pass
    finally:
        node.close()
        if blocking.rclpy.ok():
            node.destroy_node()
            blocking.rclpy.shutdown()


if __name__ == "__main__":
    main()
