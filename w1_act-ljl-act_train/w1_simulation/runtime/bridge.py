#!/usr/bin/env python3

from __future__ import annotations

import os
import threading
import time
from collections import deque

import numpy as np

from w1_simulation.runtime import bridge_base as blocking
from w1_simulation.runtime.policy_bridge_act_lipo import LipoPolicyBridgeNode

BRIDGE_MODES = ("sync", "async")


def normalize_bridge_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in BRIDGE_MODES:
        raise ValueError(f"W1_SIMULATION_BRIDGE_MODE must be one of {BRIDGE_MODES}, got {value!r}")
    return mode


class SynchronousChunkQueue:
    def __init__(self, horizon: int, action_dim: int) -> None:
        if horizon < 1 or action_dim < 1:
            raise ValueError("horizon and action_dim must be positive")
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self._actions: deque[np.ndarray] = deque()
        self._next_index = 0

    def __len__(self) -> int:
        return len(self._actions)

    def reset(self) -> None:
        self._actions.clear()
        self._next_index = 0

    def install(self, actions: np.ndarray) -> None:
        if self._actions:
            raise RuntimeError("synchronous mode cannot replace a chunk before it is exhausted")
        values = np.asarray(actions, dtype=np.float32)
        expected_shape = (self.horizon, self.action_dim)
        if values.shape != expected_shape or not np.isfinite(values).all():
            raise ValueError(
                f"expected a finite action chunk with shape {expected_shape}, got {values.shape}"
            )
        self._actions.extend(action.copy() for action in values)
        self._next_index = 0

    def pop(self) -> tuple[int, np.ndarray]:
        if not self._actions:
            raise RuntimeError("synchronous action chunk is exhausted")
        action_index = self._next_index
        self._next_index += 1
        return action_index, self._actions.popleft()


class SynchronousPolicyBridgeNode(blocking.W1PolicyBridgeBase):
    def __init__(self) -> None:
        self._sync_initialized = threading.Event()
        super().__init__()
        if self.sample_factor != 1:
            raise ValueError("synchronous raw mode requires sample_factor=1")
        self.sync_queue = SynchronousChunkQueue(blocking.CHUNK_SIZE, self.full_dim)
        self.sync_session = 0
        self.control_step = 0
        self.input_wait_logged = False
        self.get_logger().info(
            f"Synchronous Policy Bridge ready: dim={self.full_dim} "
            f"policy={self.policy_hz:.1f}Hz chunk={blocking.CHUNK_SIZE}policy_points "
            f"shadow={self.shadow_mode}"
        )
        self._sync_initialized.set()

    def _reset_sync_state(self, session_id: int) -> None:
        self.sync_queue.reset()
        self.sync_session = session_id
        self.control_step = 0
        self.input_wait_logged = False
        self.get_logger().info(f"Synchronous state reset for subscriber session={session_id}")

    def _infer_next_chunk(self, session_id: int) -> bool:
        if not self.ipc.connected:
            return False
        snapshot = self._latest_snapshot()
        if snapshot is None:
            if not self.input_wait_logged:
                self.get_logger().warning(
                    "Synchronous inference waiting for fresh latest images and joint feedback"
                )
                self.input_wait_logged = True
            return False
        self.input_wait_logged = False
        observation = self._build_observation(snapshot)
        try:
            actions, inference_ms = self.ipc.infer_chunk(observation)
            self.sync_queue.install(actions)
        except Exception as exc:
            self.get_logger().error(f"Synchronous inference failed: {exc}")
            return False

        subscriber_present, current_session = self._subscriber_state()
        if not subscriber_present or current_session != session_id:
            self.sync_queue.reset()
            self.get_logger().info(
                "Synchronous result discarded because the action subscriber session changed"
            )
            return False

        self.block_count += 1
        self.get_logger().info(
            f"Synchronous block installed: block={self.block_count} session={session_id} "
            f"source_frame={snapshot.anchor_time:.6f} state_source={snapshot.state_source} "
            f"latency={inference_ms:.2f}ms"
        )
        return True

    def _execution_loop(self) -> None:
        self._sync_initialized.wait()
        period_s = 1.0 / self.control_hz
        deadline = time.monotonic()
        while not self.stop_event.is_set() and blocking.rclpy.ok():
            subscriber_present, session_id = self._subscriber_state()
            if self._consume_subscriber_reset():
                self._reset_sync_state(session_id)
            if not subscriber_present:
                if not self.waiting_for_subscriber_logged:
                    self.get_logger().info("Waiting for an action subscriber before synchronous inference")
                    self.waiting_for_subscriber_logged = True
                deadline = time.monotonic()
                self.stop_event.wait(0.05)
                continue
            self.waiting_for_subscriber_logged = False
            if self.sync_session != session_id:
                self._reset_sync_state(session_id)

            if not self.sync_queue:
                if not self._infer_next_chunk(session_id):
                    deadline = time.monotonic()
                    self.stop_event.wait(0.01)
                    continue
                deadline = time.monotonic()

            action_index, command = self.sync_queue.pop()
            self._publish_action(command)
            self.get_logger().debug(
                f"Synchronous action emitted: block={self.block_count} "
                f"action_index={action_index} step={self.control_step}"
            )
            if not self.sync_queue:
                self.get_logger().info(
                    f"Synchronous block completed: block={self.block_count} action_index={action_index}"
                )
            self.control_step += 1
            deadline += period_s
            now = time.monotonic()
            if deadline < now - period_s:
                deadline = now
            self.stop_event.wait(max(0.0, deadline - now))

    def close(self) -> None:
        self._sync_initialized.set()
        super().close()


def bridge_node_type(mode: str) -> type[blocking.W1PolicyBridgeBase]:
    selected = normalize_bridge_mode(mode)
    return SynchronousPolicyBridgeNode if selected == "sync" else LipoPolicyBridgeNode


def main() -> None:
    if blocking.ROS_IMPORT_ERROR is not None:
        raise RuntimeError(
            "ROS2 Python dependencies are required to run bridge.py"
        ) from blocking.ROS_IMPORT_ERROR
    mode = normalize_bridge_mode(os.environ.get("W1_SIMULATION_BRIDGE_MODE", "async"))
    blocking.rclpy.init()
    node = bridge_node_type(mode)()
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
