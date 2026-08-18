from __future__ import annotations

import numpy as np
import pytest
from w1_simulation.runtime.bridge import (
    SynchronousChunkQueue,
    SynchronousPolicyBridgeNode,
    bridge_node_type,
    normalize_bridge_mode,
)
from w1_simulation.runtime.policy_bridge_act_lipo import (
    LipoPolicyBridgeNode,
    RuntimeLipoConfig,
    TrajectoryBlock,
)


def test_bridge_mode_selects_independent_schedulers() -> None:
    assert normalize_bridge_mode("sync") == "sync"
    assert normalize_bridge_mode("ASYNC") == "async"
    assert bridge_node_type("sync") is SynchronousPolicyBridgeNode
    assert bridge_node_type("async") is LipoPolicyBridgeNode
    with pytest.raises(ValueError, match="W1_SIMULATION_BRIDGE_MODE"):
        normalize_bridge_mode("raw")


def test_synchronous_queue_consumes_the_full_chunk_without_replacement() -> None:
    queue = SynchronousChunkQueue(horizon=100, action_dim=19)
    actions = np.arange(100 * 19, dtype=np.float32).reshape(100, 19)

    queue.install(actions)
    with pytest.raises(RuntimeError, match="before it is exhausted"):
        queue.install(actions)

    for expected_index in range(100):
        action_index, action = queue.pop()
        assert action_index == expected_index
        np.testing.assert_array_equal(action, actions[expected_index])

    assert len(queue) == 0
    with pytest.raises(RuntimeError, match="exhausted"):
        queue.pop()


def test_runtime_lipo_threshold_matches_async_simulation_contract() -> None:
    config = RuntimeLipoConfig(
        execution_horizon=100,
        replan_threshold=0.5,
        inference_budget_ms=300.0,
        lipo_blend_policy_points=5,
        replan_margin_policy_points=2,
        policy_hz=20.0,
        sample_factor=2,
    )

    assert config.trigger_policy_points == 50
    assert config.trigger_control_points == 100
    assert config.lipo_blend_control_points == 10
    assert config.inference_budget_policy_points == 6
    assert config.required_policy_points == 13


def test_runtime_lipo_submits_when_half_of_interpolated_chunk_remains() -> None:
    node = object.__new__(LipoPolicyBridgeNode)
    node.active_block = TrajectoryBlock(
        actions=np.zeros((200, 19), dtype=np.float32),
        origin_step=0,
        session_id=7,
        block_id=1,
        source_frame=0.0,
        state_source="test",
        inference_ms=200.0,
    )
    node.transition = None
    node.lipo_trigger_control_points = 100
    node._replan_busy = lambda: False
    submissions: list[tuple[int, int]] = []
    node._submit_replan = lambda session_id, step: not submissions.append((session_id, step))

    node.control_step = 99
    node._maybe_submit_replan(7)
    assert submissions == []

    node.control_step = 100
    node._maybe_submit_replan(7)
    assert submissions == [(7, 100)]


def test_runtime_lipo_rejects_threshold_without_safety_capacity() -> None:
    with pytest.raises(ValueError, match="does not leave enough policy points"):
        RuntimeLipoConfig(
            execution_horizon=100,
            replan_threshold=0.1,
            inference_budget_ms=300.0,
            lipo_blend_policy_points=5,
            replan_margin_policy_points=2,
            policy_hz=20.0,
            sample_factor=2,
        )
