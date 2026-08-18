from __future__ import annotations

import threading

import numpy as np
from w1_simulation.control.raw import RecedingHorizonController


class BlockingFakePolicy:
    def __init__(self, chunk_size: int = 6) -> None:
        self.chunk_size = chunk_size
        self.calls = 0
        self.release = threading.Event()
        self.seen_states: list[np.ndarray] = []
        self.seen_images: list[dict[str, np.ndarray]] = []

    def predict_chunk(self, state, images):
        call = self.calls
        self.calls += 1
        self.seen_states.append(state)
        self.seen_images.append(images)
        if call > 0:
            assert self.release.wait(timeout=5.0)
        base = call * 100
        chunk = np.repeat(np.arange(base, base + self.chunk_size)[:, None], 19, axis=1)
        return chunk.astype(np.float32), float(call + 1)


def _inputs() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    return np.zeros(19, dtype=np.float32), {"camera": np.zeros((2, 2, 3), dtype=np.uint8)}


def test_controller_uses_active_chunk_for_each_step_before_async_result_is_ready() -> None:
    policy = BlockingFakePolicy()
    controller = RecedingHorizonController(policy, replan_interval=2)
    state, images = _inputs()
    try:
        outputs = [controller.step(step, state, images) for step in range(4)]

        assert [int(output.action[0]) for output in outputs] == [0, 1, 2, 3]
        assert outputs[2].replan_submitted is True
        assert outputs[3].replan_installed is False
    finally:
        policy.release.set()
        controller.close()


def test_controller_installs_ready_chunk_from_first_action() -> None:
    policy = BlockingFakePolicy()
    controller = RecedingHorizonController(policy, replan_interval=2)
    state, images = _inputs()
    try:
        controller.step(0, state, images)
        controller.step(1, state, images)
        controller.step(2, state, images)
        policy.release.set()
        controller.pending.result(timeout=5.0)

        output = controller.step(3, state, images)

        assert output.replan_installed is True
        assert output.chunk_origin_step == 2
        assert output.chunk_install_step == 3
        assert output.action_index == 0
        assert output.action[0] == 100.0
    finally:
        controller.close()


def test_controller_installs_late_chunk_from_first_action() -> None:
    policy = BlockingFakePolicy(chunk_size=3)
    controller = RecedingHorizonController(policy, replan_interval=2)
    state, images = _inputs()
    try:
        controller.step(0, state, images)
        controller.step(1, state, images)
        controller.step(2, state, images)
        policy.release.set()
        controller.pending.result(timeout=5.0)

        output = controller.step(5, state, images)

        assert output.replan_installed is True
        assert output.chunk_origin_step == 2
        assert output.chunk_install_step == 5
        assert output.action_index == 0
        assert output.action[0] == 100.0
    finally:
        controller.close()


def test_controller_records_candidate_chunk_submit_and_install_steps() -> None:
    policy = BlockingFakePolicy()
    controller = RecedingHorizonController(policy, replan_interval=2)
    state, images = _inputs()
    try:
        controller.step(0, state, images)
        controller.step(1, state, images)
        controller.step(2, state, images)
        policy.release.set()
        controller.pending.result(timeout=5.0)

        controller.step(3, state, images)

        record = controller.inference_records[1]
        assert record.submit_step == 2
        assert record.install_step == 3
        expected = np.repeat(np.arange(100, 106, dtype=np.float32)[:, None], 19, axis=1)
        np.testing.assert_array_equal(record.chunk, expected)
    finally:
        controller.close()


def test_controller_close_records_completed_pending_chunk_as_uninstalled() -> None:
    policy = BlockingFakePolicy()
    controller = RecedingHorizonController(policy, replan_interval=2)
    state, images = _inputs()
    closed = False
    try:
        controller.step(0, state, images)
        controller.step(1, state, images)
        controller.step(2, state, images)
        policy.release.set()

        controller.close()
        closed = True

        record = controller.inference_records[1]
        assert record.submit_step == 2
        assert record.install_step == -1
        expected = np.repeat(np.arange(100, 106, dtype=np.float32)[:, None], 19, axis=1)
        np.testing.assert_array_equal(record.chunk, expected)
    finally:
        policy.release.set()
        if not closed:
            controller.close()


def test_controller_reset_restores_origin_fields_and_replan_count() -> None:
    policy = BlockingFakePolicy()
    controller = RecedingHorizonController(policy, replan_interval=2)
    state, images = _inputs()
    try:
        controller.step(0, state, images)
        policy.release.set()
        controller.reset(state, images)
        output = controller.step(0, state, images)

        assert output.chunk_origin_step == 0
        assert output.action_index == 0
        assert output.policy_latency_ms == 2.0
        assert controller.replan_count == 1
    finally:
        controller.close()


def test_controller_copies_latest_policy_inputs_for_every_async_submission() -> None:
    policy = BlockingFakePolicy()
    controller = RecedingHorizonController(policy, replan_interval=2)
    state, images = _inputs()
    try:
        controller.step(0, state, images)
        controller.step(1, state, images)

        state.fill(2.0)
        images["camera"].fill(2)
        controller.step(2, state, images)
        state.fill(20.0)
        images["camera"].fill(20)
        policy.release.set()
        controller.pending.result(timeout=5.0)

        state.fill(3.0)
        images["camera"].fill(3)
        controller.step(3, state, images)

        state.fill(4.0)
        images["camera"].fill(4)
        controller.step(4, state, images)
        state.fill(40.0)
        images["camera"].fill(40)
        controller.pending.result(timeout=5.0)

        np.testing.assert_array_equal(policy.seen_states[1], np.full(19, 2.0, dtype=np.float32))
        np.testing.assert_array_equal(policy.seen_images[1]["camera"], np.full((2, 2, 3), 2, dtype=np.uint8))
        np.testing.assert_array_equal(policy.seen_states[2], np.full(19, 4.0, dtype=np.float32))
        np.testing.assert_array_equal(policy.seen_images[2]["camera"], np.full((2, 2, 3), 4, dtype=np.uint8))
    finally:
        controller.close()


def test_synchronous_controller_uses_latest_inputs_and_installs_on_submit_step() -> None:
    policy = BlockingFakePolicy()
    controller = RecedingHorizonController(policy, replan_interval=2, asynchronous=False)
    state, images = _inputs()
    try:
        controller.step(0, state, images)
        controller.step(1, state, images)
        state.fill(2.0)
        images["camera"].fill(2)
        policy.release.set()

        output = controller.step(2, state, images)

        assert output.replan_submitted is True
        assert output.replan_installed is True
        assert output.chunk_origin_step == 2
        assert output.chunk_install_step == 2
        assert output.action_index == 0
        assert output.action[0] == 100.0
        np.testing.assert_array_equal(policy.seen_states[1], np.full(19, 2.0, dtype=np.float32))
        np.testing.assert_array_equal(policy.seen_images[1]["camera"], np.full((2, 2, 3), 2, dtype=np.uint8))
    finally:
        controller.close()
