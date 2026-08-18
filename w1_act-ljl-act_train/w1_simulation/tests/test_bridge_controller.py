from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
from w1_simulation.control.bridge import (
    ActionChunkController,
    LipoActionChunkController,
    LipoControllerConfig,
)
from w1_simulation.control.processing import BridgeActionChunkProcessor, IdentityActionChunkProcessor
from w1_simulation.control.scheduling import format_bridge_inference_log as _format_bridge_inference_log


class RecordingPolicy:
    def __init__(self, chunks: list[np.ndarray]) -> None:
        self.chunks = [np.asarray(chunk, dtype=np.float32) for chunk in chunks]
        self.calls: list[tuple[np.ndarray, dict[str, np.ndarray]]] = []

    def predict_chunk(self, state: np.ndarray, images: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
        self.calls.append(
            (
                np.asarray(state, dtype=np.float32).copy(),
                {key: np.asarray(image).copy() for key, image in images.items()},
            )
        )
        return self.chunks[len(self.calls) - 1].copy(), 1.0


class BlockingPolicy(RecordingPolicy):
    def __init__(self, chunks: list[np.ndarray]) -> None:
        super().__init__(chunks)
        self.started = threading.Event()
        self.release = threading.Event()

    def predict_chunk(self, state: np.ndarray, images: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
        if self.calls:
            self.started.set()
            self.release.wait(timeout=5)
        return super().predict_chunk(state, images)


def _images(value: int) -> dict[str, np.ndarray]:
    return {"camera": np.full((2, 2, 3), value, dtype=np.uint8)}


def _chunk(offset: float = 0.0, length: int = 30) -> np.ndarray:
    values = np.zeros((length, 19), dtype=np.float32)
    for index in range(length):
        values[index, :17] = offset + index
        values[index, 17] = offset + 1000.0 + index
        values[index, 18] = offset + 2000.0 + index
    return values


def _config() -> LipoControllerConfig:
    return LipoControllerConfig(
        simulated_inference_ms=0.0,
        inference_budget_ms=0.0,
        replan_threshold=0.5,
        lipo_blend_policy_points=5,
        replan_margin_policy_points=2,
        policy_hz=30.0,
        sample_factor=2,
        body_dimensions=17,
        execution_horizon=30,
    )


def _controller(
    policy: RecordingPolicy,
    *,
    asynchronous: bool = True,
) -> LipoActionChunkController:
    return LipoActionChunkController(
        policy,
        BridgeActionChunkProcessor(sample_factor=2),
        config=_config(),
        asynchronous=asynchronous,
    )


def test_reset_bootstraps_first_chunk_synchronously_from_feedback_state() -> None:
    feedback = np.arange(19, dtype=np.float32)
    policy = RecordingPolicy([_chunk()])
    controller = _controller(policy)

    controller.reset(feedback, _images(7))
    first = controller.step(0, np.full(19, -1.0, dtype=np.float32), _images(8))
    controller.close()

    assert len(policy.calls) == 1
    np.testing.assert_array_equal(policy.calls[0][0], feedback)
    assert int(policy.calls[0][1]["camera"][0, 0, 0]) == 7
    assert first.record_index == 0
    assert first.action_index == 0
    assert first.replan_submitted is False


def test_replan_submits_when_active_trajectory_has_thirty_control_points_remaining() -> None:
    policy = BlockingPolicy([_chunk(), _chunk(100.0)])
    controller = _controller(policy)
    controller.reset(np.zeros(19, dtype=np.float32), _images(0))

    controls = [controller.step(step, np.zeros(19, dtype=np.float32), _images(step)) for step in range(31)]
    assert policy.started.wait(timeout=5)
    policy.release.set()
    controller.close()

    assert not any(control.replan_submitted for control in controls[:30])
    assert controls[30].replan_submitted is True
    assert controller.inference_records[1].submit_step == 30


def test_hundred_point_chunk_replans_at_half_horizon_remaining() -> None:
    chunks = [_chunk(length=100), _chunk(100.0, length=100)]
    policy = BlockingPolicy(chunks)
    controller = LipoActionChunkController(
        policy,
        BridgeActionChunkProcessor(sample_factor=2),
        config=LipoControllerConfig(
            simulated_inference_ms=0.0,
            inference_budget_ms=300.0,
            replan_threshold=0.5,
            lipo_blend_policy_points=5,
            replan_margin_policy_points=2,
            policy_hz=30.0,
            sample_factor=2,
            body_dimensions=17,
            execution_horizon=100,
        ),
        asynchronous=True,
    )
    controller.reset(np.zeros(19, dtype=np.float32), _images(0))

    controls = [controller.step(step, np.zeros(19, dtype=np.float32), _images(step)) for step in range(101)]
    assert policy.started.wait(timeout=5)
    policy.release.set()
    controller.close()

    assert controller.inference_records[0].trace.raw.shape == (100, 19)
    assert controller.inference_records[0].trace.processed.shape == (200, 19)
    assert not any(control.replan_submitted for control in controls[:100])
    assert controls[100].replan_submitted is True
    assert controller.inference_records[1].submit_step == 100


def test_bridge_inference_log_reports_one_completed_model_call() -> None:
    record = SimpleNamespace(
        submit_step=170,
        install_step=183,
        latency_ms=200.284,
        trace=SimpleNamespace(
            raw=np.zeros((100, 19), dtype=np.float32),
            processed=np.zeros((200, 19), dtype=np.float32),
        ),
    )

    assert _format_bridge_inference_log(1, record, 13) == (
        "ACT_SIM_INFERENCE pipeline=bridge inference_index=1 submit_step=170 install_step=183 "
        "raw_points=100 control_points=200 discarded_prefix_steps=13 action_index=13 e2e_ms=200.28"
    )


def test_async_controller_allows_only_one_request_in_flight() -> None:
    policy = BlockingPolicy([_chunk(), _chunk(100.0), _chunk(200.0)])
    controller = _controller(policy)
    controller.reset(np.zeros(19, dtype=np.float32), _images(0))

    controls = [controller.step(step, np.zeros(19, dtype=np.float32), _images(step)) for step in range(61)]
    assert policy.started.wait(timeout=5)
    policy.release.set()
    controller.close()

    assert sum(control.replan_submitted for control in controls) == 1
    assert len(policy.calls) == 2
    assert controller.replan_count == 2


def test_install_step_skips_expired_prefix_by_absolute_action_index() -> None:
    first = _chunk()
    second = _chunk(100.0)
    policy = BlockingPolicy([first, second])
    controller = _controller(policy)
    controller.reset(np.zeros(19, dtype=np.float32), _images(0))
    controls = [controller.step(step, np.zeros(19, dtype=np.float32), _images(step)) for step in range(31)]
    assert controls[-1].replan_submitted is True
    assert policy.started.wait(timeout=5)
    for step in range(31, 38):
        controller.step(step, np.zeros(19, dtype=np.float32), _images(step))
    policy.release.set()
    assert controller.pending is not None
    controller.pending.future.result(timeout=5)

    installed = controller.step(38, np.zeros(19, dtype=np.float32), _images(38))
    controller.close()

    assert installed.replan_installed is True
    assert controller.inference_records[1].install_step == 38
    assert installed.record_index == 1
    assert installed.action_index == 8
    assert installed.observation_age_steps == 8
    assert installed.discarded_prefix_steps == 8
    assert installed.target_step_error == 0


def test_lipo_blends_body_for_ten_control_steps_but_uses_new_gripper_directly() -> None:
    first = _chunk()
    second = _chunk(100.0)
    policy = BlockingPolicy([first, second])
    controller = _controller(policy)
    controller.reset(np.zeros(19, dtype=np.float32), _images(0))
    for step in range(31):
        controller.step(step, np.zeros(19, dtype=np.float32), _images(step))
    assert policy.started.wait(timeout=5)
    policy.release.set()
    assert controller.pending is not None
    controller.pending.future.result(timeout=5)

    controls = [
        controller.step(step, np.zeros(19, dtype=np.float32), _images(step)) for step in range(31, 41)
    ]
    controller.close()

    old_actions = controller.inference_records[0].trace.processed
    new_actions = controller.inference_records[1].trace.processed
    for offset, control in enumerate(controls):
        step = 31 + offset
        alpha = float(offset + 1) / 10.0
        expected_body = old_actions[step, :17] * (1.0 - alpha) + new_actions[step - 30, :17] * alpha
        assert control.blend_active is True
        assert control.blend_alpha == alpha
        np.testing.assert_allclose(control.action[:17], expected_body, atol=1e-6)
        np.testing.assert_array_equal(control.action[17:], new_actions[step - 30, 17:])


def test_replan_uses_last_command_state_and_latest_images_instead_of_feedback() -> None:
    policy = BlockingPolicy([_chunk(), _chunk(100.0)])
    controller = _controller(policy)
    initial_feedback = np.full(19, -5.0, dtype=np.float32)
    controller.reset(initial_feedback, _images(0))
    last_control = None
    for step in range(31):
        last_control = controller.step(step, np.full(19, 999.0, dtype=np.float32), _images(step))
    assert last_control is not None
    assert policy.started.wait(timeout=5)
    policy.release.set()
    controller.close()

    np.testing.assert_array_equal(policy.calls[0][0], initial_feedback)
    np.testing.assert_array_equal(policy.calls[1][0], controller.inference_records[0].trace.processed[29])
    assert int(policy.calls[1][1]["camera"][0, 0, 0]) == 30


def test_replan_keeps_unpublished_body_at_session_initial_state() -> None:
    policy = BlockingPolicy([_chunk(), _chunk(100.0)])
    selected_indices = np.asarray([1, 5, 12], dtype=np.int64)
    controller = LipoActionChunkController(
        policy,
        BridgeActionChunkProcessor(sample_factor=2),
        config=_config(),
        asynchronous=True,
        published_body_indices=selected_indices,
    )
    initial_feedback = np.full(19, -5.0, dtype=np.float32)
    controller.reset(initial_feedback, _images(0))
    for step in range(31):
        controller.step(step, np.full(19, 999.0, dtype=np.float32), _images(step))
    assert policy.started.wait(timeout=5)
    policy.release.set()
    controller.close()

    expected = initial_feedback.copy()
    last_published = controller.inference_records[0].trace.processed[29]
    expected[selected_indices] = last_published[selected_indices]
    expected[17:] = last_published[17:]
    np.testing.assert_array_equal(policy.calls[1][0], expected)


def test_controller_holds_last_command_after_trajectory_is_exhausted() -> None:
    policy = BlockingPolicy([_chunk(), _chunk(100.0)])
    controller = _controller(policy)
    controller.reset(np.zeros(19, dtype=np.float32), _images(0))
    controls = [controller.step(step, np.zeros(19, dtype=np.float32), _images(step)) for step in range(61)]
    assert policy.started.wait(timeout=5)
    policy.release.set()
    controller.close()

    assert controls[60].held_last_command is True
    np.testing.assert_array_equal(controls[60].action, controls[59].action)


def test_raw_controller_replaces_old_chunk_at_fixed_replan_interval() -> None:
    first = _chunk()[:4]
    second = _chunk(100.0)[:4]
    policy = RecordingPolicy([first, second])
    controller = ActionChunkController(
        policy,
        IdentityActionChunkProcessor(),
        replan_interval=2,
        asynchronous=False,
    )

    controller.reset(np.zeros(19, dtype=np.float32), _images(0))
    controls = [
        controller.step(step, np.full(19, step, dtype=np.float32), _images(step)) for step in range(4)
    ]
    controller.close()

    np.testing.assert_array_equal(
        np.asarray([control.action for control in controls]),
        np.asarray([first[0], first[1], second[0], second[1]]),
    )
    np.testing.assert_array_equal([control.record_index for control in controls], [0, 0, 1, 1])
    np.testing.assert_array_equal([control.action_index for control in controls], [0, 1, 0, 1])
