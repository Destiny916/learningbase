from pathlib import Path


def test_async100_runtime_does_not_initialize_disabled_action_lipo():
    source = Path("w1_act-ljl-act_train/xwiz_real_runtime/client_service.py").read_text()
    # The vendor ActionLiPo is stubbed out in this runtime.  Async100 uses its
    # own timestep-aligned aggregator and must never enter the vendor init path.
    assert "def _async_observation_check_loop" in source
    assert "OptimizedRobotClient.observation_check_loop = _async_observation_check_loop" in source


def test_async100_runtime_accepts_chunk_while_previous_queue_is_live():
    source = Path("w1_act-ljl-act_train/xwiz_real_runtime/client_service.py").read_text()
    assert "and os.environ.get(\"XWIZ_ASYNC_REPLAN\") != \"1\"" in source


def test_real_preflight_waits_for_feedback_after_fresh_client_start():
    source = Path("w1_act-ljl-act_train/xwiz_real_runtime/client_service.py").read_text()
    assert "_wait_for_real_feedback" in source
    assert "_wait_for_real_feedback(self)" in source


def test_client_uses_dedicated_ros_executor_thread():
    source = Path("w1_act-ljl-act_train/xwiz_real_runtime/client_service.py").read_text()
    assert "MultiThreadedExecutor" in source
    assert "executor.spin" in source


def test_body_limits_keep_small_safety_margin():
    source = Path("w1_act-ljl-act_train/xwiz_real_runtime/runtime.py").read_text()
    assert "LIMIT_SAFETY_MARGIN" in source


def test_body_command_margin_matches_v024_wrist_envelope():
    import sys
    sys.path.insert(0, "w1_act-ljl-act_train")
    from xwiz_real_runtime.runtime import BODY_LIMITS, LIMIT_SAFETY_MARGIN, action_to_commands
    import numpy as np

    action = np.zeros(19, dtype=np.float64)
    action[6] = -10.0   # LEFT_J6
    action[15] = 10.0   # RIGHT_J6
    command = action_to_commands(action)
    assert np.isclose(command.body_positions[6], BODY_LIMITS["LEFT_J6"][0] + LIMIT_SAFETY_MARGIN)
    assert np.isclose(command.body_positions[15], BODY_LIMITS["RIGHT_J6"][1] - LIMIT_SAFETY_MARGIN)
    assert command.body_positions[6] > -0.776
    assert command.body_positions[15] < 0.776
