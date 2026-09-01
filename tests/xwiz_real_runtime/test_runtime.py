import numpy as np
import pytest
import xwiz_real_runtime.runtime as runtime

from xwiz_real_runtime.runtime import (
    ACT_DEFAULT_20,
    BODY_ORDER,
    LEFT_CLOSED,
    LEFT_OPEN,
    RIGHT_CLOSED,
    RIGHT_OPEN,
    RuntimeContractError,
    SingleChunkGate,
    action_to_commands,
    feedback_positions_by_name,
    gripper_scalars_from_feedback,
    hand_command_from_openness,
    hand_command_to_wire,
    mode_topics,
    normalize_hand_feedback,
    prepare_client_config,
    scalar_from_hand_command,
    validate_action_chunk,
    validate_observation_buffers,
    validate_robot_health,
    validate_robot_ready,
    validate_timed_actions,
)


def test_zero_is_closed_and_one_hundred_is_open_for_each_hand():
    assert np.allclose(hand_command_from_openness(0, LEFT_CLOSED, LEFT_OPEN), LEFT_CLOSED)
    assert np.allclose(hand_command_from_openness(100, LEFT_CLOSED, LEFT_OPEN), LEFT_OPEN)
    assert np.allclose(hand_command_from_openness(0, RIGHT_CLOSED, RIGHT_OPEN), RIGHT_CLOSED)
    assert np.allclose(hand_command_from_openness(100, RIGHT_CLOSED, RIGHT_OPEN), RIGHT_OPEN)


def test_hand_commands_are_reordered_for_pc1_wire_protocol():
    # PC1 v0.4.6 Linker_L6 uses the same canonical order as the ACT bridge.
    assert hand_command_to_wire((100, 0, 35, 45, 47, 37)) == (100, 0, 35, 45, 47, 37)


def test_hand_mapping_clips_range_and_interpolates_each_joint():
    assert np.allclose(hand_command_from_openness(-10, LEFT_CLOSED, LEFT_OPEN), LEFT_CLOSED)
    assert np.allclose(hand_command_from_openness(110, LEFT_CLOSED, LEFT_OPEN), LEFT_OPEN)
    expected = (np.asarray(LEFT_CLOSED) + np.asarray(LEFT_OPEN)) / 2.0
    assert np.allclose(hand_command_from_openness(50, LEFT_CLOSED, LEFT_OPEN), expected)


def test_hand_feedback_roundtrips_closed_open_and_midpoint_scalars():
    for closed, opened in ((LEFT_CLOSED, LEFT_OPEN), (RIGHT_CLOSED, RIGHT_OPEN)):
        assert scalar_from_hand_command(closed, closed, opened) == pytest.approx(0.0)
        assert scalar_from_hand_command(opened, closed, opened) == pytest.approx(100.0)
        midpoint = hand_command_from_openness(50.0, closed, opened)
        assert scalar_from_hand_command(midpoint, closed, opened) == pytest.approx(50.0)


def test_linker_feedback_is_reordered_by_name_not_message_position():
    states = [
        type("Joint", (), {"name": "T_MCP", "position": 10.0})(),
        type("Joint", (), {"name": "T_CMC_YAW", "position": 20.0})(),
        type("Joint", (), {"name": "IF_MCP_PITCH", "position": 30.0})(),
        type("Joint", (), {"name": "MF_MCP_PITCH", "position": 40.0})(),
        type("Joint", (), {"name": "RF_MCP_PITCH", "position": 50.0})(),
        type("Joint", (), {"name": "LF_MCP_PITCH", "position": 60.0})(),
    ]

    assert feedback_positions_by_name(states) == (10.0, 20.0, 30.0, 40.0, 50.0, 60.0)


def test_linker_feedback_rejects_missing_or_non_finite_joints():
    with pytest.raises(RuntimeContractError, match="missing"):
        feedback_positions_by_name([])
    states = [type("Joint", (), {"name": name, "position": 1.0})() for name in (
        "T_CMC_YAW", "T_MCP", "IF_MCP_PITCH", "MF_MCP_PITCH", "RF_MCP_PITCH", "LF_MCP_PITCH"
    )]
    states[-1].position = np.nan
    with pytest.raises(RuntimeContractError, match="finite"):
        feedback_positions_by_name(states)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_hand_mapping_rejects_non_finite_scalar(value):
    with pytest.raises(RuntimeContractError, match="finite"):
        hand_command_from_openness(value, LEFT_CLOSED, LEFT_OPEN)


def test_action_chunk_requires_exactly_one_finite_16_by_19_chunk():
    chunk = np.zeros((16, 19), dtype=np.float32)
    assert validate_action_chunk(chunk).shape == (16, 19)
    with pytest.raises(RuntimeContractError, match="16, 19"):
        validate_action_chunk(np.zeros((15, 19), dtype=np.float32))
    chunk[3, 4] = np.nan
    with pytest.raises(RuntimeContractError, match="finite"):
        validate_action_chunk(chunk)


def test_action_frame_maps_17_body_joints_and_two_hands_with_limits():
    action = np.zeros(19, dtype=np.float64)
    action[0] = 99.0
    action[-2:] = [0.0, 100.0]

    command = action_to_commands(action)

    assert command.body_names == BODY_ORDER
    assert command.body_positions[0] < 3.0
    assert command.left_hand == LEFT_CLOSED
    assert command.right_hand == RIGHT_OPEN


def ready_payload():
    return {
        "status": "Idle",
        "motor_status": ["OP"] * 20,
        "motor_error_code": [0] * 20,
        "server_error_code": ["None"] * 20,
        "joint_position": list(ACT_DEFAULT_20),
    }


def test_robot_ready_requires_idle_op_zero_errors_and_act_default_pose():
    assert validate_robot_ready(ready_payload(), tolerance_rad=0.05) < 1e-9

    payload = ready_payload()
    payload["joint_position"][6] += 0.051
    with pytest.raises(RuntimeContractError, match="ACT default"):
        validate_robot_ready(payload, tolerance_rad=0.05)

    payload = ready_payload()
    payload["motor_status"][4] = "INIT"
    with pytest.raises(RuntimeContractError, match="OP"):
        validate_robot_ready(payload)


def test_running_robot_health_accepts_running_but_rejects_any_motor_error():
    payload = ready_payload()
    payload["status"] = "Running"
    validate_robot_health(payload, allowed_status=("Idle", "Running"))
    payload["motor_error_code"][2] = 9
    with pytest.raises(RuntimeContractError, match="motor error"):
        validate_robot_health(payload, allowed_status=("Idle", "Running"))


def test_real_and_sim_configs_are_single_16_step_chunk_and_never_execute_home():
    source = {"max_steps": 600, "sample_factor": 2, "home_position": "unsafe"}
    simulation = prepare_client_config(source, mode=1)
    real = prepare_client_config(source, mode=2)

    for config in (simulation, real):
        assert config["action_horizon"] == 16
        assert config["max_steps"] == 16
        assert config["sample_factor"] == 1.0
        assert config["chunk_size_threshold"] == 0.0
        assert config["home_position"] == ""
    assert simulation["mode"] == 1
    assert real["mode"] == 2


def test_single_chunk_gate_stops_exactly_after_frame_16():
    gate = SingleChunkGate(limit=16)
    assert all(gate.mark_published() is False for _ in range(15))
    assert gate.mark_published() is True
    with pytest.raises(RuntimeContractError, match="already complete"):
        gate.mark_published()


def test_continuous_gate_crosses_chunk_boundary_without_completing_session():
    assert hasattr(runtime, "ChunkExecutionGate")
    assert hasattr(runtime, "EXECUTION_CONTINUOUS")
    gate = runtime.ChunkExecutionGate(runtime.EXECUTION_CONTINUOUS, chunk_size=16)

    progress = None
    for _ in range(16):
        progress = gate.mark_published()

    assert progress.chunk_index == 1
    assert progress.frame_in_chunk == 16
    assert progress.chunk_complete is True
    assert progress.session_complete is False

    next_progress = gate.mark_published()
    assert next_progress.chunk_index == 2
    assert next_progress.frame_in_chunk == 1
    assert next_progress.session_complete is False


def test_single_execution_mode_remains_bounded_to_one_chunk():
    assert hasattr(runtime, "ChunkExecutionGate")
    assert hasattr(runtime, "EXECUTION_SINGLE")
    gate = runtime.ChunkExecutionGate(runtime.EXECUTION_SINGLE, chunk_size=16)

    for _ in range(15):
        assert gate.mark_published().session_complete is False
    assert gate.mark_published().session_complete is True
    with pytest.raises(RuntimeContractError, match="already complete"):
        gate.mark_published()


def test_continuous_config_removes_the_100_frame_session_limit():
    config = prepare_client_config(
        {"execution_mode": "continuous", "max_steps": 16},
        mode=2,
    )

    assert config["execution_mode"] == "continuous"
    assert config["action_horizon"] == 16
    assert config["max_steps"] > 16
    assert config["chunk_size_threshold"] == 0.0


def test_continuous_mode_is_rejected_for_simulation():
    with pytest.raises(RuntimeContractError, match="only available in real mode"):
        prepare_client_config({"execution_mode": "continuous"}, mode=1)


def test_next_chunk_waits_for_empty_queue_and_feedback_after_completion():
    assert hasattr(runtime, "should_request_next_chunk")
    common = {
        "execution_mode": "continuous",
        "frame_in_chunk": 16,
        "chunk_completed_at": 20.0,
    }

    assert runtime.should_request_next_chunk(queue_size=1, feedback_received_at=21.0, **common) is False
    assert runtime.should_request_next_chunk(queue_size=0, feedback_received_at=19.0, **common) is False
    assert runtime.should_request_next_chunk(queue_size=0, feedback_received_at=21.0, **common) is True


def test_simulation_and_real_topics_are_strictly_isolated():
    simulation = mode_topics(1)
    real = mode_topics(2)

    assert all(topic.startswith("/mj_sim/control/") for topic in simulation.values())
    assert real == {
        "body": "/control/joint_position",
        "left_hand": "/control/ee/left",
        "right_hand": "/control/ee/right",
    }


def test_real_observation_gate_requires_every_model_and_feedback_buffer():
    buffers = {
        "head_left": [1],
        "head_right": [1],
        "wrist_left": [1],
        "wrist_right": [1],
        "joint_state": [0.0] * 20,
        "left_hand": [LEFT_OPEN],
        "right_hand": [RIGHT_OPEN],
    }
    validate_observation_buffers(buffers, use_wrist_images=True)
    buffers["head_left"] = []
    with pytest.raises(RuntimeContractError, match="head_left"):
        validate_observation_buffers(buffers, use_wrist_images=True)


def test_real_hand_feedback_is_converted_back_to_model_openness():
    assert gripper_scalars_from_feedback(
        np.asarray(LEFT_CLOSED),
        np.asarray(RIGHT_OPEN),
    ) == pytest.approx((0.0, 100.0))


def test_real_linker_feedback_uses_deployed_percentage_scale():
    feedback = np.asarray(LEFT_CLOSED, dtype=np.float64)
    assert normalize_hand_feedback(feedback) == pytest.approx(feedback)
    assert gripper_scalars_from_feedback(feedback, np.asarray(RIGHT_OPEN)) == pytest.approx(
        (0.0, 100.0)
    )


def test_hand_feedback_scale_accepts_ratio_or_percentage_and_rejects_outside_range():
    with pytest.raises(RuntimeContractError, match="0..100"):
        normalize_hand_feedback([0.0, -0.01, 0.0, 0.0, 0.0, 0.0])
    assert normalize_hand_feedback([20.0, 10.0, 30.0, 40.0, 50.0, 60.0]) == (20.0, 10.0, 30.0, 40.0, 50.0, 60.0)
    with pytest.raises(RuntimeContractError, match="0..100"):
        normalize_hand_feedback([0.0, 100.01, 0.0, 0.0, 0.0, 0.0])


def test_vendor_timed_actions_must_form_one_complete_chunk():
    class Timed:
        def __init__(self, value):
            self.value = value

        def get_action(self):
            return self.value

    actions = [Timed(np.zeros(19, dtype=np.float32)) for _ in range(16)]
    assert validate_timed_actions(actions).shape == (16, 19)
    with pytest.raises(RuntimeContractError, match="16, 19"):
        validate_timed_actions(actions[:-1])
