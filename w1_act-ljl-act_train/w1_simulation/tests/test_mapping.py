from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import w1_simulation.robot.mapping as mapping_module
from w1_simulation.robot.joints import ACT_STATE_JOINTS, HAND_POSITION_JOINTS
from w1_simulation.robot.mapping import ActJointMapper
from w1_simulation.simulation.config import BODY_JOINTS, CONTROLLED_JOINTS


@pytest.fixture
def mapper(monkeypatch) -> ActJointMapper:
    joint_count = len(CONTROLLED_JOINTS)
    name_to_id = {name: index for index, name in enumerate(CONTROLLED_JOINTS)}
    model = SimpleNamespace(
        actuator_trnid=np.column_stack((np.arange(joint_count), np.zeros(joint_count, dtype=int))),
        jnt_qposadr=np.arange(joint_count),
        jnt_range=np.column_stack((np.full(joint_count, -1.0), np.full(joint_count, 1.0))),
    )
    monkeypatch.setattr(mapping_module.mujoco, "mj_name2id", lambda _model, _kind, name: name_to_id[name])
    monkeypatch.setattr(
        mapping_module.mujoco,
        "mj_id2name",
        lambda _model, _kind, joint_id: CONTROLLED_JOINTS[joint_id],
    )
    return ActJointMapper(model)


@pytest.mark.parametrize("scalar", [0.0, 50.0, 100.0])
def test_hand_mapping_round_trips_zero_midpoint_and_full_scale(mapper, scalar: float) -> None:
    action = np.zeros(len(ACT_STATE_JOINTS), dtype=np.float32)
    action[-2:] = scalar

    recovered = mapper.act_state_from_sim(mapper.act_action_to_target(action))

    np.testing.assert_allclose(recovered[-2:], scalar, atol=1e-5)


def test_act_19d_action_maps_to_29d_simulator_target(mapper) -> None:
    action = np.linspace(-0.5, 0.5, len(ACT_STATE_JOINTS), dtype=np.float32)
    action[-2:] = (25.0, 75.0)

    target = mapper.act_action_to_target(action)

    assert target.shape == (len(CONTROLLED_JOINTS),)
    assert len(ACT_STATE_JOINTS) == 19
    assert len(CONTROLLED_JOINTS) == 29


def test_act_action_maps_to_standard_body_and_dexterous_hand_commands(mapper) -> None:
    action = np.linspace(-0.5, 0.5, len(ACT_STATE_JOINTS), dtype=np.float32)
    action[-2:] = (25.0, 75.0)

    command = mapper.act_action_to_command(action)

    assert command.body.name == BODY_JOINTS
    assert command.body.position.shape == (17,)
    assert command.left_hand.name == HAND_POSITION_JOINTS
    assert command.right_hand.name == HAND_POSITION_JOINTS
    assert command.left_hand.value.shape == (6,)
    assert command.right_hand.value.shape == (6,)
    np.testing.assert_allclose(command.body.position, action[: len(BODY_JOINTS)])


def test_standard_hand_command_preserves_thumb_mcp_cmc_protocol_order(mapper) -> None:
    action = np.zeros(len(ACT_STATE_JOINTS), dtype=np.float32)

    command = mapper.act_action_to_command(action)

    np.testing.assert_allclose(command.left_hand.value[:2], [0.0, 100.0])
    np.testing.assert_allclose(command.right_hand.value[:2], [65.0, 100.0])


def test_standard_command_extracts_selected_body_names_in_requested_order(mapper) -> None:
    selected = ("RIGHT_J2", "WAIST", "LEFT_J4")
    sparse_mapper = ActJointMapper(mapper.model, selected_body_names=selected)
    action = np.arange(len(ACT_STATE_JOINTS), dtype=np.float32) * 0.01

    command = sparse_mapper.act_action_to_command(action)

    body_index = {name: index for index, name in enumerate(BODY_JOINTS)}
    assert command.body.name == selected
    np.testing.assert_allclose(
        command.body.position,
        [action[body_index[name]] for name in selected],
    )


def test_thumb1_maps_to_mcp_and_thumb2_maps_to_cmc(mapper) -> None:
    action = np.zeros(len(ACT_STATE_JOINTS), dtype=np.float32)

    target = mapper.act_action_to_target(action)

    np.testing.assert_allclose(target[mapper.left_slice][:2], [1.0, -1.0])
    np.testing.assert_allclose(target[mapper.right_slice][:2], [1.0, 0.3])


def test_initial_pose_swaps_thumb_protocol_fields_into_urdf_joint_order(mapper) -> None:
    pose = dict.fromkeys(BODY_JOINTS, 0.0)
    pose.update(
        {
            "LEFT_HAND_THUMB1": 25.0,
            "LEFT_HAND_THUMB2": 75.0,
            "LEFT_HAND_INDEX": 0.0,
            "LEFT_HAND_MIDDLE": 0.0,
            "LEFT_HAND_RING": 0.0,
            "LEFT_HAND_PINKY": 0.0,
            "RIGHT_HAND_THUMB1": 40.0,
            "RIGHT_HAND_THUMB2": 60.0,
            "RIGHT_HAND_INDEX": 0.0,
            "RIGHT_HAND_MIDDLE": 0.0,
            "RIGHT_HAND_RING": 0.0,
            "RIGHT_HAND_PINKY": 0.0,
        }
    )

    target = mapper.initial_target_from_pose(pose)

    np.testing.assert_allclose(target[mapper.left_slice][:2], [0.5, -0.5])
    np.testing.assert_allclose(target[mapper.right_slice][:2], [0.2, -0.2])


def test_simulator_feedback_swaps_urdf_thumb_joints_back_to_protocol_order(mapper) -> None:
    qpos = np.zeros(len(CONTROLLED_JOINTS), dtype=np.float64)
    qpos[mapper.left_slice] = [1.0, -1.0, -0.3, -0.1, -0.06, -0.26]
    qpos[mapper.right_slice] = [1.0, 0.3, 0.4, 0.5, 1.0, 1.0]

    state = mapper.act_state_from_sim(qpos)

    np.testing.assert_allclose(state[-2:], [0.0, 0.0], atol=1e-5)


def test_body_joint_mapping_preserves_act_values(mapper) -> None:
    action = np.zeros(len(ACT_STATE_JOINTS), dtype=np.float32)
    action[: len(BODY_JOINTS)] = np.linspace(-0.75, 0.75, len(BODY_JOINTS))

    target = mapper.act_action_to_target(action)

    np.testing.assert_allclose(target[: len(BODY_JOINTS)], action[: len(BODY_JOINTS)])


def test_left_gripper_command_does_not_change_right_hand_target(mapper) -> None:
    closed = np.zeros(len(ACT_STATE_JOINTS), dtype=np.float32)
    left_open = closed.copy()
    left_open[-2] = 100.0

    closed_target = mapper.act_action_to_target(closed)
    left_open_target = mapper.act_action_to_target(left_open)

    np.testing.assert_array_equal(closed_target[mapper.right_slice], left_open_target[mapper.right_slice])


def test_right_gripper_command_does_not_change_left_hand_target(mapper) -> None:
    closed = np.zeros(len(ACT_STATE_JOINTS), dtype=np.float32)
    right_open = closed.copy()
    right_open[-1] = 100.0

    closed_target = mapper.act_action_to_target(closed)
    right_open_target = mapper.act_action_to_target(right_open)

    np.testing.assert_array_equal(closed_target[mapper.left_slice], right_open_target[mapper.left_slice])


def test_action_mapping_clips_body_and_hand_commands_to_joint_limits(mapper) -> None:
    action = np.full(len(ACT_STATE_JOINTS), 1_000.0, dtype=np.float32)

    target = mapper.act_action_to_target(action)

    assert np.all(target <= mapper.upper)
    assert np.all(target >= mapper.lower)


def test_simulator_state_mapping_clips_gripper_scalars_to_percent_limits(mapper) -> None:
    qpos = np.zeros(len(CONTROLLED_JOINTS), dtype=np.float64)
    qpos[mapper.left_slice] = -100.0
    qpos[mapper.right_slice] = 100.0

    state = mapper.act_state_from_sim(qpos)

    assert np.all((state[-2:] >= 0.0) & (state[-2:] <= 100.0))
