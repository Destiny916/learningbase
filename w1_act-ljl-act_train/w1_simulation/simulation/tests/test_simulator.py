import sys
from types import SimpleNamespace

import numpy as np
import pytest

from w1_simulation.robot.commands import BodyPositionCommand, HandPositionCommand, W1PositionCommand
from w1_simulation.robot.joints import BODY_JOINTS, HAND_POSITION_JOINTS
from w1_simulation.simulation.config import (
    ACTIVE_JOINTS,
    HAND_MIMIC_JOINTS,
    SimulationConfig,
)


class _FakeModel:
    def __init__(self) -> None:
        count = len(ACTIVE_JOINTS) + len(HAND_MIMIC_JOINTS)
        self.nu = len(ACTIVE_JOINTS)
        self.opt = SimpleNamespace(timestep=0.002)
        self.jnt_qposadr = np.arange(count)
        self.jnt_dofadr = np.arange(count)
        self.jnt_range = np.tile([-1.0, 1.0], (count, 1)).astype(np.float64)


class _FakeData:
    def __init__(self, model: _FakeModel) -> None:
        count = len(model.jnt_qposadr)
        self.qpos = np.zeros(count, dtype=np.float64)
        self.qvel = np.zeros(count, dtype=np.float64)
        self.ctrl = np.zeros(model.nu, dtype=np.float64)


def _position_command(
    body: np.ndarray | None = None,
    left_hand: np.ndarray | None = None,
    right_hand: np.ndarray | None = None,
) -> W1PositionCommand:
    return W1PositionCommand(
        BodyPositionCommand(
            BODY_JOINTS,
            np.zeros(len(BODY_JOINTS)) if body is None else body,
        ),
        HandPositionCommand(
            HAND_POSITION_JOINTS,
            np.zeros(6) if left_hand is None else left_hand,
        ),
        HandPositionCommand(
            HAND_POSITION_JOINTS,
            np.zeros(6) if right_hand is None else right_hand,
        ),
    )


@pytest.fixture
def fake_mujoco(monkeypatch: pytest.MonkeyPatch) -> None:
    names = {name: index for index, name in enumerate(ACTIVE_JOINTS + tuple(HAND_MIMIC_JOINTS))}
    module = SimpleNamespace(
        MjModel=SimpleNamespace(from_xml_path=lambda _: _FakeModel()),
        MjData=_FakeData,
        mjtObj=SimpleNamespace(mjOBJ_JOINT=0),
        mj_name2id=lambda model, object_type, name: names.get(name, -1),
        mj_resetData=lambda model, data: (
            data.qpos.fill(0.0),
            data.qvel.fill(0.0),
            data.ctrl.fill(0.0),
        ),
        mj_forward=lambda model, data: None,
        mj_step=lambda model, data: data.qpos.__setitem__(
            slice(0, model.nu), data.qpos[: model.nu] + 0.5 * (data.ctrl - data.qpos[: model.nu])
        ),
    )
    monkeypatch.setitem(sys.modules, "mujoco", module)


def test_raw_29d_target_reset_step_limits_and_mimic(fake_mujoco: None) -> None:
    from w1_simulation.simulation.simulator import W1Simulator

    simulator = W1Simulator("unused.xml", SimulationConfig(frame_skip=2))
    assert simulator.current_qpos.shape == (29,)
    assert simulator.joint_limits.shape == (29, 2)
    np.testing.assert_array_equal(simulator.joint_limits[:, 0], -1.0)
    np.testing.assert_array_equal(simulator.joint_limits[:, 1], 1.0)

    raw = np.linspace(-2.0, 2.0, 29)
    np.testing.assert_array_equal(simulator.set_target(raw), np.clip(raw, -1.0, 1.0))
    qpos = simulator.reset(qpos=np.zeros(29), target=raw)
    np.testing.assert_array_equal(qpos, np.zeros(29))
    body = np.linspace(-2.0, 2.0, len(BODY_JOINTS))
    left_hand = np.linspace(0.0, 100.0, 6)
    right_hand = np.linspace(100.0, 0.0, 6)
    command = _position_command(body, left_hand, right_hand)
    target = simulator.target_from_command(command)
    stepped = simulator.step(command)
    np.testing.assert_allclose(stepped, 0.75 * target)

    source_index = ACTIVE_JOINTS.index("LEFT_IF_MCP_PITCH")
    expected = np.clip(
        0.19373154697137057 + 1.2 * stepped[source_index],
        -1.0,
        1.0,
    )
    assert simulator.current_mimic_qpos["LEFT_IF_DIP"] == pytest.approx(expected)
    assert simulator.control_dt == pytest.approx(0.004)


def test_raw_interface_rejects_wrong_shape_and_non_finite_values(fake_mujoco: None) -> None:
    from w1_simulation.simulation.simulator import W1Simulator

    simulator = W1Simulator("unused.xml", SimulationConfig())
    with pytest.raises(ValueError, match="shape"):
        simulator.set_target(np.zeros(28))
    invalid = np.zeros(29)
    invalid[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        simulator.reset(qpos=invalid)
    with pytest.raises(TypeError, match="W1PositionCommand"):
        simulator.step(np.zeros(29))


def test_kinematic_step_places_controlled_qpos_at_clipped_target(fake_mujoco: None) -> None:
    from w1_simulation.simulation.simulator import W1Simulator

    simulator = W1Simulator("unused.xml", SimulationConfig())
    command = _position_command(
        np.linspace(-2.0, 2.0, len(BODY_JOINTS)),
        np.linspace(0.0, 100.0, 6),
        np.linspace(100.0, 0.0, 6),
    )
    target = simulator.target_from_command(command)

    actual = simulator.step_kinematic(command)

    np.testing.assert_array_equal(actual, target)
    np.testing.assert_array_equal(simulator.data.ctrl, target)
    np.testing.assert_array_equal(simulator.data.qvel[: len(ACTIVE_JOINTS)], 0.0)


def test_sparse_position_command_holds_unpublished_body_targets(fake_mujoco: None) -> None:
    from w1_simulation.simulation.simulator import W1Simulator

    simulator = W1Simulator("unused.xml", SimulationConfig())
    initial = np.linspace(-0.8, 0.8, len(ACTIVE_JOINTS))
    simulator.set_target(initial)
    command = W1PositionCommand(
        BodyPositionCommand(("RIGHT_J2", "LEFT_J1"), np.asarray([0.9, -0.9])),
        HandPositionCommand(HAND_POSITION_JOINTS, np.zeros(6)),
        HandPositionCommand(HAND_POSITION_JOINTS, np.zeros(6)),
    )

    target = simulator.target_from_command(command)

    body_index = {name: index for index, name in enumerate(BODY_JOINTS)}
    assert target[body_index["RIGHT_J2"]] == pytest.approx(0.9)
    assert target[body_index["LEFT_J1"]] == pytest.approx(-0.9)
    omitted = [index for index, name in enumerate(BODY_JOINTS) if name not in {"RIGHT_J2", "LEFT_J1"}]
    np.testing.assert_allclose(target[omitted], initial[omitted])
