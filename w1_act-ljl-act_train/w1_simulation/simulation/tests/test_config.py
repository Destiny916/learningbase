import pytest

from w1_simulation.simulation.config import (
    ACTIVE_JOINTS,
    BODY_JOINTS,
    DEFAULT_LOCKED_JOINT_VALUES,
    HAND_MIMIC_JOINTS,
    LEFT_HAND_JOINTS,
    RIGHT_HAND_JOINTS,
    SimulationConfig,
)


def test_joint_contract_has_17_body_and_two_6_dof_hands() -> None:
    assert len(BODY_JOINTS) == 17
    assert len(LEFT_HAND_JOINTS) == 6
    assert len(RIGHT_HAND_JOINTS) == 6
    assert len(ACTIVE_JOINTS) == len(set(ACTIVE_JOINTS)) == 29
    assert len(HAND_MIMIC_JOINTS) == 8
    assert set(HAND_MIMIC_JOINTS).isdisjoint(ACTIVE_JOINTS)
    assert {source for source, _, _ in HAND_MIMIC_JOINTS.values()} <= set(ACTIVE_JOINTS)


def test_simulation_config_owns_an_independent_dynamic_locked_default() -> None:
    first = SimulationConfig()
    second = SimulationConfig()
    assert first.locked_joint_values == DEFAULT_LOCKED_JOINT_VALUES
    assert first.locked_joint_values is not second.locked_joint_values
    first.locked_joint_values["ANKLE"] = 0.0
    assert second.locked_joint_values["ANKLE"] == DEFAULT_LOCKED_JOINT_VALUES["ANKLE"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"timestep": 0.0}, "timestep"),
        ({"frame_skip": 0}, "frame_skip"),
        ({"body_kp": -1.0}, "controller gains"),
        ({"hand_kp": 0.0}, "controller gains"),
    ],
)
def test_simulation_config_rejects_invalid_dynamics(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SimulationConfig(**kwargs)
