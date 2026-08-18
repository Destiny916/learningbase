from pathlib import Path

import numpy as np
from w1_simulation.evaluation.ee_pose import END_EFFECTOR_LINKS, REFERENCE_LINK, UrdfForwardKinematics
from w1_simulation.simulation.config import SOURCE_URDF


def test_ee_chains_include_waist_and_the_correct_arm() -> None:
    fk = UrdfForwardKinematics(SOURCE_URDF)
    for side, target in zip(("LEFT", "RIGHT"), END_EFFECTOR_LINKS, strict=True):
        names = tuple(joint.name for joint in fk.path(REFERENCE_LINK, target))
        assert names[0] == "WAIST"
        assert f"{side}_J1" in names
        assert f"{side}_J7" in names
        assert names[-1] == f"{side}_HAND_BASE"


def test_buttock_reference_excludes_lower_body_joints() -> None:
    fk = UrdfForwardKinematics(SOURCE_URDF)
    names = {joint.name for target in END_EFFECTOR_LINKS for joint in fk.path(REFERENCE_LINK, target)}
    assert not names.intersection({"ANKLE", "KNEE", "BUTTOCK"})
    assert "WAIST" in names


def test_waist_changes_both_end_effector_poses() -> None:
    fk = UrdfForwardKinematics(SOURCE_URDF)
    zeros = {
        "WAIST": 0.0,
        **{f"LEFT_J{index}": 0.0 for index in range(1, 8)},
        **{f"RIGHT_J{index}": 0.0 for index in range(1, 8)},
    }
    turned = dict(zeros)
    turned["WAIST"] = 0.2
    for target in END_EFFECTOR_LINKS:
        original = fk.pose(REFERENCE_LINK, target, zeros)
        changed = fk.pose(REFERENCE_LINK, target, turned)
        assert np.linalg.norm(changed[:3, 3] - original[:3, 3]) > 0.01
        assert not np.allclose(changed[:3, :3], original[:3, :3])


def test_source_urdf_is_the_requested_brainco_model() -> None:
    expected = Path("w1_simulation/urdf/dexforce_w1_v024_brainco_revo1_r/robot_with_ee.urdf").resolve()
    assert SOURCE_URDF.resolve() == expected
