import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from w1_simulation.simulation.config import ACTIVE_JOINTS, HAND_MIMIC_JOINTS, SOURCE_URDF
from w1_simulation.simulation.urdf import create_mujoco_compatible_urdf, inspect_urdf


def test_source_urdf_contract() -> None:
    report = inspect_urdf(SOURCE_URDF, {"ANKLE": 0.25})
    assert report.links == 64
    assert report.joints == 63
    assert report.active_joints == report.controlled_joints == len(ACTIVE_JOINTS) == 29
    assert report.mimic_joints == len(HAND_MIMIC_JOINTS) == 8
    assert report.locked_joints == 1
    assert report.mesh_references == 90
    assert report.missing_mesh_references == ()


def test_adapter_bakes_the_requested_dynamic_locked_value(tmp_path: Path) -> None:
    destination = tmp_path / "robot.urdf"
    create_mujoco_compatible_urdf(SOURCE_URDF, destination, {"ANKLE": 0.25})
    root = ET.parse(destination).getroot()
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    ankle = joints["ANKLE"]
    assert ankle.get("type") == "fixed"
    assert ankle.find("limit") is None
    np.testing.assert_allclose(
        [float(value) for value in ankle.find("origin").get("rpy").split()],
        [np.pi / 2.0, 0.25, 0.0],
        atol=1e-12,
        rtol=0.0,
    )
    assert joints["KNEE"].get("type") == "revolute"
    assert all(joint.find("mimic") is None for joint in root.findall("joint"))
    visual_meshes = [Path(mesh.get("filename")) for mesh in root.findall("./link/visual/geometry/mesh")]
    collision_meshes = [Path(mesh.get("filename")) for mesh in root.findall("./link/collision/geometry/mesh")]
    assert len(visual_meshes) == 68
    assert len(collision_meshes) == 45
    assert len(set(visual_meshes)) == len(visual_meshes)
    assert all(mesh.suffix == ".msh" and mesh.is_file() for mesh in visual_meshes)
    assert set(visual_meshes).isdisjoint(collision_meshes)
    visual_colors = root.findall("./link/visual/material/color")
    assert len(visual_colors) == len(visual_meshes)
    assert all(
        len(values := [float(value) for value in color.get("rgba", "").split()]) == 4
        and all(0.0 <= value <= 1.0 for value in values)
        for color in visual_colors
    )
    chassis = next(link for link in root.findall("link") if link.get("name") == "chassis_base_link")
    np.testing.assert_allclose(
        [
            [float(value) for value in color.get("rgba", "").split()]
            for color in chassis.findall("visual/material/color")
        ],
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.7843137383460999, 0.7843137383460999, 0.7843137383460999, 1.0],
            [0.686274528503418, 0.46666666865348816, 0.0, 1.0],
            [0.0235294122248888, 0.0235294122248888, 0.0235294122248888, 1.0],
        ],
        atol=1e-8,
        rtol=0.0,
    )


def test_adapter_rejects_missing_overlap_and_out_of_range_locks() -> None:
    with pytest.raises(ValueError, match="locked=.*NOT_A_JOINT"):
        inspect_urdf(SOURCE_URDF, {"NOT_A_JOINT": 0.0})
    with pytest.raises(ValueError, match="overlap=.*WAIST"):
        inspect_urdf(SOURCE_URDF, {"WAIST": 0.0})
    with pytest.raises(ValueError, match="outside"):
        inspect_urdf(SOURCE_URDF, {"ANKLE": 2.0})
