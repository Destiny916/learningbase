from __future__ import annotations

from pathlib import Path


def test_legacy_python_module_aliases_are_removed() -> None:
    root = Path(__file__).resolve().parents[1]
    aliases = (
        "bridge_controller.py",
        "controller.py",
        "deployment.py",
        "mapping.py",
        "origin.py",
        "policy.py",
        "quality.py",
        "scoring.py",
        "script_policy.py",
        "verify.py",
        "ee_pose_validation.py",
    )

    assert all(not (root / alias).exists() for alias in aliases)


def test_domain_config_does_not_depend_on_simulation_package() -> None:
    profile_source = Path(__file__).resolve().parents[1] / "w1_profile.py"
    assert "w1_simulation.simulation" not in profile_source.read_text(encoding="utf-8")


def test_simulation_backend_does_not_depend_on_policy_chunk_or_ros_packages() -> None:
    simulation_root = Path(__file__).resolve().parents[1] / "simulation"
    sources = "\n".join(path.read_text(encoding="utf-8") for path in simulation_root.glob("*.py"))

    assert "w1_simulation.inference" not in sources
    assert "w1_simulation.control" not in sources
    assert "w1_simulation.action_processor" not in sources
    assert "import rclpy" not in sources
    assert "joint_interfaces" not in sources
    assert "end_effector_interfaces" not in sources
