from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from w1_simulation.runtime.bridge_base import (
    interpolate_actions,
    load_revolute_joint_limits,
)


def test_sim2real_interpolation_matches_sample_factor_contract() -> None:
    chunk = np.zeros((30, 19), dtype=np.float32)
    chunk[:, 0] = np.arange(30, dtype=np.float32)
    chunk[:, 17] = np.linspace(0.0, 100.0, 30, dtype=np.float32)

    processed = interpolate_actions(chunk, 2)

    assert processed.shape == (60, 19)
    np.testing.assert_array_equal(processed[0], chunk[0])
    np.testing.assert_array_equal(processed[-1], chunk[-1])


def test_urdf_limits_fail_closed_when_file_is_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="URDF does not exist"):
        load_revolute_joint_limits(str(tmp_path / "missing.urdf"))


def test_lipo_launcher_pins_the_deployed_runtime_contract() -> None:
    script = Path("w1_simulation/runtime/start_infer_lipo.sh").read_text(encoding="utf-8")

    assert "-m w1_simulation.runtime.policy_infer_act" in script
    assert 'BRIDGE_MODE="async"' in script
    assert "-m w1_simulation.runtime.bridge" in script
    assert 'W1_SIMULATION_BRIDGE_MODE="$BRIDGE_MODE"' in script
    assert "sample_factor:=1" in script
    assert "w1_popcorn_v1.json" in script


def test_sim2real_bridge_does_not_depend_on_simulator_modules() -> None:
    source = Path("w1_simulation/runtime/bridge_base.py").read_text(encoding="utf-8")

    assert "w1_simulation.simulation" not in source
    assert "import mujoco" not in source
