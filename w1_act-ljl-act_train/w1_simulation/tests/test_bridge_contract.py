from __future__ import annotations

from pathlib import Path

from safetensors import safe_open
from w1_simulation.w1_profile import DEFAULT_PROFILE


def test_default_bridge_profile_matches_runtime_contract() -> None:
    profile = DEFAULT_PROFILE
    bridge = profile.simulation["bridge"]

    assert bridge["policy_hz"] == 30.0
    assert bridge["sample_factor"] == 2
    assert bridge["replan_threshold"] == 0.5
    assert bridge["lipo_blend_policy_points"] == 5
    assert profile.runtime["policy_hz"] == 20.0
    assert profile.runtime["sample_factor"] == 2
    assert profile.runtime["replan_threshold"] == 0.5
    assert profile.runtime["inference_budget_ms"] == 300.0
    assert profile.runtime["lipo_blend_policy_points"] == 5
    assert profile.runtime["replan_margin_policy_points"] == 2


def test_ros_bridge_imports_the_same_core_used_by_simulation() -> None:
    source = DEFAULT_PROFILE.bridge_script.read_text(encoding="utf-8")

    assert "from w1_simulation.runtime import bridge_base as blocking" in source


def test_hardware_launcher_pins_bridge_profile_and_thumb_protocol_order() -> None:
    launcher = DEFAULT_PROFILE.whole_script.read_text(encoding="utf-8")
    runtime_source = Path("w1_simulation/runtime/bridge_base.py").read_text(encoding="utf-8")

    assert 'BRIDGE_MODE="async"' in launcher
    assert "-m w1_simulation.runtime.bridge" in launcher
    assert "w1_popcorn_v1.json" in launcher
    assert "DEFAULT_PROFILE.endpoints.body" in runtime_source
    assert "DEFAULT_PROFILE.endpoints.left_hand" in runtime_source
    assert "DEFAULT_PROFILE.endpoints.right_hand" in runtime_source


def test_checkpoint_declares_grippers_on_percent_scale(checkpoint_root) -> None:
    statistics = checkpoint_root / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    with safe_open(statistics, framework="pt", device="cpu") as stream:
        action_min = stream.get_tensor("action.min")
        action_max = stream.get_tensor("action.max")

    assert action_min[-2:].tolist() == [0.0, 0.0]
    assert action_max[-2:].tolist() == [100.0, 100.0]
