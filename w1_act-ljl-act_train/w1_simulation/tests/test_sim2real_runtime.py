from __future__ import annotations

import hashlib
import importlib.util
import sys
import threading
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SIM2REAL_ROOT = REPO_ROOT / "inference_codes" / "act" / "sim2real"


def _load_bridge_module():
    path = SIM2REAL_ROOT / "policy_bridge.py"
    spec = importlib.util.spec_from_file_location("w1_simulation_sim2real_policy_bridge", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bridge = _load_bridge_module()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_sim2real_runtime_uses_only_the_three_renamed_entrypoints() -> None:
    assert (SIM2REAL_ROOT / "policy_infer.py").is_file()
    assert (SIM2REAL_ROOT / "policy_bridge.py").is_file()
    assert (SIM2REAL_ROOT / "start_act.sh").is_file()
    assert not (SIM2REAL_ROOT / "policy_infer_act.py").exists()
    assert not (SIM2REAL_ROOT / "policy_bridge_act_lipo.py").exists()
    assert not (SIM2REAL_ROOT / "start_infer.sh").exists()

    source = (SIM2REAL_ROOT / "policy_bridge.py").read_text(encoding="utf-8")
    assert "import policy_bridge_act" not in source
    assert "w1_simulation.runtime" not in source


def test_policy_infer_was_renamed_without_behavior_changes() -> None:
    assert _sha256(SIM2REAL_ROOT / "policy_infer.py") == _sha256(
        SIM2REAL_ROOT / "bak" / "policy_infer_act.py"
    )


def test_start_act_exposes_one_mode_switch_without_overriding_sample_factor() -> None:
    source = (SIM2REAL_ROOT / "start_act.sh").read_text(encoding="utf-8")

    assert 'BRIDGE_MODE="async"' in source
    assert "policy_infer.py" in source
    assert "policy_bridge.py" in source
    assert 'W1_ACT_BRIDGE_MODE="$BRIDGE_MODE"' in source
    assert "inference_steps" in source
    assert "ordered_body_names" in source
    assert "selected_body_names" in source
    assert "drop_joint_names" not in source
    sync_case = source[source.index('case "$BRIDGE_MODE"') : source.index('MODEL_PID=""')]
    assert "sample_factor" not in sync_case


def test_selected_body_names_are_mapped_by_name_without_changing_model_order() -> None:
    ordered = ("WAIST", "LEFT_J1", "RIGHT_J2")
    selected = ("RIGHT_J2", "WAIST")

    actual_ordered, actual_selected, indices = bridge.validate_body_orders(ordered, selected)

    assert actual_ordered == ordered
    assert actual_selected == selected
    np.testing.assert_array_equal(indices, [2, 0])
    with pytest.raises(ValueError, match="not supported"):
        bridge.validate_body_orders(ordered, ("ANKLE",))
    with pytest.raises(ValueError, match="unique"):
        bridge.validate_body_orders(ordered, ("WAIST", "WAIST"))


def test_unpublished_body_state_stays_at_session_initial_value() -> None:
    state = bridge.compose_policy_state(
        ("WAIST", "LEFT_J1", "RIGHT_J2"),
        ("LEFT_J1",),
        {"WAIST": 0.1, "LEFT_J1": 0.2, "RIGHT_J2": 0.3},
        {"WAIST": 9.1, "LEFT_J1": 0.8, "RIGHT_J2": 9.3},
        ("left", "right"),
        {"left": 25.0, "right": 75.0},
    )

    np.testing.assert_allclose(state, [0.1, 0.8, 0.3, 25.0, 75.0])


def test_body_message_preserves_selected_name_position_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeBodyMessage:
        def __init__(self) -> None:
            self.header = SimpleNamespace(stamp=None)
            self.name = []
            self.position = []

    class FakePublisher:
        def __init__(self) -> None:
            self.messages = []

        def publish(self, message) -> None:
            self.messages.append(message)

    monkeypatch.setattr(bridge, "JointPositionControl", FakeBodyMessage, raising=False)
    node = object.__new__(bridge.PolicyBridgeBase)
    node.full_dim = 3
    node.action_clip_mask = np.zeros(3, dtype=bool)
    node.action_lower = np.full(3, -np.inf, dtype=np.float32)
    node.action_upper = np.full(3, np.inf, dtype=np.float32)
    node.left_gripper_index = None
    node.right_gripper_index = None
    node.selected_body_names = ("RIGHT_J2", "WAIST")
    node.selected_body_indices = np.asarray([2, 0], dtype=np.int64)
    node.shadow_mode = False
    node.pub_action = FakePublisher()
    node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: "stamp"))
    node.state_lock = threading.RLock()
    node.last_published_body = {}
    node.gripper_command_state = {"left": 0.0, "right": 0.0}
    node.last_command_state = None

    node._publish_action(np.asarray([0.1, 0.2, 0.3], dtype=np.float32))

    message = node.pub_action.messages[0]
    assert message.header.stamp == "stamp"
    assert message.name == ["RIGHT_J2", "WAIST"]
    np.testing.assert_allclose(message.position, [0.3, 0.1])
    assert node.last_published_body == pytest.approx({"RIGHT_J2": 0.3, "WAIST": 0.1})


def test_subscriber_session_reset_invalidates_cached_body_and_trajectory_state() -> None:
    node = object.__new__(bridge.PolicyBridgeBase)
    node.subscriber_lock = threading.Lock()
    node.subscriber_reset_pending = True
    node.subscriber_session = 4
    node.state_lock = threading.RLock()
    node.session_initial_body = {"WAIST": 0.1}
    node.last_published_body = {"WAIST": 0.2}
    node.last_command_state = np.ones(19, dtype=np.float32)
    node.feedback_state_source = "session_command"
    node.gripper_command_state = {"left": 25.0, "right": 75.0}
    node.get_logger = lambda: SimpleNamespace(info=lambda _message: None)

    assert node._consume_subscriber_reset() is True
    assert node.session_initial_body is None
    assert node.last_published_body == {}
    assert node.last_command_state is None
    assert node.feedback_state_source == "feedback_resubscribe"
    assert node.gripper_command_state == {"left": 25.0, "right": 75.0}


def test_old_async_result_is_discarded_after_session_restart() -> None:
    node = object.__new__(bridge.AsynchronousPolicyBridgeNode)
    node.lipo_lock = threading.RLock()
    node.inference_results = deque(
        [
            bridge.InferenceResult(
                submit_step=50,
                session_id=3,
                source_frame=1.0,
                state_source="session_command",
                actions=np.zeros((200, 19), dtype=np.float32),
                inference_ms=200.0,
                error=None,
            )
        ]
    )
    node.get_logger = lambda: SimpleNamespace(info=lambda _message: None)

    node._install_ready_result(session_id=4)

    assert not node.inference_results


def test_sample_factor_only_scales_control_points() -> None:
    base = {
        "execution_horizon": 100,
        "replan_threshold": 0.5,
        "inference_budget_ms": 300.0,
        "lipo_blend_policy_points": 5,
        "replan_margin_policy_points": 2,
        "policy_hz": 20.0,
    }
    one = bridge.RuntimeLipoConfig(**base, sample_factor=1)
    two = bridge.RuntimeLipoConfig(**base, sample_factor=2)

    assert one.trigger_policy_points == two.trigger_policy_points == 50
    assert one.required_policy_points == two.required_policy_points == 13
    assert two.trigger_control_points == one.trigger_control_points * 2 == 100
    assert two.lipo_blend_control_points == one.lipo_blend_control_points * 2 == 10


def test_lipo_blends_body_and_passes_new_hand_scalars_through() -> None:
    old = np.zeros(19, dtype=np.float32)
    new = np.full(19, 10.0, dtype=np.float32)
    old[-2:] = (20.0, 30.0)
    new[-2:] = (70.0, 80.0)

    blended = bridge.lipo_body_action(old, new, 0.5, np.arange(17, dtype=np.int64))

    np.testing.assert_allclose(blended[:17], 5.0)
    np.testing.assert_allclose(blended[-2:], [70.0, 80.0])


def test_hand_mapping_keeps_existing_inverted_endpoints() -> None:
    start = np.asarray([0.0, 70.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    end = np.asarray([0.0, 100.0, 35.0, 45.0, 47.0, 37.0], dtype=np.float32)

    np.testing.assert_allclose(bridge.map_gripper_to_hand(0.0, start, end, invert=True), end)
    np.testing.assert_allclose(bridge.map_gripper_to_hand(100.0, start, end, invert=True), start)
