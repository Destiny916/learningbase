import numpy as np
import pytest
import importlib.util
import sys
from pathlib import Path

_MODULE_PATH = Path(__file__).parents[1] / "prismatic" / "vla" / "datasets" / "lerobot_w1.py"
_SPEC = importlib.util.spec_from_file_location("lerobot_w1_contract", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

W1_ACTION_HORIZON = _MODULE.W1_ACTION_HORIZON
build_action_chunk = _MODULE.build_action_chunk
build_pair_indices = _MODULE.build_pair_indices
normalize_with_quantiles = _MODULE.normalize_with_quantiles
validate_w1_info = _MODULE.validate_w1_info
relative_state_representation = _MODULE.relative_state_representation
relative_action_representation = _MODULE.relative_action_representation



def test_w1_contract_requires_19d_three_views_and_horizon_20():
    info = {
        "codebase_version": "v3.0",
        "total_episodes": 50,
        "features": {
            "observation.state": {"shape": [19]},
            "action": {"shape": [19]},
            "observation.images.cam_high_right": {"shape": [3, 224, 224]},
            "observation.images.cam_hand_left": {"shape": [3, 224, 224]},
            "observation.images.cam_hand_right": {"shape": [3, 224, 224]},
        },
    }
    assert W1_ACTION_HORIZON == 20
    validate_w1_info(info)

    info["features"]["action"]["shape"] = [7]
    with pytest.raises(ValueError, match="action.*19"):
        validate_w1_info(info)


def test_state_and_action_quantiles_are_independent_and_clipped():
    state = np.array([[-10.0, 0.0, 10.0]], dtype=np.float32)
    action = np.array([[-100.0, 0.0, 100.0]], dtype=np.float32)
    state_q01 = np.array([-1.0, -1.0, -1.0])
    state_q99 = np.array([1.0, 1.0, 1.0])
    action_q01 = np.array([-10.0, -10.0, -10.0])
    action_q99 = np.array([10.0, 10.0, 10.0])

    state_norm = normalize_with_quantiles(state, state_q01, state_q99)
    action_norm = normalize_with_quantiles(action, action_q01, action_q99)

    np.testing.assert_allclose(state_norm, [[-1.0, 0.0, 1.0]])
    np.testing.assert_allclose(action_norm, [[-1.0, 0.0, 1.0]])
    assert not np.array_equal(state_q01, action_q01)


def test_action_chunk_pads_only_inside_episode_and_marks_invalid_tail():
    actions = np.arange(3 * 19, dtype=np.float32).reshape(3, 19)
    chunk, valid = build_action_chunk(actions, start=1, horizon=20)
    assert chunk.shape == (20, 19)
    np.testing.assert_array_equal(chunk[0], actions[1])
    np.testing.assert_array_equal(chunk[1], actions[2])
    np.testing.assert_array_equal(chunk[2:], np.repeat(actions[2][None, :], 18, axis=0))
    np.testing.assert_array_equal(valid, np.array([True, True, False] + [False] * 17))


def test_pair_indices_clamp_at_episode_end():
    np.testing.assert_array_equal(build_pair_indices(length=5, start=3, offset=31), np.array([3, 4]))


def test_only_arm_joints_are_relative_other_w1_dimensions_stay_absolute():
    states = np.arange(3 * 19, dtype=np.float32).reshape(3, 19)
    converted_state = relative_state_representation(states)
    arm = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16]
    absolute = [0, 8, 9, 17, 18]
    np.testing.assert_array_equal(converted_state[0, arm], np.zeros(len(arm), dtype=np.float32))
    np.testing.assert_array_equal(converted_state[1, arm], states[1, arm] - states[0, arm])
    np.testing.assert_array_equal(converted_state[:, absolute], states[:, absolute])

    chunk = states[1:]
    converted_action = relative_action_representation(chunk, states[1])
    np.testing.assert_array_equal(converted_action[:, arm], chunk[:, arm] - states[1, arm])
    np.testing.assert_array_equal(converted_action[:, absolute], chunk[:, absolute])
