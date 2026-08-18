import numpy as np

from lerobot.scripts.prepare_0806swap_pi052_dataset import _relative_stats


def test_relative_stats_aligns_next_state_actions_with_current_state() -> None:
    states = [
        np.array(
            [
                [0.0] * 20,
                [100.0] * 20,
                [200.0] * 20,
            ],
            dtype=np.float32,
        )
    ]
    # The conversion stores action[t] = joint state[t + 1], with the final action repeated.
    actions = [
        np.array(
            [
                [1.0] * 14,
                [17.0] * 14,
                [17.0] * 14,
            ],
            dtype=np.float32,
        )
    ]

    _, action_stats = _relative_stats(states, actions, state_absolute=[6, 7, 8, 9, 16, 17, 18, 19])

    # The first future target is action[0] relative to state[0]. The old
    # action[offset:] formula instead started from action[1].
    assert action_stats["count"] == 3
    expected_arm = np.array([1.0, 17.0 - 100.0, 17.0])
    np.testing.assert_allclose(action_stats["q01"][:6], np.quantile(expected_arm, 0.01))
    np.testing.assert_allclose(action_stats["q99"][:6], np.quantile(expected_arm, 0.99))
    expected_gripper = np.array([1.0, 17.0, 17.0])
    np.testing.assert_allclose(action_stats["q01"][6], np.quantile(expected_gripper, 0.01))
    np.testing.assert_allclose(action_stats["q99"][6], np.quantile(expected_gripper, 0.99))
