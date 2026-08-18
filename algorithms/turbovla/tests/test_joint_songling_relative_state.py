import numpy as np

from starVLA.dataloader.gr00t_lerobot.datasets import relative_arm_state_from_history


def test_relative_arm_state_keeps_both_grippers_absolute():
    previous = np.arange(14, dtype=np.float32)
    current = previous + 10
    current[6] = 0.03
    current[13] = 0.08

    output = relative_arm_state_from_history(
        np.stack([previous, current]),
        arm_indices=np.array([*range(6), *range(7, 13)]),
    )

    np.testing.assert_array_equal(output.shape, (1, 14))
    np.testing.assert_array_equal(output[0, :6], np.full(6, 10.0, dtype=np.float32))
    np.testing.assert_array_equal(output[0, 7:13], np.full(6, 10.0, dtype=np.float32))
    assert output[0, 6] == 0.03
    assert output[0, 13] == 0.08
