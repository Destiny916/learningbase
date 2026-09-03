import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "w1_act-ljl-act_train"))


def test_nointerp_parameters_keep_one_to_one_100_point_chunk():
    from xwiz_real_runtime.async_chunk100 import (
        execution_parameters,
        expand_policy_chunk,
    )

    params = execution_parameters(sample_factor=1, replan_remaining=15, blend_points=15)
    actions = np.arange(100 * 19, dtype=np.float32).reshape(100, 19)
    assert params == {"sample_factor": 1, "control_horizon": 100, "replan_remaining": 15, "blend_points": 15}
    np.testing.assert_array_equal(expand_policy_chunk(actions, sample_factor=1), actions)


def test_nointerp_trigger_uses_fifteen_remaining_points():
    from start.async_chunk100_runtime import should_prefetch

    assert should_prefetch(15, remaining=15)
    assert should_prefetch(1, remaining=15)
    assert not should_prefetch(16, remaining=15)
