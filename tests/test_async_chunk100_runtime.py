import numpy as np

from start.async_chunk100_runtime import (
    CONTROL_HORIZON,
    BLEND_CONTROL_POINTS,
    expand_policy_chunk,
    should_prefetch,
    align_and_blend,
)


def test_expand_100_policy_points_to_200_control_points():
    actions = np.zeros((100, 19), dtype=np.float32)
    actions[-1] = 99.0
    expanded = expand_policy_chunk(actions, sample_factor=2)
    assert expanded.shape == (CONTROL_HORIZON, 19)
    np.testing.assert_allclose(expanded[0], 0.0)
    np.testing.assert_allclose(expanded[-1], 99.0)


def test_prefetch_when_30_control_points_remain():
    assert should_prefetch(queue_size=30)
    assert not should_prefetch(queue_size=31)


def test_late_chunk_skips_expired_prefix_and_blends_remaining_overlap():
    old = {t: np.zeros(19, dtype=np.float32) for t in range(1, 231)}
    new = np.ones((200, 19), dtype=np.float32)
    # New chunk starts at 171; latest execution already reached 175.
    merged, first_live, blend_len = align_and_blend(
        old_queue=old, new_actions=new, chunk_start_timestep=171,
        latest_executed_timestep=175,
    )
    assert first_live == 5
    assert blend_len == BLEND_CONTROL_POINTS
    np.testing.assert_allclose(merged[0], 1.0 / 30.0)
    np.testing.assert_allclose(merged[29], 1.0)


def test_no_blend_when_old_chunk_has_finished():
    old = {t: np.zeros(19, dtype=np.float32) for t in range(1, 200)}
    new = np.ones((200, 19), dtype=np.float32)
    merged, first_live, blend_len = align_and_blend(
        old_queue=old, new_actions=new, chunk_start_timestep=170,
        latest_executed_timestep=199,
    )
    assert first_live == 30
    assert blend_len == 0
    np.testing.assert_allclose(merged[0], 1.0)


def test_blend_excludes_left_and_right_hand_scalar_dimensions():
    old = {t: np.zeros(19, dtype=np.float32) for t in range(0, 220)}
    new = np.ones((200, 19), dtype=np.float32)
    merged, _, blend_len = align_and_blend(
        old_queue=old,
        new_actions=new,
        chunk_start_timestep=0,
        latest_executed_timestep=-1,
        blend_indices=np.arange(17),
    )
    assert blend_len == BLEND_CONTROL_POINTS
    np.testing.assert_allclose(merged[0, :17], 1.0 / 30.0)
    np.testing.assert_allclose(merged[0, 17:], 1.0)
