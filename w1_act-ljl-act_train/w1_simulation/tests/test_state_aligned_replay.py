from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from w1_simulation.replay.origin import StateAlignedFrameSelector


def _selector(**overrides: object) -> StateAlignedFrameSelector:
    frames = [SimpleNamespace(frame_id=index) for index in range(8)]
    references = np.zeros((8, 19), dtype=np.float64)
    references[:, 0] = np.arange(8, dtype=np.float64)
    parameters = {
        "search_ahead_frames": 4,
        "max_advance_frames": 2,
        "match_threshold": 0.2,
        "similarity_slack": 0.0,
    }
    parameters.update(overrides)
    return StateAlignedFrameSelector(
        frames,
        references,
        np.ones(19, dtype=np.float64),
        **parameters,
    )


def test_state_aligned_replay_uses_current_frame_for_bootstrap() -> None:
    selector = _selector()

    selection = selector.current(np.zeros(19, dtype=np.float64))

    assert selection.index == 0
    assert selection.match_distance == 0.0
    assert selection.frozen is False


def test_state_aligned_replay_searches_forward_and_limits_each_advance() -> None:
    selector = _selector()
    state = np.zeros(19, dtype=np.float64)
    state[0] = 4.0

    first = selector.select(state)
    second = selector.select(state)

    assert first.index == 2
    assert second.index == 4
    assert first.frozen is False
    assert second.match_distance == 0.0


def test_state_aligned_replay_never_moves_backward() -> None:
    selector = _selector()
    forward = np.zeros(19, dtype=np.float64)
    forward[0] = 4.0
    selector.select(forward)
    selector.select(forward)

    backward = np.zeros(19, dtype=np.float64)
    selection = selector.select(backward)

    assert selection.index == 4
    assert selection.frozen is True


def test_state_aligned_replay_freezes_outside_reference_corridor() -> None:
    selector = _selector(match_threshold=0.1)
    state = np.full(19, 10.0, dtype=np.float64)

    selection = selector.select(state)

    assert selection.index == 0
    assert selection.frozen is True
    assert selector.summary()["freeze_count"] == 1


def test_state_aligned_replay_prefers_later_similar_frame() -> None:
    selector = _selector(similarity_slack=0.01)
    selector.reference_states[:, 0] = 0.0

    selection = selector.select(np.zeros(19, dtype=np.float64))

    assert selection.index == 2


def test_state_aligned_replay_rejects_reference_length_mismatch() -> None:
    with pytest.raises(ValueError, match="one reference state"):
        StateAlignedFrameSelector(
            [SimpleNamespace(frame_id=0)],
            np.zeros((2, 19), dtype=np.float64),
            np.ones(19, dtype=np.float64),
        )
