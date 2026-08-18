from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from w1_simulation.control.scheduling import control_ticks as _control_ticks


def test_control_ticks_use_independent_control_clock_and_latest_image() -> None:
    frames = [SimpleNamespace(timestamp=10.0), SimpleNamespace(timestamp=10.04)]

    ticks = _control_ticks(frames, control_hz=40.0)

    assert [source_index for _, _, source_index in ticks] == [0, 0, 1]
    assert ticks[0][0] is frames[0]
    assert ticks[1][0] is frames[0]
    assert ticks[2][0] is frames[1]
    np.testing.assert_allclose(
        [timestamp for _, timestamp, _ in ticks],
        [10.0, 10.025, 10.05],
    )


@pytest.mark.parametrize("invalid", [0, -1])
def test_control_ticks_reject_invalid_control_hz(invalid: int) -> None:
    with pytest.raises(ValueError, match="control_hz must be positive"):
        _control_ticks([SimpleNamespace(timestamp=10.0)], invalid)
