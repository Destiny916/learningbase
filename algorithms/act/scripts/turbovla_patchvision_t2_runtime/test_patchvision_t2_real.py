from __future__ import annotations

from argparse import Namespace

import numpy as np
import pytest

from dual_turbovla_patchvision_t2_real import execute_action_chunk, validate_real_args


class FakeHardware:
    def __init__(self) -> None:
        self.actions: list[dict[str, float]] = []

    def send_action(self, action: dict[str, float]) -> None:
        self.actions.append(action)


class NeverStop:
    def is_set(self) -> bool:
        return False


@pytest.mark.parametrize("missing", ["enable_arms", "enable_grippers", "execute_robot_actions"])
def test_real_client_requires_every_motion_flag(missing: str) -> None:
    args = Namespace(enable_arms=True, enable_grippers=True, execute_robot_actions=True)
    setattr(args, missing, False)

    with pytest.raises(SystemExit, match="requires --enable-arms"):
        validate_real_args(args)


def test_execute_action_chunk_sends_exactly_50_ordered_rows() -> None:
    hardware = FakeHardware()
    chunk = np.arange(50 * 14, dtype=np.float32).reshape(50, 14)

    execute_action_chunk(
        hardware,
        chunk,
        fps=30.0,
        stop=NeverStop(),
        clock=lambda: 0.0,
        sleep=lambda _: None,
    )

    assert len(hardware.actions) == 50
    assert list(hardware.actions[0].values()) == pytest.approx(chunk[0].tolist())
    assert list(hardware.actions[-1].values()) == pytest.approx(chunk[-1].tolist())


def test_execute_action_chunk_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match=r"\[50,14\]"):
        execute_action_chunk(
            FakeHardware(),
            np.zeros((49, 14), dtype=np.float32),
            fps=30.0,
            stop=NeverStop(),
        )
