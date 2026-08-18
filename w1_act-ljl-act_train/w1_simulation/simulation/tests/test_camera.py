from __future__ import annotations

import pytest

from w1_simulation.simulation.camera import EyeCameraConfig, EyeCameraSchedule


@pytest.mark.parametrize(
    "kwargs",
    (
        {"width": 15},
        {"height": 15},
        {"fps": -1.0},
        {"fovy_degrees": 0.0},
        {"fovy_degrees": 179.0},
        {"scene": "unknown"},
        {"jpeg_quality": 0},
    ),
)
def test_eye_camera_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        EyeCameraConfig(**kwargs)


def test_eye_camera_control_rate_and_explicit_rate_schedules() -> None:
    config = EyeCameraConfig(fps=0.0)
    assert config.effective_fps(60.0) == 60.0
    control_rate = EyeCameraSchedule(config.effective_fps(60.0), 60.0)
    half_rate = EyeCameraSchedule(30.0, 60.0)

    assert [step for step in range(6) if control_rate.due(step)] == [0, 1, 2, 3, 4, 5]
    assert [step for step in range(6) if half_rate.due(step)] == [0, 2, 4]


def test_eye_camera_fps_cannot_exceed_control_rate() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        EyeCameraConfig(fps=61.0).effective_fps(60.0)
