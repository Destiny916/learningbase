from __future__ import annotations

from dataclasses import dataclass

RERUN_VIEW_MODES = ("standard", "eye", "both")
EYE_CAMERA_SCENES = ("robot", "grid")


@dataclass(frozen=True)
class EyeCameraConfig:
    enabled: bool = True
    name: str = "w1_eye_camera"
    parent_body: str = "eyes"
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    fovy_degrees: float = 70.0
    scene: str = "grid"
    jpeg_quality: int = 90

    def __post_init__(self) -> None:
        if not self.name or "/" in self.name:
            raise ValueError("eye camera name must be a non-empty entity-safe name")
        if not self.parent_body or "/" in self.parent_body:
            raise ValueError("eye camera parent body must be a non-empty body name")
        if self.width < 16 or self.height < 16:
            raise ValueError("eye camera dimensions must be at least 16 pixels")
        if self.fps < 0.0:
            raise ValueError("eye camera fps must be zero (control rate) or positive")
        if not 1.0 <= self.fovy_degrees < 179.0:
            raise ValueError("eye camera vertical field of view must be in [1, 179) degrees")
        if self.scene not in EYE_CAMERA_SCENES:
            raise ValueError(f"Unknown eye camera scene: {self.scene}")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("eye camera JPEG quality must be in [1, 100]")

    def effective_fps(self, control_hz: float) -> float:
        if control_hz <= 0.0:
            raise ValueError("control_hz must be positive")
        effective = control_hz if self.fps == 0.0 else self.fps
        if effective > control_hz + 1e-9:
            raise ValueError(
                f"eye camera fps cannot exceed the control rate: camera={effective}, control={control_hz}"
            )
        return effective


class EyeCameraSchedule:
    def __init__(self, camera_fps: float, control_hz: float) -> None:
        if camera_fps <= 0.0 or control_hz <= 0.0 or camera_fps > control_hz + 1e-9:
            raise ValueError("camera_fps must be within (0, control_hz]")
        self.camera_fps = float(camera_fps)
        self.control_hz = float(control_hz)

    def due(self, step: int) -> bool:
        if step < 0:
            raise ValueError("step must be non-negative")
        if step == 0:
            return True
        previous_frame = int((step - 1) * self.camera_fps / self.control_hz + 1e-12)
        current_frame = int(step * self.camera_fps / self.control_hz + 1e-12)
        return current_frame > previous_frame
