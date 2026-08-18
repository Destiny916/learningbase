from __future__ import annotations

import math
import queue
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from w1_simulation.artifacts import sha256_array as sha256_array
from w1_simulation.artifacts import sha256_file as sha256_file
from w1_simulation.artifacts import write_json as write_json
from w1_simulation.observability.system import collect_gpu_metrics
from w1_simulation.simulation.camera import RERUN_VIEW_MODES, EyeCameraConfig
from w1_simulation.simulation.config import ACTIVE_JOINTS, SOURCE_URDF

CAMERA_STREAMS = ("cam_high_left", "cam_hand_left", "cam_hand_right")
ROBOT_VIEW_SHARE = 2.0
VIDEO_COLUMN_SHARE = 1.0
DEFAULT_EYE_CAMERA_ENTITY = "world/robot/eyes/w1_eye_camera"


def build_act_blueprint(
    rrb: Any,
    camera_streams: tuple[str, ...],
    view_mode: str = "standard",
    eye_camera_entity: str = DEFAULT_EYE_CAMERA_ENTITY,
) -> Any:
    if view_mode not in RERUN_VIEW_MODES:
        raise ValueError(f"Unknown Rerun view mode: {view_mode}")
    robot_view = rrb.Spatial3DView(origin="world", contents="world/robot/**", name="W1 Robot")
    eye_view = rrb.Spatial2DView(
        origin=eye_camera_entity,
        contents=f"{eye_camera_entity}/rgb",
        name="MuJoCo Eye Camera",
    )
    if view_mode == "eye":
        primary_view = eye_view
    elif view_mode == "both":
        primary_view = rrb.Vertical(
            robot_view,
            eye_view,
            row_shares=[1.0, 1.0],
            name="Simulation Views",
        )
    else:
        primary_view = robot_view
    camera_views = [
        rrb.Spatial2DView(
            origin=f"observation/{camera}",
            contents=f"observation/{camera}",
            name=camera,
        )
        for camera in camera_streams
    ]
    if camera_views:
        layout = rrb.Horizontal(
            primary_view,
            rrb.Vertical(
                *camera_views,
                row_shares=[1.0] * len(camera_views),
                name="Model Inputs",
            ),
            column_shares=[ROBOT_VIEW_SHARE, VIDEO_COLUMN_SHARE],
            name="ACT Simulation",
        )
    else:
        layout = primary_view
    return rrb.Blueprint(
        layout,
        rrb.TimePanel(state="collapsed", timeline="sim_time"),
        auto_layout=False,
        auto_views=False,
        collapse_panels=True,
    )


@dataclass(frozen=True)
class TelemetrySnapshot:
    step: int
    time_seconds: float | None
    links: tuple[tuple[str, npt.NDArray[np.float64], npt.NDArray[np.float64]], ...]
    qpos: npt.NDArray[np.float64]
    target: npt.NDArray[np.float64]
    action: npt.NDArray[np.float64]
    metrics: dict[str, float]
    images: dict[str, npt.NDArray[np.uint8]]
    model_qpos: npt.NDArray[np.float64] | None


def _quaternion_from_rpy(rpy: str) -> list[float]:
    roll, pitch, yaw = (float(value) for value in rpy.split())
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]


class RerunTelemetry:
    def __init__(
        self,
        model: Any,
        grpc_url: str | None = None,
        recording_path: Path | None = None,
        source_urdf: Path = SOURCE_URDF,
        application_id: str = "dexforce_w1_simulation",
        camera_streams: tuple[str, ...] = CAMERA_STREAMS,
        view_mode: str = "standard",
        eye_camera: EyeCameraConfig | None = None,
    ) -> None:
        import mujoco
        import rerun as rr
        import rerun.blueprint as rrb

        self._rr = rr
        self.model = model
        self.view_mode = view_mode
        self.camera_streams = tuple(camera_streams)
        self.eye_camera = eye_camera
        self.eye_camera_entity = (
            f"world/robot/{eye_camera.parent_body}/{eye_camera.name}"
            if eye_camera is not None and eye_camera.enabled
            else DEFAULT_EYE_CAMERA_ENTITY
        )
        self.streams: list[Any] = []
        self.link_entities: list[tuple[int, str]] = []
        self._renderer: Any | None = None
        self._render_data: Any | None = None
        self._scene_option: Any | None = None
        self.eye_frames_submitted = 0
        self.eye_frames_rendered = 0
        self.eye_render_times_ms: list[float] = []
        if view_mode not in RERUN_VIEW_MODES:
            raise ValueError(f"Unknown Rerun view mode: {view_mode}")
        if view_mode != "standard" and (eye_camera is None or not eye_camera.enabled):
            raise ValueError("eye and both Rerun modes require an enabled MuJoCo eye camera")
        self.active_qpos_ids = [
            model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in ACTIVE_JOINTS
        ]
        try:
            if grpc_url:
                stream = rr.RecordingStream(f"{application_id}_live")
                stream.connect_grpc(url=grpc_url)
                self.streams.append(stream)
            if recording_path is not None:
                recording_path.parent.mkdir(parents=True, exist_ok=True)
                stream = rr.RecordingStream(f"{application_id}_recording")
                stream.save(recording_path)
                self.streams.append(stream)
            self.enabled = bool(self.streams)
            if self.enabled:
                blueprint = build_act_blueprint(
                    rrb,
                    camera_streams,
                    view_mode=view_mode,
                    eye_camera_entity=self.eye_camera_entity,
                )
                for stream in self.streams:
                    stream.send_blueprint(blueprint)
                self._initialize_scene(source_urdf)
        except BaseException:
            self.close()
            raise

    def _log(self, entity: str, value: Any, static: bool = False) -> None:
        for stream in self.streams:
            stream.log(entity, value, static=static)

    def _initialize_scene(self, source_urdf: Path) -> None:
        import mujoco

        rr = self._rr
        self._log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        root = ET.parse(source_urdf).getroot()
        for link in root.findall("link"):
            name = link.get("name", "")
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id < 0:
                continue
            self.link_entities.append((body_id, name))
            mesh = link.find("visual/geometry/mesh")
            if mesh is None:
                continue
            mesh_path = source_urdf.parent / mesh.get("filename", "")
            entity = f"world/robot/{name}/visual"
            self._log(entity, rr.Asset3D(path=mesh_path), static=True)
            origin = link.find("visual/origin")
            if origin is not None:
                xyz = [float(value) for value in origin.get("xyz", "0 0 0").split()]
                quaternion = _quaternion_from_rpy(origin.get("rpy", "0 0 0"))
                self._log(
                    entity,
                    rr.Transform3D(translation=xyz, quaternion=rr.Quaternion(xyzw=quaternion)),
                    static=True,
                )
        if self.eye_camera is not None and self.eye_camera.enabled:
            focal_length = self.eye_camera.height / (
                2.0 * math.tan(math.radians(self.eye_camera.fovy_degrees) / 2.0)
            )
            self._log(
                self.eye_camera_entity,
                rr.Transform3D(quaternion=rr.Quaternion(xyzw=[1.0, 0.0, 0.0, 0.0])),
                static=True,
            )
            self._log(
                self.eye_camera_entity,
                rr.Pinhole(
                    resolution=[self.eye_camera.width, self.eye_camera.height],
                    focal_length=[focal_length, focal_length],
                    camera_xyz=rr.ViewCoordinates.RDF,
                    image_plane_distance=0.15,
                ),
                static=True,
            )

    def log_state(
        self,
        step: int,
        data: Any,
        target: npt.ArrayLike,
        action: npt.ArrayLike,
        metrics: Mapping[str, float],
        images: Mapping[str, npt.ArrayLike] | None = None,
        time_seconds: float | None = None,
        render_eye: bool = False,
    ) -> None:
        if not self.enabled:
            return
        self.log_snapshot(
            self.capture_state(step, data, target, action, metrics, images, time_seconds, render_eye)
        )

    def capture_state(
        self,
        step: int,
        data: Any,
        target: npt.ArrayLike,
        action: npt.ArrayLike,
        metrics: Mapping[str, float],
        images: Mapping[str, npt.ArrayLike] | None = None,
        time_seconds: float | None = None,
        render_eye: bool = False,
    ) -> TelemetrySnapshot:
        if render_eye and (self.eye_camera is None or not self.eye_camera.enabled):
            raise ValueError("render_eye requires an enabled MuJoCo eye camera")
        links = tuple(
            (
                name,
                np.asarray(data.xpos[body_id], dtype=np.float64).copy(),
                np.asarray(data.xquat[body_id], dtype=np.float64).copy(),
            )
            for body_id, name in self.link_entities
        )
        image_arrays: dict[str, npt.NDArray[np.uint8]] = {}
        for camera, image_rgb in (images or {}).items():
            if not camera or "/" in camera:
                raise ValueError(f"invalid camera stream: {camera!r}")
            image = np.asarray(image_rgb, dtype=np.uint8)
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"expected an HWC RGB image for {camera}, got {image.shape}")
            image_arrays[camera] = image
        return TelemetrySnapshot(
            step=step,
            time_seconds=float(time_seconds) if time_seconds is not None else None,
            links=links,
            qpos=np.asarray(data.qpos[self.active_qpos_ids], dtype=np.float64).copy(),
            target=np.asarray(target, dtype=np.float64).copy(),
            action=np.asarray(action, dtype=np.float64).copy(),
            metrics={name: float(value) for name, value in metrics.items()},
            images=image_arrays,
            model_qpos=np.asarray(data.qpos, dtype=np.float64).copy() if render_eye else None,
        )

    def _initialize_eye_renderer(self) -> None:
        import mujoco

        if self.eye_camera is None or not self.eye_camera.enabled:
            raise RuntimeError("MuJoCo eye camera is not enabled")
        if self._renderer is not None:
            return
        self._render_data = mujoco.MjData(self.model)
        self._renderer = mujoco.Renderer(
            self.model,
            height=self.eye_camera.height,
            width=self.eye_camera.width,
        )
        self._scene_option = mujoco.MjvOption()
        self._scene_option.geomgroup[:] = 0
        self._scene_option.geomgroup[1] = 1
        if self.eye_camera.scene == "grid":
            self._scene_option.geomgroup[2] = 1

    def warm_up_eye_renderer(self) -> None:
        import mujoco

        if self.eye_camera is None or not self.eye_camera.enabled:
            return
        self._initialize_eye_renderer()
        mujoco.mj_forward(self.model, self._render_data)
        self._renderer.update_scene(
            self._render_data,
            camera=self.eye_camera.name,
            scene_option=self._scene_option,
        )
        self._renderer.render()

    def _render_eye_image(self, model_qpos: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.uint8], float]:
        import mujoco

        if self.eye_camera is None or not self.eye_camera.enabled:
            raise RuntimeError("MuJoCo eye camera is not enabled")
        self._initialize_eye_renderer()
        started = time.monotonic()
        self._render_data.qpos[:] = model_qpos
        mujoco.mj_forward(self.model, self._render_data)
        self._renderer.update_scene(
            self._render_data,
            camera=self.eye_camera.name,
            scene_option=self._scene_option,
        )
        image = np.asarray(self._renderer.render(), dtype=np.uint8).copy()
        return image, (time.monotonic() - started) * 1000.0

    def log_snapshot(self, snapshot: TelemetrySnapshot) -> None:
        if not self.enabled:
            return
        rr = self._rr
        for stream in self.streams:
            stream.set_time("step", sequence=snapshot.step)
            if snapshot.time_seconds is not None:
                stream.set_time("sim_time", duration=snapshot.time_seconds)
        for name, position, quaternion in snapshot.links:
            self._log(
                f"world/robot/{name}",
                rr.Transform3D(
                    translation=position,
                    quaternion=rr.Quaternion(
                        xyzw=[quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
                    ),
                ),
            )
        self._log("joints/qpos", rr.Scalars(snapshot.qpos))
        self._log("joints/target", rr.Scalars(snapshot.target))
        self._log("joints/action", rr.Scalars(snapshot.action))
        for name, value in snapshot.metrics.items():
            self._log(f"metrics/{name}", rr.Scalars(float(value)))
        for camera, image in snapshot.images.items():
            encoded = rr.Image(image, color_model="RGB").compress(jpeg_quality=85)
            self._log(f"observation/{camera}", encoded)
            self._log(f"observation/{camera}/sha256", rr.TextDocument(sha256_array(image)))
        if snapshot.model_qpos is not None:
            image, render_ms = self._render_eye_image(snapshot.model_qpos)
            encoded = rr.Image(image, color_model="RGB").compress(jpeg_quality=self.eye_camera.jpeg_quality)
            self._log(f"{self.eye_camera_entity}/rgb", encoded)
            self._log("metrics/eye_camera/render_ms", rr.Scalars(render_ms))
            self.eye_frames_rendered += 1
            self.eye_render_times_ms.append(render_ms)

    def eye_camera_summary(self, effective_fps: float) -> dict[str, Any]:
        config = self.eye_camera
        enabled = config is not None and config.enabled
        render_times = np.asarray(self.eye_render_times_ms, dtype=np.float64)
        return {
            "enabled": enabled,
            "used_by_policy": False,
            "source": "mujoco_named_camera" if enabled else "disabled",
            "camera_mount_source": "source_urdf_link" if enabled else None,
            "robot_geometry_source": "source_urdf_visual_mesh" if enabled else None,
            "name": config.name if enabled else None,
            "parent_body": config.parent_body if enabled else None,
            "resolution": [config.width, config.height] if enabled else None,
            "requested_fps": config.fps if enabled else None,
            "effective_fps": effective_fps if enabled else 0.0,
            "fovy_degrees": config.fovy_degrees if enabled else None,
            "intrinsics_source": "visualization_default" if enabled else None,
            "scene": config.scene if enabled else None,
            "jpeg_quality": config.jpeg_quality if enabled else None,
            "frames_submitted": self.eye_frames_submitted,
            "frames_rendered": self.eye_frames_rendered,
            "frames_dropped": self.eye_frames_submitted - self.eye_frames_rendered,
            "mean_render_ms": float(np.mean(render_times)) if render_times.size else 0.0,
            "p95_render_ms": float(np.percentile(render_times, 95)) if render_times.size else 0.0,
            "max_render_ms": float(np.max(render_times)) if render_times.size else 0.0,
        }

    def layout_summary(self) -> dict[str, Any]:
        primary_views = {
            "standard": ["mujoco_robot_3d"],
            "eye": ["mujoco_eye_camera"],
            "both": ["mujoco_robot_3d", "mujoco_eye_camera"],
        }
        return {
            "primary_views": primary_views[self.view_mode],
            "dataset_camera_streams": list(self.camera_streams),
            "dataset_camera_column_visible": bool(self.camera_streams),
        }

    def close(self) -> None:
        if self._renderer is not None:
            with suppress(Exception):
                self._renderer.close()
            self._renderer = None
            self._render_data = None
            self._scene_option = None
        for stream in self.streams:
            with suppress(Exception):
                stream.flush(timeout_sec=3.0)
            with suppress(Exception):
                stream.disconnect()
        self.streams.clear()
        self.enabled = False


class AsyncRerunTelemetry:
    def __init__(self, *args: Any, max_pending_frames: int = 180, **kwargs: Any) -> None:
        if max_pending_frames <= 0:
            raise ValueError("max_pending_frames must be positive")
        self.telemetry = RerunTelemetry(*args, **kwargs)
        self._queue: queue.Queue[TelemetrySnapshot | None] = queue.Queue(maxsize=max_pending_frames)
        self._error: BaseException | None = None
        self._closed = False
        self._ready = threading.Event()
        self._worker = threading.Thread(target=self._run, name="rerun-telemetry", daemon=True)
        self._worker.start()
        self._ready.wait()
        self._raise_if_failed()

    def _run(self) -> None:
        try:
            self.telemetry.warm_up_eye_renderer()
            self._ready.set()
            while True:
                snapshot = self._queue.get()
                if snapshot is None:
                    return
                self.telemetry.log_snapshot(snapshot)
        except BaseException as exc:
            self._error = exc
        finally:
            self.telemetry.close()
            self._ready.set()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("Asynchronous Rerun telemetry failed") from self._error

    def log_state(
        self,
        step: int,
        data: Any,
        target: npt.ArrayLike,
        action: npt.ArrayLike,
        metrics: Mapping[str, float],
        images: Mapping[str, npt.ArrayLike] | None = None,
        time_seconds: float | None = None,
        render_eye: bool = False,
    ) -> None:
        if self._closed:
            raise RuntimeError("Asynchronous Rerun telemetry is closed")
        self._raise_if_failed()
        snapshot = self.telemetry.capture_state(
            step, data, target, action, metrics, images, time_seconds, render_eye
        )
        if render_eye:
            self.telemetry.eye_frames_submitted += 1
        self._queue.put(snapshot)

    def eye_camera_summary(self, effective_fps: float) -> dict[str, Any]:
        return self.telemetry.eye_camera_summary(effective_fps)

    def layout_summary(self) -> dict[str, Any]:
        return self.telemetry.layout_summary()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._worker.join()
        try:
            self._raise_if_failed()
        finally:
            self.telemetry.close()


def gpu_metrics() -> dict[str, float]:
    return collect_gpu_metrics(subprocess.run)
