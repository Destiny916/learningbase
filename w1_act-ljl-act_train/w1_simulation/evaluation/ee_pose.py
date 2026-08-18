from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2
import mujoco
import numpy as np
import pyarrow.parquet as pq

from w1_simulation.artifacts import sha256_file
from w1_simulation.robot.joints import ACT_STATE_JOINTS
from w1_simulation.robot.mapping import ActHandGestureConfig, ActJointMapper
from w1_simulation.simulation.config import DEFAULT_LOCKED_JOINT_VALUES, SOURCE_URDF, SimulationConfig
from w1_simulation.simulation.model import build_runtime_model
from w1_simulation.simulation.simulator import W1Simulator
from w1_simulation.simulation.telemetry import RerunTelemetry
from w1_simulation.w1_profile import DEFAULT_ARTIFACT_ROOT, DEFAULT_PROFILE

REFERENCE_LINK = "buttock"
END_EFFECTOR_LINKS = ("left_hand_base_link", "right_hand_base_link")
POSITION_TOLERANCE_M = 1e-6
ORIENTATION_TOLERANCE_RAD = 1e-6


@dataclass(frozen=True)
class JointSpec:
    name: str
    joint_type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray


def _vector(text: str, length: int) -> np.ndarray:
    values = np.asarray([float(value) for value in text.split()], dtype=np.float64)
    if values.shape != (length,):
        raise ValueError(f"Expected {length} values, got {text!r}")
    return values


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation_x = np.asarray(((1, 0, 0), (0, cr, -sr), (0, sr, cr)), dtype=np.float64)
    rotation_y = np.asarray(((cp, 0, sp), (0, 1, 0), (-sp, 0, cp)), dtype=np.float64)
    rotation_z = np.asarray(((cy, -sy, 0), (sy, cy, 0), (0, 0, 1)), dtype=np.float64)
    return rotation_z @ rotation_y @ rotation_x


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    normalized = np.asarray(axis, dtype=np.float64)
    normalized /= np.linalg.norm(normalized)
    x, y, z = normalized
    skew = np.asarray(((0, -z, y), (z, 0, -x), (-y, x, 0)), dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)
    return identity + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def _transform(translation: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = rotation
    value[:3, 3] = translation
    return value


class UrdfForwardKinematics:
    def __init__(self, urdf_path: Path) -> None:
        self.urdf_path = Path(urdf_path).resolve()
        root = ET.parse(self.urdf_path).getroot()
        self.joints_by_child: dict[str, JointSpec] = {}
        for element in root.findall("joint"):
            parent = element.find("parent")
            child = element.find("child")
            if parent is None or child is None:
                raise ValueError(f"Joint {element.get('name')} lacks parent or child")
            origin = element.find("origin")
            axis = element.find("axis")
            xyz = _vector(origin.get("xyz", "0 0 0"), 3) if origin is not None else np.zeros(3)
            rpy = _vector(origin.get("rpy", "0 0 0"), 3) if origin is not None else np.zeros(3)
            axis_xyz = _vector(axis.get("xyz", "1 0 0"), 3) if axis is not None else np.ones(3)
            spec = JointSpec(
                name=element.get("name", ""),
                joint_type=element.get("type", "fixed"),
                parent=parent.get("link", ""),
                child=child.get("link", ""),
                origin=_transform(xyz, _rpy_matrix(rpy)),
                axis=axis_xyz,
            )
            if spec.child in self.joints_by_child:
                raise ValueError(f"Link {spec.child} has more than one parent joint")
            self.joints_by_child[spec.child] = spec

    def path(self, reference_link: str, target_link: str) -> tuple[JointSpec, ...]:
        reversed_path: list[JointSpec] = []
        current = target_link
        while current != reference_link:
            try:
                joint = self.joints_by_child[current]
            except KeyError as exc:
                raise ValueError(f"{target_link} is not a descendant of {reference_link}") from exc
            reversed_path.append(joint)
            current = joint.parent
        return tuple(reversed(reversed_path))

    def pose(
        self,
        reference_link: str,
        target_link: str,
        joint_positions: dict[str, float],
    ) -> np.ndarray:
        result = np.eye(4, dtype=np.float64)
        for joint in self.path(reference_link, target_link):
            result = result @ joint.origin
            if joint.joint_type in {"revolute", "continuous"}:
                if joint.name not in joint_positions:
                    raise ValueError(f"Missing position for movable joint {joint.name}")
                result = result @ _transform(
                    np.zeros(3), _axis_angle_matrix(joint.axis, joint_positions[joint.name])
                )
            elif joint.joint_type == "prismatic":
                if joint.name not in joint_positions:
                    raise ValueError(f"Missing position for movable joint {joint.name}")
                axis = joint.axis / np.linalg.norm(joint.axis)
                result = result @ _transform(axis * joint_positions[joint.name], np.eye(3))
            elif joint.joint_type != "fixed":
                raise ValueError(f"Unsupported joint type {joint.joint_type!r}")
        return result


def _body_pose(model: mujoco.MjModel, data: mujoco.MjData, link: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link)
    if body_id < 0:
        raise ValueError(f"MuJoCo model lacks body {link}")
    return _transform(
        np.asarray(data.xpos[body_id], dtype=np.float64),
        np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3),
    )


def _relative_body_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    reference_link: str,
    target_link: str,
) -> np.ndarray:
    return np.linalg.inv(_body_pose(model, data, reference_link)) @ _body_pose(model, data, target_link)


def _rotation_error(left: np.ndarray, right: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(left.T @ right) - 1.0) * 0.5, -1.0, 1.0))
    return math.acos(cosine)


def _load_episode(dataset_root: Path, episode_id: int) -> tuple[np.ndarray, np.ndarray]:
    info_path = dataset_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    state_names = tuple(info["features"]["observation.state"]["names"])
    action_names = tuple(info["features"]["action"]["names"])
    if state_names != ACT_STATE_JOINTS or action_names != ACT_STATE_JOINTS:
        raise ValueError(
            "Dataset joint order differs from ACT contract: "
            f"state={state_names}, action={action_names}, expected={ACT_STATE_JOINTS}"
        )
    parquet_path = dataset_root / "data" / "chunk-000" / "file-000.parquet"
    parquet = pq.ParquetFile(parquet_path)
    if not 0 <= episode_id < parquet.num_row_groups:
        raise ValueError(f"Episode {episode_id} is outside [0, {parquet.num_row_groups - 1}]")
    table = parquet.read_row_group(
        episode_id,
        columns=["observation.state", "action", "episode_index", "frame_index"],
    )
    episode_values = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
    if np.any(episode_values != episode_id):
        raise ValueError(f"Row group {episode_id} contains episodes {np.unique(episode_values)}")
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    if states.shape != actions.shape or states.shape[1:] != (len(ACT_STATE_JOINTS),):
        raise ValueError(f"Unexpected episode arrays: state={states.shape}, action={actions.shape}")
    return states, actions


def _open_video(dataset_root: Path, episode_id: int) -> cv2.VideoCapture:
    video_path = (
        dataset_root
        / "videos"
        / "observation.images.cam_high_left"
        / "chunk-000"
        / f"file-{episode_id:03d}.mp4"
    )
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open dataset video {video_path}")
    return capture


def _validation_blueprint() -> object:
    import rerun.blueprint as rrb

    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="world",
                contents=["world/robot/**", "world/validation/**"],
                name="URDF Robot and End-effector Frames",
            ),
            rrb.Vertical(
                rrb.Spatial2DView(
                    origin="observation/cam_high_left",
                    contents="observation/cam_high_left",
                    name="Dataset Camera Frame",
                ),
                rrb.TextDocumentView(
                    origin="validation/report",
                    contents="validation/report",
                    name="Coordinate Validation",
                ),
                rrb.TimeSeriesView(
                    origin="metrics",
                    contents="metrics/**",
                    name="Independent FK vs MuJoCo Error",
                ),
                row_shares=[1.4, 1.0, 1.0],
                name="Dataset and Validation",
            ),
            column_shares=[2.1, 1.0],
            name="EE Pose / FK Loss Coordinate Validation",
        ),
        auto_layout=False,
        auto_views=False,
        collapse_panels=True,
    )


def _launch_viewer(recording_path: Path) -> int:
    process = subprocess.Popen(
        [
            "rerun",
            str(recording_path),
            "--renderer",
            "gl",
            "--window-size",
            "1600x900",
            "--memory-limit",
            "3GB",
            "--hide-welcome-screen",
        ],
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    return process.pid


def validate_episode(
    dataset_root: Path,
    artifact_root: Path,
    episode_id: int,
    joint_source: str,
    start_frame: int,
    max_frames: int,
    frame_stride: int,
    open_viewer: bool,
) -> tuple[Path, Path, int | None]:
    import rerun as rr

    dataset_root = Path(dataset_root).resolve()
    artifact_root = Path(artifact_root).resolve()
    if joint_source not in {"state", "action"}:
        raise ValueError("joint_source must be state or action")
    if start_frame < 0 or max_frames < 0 or frame_stride < 1:
        raise ValueError("start_frame/max_frames must be non-negative and frame_stride positive")
    states, actions = _load_episode(dataset_root, episode_id)
    source = states if joint_source == "state" else actions
    stop = len(source) if max_frames == 0 else min(len(source), start_frame + max_frames)
    frame_ids = np.arange(start_frame, stop, frame_stride, dtype=np.int64)
    if not len(frame_ids):
        raise ValueError("Selected frame range is empty")

    run_name = f"ee_coordinates_ep{episode_id}_{joint_source}_{time.strftime('%Y%m%d_%H%M%S')}"
    generated_dir = artifact_root / "generated" / run_name
    recording_path = artifact_root / "rerun" / f"{run_name}.rrd"
    report_path = artifact_root / "reports" / f"{run_name}.json"
    recording_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_path = build_runtime_model(
        generated_dir,
        source=SOURCE_URDF,
        config=SimulationConfig(),
        locked_joint_values=DEFAULT_LOCKED_JOINT_VALUES,
    )
    simulator = W1Simulator(runtime_path)
    mapper = ActJointMapper(
        simulator.model,
        ActHandGestureConfig.from_dict(DEFAULT_PROFILE.hands),
        selected_body_names=DEFAULT_PROFILE.body_command_names,
    )
    fk = UrdfForwardKinematics(SOURCE_URDF)
    for target_link in END_EFFECTOR_LINKS:
        path_names = tuple(joint.name for joint in fk.path(REFERENCE_LINK, target_link))
        required = "LEFT_J1" if target_link.startswith("left") else "RIGHT_J1"
        if "WAIST" not in path_names or required not in path_names:
            raise AssertionError(f"Unexpected FK chain for {target_link}: {path_names}")

    telemetry = RerunTelemetry(
        simulator.model,
        recording_path=recording_path,
        source_urdf=SOURCE_URDF,
        application_id="w1_ee_coordinate_validation",
        camera_streams=("cam_high_left",),
    )
    for stream in telemetry.streams:
        stream.send_blueprint(_validation_blueprint())
    capture = _open_video(dataset_root, episode_id)
    current_video_frame = -1
    current_image: np.ndarray | None = None
    position_errors: dict[str, list[float]] = {link: [] for link in END_EFFECTOR_LINKS}
    orientation_errors: dict[str, list[float]] = {link: [] for link in END_EFFECTOR_LINKS}
    positions: dict[str, list[np.ndarray]] = {link: [] for link in END_EFFECTOR_LINKS}
    try:
        for output_step, frame_id in enumerate(frame_ids):
            while current_video_frame < int(frame_id):
                ok, image_bgr = capture.read()
                if not ok:
                    raise RuntimeError(f"Video ended before frame {frame_id}")
                current_video_frame += 1
                current_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            if current_image is None:
                raise RuntimeError("No video frame decoded")
            act_values = source[frame_id]
            command = mapper.act_action_to_command(act_values)
            simulator.step_kinematic(command)
            target = simulator.target.copy()
            joint_positions = {
                name: float(value) for name, value in zip(ACT_STATE_JOINTS, act_values, strict=True)
            }
            reference_world = _body_pose(simulator.model, simulator.data, REFERENCE_LINK)
            frame_text = [
                f"## EE coordinate validation: episode {episode_id} / frame {frame_id}",
                "",
                f"- Joint source: `{joint_source}` (19D; radians; grippers 0-100)",
                f"- Reference frame: `{REFERENCE_LINK}`",
                "- Targets: `left_hand_base_link` / `right_hand_base_link`",
                "- Colored axes: independent URDF FK from dataset joints",
                "- Large green point: corresponding MuJoCo palm body",
                "- Small red point: independent URDF FK transformed to world",
                "",
            ]
            metrics: dict[str, float] = {}
            for side, link in zip(("left", "right"), END_EFFECTOR_LINKS, strict=True):
                urdf_relative = fk.pose(REFERENCE_LINK, link, joint_positions)
                mujoco_relative = _relative_body_pose(simulator.model, simulator.data, REFERENCE_LINK, link)
                position_error = float(np.linalg.norm(urdf_relative[:3, 3] - mujoco_relative[:3, 3]))
                orientation_error = _rotation_error(urdf_relative[:3, :3], mujoco_relative[:3, :3])
                position_errors[link].append(position_error)
                orientation_errors[link].append(orientation_error)
                fk_world = reference_world @ urdf_relative
                positions[link].append(fk_world[:3, 3].copy())
                metrics[f"{side}_position_error_um"] = position_error * 1e6
                metrics[f"{side}_orientation_error_urad"] = orientation_error * 1e6
                frame_text.extend(
                    (
                        f"### {side}",
                        f"`buttock -> {link}` position (m): "
                        f"`[{urdf_relative[0, 3]:+.5f}, {urdf_relative[1, 3]:+.5f}, "
                        f"{urdf_relative[2, 3]:+.5f}]`",
                        f"Position error: **{position_error * 1e6:.3f} um**; "
                        f"orientation error: **{orientation_error * 1e6:.3f} urad**",
                        "",
                    )
                )
                actual_world = _body_pose(simulator.model, simulator.data, link)
                for stream in telemetry.streams:
                    stream.log(
                        f"world/validation/urdf_fk/{link}",
                        rr.Transform3D(
                            translation=fk_world[:3, 3],
                            mat3x3=fk_world[:3, :3],
                            axis_length=0.12,
                        ),
                    )
                    stream.log(
                        f"world/validation/points/{side}",
                        rr.Points3D(
                            [actual_world[:3, 3], fk_world[:3, 3]],
                            radii=[0.014, 0.007],
                            colors=[[20, 220, 80], [240, 40, 40]],
                            labels=["MuJoCo palm", "URDF FK"],
                        ),
                    )
                    if output_step % 10 == 0:
                        stream.log(
                            f"world/validation/trail/{side}",
                            rr.LineStrips3D(
                                [np.asarray(positions[link])],
                                radii=0.002,
                                colors=[40, 180, 255] if side == "left" else [255, 180, 40],
                            ),
                        )
            telemetry.log_state(
                output_step,
                simulator.data,
                target,
                act_values,
                metrics,
                images={"cam_high_left": current_image},
                time_seconds=float(frame_id) / 30.0,
            )
            for stream in telemetry.streams:
                stream.log(
                    "validation/report",
                    rr.TextDocument("\n".join(frame_text), media_type=rr.MediaType.MARKDOWN),
                )
    finally:
        capture.release()
        telemetry.close()

    max_position_error = max(max(values) for values in position_errors.values())
    max_orientation_error = max(max(values) for values in orientation_errors.values())
    waist_probe = source[frame_ids[len(frame_ids) // 2]].copy()
    waist_joint_positions = {
        name: float(value) for name, value in zip(ACT_STATE_JOINTS, waist_probe, strict=True)
    }
    perturbed_positions = dict(waist_joint_positions)
    perturbed_positions["WAIST"] += 0.1
    waist_sensitivity = {
        link: float(
            np.linalg.norm(
                fk.pose(REFERENCE_LINK, link, perturbed_positions)[:3, 3]
                - fk.pose(REFERENCE_LINK, link, waist_joint_positions)[:3, 3]
            )
        )
        for link in END_EFFECTOR_LINKS
    }
    passed = (
        max_position_error <= POSITION_TOLERANCE_M
        and max_orientation_error <= ORIENTATION_TOLERANCE_RAD
        and all(value > 0.0 for value in waist_sensitivity.values())
    )
    report = {
        "status": "passed" if passed else "failed",
        "dataset": str(dataset_root),
        "dataset_info_sha256": sha256_file(dataset_root / "meta" / "info.json"),
        "urdf": str(SOURCE_URDF),
        "urdf_sha256": sha256_file(SOURCE_URDF),
        "episode_id": episode_id,
        "joint_source": joint_source,
        "reference_frame": REFERENCE_LINK,
        "end_effector_frames": list(END_EFFECTOR_LINKS),
        "frames_checked": int(len(frame_ids)),
        "first_frame": int(frame_ids[0]),
        "last_frame": int(frame_ids[-1]),
        "max_position_error_m": max_position_error,
        "max_orientation_error_rad": max_orientation_error,
        "position_tolerance_m": POSITION_TOLERANCE_M,
        "orientation_tolerance_rad": ORIENTATION_TOLERANCE_RAD,
        "waist_perturbation_rad": 0.1,
        "waist_position_response_m": waist_sensitivity,
        "lower_body_contract": (
            "Reference starts at buttock, so untrained ANKLE/KNEE/BUTTOCK joints are excluded "
            "while trained WAIST remains in both arm chains."
        ),
        "recording": str(recording_path),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if not passed:
        raise AssertionError(f"EE coordinate validation failed; see {report_path}")
    viewer_pid = _launch_viewer(recording_path) if open_viewer else None
    return report_path, recording_path, viewer_pid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate dataset-to-URDF EE coordinates and create a native Rerun visualization"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dataset/lerobotv30/fist_pound_lerobotv30"),
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT / "ee_pose_validation",
    )
    parser.add_argument("--episode-id", type=int, default=56)
    parser.add_argument("--joint-source", choices=("state", "action"), default="action")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--open-viewer", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    report_path, recording_path, viewer_pid = validate_episode(
        dataset_root=args.dataset,
        artifact_root=args.artifacts,
        episode_id=args.episode_id,
        joint_source=args.joint_source,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        open_viewer=args.open_viewer,
    )
    print(f"EE_VALIDATION_REPORT={report_path}", flush=True)
    print(f"EE_VALIDATION_RRD={recording_path}", flush=True)
    print(f"EE_VALIDATION_VIEWER_PID={viewer_pid or 'not-opened'}", flush=True)


if __name__ == "__main__":
    main()
