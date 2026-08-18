"""Convert right end-effector pose captures into a standalone LeRobot v3 dataset.

The source end pose is absolute ``xyz + roll/pitch/yaw`` and the gripper width
lives in the seventh right-arm joint sample.  This module deliberately keeps
the raw LeRobot dataset absolute; relative SE(3) labels are made later by the
pose-specific policy processors after temporal sampling.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


POSE10D_NAMES = ["x", "y", "z", "rot6d_0", "rot6d_1", "rot6d_2", "rot6d_3", "rot6d_4", "rot6d_5", "gripper"]
RIGHT_END_POSE_RELATIVE_PATH = Path("arm/endPose/puppetRight")
RIGHT_JOINT_RELATIVE_PATH = Path("arm/jointState/puppetRight")
RIGHT_FISHEYE_RELATIVE_PATH = Path("camera/color/pikaGripperFisheyeCamera_r")


@dataclass(frozen=True)
class AlignedPoseFrame:
    camera_path: Path
    end_pose_path: Path
    gripper_path: Path
    camera_timestamp: float
    end_pose_timestamp: float
    gripper_timestamp: float


@dataclass(frozen=True)
class EpisodeSplit:
    train: dict[Path, list[Path]]
    test: dict[Path, list[Path]]


@dataclass(frozen=True)
class EpisodeConversionReport:
    source_root: str
    source_episode: str
    output_episode_index: int
    task: str
    kept_frames: int
    discarded_camera_frames: int
    max_end_pose_alignment_delta_sec: float
    max_gripper_alignment_delta_sec: float


@dataclass(frozen=True)
class ConversionReport:
    output_root: str
    repo_id: str
    total_source_episodes: int
    total_output_episodes: int
    total_kept_frames: int
    episodes: list[EpisodeConversionReport]
    skipped_episodes: list[str]


@dataclass(frozen=True)
class TrainTestConversionResult:
    train: ConversionReport
    test: ConversionReport
    output_root: str


def _timestamp_from_path(path: Path) -> float:
    try:
        return float(path.stem)
    except ValueError as exc:
        raise ValueError(f"Expected a numeric timestamp filename, got {path.name!r}") from exc


def _load_timestamped_paths(directory: Path, suffix: str) -> tuple[list[Path], np.ndarray]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Required modality directory was not found: {directory}")
    paths = [path for path in directory.glob(f"*{suffix}") if path.stem.replace(".", "", 1).isdigit()]
    paths.sort(key=_timestamp_from_path)
    if not paths:
        raise ValueError(f"No timestamped {suffix} files found in {directory}")
    return paths, np.asarray([_timestamp_from_path(path) for path in paths], dtype=np.float64)


def _nearest_indices(reference_timestamps: np.ndarray, query_timestamps: np.ndarray) -> np.ndarray:
    insertion = np.searchsorted(reference_timestamps, query_timestamps)
    previous = np.clip(insertion - 1, 0, len(reference_timestamps) - 1)
    following = np.clip(insertion, 0, len(reference_timestamps) - 1)
    return np.where(
        np.abs(reference_timestamps[previous] - query_timestamps)
        <= np.abs(reference_timestamps[following] - query_timestamps),
        previous,
        following,
    )


def align_end_pose_and_gripper_to_camera(
    end_pose_dir: Path,
    gripper_dir: Path,
    camera_dir: Path,
    *,
    max_alignment_delta_sec: float,
) -> list[AlignedPoseFrame]:
    """Keep camera frames having both an end pose and gripper sample within tolerance."""
    if max_alignment_delta_sec < 0:
        raise ValueError("max_alignment_delta_sec must be non-negative")
    end_paths, end_times = _load_timestamped_paths(end_pose_dir, ".json")
    gripper_paths, gripper_times = _load_timestamped_paths(gripper_dir, ".json")
    camera_paths, camera_times = _load_timestamped_paths(camera_dir, ".jpg")
    end_indices = _nearest_indices(end_times, camera_times)
    gripper_indices = _nearest_indices(gripper_times, camera_times)
    tolerance = max_alignment_delta_sec + np.finfo(np.float64).eps * max(1.0, abs(max_alignment_delta_sec))
    frames: list[AlignedPoseFrame] = []
    for camera_path, camera_time, end_idx, gripper_idx in zip(
        camera_paths, camera_times, end_indices, gripper_indices, strict=True
    ):
        end_time = float(end_times[int(end_idx)])
        gripper_time = float(gripper_times[int(gripper_idx)])
        if abs(end_time - camera_time) <= tolerance and abs(gripper_time - camera_time) <= tolerance:
            frames.append(
                AlignedPoseFrame(
                    camera_path=camera_path,
                    end_pose_path=end_paths[int(end_idx)],
                    gripper_path=gripper_paths[int(gripper_idx)],
                    camera_timestamp=float(camera_time),
                    end_pose_timestamp=end_time,
                    gripper_timestamp=gripper_time,
                )
            )
    if not frames:
        raise ValueError(
            "No camera frames have both end-pose and gripper samples within the requested timestamp tolerance "
            f"({max_alignment_delta_sec:.6f}s)"
        )
    return frames


def _episode_sort_key(path: Path) -> tuple[int, int | str]:
    suffix = path.name.removeprefix("episode")
    return (0, int(suffix)) if suffix.isdigit() else (1, path.name)


def discover_episode_dirs(input_root: Path, *, episode_index_divisor: int | None = None) -> list[Path]:
    """Discover source episode directories in stable order, optionally retaining multiples only."""
    if episode_index_divisor is not None and (
        type(episode_index_divisor) is not int or episode_index_divisor <= 0
    ):
        raise ValueError("episode_index_divisor must be a positive integer")
    input_root = Path(input_root).resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    episodes = [path for path in input_root.iterdir() if path.is_dir() and path.name.startswith("episode")]
    if episode_index_divisor is not None:
        episodes = [
            path
            for path in episodes
            if path.name.removeprefix("episode").isdigit()
            and int(path.name.removeprefix("episode")) % episode_index_divisor == 0
        ]
    episodes.sort(key=_episode_sort_key)
    if not episodes:
        raise ValueError(f"No matching episode directories found in {input_root}")
    return episodes


def filter_convertible_episode_dirs(
    episodes: list[Path], *, max_alignment_delta_sec: float
) -> tuple[list[Path], dict[str, str]]:
    """Retain only episodes with aligned end-pose, gripper, and right-fisheye data."""
    usable: list[Path] = []
    rejected: dict[str, str] = {}
    for episode in episodes:
        try:
            align_end_pose_and_gripper_to_camera(
                episode / RIGHT_END_POSE_RELATIVE_PATH,
                episode / RIGHT_JOINT_RELATIVE_PATH,
                episode / RIGHT_FISHEYE_RELATIVE_PATH,
                max_alignment_delta_sec=max_alignment_delta_sec,
            )
        except (FileNotFoundError, ValueError) as exc:
            rejected[episode.name] = str(exc)
        else:
            usable.append(episode)
    return usable, rejected


def split_episode_dirs_by_source(
    episodes_by_source: dict[Path, list[Path]], *, test_fraction: float, seed: int
) -> EpisodeSplit:
    """Create a reproducible per-source train/test split without cross-source leakage."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    generator = np.random.default_rng(seed)
    train: dict[Path, list[Path]] = {}
    test: dict[Path, list[Path]] = {}
    for source, episodes in episodes_by_source.items():
        if len(episodes) < 2:
            raise ValueError(f"Source {source} needs at least two episodes for a train/test split")
        indices = generator.permutation(len(episodes))
        test_count = max(1, int(round(len(episodes) * test_fraction)))
        test_indices = set(indices[:test_count].tolist())
        train[source] = [episode for index, episode in enumerate(episodes) if index not in test_indices]
        test[source] = [episode for index, episode in enumerate(episodes) if index in test_indices]
    return EpisodeSplit(train=train, test=test)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload in {path} must be an object")
    return payload


def _load_end_pose(path: Path) -> dict[str, float]:
    payload = _load_json(path)
    return {key: float(value) for key, value in payload.items()}


def _load_gripper(path: Path) -> float:
    payload = _load_json(path)
    position = np.asarray(payload.get("position"), dtype=np.float64)
    if position.shape != (7,):
        raise ValueError(f"Joint JSON {path} must contain seven positions, got shape {position.shape}")
    if not np.isfinite(position).all():
        raise ValueError(f"Joint JSON {path} contains non-finite positions")
    return float(position[6])


def _load_rgb_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Camera image {path} does not decode as HxWx3 RGB")
    return rgb


def _make_features(image_shape: tuple[int, int, int]) -> dict[str, dict[str, Any]]:
    return {
        "observation.images.right_fisheye": {"dtype": "video", "shape": image_shape, "names": None},
        "observation.state": {"dtype": "float32", "shape": (10,), "names": POSE10D_NAMES},
        "action": {"dtype": "float32", "shape": (10,), "names": POSE10D_NAMES},
    }


def _streaming_encoder_drops(dataset: Any) -> dict[str, int]:
    encoder = getattr(dataset.writer, "_streaming_encoder", None)
    return dict(getattr(encoder, "_dropped_frames", {}))


def _flatten_episode_sources(episodes_by_source: dict[Path, list[Path]]) -> list[tuple[Path, Path]]:
    return [(source, episode) for source, episodes in episodes_by_source.items() for episode in episodes]


def encoder_queue_maxsize_for_episodes(source_episodes: list[tuple[Path, Path]]) -> int:
    """Size the video queue so a complete selected episode cannot overflow it."""
    frame_counts: list[int] = []
    for _, episode in source_episodes:
        camera_dir = episode / RIGHT_FISHEYE_RELATIVE_PATH
        try:
            camera_paths, _ = _load_timestamped_paths(camera_dir, ".jpg")
        except (FileNotFoundError, ValueError):
            continue
        frame_counts.append(len(camera_paths))
    if not frame_counts:
        raise ValueError("No selected episode contains a timestamped right-fisheye JPEG frame")
    return max(max(frame_counts) + 8, 32)


def convert_capture_episodes(
    episodes_by_source: dict[Path, list[Path]],
    output_root: Path,
    *,
    repo_id: str,
    fps: int = 30,
    max_alignment_delta_sec: float = 0.01,
    encoder_threads: int = 4,
) -> ConversionReport:
    """Write selected source episodes into one standalone pose10d LeRobot v3 dataset."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    if encoder_threads <= 0:
        raise ValueError("encoder_threads must be positive")
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output root: {output_root}")
    source_episodes = _flatten_episode_sources(episodes_by_source)
    if not source_episodes:
        raise ValueError("No source episodes were selected")

    build_root = output_root.parent / f".{output_root.name}.building"
    if build_root.exists():
        raise FileExistsError(f"Stale build root exists: {build_root}")

    from lerobot.configs import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = None
    reports: list[EpisodeConversionReport] = []
    skipped: list[str] = []
    try:
        for _, episode in source_episodes:
            camera_dir = episode / RIGHT_FISHEYE_RELATIVE_PATH
            if camera_dir.is_dir() and any(camera_dir.glob("*.jpg")):
                image_paths, _ = _load_timestamped_paths(camera_dir, ".jpg")
                features = _make_features(tuple(_load_rgb_image(image_paths[0]).shape))
                break
        else:
            raise ValueError("No selected episode contains a timestamped right-fisheye JPEG frame")

        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=features,
            root=build_root,
            use_videos=True,
            rgb_encoder=RGBEncoderConfig(vcodec="h264"),
            streaming_encoding=True,
            encoder_queue_maxsize=encoder_queue_maxsize_for_episodes(source_episodes),
            encoder_threads=encoder_threads,
            image_writer_threads=4,
        )
        for source_root, episode in source_episodes:
            try:
                aligned = align_end_pose_and_gripper_to_camera(
                    episode / RIGHT_END_POSE_RELATIVE_PATH,
                    episode / RIGHT_JOINT_RELATIVE_PATH,
                    episode / RIGHT_FISHEYE_RELATIVE_PATH,
                    max_alignment_delta_sec=max_alignment_delta_sec,
                )
            except (FileNotFoundError, ValueError) as exc:
                skipped.append(f"{episode}: {exc}")
                continue
            states = np.asarray(
                [
                    pose10d_from_end_pose(_load_end_pose(frame.end_pose_path), _load_gripper(frame.gripper_path))
                    for frame in aligned
                ],
                dtype=np.float32,
            )
            actions = build_next_actions(states)
            images = [_load_rgb_image(frame.camera_path) for frame in aligned]
            if any(tuple(image.shape) != tuple(features["observation.images.right_fisheye"]["shape"]) for image in images):
                raise ValueError(f"Right-fisheye image shape differs within selected episode {episode}")
            for state, action, image in zip(states, actions, images, strict=True):
                dataset.add_frame(
                    {
                        "observation.state": state,
                        "action": action,
                        "observation.images.right_fisheye": image,
                        "task": source_root.name,
                    }
                )
            dataset.save_episode(parallel_encoding=False)
            dropped = _streaming_encoder_drops(dataset)
            if any(dropped.values()):
                raise RuntimeError(f"Streaming encoder dropped frames for {episode}: {dropped}")
            reports.append(
                EpisodeConversionReport(
                    source_root=str(source_root),
                    source_episode=str(episode),
                    output_episode_index=len(reports),
                    task=source_root.name,
                    kept_frames=len(aligned),
                    discarded_camera_frames=len(_load_timestamped_paths(episode / RIGHT_FISHEYE_RELATIVE_PATH, ".jpg")[0])
                    - len(aligned),
                    max_end_pose_alignment_delta_sec=max(
                        abs(frame.end_pose_timestamp - frame.camera_timestamp) for frame in aligned
                    ),
                    max_gripper_alignment_delta_sec=max(
                        abs(frame.gripper_timestamp - frame.camera_timestamp) for frame in aligned
                    ),
                )
            )
        if not reports:
            raise ValueError("No selected episode yielded aligned end-pose, gripper, and camera frames")
        dataset.finalize()
        summary = ConversionReport(
            output_root=str(output_root),
            repo_id=repo_id,
            total_source_episodes=len(source_episodes),
            total_output_episodes=len(reports),
            total_kept_frames=sum(report.kept_frames for report in reports),
            episodes=reports,
            skipped_episodes=skipped,
        )
        (build_root / "conversion_summary.json").write_text(
            json.dumps(
                {
                    "output_root": summary.output_root,
                    "repo_id": summary.repo_id,
                    "total_source_episodes": summary.total_source_episodes,
                    "total_output_episodes": summary.total_output_episodes,
                    "total_kept_frames": summary.total_kept_frames,
                    "pose_representation": "absolute xyz + rot6d + absolute gripper",
                    "action_semantics": "action[t] = state[t+1]; action[T] = state[T]",
                    "episodes": [report.__dict__ for report in reports],
                    "skipped_episodes": skipped,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        build_root.rename(output_root)
        return summary
    except BaseException:
        if dataset is not None:
            try:
                if dataset.has_pending_frames():
                    dataset.clear_episode_buffer()
                dataset.finalize()
            except BaseException:
                pass
        shutil.rmtree(build_root, ignore_errors=True)
        raise


def _split_manifest_section(output_root: Path, episodes_by_source: dict[Path, list[Path]]) -> dict[str, Any]:
    return {
        "root": str(output_root),
        "episodes": {
            str(source): [episode.name for episode in episodes] for source, episodes in episodes_by_source.items()
        },
    }


def convert_capture_roots_with_split(
    input_roots: list[Path],
    output_root: Path,
    *,
    train_repo_id: str,
    test_repo_id: str,
    episode_index_divisor: int = 2,
    test_fraction: float = 0.2,
    split_seed: int = 42,
    fps: int = 30,
    max_alignment_delta_sec: float = 0.01,
    encoder_threads: int = 4,
) -> TrainTestConversionResult:
    """Convert even source episodes into isolated train/test pose10d v3 roots."""
    if not input_roots:
        raise ValueError("At least one input root is required")
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output root: {output_root}")
    build_root = output_root.parent / f".{output_root.name}.building"
    if build_root.exists():
        raise FileExistsError(f"Stale build root exists: {build_root}")
    candidate_episodes = {
        Path(source_root).resolve(): discover_episode_dirs(
            Path(source_root), episode_index_divisor=episode_index_divisor
        )
        for source_root in input_roots
    }
    source_episodes: dict[Path, list[Path]] = {}
    rejected_episodes: dict[str, dict[str, str]] = {}
    for source_root, candidates in candidate_episodes.items():
        usable, rejected = filter_convertible_episode_dirs(
            candidates, max_alignment_delta_sec=max_alignment_delta_sec
        )
        if len(usable) < 2:
            raise ValueError(
                f"Source {source_root} has only {len(usable)} convertible selected episodes; at least two are required"
            )
        source_episodes[source_root] = usable
        rejected_episodes[str(source_root)] = rejected
    split = split_episode_dirs_by_source(source_episodes, test_fraction=test_fraction, seed=split_seed)
    try:
        build_root.mkdir(parents=True)
        train_report = convert_capture_episodes(
            split.train,
            build_root / "train",
            repo_id=train_repo_id,
            fps=fps,
            max_alignment_delta_sec=max_alignment_delta_sec,
            encoder_threads=encoder_threads,
        )
        test_report = convert_capture_episodes(
            split.test,
            build_root / "test",
            repo_id=test_repo_id,
            fps=fps,
            max_alignment_delta_sec=max_alignment_delta_sec,
            encoder_threads=encoder_threads,
        )
        (build_root / "split_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "episode_index_divisor": episode_index_divisor,
                    "test_fraction": test_fraction,
                    "split_seed": split_seed,
                    "pose_representation": "absolute xyz + rot6d + absolute gripper",
                    "rejected_selected_episodes": rejected_episodes,
                    "splits": {
                        "train": _split_manifest_section(output_root / "train", split.train),
                        "test": _split_manifest_section(output_root / "test", split.test),
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        build_root.rename(output_root)
        return TrainTestConversionResult(train=train_report, test=test_report, output_root=str(output_root))
    except BaseException:
        shutil.rmtree(build_root, ignore_errors=True)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", action="append", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--train-repo-id", required=True)
    parser.add_argument("--test-repo-id", required=True)
    parser.add_argument("--episode-index-divisor", default=2, type=int)
    parser.add_argument("--test-fraction", default=0.2, type=float)
    parser.add_argument("--split-seed", default=42, type=int)
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--max-alignment-delta-sec", default=0.01, type=float)
    parser.add_argument("--encoder-threads", default=4, type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = convert_capture_roots_with_split(
        args.input_root,
        args.output_root,
        train_repo_id=args.train_repo_id,
        test_repo_id=args.test_repo_id,
        episode_index_divisor=args.episode_index_divisor,
        test_fraction=args.test_fraction,
        split_seed=args.split_seed,
        fps=args.fps,
        max_alignment_delta_sec=args.max_alignment_delta_sec,
        encoder_threads=args.encoder_threads,
    )
    print(json.dumps({"output_root": result.output_root}, indent=2))


def euler_rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return Piper's ``Rz(yaw) @ Ry(pitch) @ Rx(roll)`` rotation matrix."""
    cos_roll, sin_roll = np.cos(roll), np.sin(roll)
    cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cos_roll, -sin_roll], [0.0, sin_roll, cos_roll]])
    rot_y = np.array([[cos_pitch, 0.0, sin_pitch], [0.0, 1.0, 0.0], [-sin_pitch, 0.0, cos_pitch]])
    rot_z = np.array([[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]])
    return rot_z @ rot_y @ rot_x


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    """Encode the first two rotation-matrix columns in the continuous 6D form."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"rotation matrix must have shape (3, 3), got {matrix.shape}")
    return np.concatenate((matrix[:, 0], matrix[:, 1])).astype(np.float32)


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Project a 6D orientation onto SO(3) with Gram-Schmidt orthonormalization."""
    rot6d = np.asarray(rot6d, dtype=np.float64)
    if rot6d.shape != (6,):
        raise ValueError(f"rot6d must have shape (6,), got {rot6d.shape}")
    first = rot6d[:3]
    second = rot6d[3:]
    first_norm = np.linalg.norm(first)
    if first_norm < 1e-12:
        raise ValueError("rot6d first column has zero norm")
    first = first / first_norm
    second = second - np.dot(first, second) * first
    second_norm = np.linalg.norm(second)
    if second_norm < 1e-12:
        raise ValueError("rot6d columns are collinear")
    second = second / second_norm
    third = np.cross(first, second)
    return np.column_stack((first, second, third)).astype(np.float32)


def pose10d_from_end_pose(end_pose: dict[str, float], gripper: float) -> np.ndarray:
    """Convert an end-pose JSON payload and absolute gripper width into pose10d."""
    required = ("x", "y", "z", "roll", "pitch", "yaw")
    missing = [name for name in required if name not in end_pose]
    if missing:
        raise ValueError(f"end-pose payload is missing fields: {missing}")
    values = np.asarray([end_pose[name] for name in required] + [gripper], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("end-pose or gripper contains non-finite values")
    rotation = euler_rpy_to_matrix(end_pose["roll"], end_pose["pitch"], end_pose["yaw"])
    return np.concatenate((values[:3], matrix_to_rot6d(rotation), values[-1:])).astype(np.float32)


def _pose10d_to_transform(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (10,):
        raise ValueError(f"pose10d must have shape (10,), got {pose.shape}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rot6d_to_matrix(pose[3:9])
    transform[:3, 3] = pose[:3]
    return transform


def _transform_to_pose10d(transform: np.ndarray, gripper: float) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"transform must have shape (4, 4), got {transform.shape}")
    return np.concatenate((transform[:3, 3], matrix_to_rot6d(transform[:3, :3]), [gripper])).astype(np.float32)


def relative_pose10d(anchor: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return ``inv(T_anchor) @ T_target`` with target's absolute gripper width."""
    anchor_transform = _pose10d_to_transform(anchor)
    target_transform = _pose10d_to_transform(target)
    relative_transform = np.linalg.inv(anchor_transform) @ target_transform
    return _transform_to_pose10d(relative_transform, float(np.asarray(target)[9]))


def absolute_pose10d(anchor: np.ndarray, relative: np.ndarray) -> np.ndarray:
    """Compose an anchored relative pose into an absolute pose10d target."""
    absolute_transform = _pose10d_to_transform(anchor) @ _pose10d_to_transform(relative)
    return _transform_to_pose10d(absolute_transform, float(np.asarray(relative)[9]))


def build_next_actions(states: np.ndarray) -> np.ndarray:
    """Build absolute ``action[t] = state[t + 1]`` labels with terminal repeat."""
    states = np.asarray(states, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != len(POSE10D_NAMES) or len(states) == 0:
        raise ValueError(f"states must have shape [N, {len(POSE10D_NAMES)}] with N > 0")
    return np.concatenate((states[1:], states[-1:]), axis=0)


if __name__ == "__main__":
    main()
