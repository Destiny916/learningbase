"""Convert timestamped right-arm and right-fisheye captures into LeRobot v3."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


RIGHT_JOINT_RELATIVE_PATH = Path("arm/jointState/puppetRight")
RIGHT_FISHEYE_RELATIVE_PATH = Path("camera/color/pikaGripperFisheyeCamera_r")
JOINT_NAMES = ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"]
RIGHT_FISHEYE_KEY = "observation.images.right_fisheye"
DEFAULT_FPS = 30
DEFAULT_MAX_ALIGNMENT_DELTA_SEC = 0.01
DEFAULT_VCODEC = "h264"
DEFAULT_ENCODER_THREADS = 4

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlignedFrame:
    """One camera frame paired with its nearest right-arm joint sample."""

    camera_path: Path
    joint_path: Path
    camera_timestamp: float
    joint_timestamp: float

    @property
    def alignment_delta_sec(self) -> float:
        return abs(self.joint_timestamp - self.camera_timestamp)


@dataclass(frozen=True)
class EpisodeConversionReport:
    """Alignment and output summary for one source episode."""

    source_root: str
    source_episode: str
    output_episode_index: int
    task: str
    raw_joint_samples: int
    raw_camera_frames: int
    kept_frames: int
    discarded_camera_frames: int
    max_alignment_delta_sec: float
    mean_alignment_delta_sec: float
    p95_alignment_delta_sec: float


@dataclass(frozen=True)
class SkippedEpisodeReport:
    """A source episode that cannot yield a right-arm/right-camera sample."""

    source_root: str
    source_episode: str
    raw_joint_samples: int
    raw_camera_frames: int
    reason: str


@dataclass(frozen=True)
class ConversionReport:
    """Summary returned after a complete LeRobot v3 conversion."""

    output_root: str
    repo_id: str
    fps: int
    max_alignment_delta_sec: float
    episode_index_divisor: int | None
    total_source_episodes: int
    total_output_episodes: int
    total_kept_frames: int
    total_discarded_camera_frames: int
    episodes: list[EpisodeConversionReport]
    skipped_episodes: list[SkippedEpisodeReport]


def _timestamp_from_path(path: Path) -> float:
    try:
        return float(path.stem)
    except ValueError as exc:
        raise ValueError(f"Expected a numeric timestamp filename, got {path.name!r}") from exc


def _load_timestamped_paths(directory: Path, suffix: str) -> tuple[list[Path], np.ndarray]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Required modality directory was not found: {directory}")

    paths: list[Path] = []
    for path in directory.glob(f"*{suffix}"):
        try:
            _timestamp_from_path(path)
        except ValueError:
            continue
        paths.append(path)

    paths.sort(key=_timestamp_from_path)
    if not paths:
        raise ValueError(f"No timestamped {suffix} files found in {directory}")
    return paths, np.asarray([_timestamp_from_path(path) for path in paths], dtype=np.float64)


def _count_timestamped_files(directory: Path, suffix: str) -> int:
    if not directory.is_dir():
        return 0
    count = 0
    for path in directory.glob(f"*{suffix}"):
        try:
            _timestamp_from_path(path)
        except ValueError:
            continue
        count += 1
    return count


def align_right_arm_to_camera(
    joint_dir: Path,
    camera_dir: Path,
    *,
    max_alignment_delta_sec: float,
) -> list[AlignedFrame]:
    """Align each camera frame with the nearest right-arm joint file.

    Camera frames outside the maximum timestamp residual are intentionally
    omitted so each produced LeRobot frame has a valid robot state.
    """
    if max_alignment_delta_sec < 0:
        raise ValueError("max_alignment_delta_sec must be non-negative")

    alignment_limit = max_alignment_delta_sec + np.finfo(np.float64).eps * max(
        1.0, abs(max_alignment_delta_sec)
    )

    joint_paths, joint_timestamps = _load_timestamped_paths(joint_dir, ".json")
    camera_paths, camera_timestamps = _load_timestamped_paths(camera_dir, ".jpg")
    insertion_indices = np.searchsorted(joint_timestamps, camera_timestamps)
    previous_indices = np.clip(insertion_indices - 1, 0, len(joint_paths) - 1)
    following_indices = np.clip(insertion_indices, 0, len(joint_paths) - 1)
    nearest_indices = np.where(
        np.abs(joint_timestamps[previous_indices] - camera_timestamps)
        <= np.abs(joint_timestamps[following_indices] - camera_timestamps),
        previous_indices,
        following_indices,
    )

    aligned: list[AlignedFrame] = []
    for camera_path, camera_timestamp, joint_idx in zip(
        camera_paths, camera_timestamps, nearest_indices, strict=True
    ):
        joint_timestamp = float(joint_timestamps[joint_idx])
        frame = AlignedFrame(
            camera_path=camera_path,
            joint_path=joint_paths[int(joint_idx)],
            camera_timestamp=float(camera_timestamp),
            joint_timestamp=joint_timestamp,
        )
        if frame.alignment_delta_sec <= alignment_limit:
            aligned.append(frame)

    if not aligned:
        raise ValueError(
            "No camera frames are within the requested timestamp alignment tolerance "
            f"({max_alignment_delta_sec:.6f}s)"
        )
    return aligned


def build_next_actions(states: np.ndarray) -> np.ndarray:
    """Build q_(t+1) actions and repeat q_T for the terminal action."""
    if states.ndim != 2 or states.shape[1] != len(JOINT_NAMES) or len(states) == 0:
        raise ValueError(f"states must have shape [N, {len(JOINT_NAMES)}] with N > 0")
    return np.concatenate([states[1:], states[-1:]], axis=0)


def _load_joint_position(path: Path) -> np.ndarray:
    try:
        with path.open(encoding="utf-8") as file:
            payload = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid joint JSON in {path}") from exc

    if not isinstance(payload, dict) or "position" not in payload:
        raise ValueError(f"Joint JSON {path} does not contain a position field")
    position = np.asarray(payload["position"], dtype=np.float32)
    if position.shape != (len(JOINT_NAMES),):
        raise ValueError(
            f"Joint JSON {path} must contain {len(JOINT_NAMES)} positions, got shape {position.shape}"
        )
    if not np.isfinite(position).all():
        raise ValueError(f"Joint JSON {path} contains non-finite positions")
    return position


def _load_rgb_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Camera image {path} does not decode as HxWx3 RGB")
    return rgb


def _episode_sort_key(path: Path) -> tuple[int, int | str]:
    suffix = path.name.removeprefix("episode")
    if suffix.isdigit():
        return (0, int(suffix))
    return (1, path.name)


def _episode_index_is_divisible(path: Path, divisor: int) -> bool:
    suffix = path.name.removeprefix("episode")
    return suffix.isdigit() and int(suffix) % divisor == 0


def _validate_episode_index_divisor(episode_index_divisor: int | None) -> None:
    if episode_index_divisor is not None and (
        type(episode_index_divisor) is not int or episode_index_divisor <= 0
    ):
        raise ValueError("episode_index_divisor must be positive integer")


def discover_episode_dirs(
    input_root: Path,
    *,
    episode_index_divisor: int | None = None,
    joint_relative_path: Path = RIGHT_JOINT_RELATIVE_PATH,
    camera_relative_path: Path = RIGHT_FISHEYE_RELATIVE_PATH,
) -> list[Path]:
    """Return source episodes in stable natural order."""
    _validate_episode_index_divisor(episode_index_divisor)

    input_root = input_root.resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    if (input_root / joint_relative_path).is_dir() or (input_root / camera_relative_path).is_dir():
        if episode_index_divisor is not None and not _episode_index_is_divisible(
            input_root, episode_index_divisor
        ):
            raise ValueError(
                f"Direct input episode {input_root.name!r} does not match "
                f"episode index divisor {episode_index_divisor}"
            )
        return [input_root]

    episodes = [path for path in input_root.iterdir() if path.is_dir() and path.name.startswith("episode")]
    if not episodes:
        raise ValueError(f"No episode directories found in {input_root}")
    if episode_index_divisor is not None:
        episodes = [
            path
            for path in episodes
            if _episode_index_is_divisible(path, episode_index_divisor)
        ]
    episodes.sort(key=_episode_sort_key)
    return episodes


def _make_features(image_shape: tuple[int, int, int], *, camera_key: str) -> dict[str, dict[str, Any]]:
    return {
        camera_key: {"dtype": "video", "shape": image_shape, "names": None},
        "observation.state": {"dtype": "float32", "shape": (7,), "names": JOINT_NAMES},
        "action": {"dtype": "float32", "shape": (7,), "names": JOINT_NAMES},
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)
        file.write("\n")


def _task_for_root(input_root: Path, task: str | None) -> str:
    return task if task is not None else input_root.name


def _streaming_encoder_drops(dataset: Any) -> dict[str, int]:
    encoder = getattr(dataset.writer, "_streaming_encoder", None)
    dropped = getattr(encoder, "_dropped_frames", {})
    return dict(dropped)


def _prepare_episode(
    episode_dir: Path,
    *,
    joint_relative_path: Path = RIGHT_JOINT_RELATIVE_PATH,
    camera_relative_path: Path = RIGHT_FISHEYE_RELATIVE_PATH,
) -> tuple[list[AlignedFrame], np.ndarray, np.ndarray, np.ndarray]:
    joint_dir = episode_dir / joint_relative_path
    camera_dir = episode_dir / camera_relative_path
    aligned = align_right_arm_to_camera(
        joint_dir,
        camera_dir,
        max_alignment_delta_sec=DEFAULT_MAX_ALIGNMENT_DELTA_SEC,
    )
    states = np.asarray([_load_joint_position(frame.joint_path) for frame in aligned], dtype=np.float32)
    actions = build_next_actions(states)
    images = np.asarray([_load_rgb_image(frame.camera_path) for frame in aligned], dtype=np.uint8)
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"Images in {episode_dir} do not have a consistent HxWx3 shape")
    return aligned, states, actions, images


def convert_capture_roots(
    input_roots: list[Path],
    output_root: Path,
    *,
    repo_id: str,
    fps: int = DEFAULT_FPS,
    max_alignment_delta_sec: float = DEFAULT_MAX_ALIGNMENT_DELTA_SEC,
    vcodec: str = DEFAULT_VCODEC,
    encoder_threads: int = DEFAULT_ENCODER_THREADS,
    task: str | None = None,
    episode_index_divisor: int | None = None,
    joint_relative_path: Path = RIGHT_JOINT_RELATIVE_PATH,
    camera_relative_path: Path = RIGHT_FISHEYE_RELATIVE_PATH,
    camera_key: str = RIGHT_FISHEYE_KEY,
) -> ConversionReport:
    """Convert one or more raw capture roots into one LeRobot v3 video dataset."""
    if not input_roots:
        raise ValueError("At least one input root is required")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if max_alignment_delta_sec < 0:
        raise ValueError("max_alignment_delta_sec must be non-negative")
    if encoder_threads <= 0:
        raise ValueError("encoder_threads must be positive")
    if joint_relative_path.is_absolute() or camera_relative_path.is_absolute():
        raise ValueError("modality paths must be relative to each episode")
    if not camera_key.startswith("observation.images."):
        raise ValueError("camera_key must start with 'observation.images.'")
    _validate_episode_index_divisor(episode_index_divisor)

    resolved_roots = [Path(path).resolve() for path in input_roots]
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output root: {output_root}")
    build_root = output_root.parent / f".{output_root.name}.building"
    if build_root.exists():
        raise FileExistsError(f"Stale build root exists: {build_root}")

    source_episodes = [
        (root, episode)
        for root in resolved_roots
        for episode in discover_episode_dirs(
            root,
            episode_index_divisor=episode_index_divisor,
            joint_relative_path=joint_relative_path,
            camera_relative_path=camera_relative_path,
        )
    ]
    if not source_episodes:
        raise ValueError("No source episodes were discovered")

    valid_source_episodes: list[tuple[Path, Path]] = []
    skipped_reports: list[SkippedEpisodeReport] = []
    for source_root, episode_dir in source_episodes:
        joint_count = _count_timestamped_files(episode_dir / joint_relative_path, ".json")
        image_count = _count_timestamped_files(episode_dir / camera_relative_path, ".jpg")
        if joint_count and image_count:
            valid_source_episodes.append((source_root, episode_dir))
            continue
        if not joint_count and not image_count:
            reason = "missing right joint JSON and right fisheye JPEG frames"
        elif not joint_count:
            reason = "missing right joint JSON samples"
        else:
            reason = "missing right fisheye JPEG frames"
        skipped_reports.append(
            SkippedEpisodeReport(
                source_root=str(source_root),
                source_episode=str(episode_dir),
                raw_joint_samples=joint_count,
                raw_camera_frames=image_count,
                reason=reason,
            )
        )

    if not valid_source_episodes:
        _write_json(
            output_root.parent / f"{output_root.name}.rejected_episodes.json",
            {
                "repo_id": repo_id,
                "total_source_episodes": len(source_episodes),
                "total_output_episodes": 0,
                "skipped_episodes": [asdict(report) for report in skipped_reports],
            },
        )
        raise ValueError("No episodes contain both right joint JSON and right fisheye JPEG frames")

    first_camera_dir = valid_source_episodes[0][1] / camera_relative_path
    first_images, _ = _load_timestamped_paths(first_camera_dir, ".jpg")
    features = _make_features(tuple(_load_rgb_image(first_images[0]).shape), camera_key=camera_key)

    from lerobot.configs import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = None
    reports: list[EpisodeConversionReport] = []
    alignment_manifests: list[dict[str, Any]] = []
    try:
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=features,
            root=build_root,
            use_videos=True,
            rgb_encoder=RGBEncoderConfig(vcodec=vcodec),
            streaming_encoding=True,
            encoder_queue_maxsize=max(
                max(
                    len(_load_timestamped_paths(episode / camera_relative_path, ".jpg")[0])
                    for _, episode in valid_source_episodes
                )
                + 8,
                32,
            ),
            encoder_threads=encoder_threads,
            image_writer_threads=4,
        )

        for source_root, episode_dir in valid_source_episodes:
            joint_dir = episode_dir / joint_relative_path
            camera_dir = episode_dir / camera_relative_path
            joint_paths, _ = _load_timestamped_paths(joint_dir, ".json")
            camera_paths, _ = _load_timestamped_paths(camera_dir, ".jpg")
            try:
                aligned = align_right_arm_to_camera(
                    joint_dir,
                    camera_dir,
                    max_alignment_delta_sec=max_alignment_delta_sec,
                )
            except ValueError as exc:
                if not str(exc).startswith("No camera frames are within"):
                    raise
                skipped_reports.append(
                    SkippedEpisodeReport(
                        source_root=str(source_root),
                        source_episode=str(episode_dir),
                        raw_joint_samples=len(joint_paths),
                        raw_camera_frames=len(camera_paths),
                        reason="no right camera frames within alignment tolerance",
                    )
                )
                continue
            states = np.asarray([_load_joint_position(frame.joint_path) for frame in aligned], dtype=np.float32)
            actions = build_next_actions(states)
            images = [_load_rgb_image(frame.camera_path) for frame in aligned]
            for image in images:
                if tuple(image.shape) != tuple(features[camera_key]["shape"]):
                    raise ValueError(
                        f"Image {episode_dir} has shape {tuple(image.shape)}, expected "
                        f"{tuple(features[camera_key]['shape'])}"
                    )

            episode_task = _task_for_root(source_root, task)
            for state, action, image in zip(states, actions, images, strict=True):
                dataset.add_frame(
                    {
                        "observation.state": state,
                        "action": action,
                        camera_key: image,
                        "task": episode_task,
                    }
                )
            dataset.save_episode(parallel_encoding=False)
            dropped = _streaming_encoder_drops(dataset)
            if any(dropped.values()):
                raise RuntimeError(f"Streaming encoder dropped frames for {episode_dir}: {dropped}")

            residuals = np.asarray([frame.alignment_delta_sec for frame in aligned], dtype=np.float64)
            report = EpisodeConversionReport(
                source_root=str(source_root),
                source_episode=str(episode_dir),
                output_episode_index=len(reports),
                task=episode_task,
                raw_joint_samples=len(joint_paths),
                raw_camera_frames=len(camera_paths),
                kept_frames=len(aligned),
                discarded_camera_frames=len(camera_paths) - len(aligned),
                max_alignment_delta_sec=float(residuals.max()),
                mean_alignment_delta_sec=float(residuals.mean()),
                p95_alignment_delta_sec=float(np.quantile(residuals, 0.95)),
            )
            reports.append(report)
            alignment_manifests.append(
                {
                    "episode": asdict(report),
                    "frames": [
                        {
                            "camera_file": frame.camera_path.name,
                            "camera_timestamp": frame.camera_timestamp,
                            "joint_file": frame.joint_path.name,
                            "joint_timestamp": frame.joint_timestamp,
                            "alignment_delta_sec": frame.alignment_delta_sec,
                        }
                        for frame in aligned
                    ],
                }
            )
            logger.info(
                "converted %s: kept %d/%d camera frames",
                episode_dir,
                report.kept_frames,
                report.raw_camera_frames,
            )

        dataset.finalize()
        summary = ConversionReport(
            output_root=str(output_root),
            repo_id=repo_id,
            fps=fps,
            max_alignment_delta_sec=max_alignment_delta_sec,
            episode_index_divisor=episode_index_divisor,
            total_source_episodes=len(source_episodes),
            total_output_episodes=len(reports),
            total_kept_frames=sum(report.kept_frames for report in reports),
            total_discarded_camera_frames=sum(report.discarded_camera_frames for report in reports),
            episodes=reports,
            skipped_episodes=skipped_reports,
        )
        _write_json(
            build_root / "conversion_summary.json",
            {
                **asdict(summary),
                "action_semantics": "action[t] = q[t+1]; action[T] = q[T]",
                "state_semantics": "observation.state[t] = nearest joint q at camera t",
                "joint_relative_path": str(joint_relative_path),
                "camera_relative_path": str(camera_relative_path),
                "camera_key": camera_key,
            },
        )
        for manifest in alignment_manifests:
            index = manifest["episode"]["output_episode_index"]
            _write_json(build_root / "alignment_reports" / f"episode-{index:06d}.json", manifest)
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", action="append", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--fps", default=DEFAULT_FPS, type=int)
    parser.add_argument("--max-alignment-delta-sec", default=DEFAULT_MAX_ALIGNMENT_DELTA_SEC, type=float)
    parser.add_argument("--vcodec", default=DEFAULT_VCODEC)
    parser.add_argument("--encoder-threads", default=DEFAULT_ENCODER_THREADS, type=int)
    parser.add_argument("--episode-index-divisor", type=int)
    parser.add_argument("--task")
    parser.add_argument("--joint-relative-path", default=str(RIGHT_JOINT_RELATIVE_PATH), type=Path)
    parser.add_argument("--camera-relative-path", default=str(RIGHT_FISHEYE_RELATIVE_PATH), type=Path)
    parser.add_argument("--camera-key", default=RIGHT_FISHEYE_KEY)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = convert_capture_roots(
        args.input_root,
        args.output_root,
        repo_id=args.repo_id,
        fps=args.fps,
        max_alignment_delta_sec=args.max_alignment_delta_sec,
        vcodec=args.vcodec,
        encoder_threads=args.encoder_threads,
        task=args.task,
        episode_index_divisor=args.episode_index_divisor,
        joint_relative_path=args.joint_relative_path,
        camera_relative_path=args.camera_relative_path,
        camera_key=args.camera_key,
    )
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
