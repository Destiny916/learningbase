"""Convert dual-arm, three-camera timestamped captures into a LeRobot v3 video dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


LEFT_JOINT_PATH = Path("arm/jointState/puppetLeft")
RIGHT_JOINT_PATH = Path("arm/jointState/puppetRight")
STEREO_RIGHT_PATH = Path("camera/color/stereoRight")
GRIPPER_RIGHT_PATH = Path("camera/color/pikaGripperDepthCamera_r")
GRIPPER_LEFT_PATH = Path("camera/color/pikaGripperDepthCamera_l")
CAMERAS = {
    "observation.images.top": STEREO_RIGHT_PATH,
    "observation.images.gripper_right": GRIPPER_RIGHT_PATH,
    "observation.images.gripper_left": GRIPPER_LEFT_PATH,
}
JOINT_NAMES = [f"left_joint_{index}" for index in range(6)] + ["left_gripper"] + [
    f"right_joint_{index}" for index in range(6)
] + ["right_gripper"]
DEFAULT_FPS = 30
DEFAULT_MAX_ALIGNMENT_DELTA_SEC = 0.01


@dataclass(frozen=True)
class AlignedFrame:
    stereo_path: Path
    left_joint_path: Path
    right_joint_path: Path
    gripper_right_path: Path
    gripper_left_path: Path
    max_alignment_delta_sec: float


@dataclass(frozen=True)
class EpisodeReport:
    source_episode: str
    output_episode_index: int
    raw_stereo_frames: int
    kept_frames: int
    discarded_stereo_frames: int
    max_alignment_delta_sec: float


@dataclass(frozen=True)
class ConversionReport:
    output_root: str
    repo_id: str
    total_source_episodes: int
    total_output_episodes: int
    total_kept_frames: int
    total_discarded_stereo_frames: int
    episodes: list[EpisodeReport]
    skipped_episodes: list[dict[str, str]]


def _timestamp(path: Path) -> float:
    try:
        return float(path.stem)
    except ValueError as exc:
        raise ValueError(f"Timestamp filename must be numeric: {path}") from exc


def _load_paths(directory: Path, suffix: str) -> tuple[list[Path], np.ndarray]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Required modality directory is absent: {directory}")
    paths = [path for path in directory.glob(f"*{suffix}") if _is_timestamped(path)]
    paths.sort(key=_timestamp)
    if not paths:
        raise ValueError(f"No timestamped {suffix} files in {directory}")
    return paths, np.asarray([_timestamp(path) for path in paths], dtype=np.float64)


def _is_timestamped(path: Path) -> bool:
    try:
        _timestamp(path)
    except ValueError:
        return False
    return True


def _nearest(paths: list[Path], timestamps: np.ndarray, reference: float) -> tuple[Path, float]:
    insertion = int(np.searchsorted(timestamps, reference))
    candidates = [max(insertion - 1, 0), min(insertion, len(paths) - 1)]
    index = min(candidates, key=lambda item: abs(float(timestamps[item]) - reference))
    return paths[index], abs(float(timestamps[index]) - reference)


def _load_joint(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    value = np.asarray(payload.get("position"), dtype=np.float32)
    if value.shape != (7,) or not np.isfinite(value).all():
        raise ValueError(f"Joint sample must have seven finite positions: {path}")
    return value


def _load_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        value = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"Image does not decode as RGB: {path}")
    return value


def _episode_sort_key(path: Path) -> tuple[int, int | str]:
    suffix = path.name.removeprefix("episode")
    return (0, int(suffix)) if suffix.isdigit() else (1, path.name)


def _discover_episodes(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {root}")
    episodes = [path for path in root.iterdir() if path.is_dir() and path.name.startswith("episode")]
    episodes.sort(key=_episode_sort_key)
    if not episodes:
        raise ValueError(f"No episode directories in {root}")
    return episodes


def align_episode(
    episode: Path, *, max_alignment_delta_sec: float, resample_to_fps: int | None = None
) -> list[AlignedFrame]:
    """Use stereoRight as anchor and retain only complete, near-synchronous frames."""
    stereo_paths, stereo_times = _load_paths(episode / STEREO_RIGHT_PATH, ".jpg")
    left_paths, left_times = _load_paths(episode / LEFT_JOINT_PATH, ".json")
    right_paths, right_times = _load_paths(episode / RIGHT_JOINT_PATH, ".json")
    gripper_right_paths, gripper_right_times = _load_paths(episode / GRIPPER_RIGHT_PATH, ".jpg")
    gripper_left_paths, gripper_left_times = _load_paths(episode / GRIPPER_LEFT_PATH, ".jpg")

    if resample_to_fps is not None:
        if resample_to_fps <= 0:
            raise ValueError("resample_to_fps must be positive")
        grid = np.arange(stereo_times[0], stereo_times[-1] + 0.5 / resample_to_fps, 1 / resample_to_fps)
        selected_indices: list[int] = []
        for target in grid:
            insertion = int(np.searchsorted(stereo_times, target))
            candidates = [max(insertion - 1, 0), min(insertion, len(stereo_times) - 1)]
            index = min(candidates, key=lambda item: abs(float(stereo_times[item]) - target))
            if not selected_indices or index != selected_indices[-1]:
                selected_indices.append(index)
        anchor_paths = [stereo_paths[index] for index in selected_indices]
        anchor_times = stereo_times[selected_indices]
    else:
        anchor_paths, anchor_times = stereo_paths, stereo_times

    aligned: list[AlignedFrame] = []
    for stereo_path, timestamp in zip(anchor_paths, anchor_times, strict=True):
        left_path, left_delta = _nearest(left_paths, left_times, float(timestamp))
        right_path, right_delta = _nearest(right_paths, right_times, float(timestamp))
        gripper_right_path, gripper_right_delta = _nearest(
            gripper_right_paths, gripper_right_times, float(timestamp)
        )
        gripper_left_path, gripper_left_delta = _nearest(
            gripper_left_paths, gripper_left_times, float(timestamp)
        )
        max_delta = max(left_delta, right_delta, gripper_right_delta, gripper_left_delta)
        if max_delta <= max_alignment_delta_sec:
            aligned.append(
                AlignedFrame(
                    stereo_path,
                    left_path,
                    right_path,
                    gripper_right_path,
                    gripper_left_path,
                    max_delta,
                )
            )
    if not aligned:
        raise ValueError(f"No fully aligned frames within {max_alignment_delta_sec:.3f}s: {episode}")
    return aligned


def _next_actions(states: np.ndarray) -> np.ndarray:
    return np.concatenate((states[1:], states[-1:]), axis=0)


def _features(images: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        **{key: {"dtype": "video", "shape": value.shape, "names": None} for key, value in images.items()},
        "observation.state": {"dtype": "float32", "shape": (14,), "names": JOINT_NAMES},
        "action": {"dtype": "float32", "shape": (14,), "names": JOINT_NAMES},
    }


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def convert_capture_root(
    input_root: Path,
    output_root: Path,
    *,
    repo_id: str,
    fps: int = DEFAULT_FPS,
    max_alignment_delta_sec: float = DEFAULT_MAX_ALIGNMENT_DELTA_SEC,
    resample_to_fps: int | None = None,
    encoder_threads: int = 4,
    task: str | None = None,
) -> ConversionReport:
    """Convert all complete dual-arm episodes into one LeRobot v3 H.264 dataset."""
    if fps <= 0 or max_alignment_delta_sec < 0 or encoder_threads <= 0:
        raise ValueError("fps and encoder_threads must be positive; alignment delta must be non-negative")
    input_root, output_root = Path(input_root).resolve(), Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite output root: {output_root}")
    build_root = output_root.parent / f".{output_root.name}.building"
    if build_root.exists():
        raise FileExistsError(f"Stale build directory exists: {build_root}")

    source_episodes = _discover_episodes(input_root)
    valid: list[tuple[Path, list[AlignedFrame]]] = []
    skipped: list[dict[str, str]] = []
    for episode in source_episodes:
        try:
            valid.append(
                (
                    episode,
                    align_episode(
                        episode,
                        max_alignment_delta_sec=max_alignment_delta_sec,
                        resample_to_fps=resample_to_fps,
                    ),
                )
            )
        except (FileNotFoundError, ValueError) as exc:
            skipped.append({"source_episode": str(episode), "reason": str(exc)})
    if not valid:
        raise ValueError("No episodes have complete, aligned dual-arm and three-camera data")

    first = valid[0][1][0]
    first_images = {
        "observation.images.top": _load_image(first.stereo_path),
        "observation.images.gripper_right": _load_image(first.gripper_right_path),
        "observation.images.gripper_left": _load_image(first.gripper_left_path),
    }

    from lerobot.configs import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = None
    reports: list[EpisodeReport] = []
    try:
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=build_root,
            fps=fps,
            features=_features(first_images),
            use_videos=True,
            # stereoRight is 720x405. yuv420p requires even image dimensions,
            # while yuv444p preserves the native odd-height RGB frame.
            rgb_encoder=RGBEncoderConfig(vcodec="h264", pix_fmt="yuv444p"),
            streaming_encoding=True,
            encoder_queue_maxsize=max(max(len(frames) for _, frames in valid) + 8, 32),
            encoder_threads=encoder_threads,
            image_writer_threads=4,
        )
        for episode, frames in valid:
            states = np.asarray(
                [np.concatenate((_load_joint(frame.left_joint_path), _load_joint(frame.right_joint_path))) for frame in frames],
                dtype=np.float32,
            )
            actions = _next_actions(states)
            for state, action, frame in zip(states, actions, frames, strict=True):
                images = {
                    "observation.images.top": _load_image(frame.stereo_path),
                    "observation.images.gripper_right": _load_image(frame.gripper_right_path),
                    "observation.images.gripper_left": _load_image(frame.gripper_left_path),
                }
                for key, image in images.items():
                    if tuple(image.shape) != tuple(first_images[key].shape):
                        raise ValueError(f"Inconsistent {key} image shape in {episode}")
                dataset.add_frame({"observation.state": state, "action": action, **images, "task": task or input_root.name})
            dataset.save_episode(parallel_encoding=False)
            reports.append(
                EpisodeReport(
                    source_episode=str(episode),
                    output_episode_index=len(reports),
                    raw_stereo_frames=len(_load_paths(episode / STEREO_RIGHT_PATH, ".jpg")[0]),
                    kept_frames=len(frames),
                    discarded_stereo_frames=len(_load_paths(episode / STEREO_RIGHT_PATH, ".jpg")[0]) - len(frames),
                    max_alignment_delta_sec=max(frame.max_alignment_delta_sec for frame in frames),
                )
            )
        dataset.finalize()
        report = ConversionReport(
            output_root=str(output_root),
            repo_id=repo_id,
            total_source_episodes=len(source_episodes),
            total_output_episodes=len(reports),
            total_kept_frames=sum(item.kept_frames for item in reports),
            total_discarded_stereo_frames=sum(item.discarded_stereo_frames for item in reports),
            episodes=reports,
            skipped_episodes=skipped,
        )
        _write_json(
            build_root / "conversion_summary.json",
            {
                **asdict(report),
                "state_action_order": JOINT_NAMES,
                "state_semantics": "state[t] = nearest left/right puppet q at stereoRight time t",
                "action_semantics": "action[t] = state[t+1]; action[T] = state[T]",
                "camera_anchor": str(STEREO_RIGHT_PATH),
                "max_alignment_delta_sec": max_alignment_delta_sec,
                "resample_to_fps": resample_to_fps,
            },
        )
        build_root.rename(output_root)
        return report
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
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--fps", default=DEFAULT_FPS, type=int)
    parser.add_argument("--max-alignment-delta-sec", default=DEFAULT_MAX_ALIGNMENT_DELTA_SEC, type=float)
    parser.add_argument("--resample-to-fps", type=int)
    parser.add_argument("--encoder-threads", default=4, type=int)
    parser.add_argument("--task")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(json.dumps(asdict(convert_capture_root(**vars(args))), indent=2))


if __name__ == "__main__":
    main()
