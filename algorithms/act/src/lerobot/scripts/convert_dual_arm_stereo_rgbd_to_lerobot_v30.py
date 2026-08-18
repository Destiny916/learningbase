"""Convert selected 0729 dual-arm frames into an isolated stereo-top RGB-D LeRobot v3 dataset."""

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
TOP_LEFT_PATH = Path("camera/color/stereoLeft")
TOP_RIGHT_PATH = Path("camera/color/stereoRight")
GRIPPER_LEFT_RGB_PATH = Path("camera/color/pikaGripperDepthCamera_l")
GRIPPER_RIGHT_RGB_PATH = Path("camera/color/pikaGripperDepthCamera_r")
GRIPPER_LEFT_DEPTH_PATH = Path("camera/depth/pikaGripperDepthCamera_l")
GRIPPER_RIGHT_DEPTH_PATH = Path("camera/depth/pikaGripperDepthCamera_r")

LEFT_DEPTH_RANGE_M = (0.07, 0.90)
RIGHT_DEPTH_RANGE_M = (0.07, 0.60)
STORAGE_DEPTH_RANGE_M = (0.07, 0.90)
# ``threads`` does not constrain x265's internal worker pool. Keep the
# lossless 12-bit depth representation while capping that pool on the shared
# training host.
DEPTH_ENCODER_EXTRA_OPTIONS = {"x265-params": "lossless=1:pools=1"}
# The supplied 0729 windows are indexes into the raw stereoRight sequence.
# Wrist depth is recorded asynchronously and can have short recording gaps, so
# validate the selected window rather than dropping anchor frames beforehand.
DEFAULT_MAX_ALIGNMENT_DELTA_SEC = 0.25
JOINT_NAMES = [f"left_joint_{index}" for index in range(6)] + ["left_gripper"] + [
    f"right_joint_{index}" for index in range(6)
] + ["right_gripper"]

# Inclusive indexes into each source episode's raw, timestamp-sorted
# stereoRight sequence. They are intentionally not indexes into a filtered
# near-synchronous subset.
EPISODE_WINDOWS: dict[int, tuple[int, int]] = {
    0: (179, 343), 1: (163, 317), 2: (131, 305), 3: (132, 262), 4: (138, 233),
    5: (161, 343), 6: (196, 363), 7: (166, 330), 8: (158, 345), 9: (167, 352),
    10: (196, 325), 11: (161, 342), 12: (165, 321), 13: (162, 306), 14: (152, 311),
    15: (170, 287), 16: (186, 320), 17: (150, 278), 18: (150, 276), 19: (157, 327),
    20: (129, 243), 21: (146, 283), 22: (148, 327), 23: (163, 292), 24: (171, 317),
    25: (209, 368), 26: (208, 388), 27: (152, 302), 28: (140, 297), 29: (146, 267),
    30: (142, 262), 31: (114, 215), 32: (132, 233), 33: (144, 231), 34: (117, 219),
    35: (131, 211), 36: (134, 265), 37: (117, 276), 38: (163, 272),
}


@dataclass(frozen=True)
class AlignedFrame:
    source_index: int
    top_left_path: Path
    top_right_path: Path
    gripper_left_rgb_path: Path
    gripper_right_rgb_path: Path
    gripper_left_depth_path: Path
    gripper_right_depth_path: Path
    left_joint_path: Path
    right_joint_path: Path
    max_alignment_delta_sec: float


@dataclass(frozen=True)
class EpisodeReport:
    source_episode: str
    output_episode_index: int
    requested_window: tuple[int, int]
    aligned_frames: int
    kept_frames: int
    max_alignment_delta_sec: float


@dataclass(frozen=True)
class ConversionReport:
    output_root: str
    repo_id: str
    total_output_episodes: int
    total_kept_frames: int
    episodes: list[EpisodeReport]


def _timestamp(path: Path) -> float:
    try:
        return float(path.stem)
    except ValueError as error:
        raise ValueError(f"timestamp filename must be numeric: {path}") from error


def _is_timestamped(path: Path) -> bool:
    try:
        _timestamp(path)
    except ValueError:
        return False
    return True


def _load_paths(directory: Path, suffix: str) -> tuple[list[Path], np.ndarray]:
    if not directory.is_dir():
        raise FileNotFoundError(f"required modality directory is absent: {directory}")
    paths = sorted((path for path in directory.glob(f"*{suffix}") if _is_timestamped(path)), key=_timestamp)
    if not paths:
        raise ValueError(f"no timestamped {suffix} files in {directory}")
    return paths, np.asarray([_timestamp(path) for path in paths], dtype=np.float64)


def _nearest(paths: list[Path], timestamps: np.ndarray, reference: float) -> tuple[Path, float]:
    insertion = int(np.searchsorted(timestamps, reference))
    candidates = (max(insertion - 1, 0), min(insertion, len(paths) - 1))
    index = min(candidates, key=lambda item: abs(float(timestamps[item]) - reference))
    return paths[index], abs(float(timestamps[index]) - reference)


def _episode_index(path: Path) -> int:
    suffix = path.name.removeprefix("episode")
    if not suffix.isdigit():
        raise ValueError(f"episode directory must be named episode<N>: {path}")
    return int(suffix)


def _discover_episodes(root: Path) -> list[Path]:
    episodes = sorted((path for path in root.glob("episode*") if path.is_dir()), key=_episode_index)
    if not episodes:
        raise ValueError(f"no episode directories under {root}")
    return episodes


def ordered_episode_windows(
    episodes: list[Path], windows: dict[int, tuple[int, int]]
) -> list[tuple[int, Path, tuple[int, int]]]:
    """Map logical window rows to the numerically sorted source episode directories."""
    if len(episodes) != len(windows):
        raise ValueError(
            f"source contains {len(episodes)} episode directories but {len(windows)} logical windows were configured"
        )
    expected = list(range(len(windows)))
    if sorted(windows) != expected:
        raise ValueError(f"logical episode windows must be indexed consecutively from 0, got {sorted(windows)}")
    return [(logical_index, episode, windows[logical_index]) for logical_index, episode in enumerate(episodes)]


def _load_joint(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = np.asarray(payload.get("position"), dtype=np.float32)
    if value.shape != (7,) or not np.isfinite(value).all():
        raise ValueError(f"joint sample must be seven finite values: {path}")
    return value


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        value = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"RGB image has invalid shape: {path}")
    return value


def _load_depth_m(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        raw = np.asarray(image)
    if raw.ndim != 2 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"depth image must be a single-channel integer image: {path}")
    return raw.astype(np.float32) / 1000.0


def clip_depth_m(depth_m: np.ndarray, *, camera: str) -> np.ndarray:
    if camera == "left":
        lower, upper = LEFT_DEPTH_RANGE_M
    elif camera == "right":
        lower, upper = RIGHT_DEPTH_RANGE_M
    else:
        raise ValueError(f"camera must be 'left' or 'right', got {camera!r}")
    return np.clip(np.asarray(depth_m, dtype=np.float32), lower, upper)


def trim_states_and_actions(states: np.ndarray, *, start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
    if states.ndim != 2 or states.shape[1] != 14:
        raise ValueError(f"states must have shape (T, 14), got {states.shape}")
    if not 0 <= start <= stop < len(states):
        raise ValueError(f"invalid inclusive window [{start}, {stop}] for {len(states)} aligned frames")
    trimmed = states[start : stop + 1]
    return trimmed, np.concatenate((trimmed[1:], trimmed[-1:]), axis=0)


def align_episode(episode: Path, *, max_alignment_delta_sec: float) -> list[AlignedFrame]:
    top_right_paths, top_right_times = _load_paths(episode / TOP_RIGHT_PATH, ".jpg")
    top_left_paths, top_left_times = _load_paths(episode / TOP_LEFT_PATH, ".jpg")
    left_rgb_paths, left_rgb_times = _load_paths(episode / GRIPPER_LEFT_RGB_PATH, ".jpg")
    right_rgb_paths, right_rgb_times = _load_paths(episode / GRIPPER_RIGHT_RGB_PATH, ".jpg")
    left_depth_paths, left_depth_times = _load_paths(episode / GRIPPER_LEFT_DEPTH_PATH, ".png")
    right_depth_paths, right_depth_times = _load_paths(episode / GRIPPER_RIGHT_DEPTH_PATH, ".png")
    left_joint_paths, left_joint_times = _load_paths(episode / LEFT_JOINT_PATH, ".json")
    right_joint_paths, right_joint_times = _load_paths(episode / RIGHT_JOINT_PATH, ".json")

    aligned: list[AlignedFrame] = []
    for source_index, (top_right_path, timestamp) in enumerate(zip(top_right_paths, top_right_times, strict=True)):
        top_left_path, top_left_delta = _nearest(top_left_paths, top_left_times, float(timestamp))
        left_rgb_path, left_rgb_delta = _nearest(left_rgb_paths, left_rgb_times, float(timestamp))
        right_rgb_path, right_rgb_delta = _nearest(right_rgb_paths, right_rgb_times, float(timestamp))
        left_depth_path, left_depth_delta = _nearest(left_depth_paths, left_depth_times, float(timestamp))
        right_depth_path, right_depth_delta = _nearest(right_depth_paths, right_depth_times, float(timestamp))
        left_joint_path, left_joint_delta = _nearest(left_joint_paths, left_joint_times, float(timestamp))
        right_joint_path, right_joint_delta = _nearest(right_joint_paths, right_joint_times, float(timestamp))
        max_delta = max(
            top_left_delta, left_rgb_delta, right_rgb_delta, left_depth_delta, right_depth_delta,
            left_joint_delta, right_joint_delta,
        )
        aligned.append(
            AlignedFrame(
                source_index=source_index,
                top_left_path=top_left_path,
                top_right_path=top_right_path,
                gripper_left_rgb_path=left_rgb_path,
                gripper_right_rgb_path=right_rgb_path,
                gripper_left_depth_path=left_depth_path,
                gripper_right_depth_path=right_depth_path,
                left_joint_path=left_joint_path,
                right_joint_path=right_joint_path,
                max_alignment_delta_sec=max_delta,
            )
        )
    if not aligned:
        raise ValueError(f"no timestamped stereoRight frames in {episode}")
    return aligned


def select_window(
    aligned: list[AlignedFrame], *, start: int, stop: int, max_alignment_delta_sec: float
) -> list[AlignedFrame]:
    """Select a raw stereoRight-indexed window and enforce its modality lag bound."""
    if not 0 <= start <= stop < len(aligned):
        raise ValueError(f"invalid inclusive window [{start}, {stop}] for {len(aligned)} stereoRight frames")
    selected = aligned[start : stop + 1]
    observed_max = max(frame.max_alignment_delta_sec for frame in selected)
    if observed_max > max_alignment_delta_sec:
        raise ValueError(
            f"window [{start}, {stop}] has modality lag {observed_max:.3f}s above "
            f"the allowed {max_alignment_delta_sec:.3f}s"
        )
    return selected


def _features(images: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    for key, image in images.items():
        is_depth = key.endswith("_depth")
        features[key] = {
            "dtype": "video",
            "shape": image.shape,
            "names": None,
            "info": {"is_depth_map": is_depth, **({"depth_unit": "m"} if is_depth else {})},
        }
    features["observation.state"] = {"dtype": "float32", "shape": (14,), "names": JOINT_NAMES}
    features["action"] = {"dtype": "float32", "shape": (14,), "names": JOINT_NAMES}
    return features


def convert_capture_root(
    input_root: Path,
    output_root: Path,
    *,
    repo_id: str,
    fps: int = 25,
    max_alignment_delta_sec: float = DEFAULT_MAX_ALIGNMENT_DELTA_SEC,
    encoder_threads: int = 4,
    task: str | None = None,
    episode_windows: dict[int, tuple[int, int]] | None = None,
) -> ConversionReport:
    """Build the isolated all-episode 0729 stereo-top RGB-D dataset."""
    if fps <= 0 or encoder_threads <= 0 or max_alignment_delta_sec < 0:
        raise ValueError("fps and encoder_threads must be positive; alignment delta must be non-negative")
    input_root, output_root = Path(input_root).resolve(), Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite output root: {output_root}")
    build_root = output_root.parent / f".{output_root.name}.building"
    if build_root.exists():
        raise FileExistsError(f"stale build root exists: {build_root}")

    windows = EPISODE_WINDOWS if episode_windows is None else episode_windows
    episodes = _discover_episodes(input_root)
    planned_episodes = ordered_episode_windows(episodes, windows)
    aligned_by_episode = [
        (logical_index, episode, window, align_episode(episode, max_alignment_delta_sec=max_alignment_delta_sec))
        for logical_index, episode, window in planned_episodes
    ]

    _, first_episode, _, first_frames = aligned_by_episode[0]
    del first_episode
    first = first_frames[0]
    first_images = {
        "observation.images.top_left": _load_rgb(first.top_left_path),
        "observation.images.top_right": _load_rgb(first.top_right_path),
        "observation.images.gripper_left": _load_rgb(first.gripper_left_rgb_path),
        "observation.images.gripper_right": _load_rgb(first.gripper_right_rgb_path),
        "observation.images.gripper_left_depth": clip_depth_m(_load_depth_m(first.gripper_left_depth_path), camera="left")[..., None],
        "observation.images.gripper_right_depth": clip_depth_m(_load_depth_m(first.gripper_right_depth_path), camera="right")[..., None],
    }

    from lerobot.configs import DepthEncoderConfig, RGBEncoderConfig
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
            rgb_encoder=RGBEncoderConfig(vcodec="h264", pix_fmt="yuv444p"),
            depth_encoder=DepthEncoderConfig(
                depth_min=STORAGE_DEPTH_RANGE_M[0],
                depth_max=STORAGE_DEPTH_RANGE_M[1],
                extra_options=DEPTH_ENCODER_EXTRA_OPTIONS,
            ),
            # Six concurrent H.264/H.265 encoders can each allocate a large
            # codec thread pool on the shared training host. Write one episode
            # of temporary frames and encode its six streams sequentially in
            # save_episode(parallel_encoding=False) below instead.
            streaming_encoding=False,
            encoder_threads=encoder_threads,
            image_writer_threads=4,
        )
        for output_episode_index, episode, (start, stop), aligned in aligned_by_episode:
            try:
                selected = select_window(
                    aligned,
                    start=start,
                    stop=stop,
                    max_alignment_delta_sec=max_alignment_delta_sec,
                )
            except ValueError as error:
                raise ValueError(f"logical episode {output_episode_index} ({episode.name}): {error}") from error
            states = np.asarray(
                [np.concatenate((_load_joint(frame.left_joint_path), _load_joint(frame.right_joint_path))) for frame in selected],
                dtype=np.float32,
            )
            states, actions = trim_states_and_actions(states, start=0, stop=len(states) - 1)
            for state, action, frame in zip(states, actions, selected, strict=True):
                images = {
                    "observation.images.top_left": _load_rgb(frame.top_left_path),
                    "observation.images.top_right": _load_rgb(frame.top_right_path),
                    "observation.images.gripper_left": _load_rgb(frame.gripper_left_rgb_path),
                    "observation.images.gripper_right": _load_rgb(frame.gripper_right_rgb_path),
                    "observation.images.gripper_left_depth": clip_depth_m(_load_depth_m(frame.gripper_left_depth_path), camera="left")[..., None],
                    "observation.images.gripper_right_depth": clip_depth_m(_load_depth_m(frame.gripper_right_depth_path), camera="right")[..., None],
                }
                if any(image.shape != first_images[key].shape for key, image in images.items()):
                    raise ValueError(f"inconsistent image shape in {episode}")
                dataset.add_frame({"observation.state": state, "action": action, **images, "task": task or input_root.name})
            dataset.save_episode(parallel_encoding=False)
            reports.append(
                EpisodeReport(
                    source_episode=str(episode), output_episode_index=output_episode_index,
                    requested_window=(start, stop), aligned_frames=len(aligned), kept_frames=len(selected),
                    max_alignment_delta_sec=max(frame.max_alignment_delta_sec for frame in selected),
                )
            )
        dataset.finalize()
        report = ConversionReport(
            output_root=str(output_root), repo_id=repo_id, total_output_episodes=len(reports),
            total_kept_frames=sum(item.kept_frames for item in reports), episodes=reports,
        )
        (build_root / "conversion_summary.json").write_text(
            json.dumps(
                {
                    **asdict(report), "camera_anchor": str(TOP_RIGHT_PATH),
                    "window_index_semantics": "inclusive indexes in raw timestamp-sorted stereoRight frames",
                    "max_alignment_delta_sec": max_alignment_delta_sec,
                    "depth_ranges_m": {"left": LEFT_DEPTH_RANGE_M, "right": RIGHT_DEPTH_RANGE_M},
                    "state_action_contract": "action[t] == state[t+1]; action[T] == state[T]",
                }, indent=2,
            ) + "\n", encoding="utf-8",
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
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--max-alignment-delta-sec", type=float, default=DEFAULT_MAX_ALIGNMENT_DELTA_SEC)
    parser.add_argument("--encoder-threads", type=int, default=4)
    parser.add_argument("--task")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(asdict(convert_capture_root(**vars(_parse_args()))), indent=2))


if __name__ == "__main__":
    main()
