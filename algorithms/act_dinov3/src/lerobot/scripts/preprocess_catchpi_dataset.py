#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import logging
import shutil
from copy import deepcopy
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq

from lerobot.configs import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

TOP_CAMERA_KEY = "observation.images.top"
LEFT_CAMERA_KEY = "observation.images.left_wrist"
RIGHT_CAMERA_KEY = "observation.images.right_wrist"
AUTO_FEATURE_KEYS = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
DEFAULT_REPO_ID = "local/0704_bread_grasp_only_songling_robot"

logger = logging.getLogger(__name__)


def trimmed_episode_length(length: int) -> int:
    """Return the number of frames retained by an exact floor(2N/3) trim."""
    if length <= 0:
        raise ValueError(f"Episode length must be positive, got {length}")
    return length * 2 // 3


def make_output_features(source_features: dict) -> dict:
    """Build the two-camera feature schema with explicit gripper names."""
    output = {
        key: deepcopy(feature)
        for key, feature in source_features.items()
        if key != TOP_CAMERA_KEY and key not in AUTO_FEATURE_KEYS
    }

    required = {"observation.state", "action", LEFT_CAMERA_KEY, RIGHT_CAMERA_KEY}
    missing = required - output.keys()
    if missing:
        raise ValueError(f"Source dataset is missing required features: {sorted(missing)}")

    for key in ("observation.state", "action"):
        feature = output[key]
        if tuple(feature.get("shape", ())) != (14,):
            raise ValueError(f"{key} must have shape [14], got {feature.get('shape')}")
        names = feature.get("names")
        if not isinstance(names, list) or len(names) != 14:
            raise ValueError(f"{key} must define 14 dimension names")
        names[6] = "left_gripper"
        names[13] = "right_gripper"

    return output


def make_terminal_action(state: np.ndarray, source_action: np.ndarray, *, is_terminal: bool) -> np.ndarray:
    """Copy the source action, or use the retained state at a new episode boundary."""
    return np.array(state if is_terminal else source_action, dtype=np.float32, copy=True)


def trim_alignment_frames(frames: list[dict], keep_length: int) -> list[dict]:
    """Trim one alignment manifest and remove all top-camera references."""
    if not 0 < keep_length <= len(frames):
        raise ValueError(f"Invalid retained length {keep_length} for {len(frames)} manifest frames")

    output = deepcopy(frames[:keep_length])
    for frame in output:
        frame.get("source_timestamps", {}).pop(TOP_CAMERA_KEY, None)
        frame.get("images", {}).pop(TOP_CAMERA_KEY, None)
    output[-1]["action"] = deepcopy(output[-1]["state"])
    return output


def _load_json(path: Path) -> dict | list:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)
        file.write("\n")


def _read_source(source: Path) -> tuple[dict, list[dict], np.ndarray, np.ndarray]:
    required_paths = [
        source / "meta/info.json",
        source / "meta/episodes/chunk-000/file-000.parquet",
        source / "data/chunk-000/file-000.parquet",
        source / "videos" / LEFT_CAMERA_KEY / "chunk-000/file-000.mp4",
        source / "videos" / RIGHT_CAMERA_KEY / "chunk-000/file-000.mp4",
    ]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Source dataset is incomplete; missing: {missing}")

    info = _load_json(required_paths[0])
    episodes = pq.read_table(required_paths[1]).to_pylist()
    data = pq.read_table(required_paths[2], columns=["observation.state", "action"])
    states = np.asarray(data["observation.state"].to_pylist(), dtype=np.float32)
    actions = np.asarray(data["action"].to_pylist(), dtype=np.float32)

    if len(states) != info["total_frames"] or len(actions) != info["total_frames"]:
        raise ValueError("Source parquet row count does not match meta/info.json")
    if len(episodes) != info["total_episodes"]:
        raise ValueError("Source episode count does not match meta/info.json")
    return info, episodes, states, actions


def _decode_rgb_frames(video_path: Path):
    with av.open(str(video_path)) as container:
        for frame in container.decode(video=0):
            yield frame.to_ndarray(format="rgb24")


def _next_video_frame(iterator, camera_key: str, source_index: int) -> np.ndarray:
    try:
        return next(iterator)
    except StopIteration as error:
        raise ValueError(f"{camera_key} video ended before source frame {source_index}") from error


def _assert_video_exhausted(iterator, camera_key: str, expected_frames: int) -> None:
    try:
        next(iterator)
    except StopIteration:
        return
    raise ValueError(f"{camera_key} video contains more than {expected_frames} frames")


def _rewrite_auxiliary_metadata(
    source: Path,
    build_root: Path,
    final_output: Path,
    source_info: dict,
    episodes: list[dict],
    keep_lengths: list[int],
) -> None:
    source_summary_path = source / "catchpi_conversion_summary.json"
    source_summary = _load_json(source_summary_path) if source_summary_path.is_file() else None
    output_manifest_dir = build_root / "catchpi_alignment_manifests"
    output_manifest_dir.mkdir(parents=True, exist_ok=True)

    per_episode = []
    for episode_index, (episode, keep_length) in enumerate(zip(episodes, keep_lengths, strict=True)):
        if source_summary is not None:
            source_episode = source_summary["episodes"][episode_index]
            manifest_name = Path(source_episode["alignment_manifest"]).name
        else:
            source_episode = None
            manifest_name = f"episode{episode_index}_alignment_manifest.json"

        source_manifest_path = source / "catchpi_alignment_manifests" / manifest_name
        if source_manifest_path.is_file():
            source_frames = _load_json(source_manifest_path)
            output_frames = trim_alignment_frames(source_frames, keep_length)
            _write_json(output_manifest_dir / manifest_name, output_frames)
        else:
            output_frames = None

        item = {
            "episode_index": episode_index,
            "old_length": int(episode["length"]),
            "new_length": keep_length,
        }
        if source_episode is not None:
            updated_episode = deepcopy(source_episode)
            updated_episode["frames"] = keep_length
            updated_episode["alignment_manifest"] = str(
                final_output / "catchpi_alignment_manifests" / manifest_name
            )
            if output_frames:
                updated_episode["start_timestamp"] = output_frames[0]["timestamp"]
                updated_episode["end_timestamp"] = output_frames[-1]["timestamp"]
                updated_episode["duration_seconds"] = (
                    updated_episode["end_timestamp"] - updated_episode["start_timestamp"]
                )
            source_summary["episodes"][episode_index] = updated_episode
        per_episode.append(item)

    new_total_frames = sum(keep_lengths)
    preprocessing = {
        "source": str(source),
        "output": str(final_output),
        "mode": "per_episode_floor_first_2of3_frames",
        "removed_features": [TOP_CAMERA_KEY],
        "renamed_dimensions": {
            "left_joint_6": "left_gripper",
            "right_joint_6": "right_gripper",
        },
        "terminal_action": "action[last] = observation.state[last]",
        "fps": source_info["fps"],
        "old_total_frames": source_info["total_frames"],
        "new_total_frames": new_total_frames,
        "total_episodes": source_info["total_episodes"],
        "per_episode": per_episode,
    }
    _write_json(build_root / "preprocessing_summary.json", preprocessing)

    if source_summary is not None:
        source_summary["repo_id"] = DEFAULT_REPO_ID
        source_summary["output_dir"] = str(final_output)
        source_summary["preprocessing"] = preprocessing
        _write_json(build_root / "catchpi_conversion_summary.json", source_summary)


def convert_dataset(source: Path, output: Path, repo_id: str = DEFAULT_REPO_ID) -> None:
    """Create the derived LeRobot v3 dataset without mutating the source."""
    source = source.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if source == output:
        raise ValueError("Source and output paths must differ")

    build_root = output.parent / f".{output.name}.building"
    if build_root.exists():
        raise FileExistsError(f"Stale build directory exists: {build_root}")

    source_info, episodes, states, actions = _read_source(source)
    features = make_output_features(source_info["features"])
    rgb_encoder = RGBEncoderConfig.from_video_info(features[LEFT_CAMERA_KEY].get("info"))
    keep_lengths = [trimmed_episode_length(int(episode["length"])) for episode in episodes]
    expected_total = sum(keep_lengths)
    max_keep_length = max(keep_lengths)

    dataset = None
    try:
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=int(source_info["fps"]),
            features=features,
            root=build_root,
            robot_type=source_info.get("robot_type"),
            use_videos=True,
            rgb_encoder=rgb_encoder,
            streaming_encoding=True,
            encoder_queue_maxsize=max_keep_length + 8,
            encoder_threads=2,
            data_files_size_in_mb=int(source_info.get("data_files_size_in_mb", 100)),
            video_files_size_in_mb=int(source_info.get("video_files_size_in_mb", 200)),
        )

        left_frames = _decode_rgb_frames(source / "videos" / LEFT_CAMERA_KEY / "chunk-000/file-000.mp4")
        right_frames = _decode_rgb_frames(source / "videos" / RIGHT_CAMERA_KEY / "chunk-000/file-000.mp4")
        source_index = 0
        written_frames = 0

        for episode_index, (episode, keep_length) in enumerate(zip(episodes, keep_lengths, strict=True)):
            source_length = int(episode["length"])
            tasks = episode.get("tasks") or []
            if len(tasks) != 1:
                raise ValueError(f"Episode {episode_index} must contain exactly one task, got {tasks}")

            for local_index in range(source_length):
                left_image = _next_video_frame(left_frames, LEFT_CAMERA_KEY, source_index)
                right_image = _next_video_frame(right_frames, RIGHT_CAMERA_KEY, source_index)
                if local_index < keep_length:
                    state = states[source_index]
                    action = make_terminal_action(
                        state,
                        actions[source_index],
                        is_terminal=local_index == keep_length - 1,
                    )
                    dataset.add_frame(
                        {
                            "observation.state": state.copy(),
                            "action": action,
                            LEFT_CAMERA_KEY: left_image,
                            RIGHT_CAMERA_KEY: right_image,
                            "task": tasks[0],
                        }
                    )
                    written_frames += 1
                source_index += 1

            dataset.save_episode(parallel_encoding=False)
            dropped = dataset.writer._streaming_encoder._dropped_frames
            if any(dropped.values()):
                raise RuntimeError(f"Streaming encoder dropped frames in episode {episode_index}: {dropped}")
            logger.info(
                "episode %d/%d: kept %d of %d frames",
                episode_index + 1,
                len(episodes),
                keep_length,
                source_length,
            )

        if source_index != source_info["total_frames"]:
            raise ValueError(f"Consumed {source_index} source rows, expected {source_info['total_frames']}")
        if written_frames != expected_total:
            raise ValueError(f"Wrote {written_frames} frames, expected {expected_total}")
        _assert_video_exhausted(left_frames, LEFT_CAMERA_KEY, source_index)
        _assert_video_exhausted(right_frames, RIGHT_CAMERA_KEY, source_index)

        dataset.finalize()
        _rewrite_auxiliary_metadata(source, build_root, output, source_info, episodes, keep_lengths)
        build_root.rename(output)
    except BaseException:
        if dataset is not None:
            try:
                if dataset.has_pending_frames():
                    dataset.clear_episode_buffer()
                dataset.finalize()
            except Exception:
                logger.exception("Failed to close partial dataset cleanly")
        shutil.rmtree(build_root, ignore_errors=True)
        raise

    logger.info("Created %s with %d episodes and %d frames", output, len(episodes), expected_total)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    convert_dataset(args.source, args.output, args.repo_id)


if __name__ == "__main__":
    main()
