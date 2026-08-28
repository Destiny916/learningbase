#!/usr/bin/env python3
"""Convert complete W1 Popcorn episodes to a local LeRobot v3.0 dataset.

The source contains four camera streams and a pose JSON.  The output keeps
the right eye plus both wrist cameras and writes absolute 19D state/action
vectors.  Source files are never modified.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


BODY_NAMES = (
    "WAIST",
    "LEFT_J1", "LEFT_J2", "LEFT_J3", "LEFT_J4", "LEFT_J5", "LEFT_J6", "LEFT_J7",
    "NECK1", "NECK2",
    "RIGHT_J1", "RIGHT_J2", "RIGHT_J3", "RIGHT_J4", "RIGHT_J5", "RIGHT_J6", "RIGHT_J7",
)
STATE_NAMES = BODY_NAMES + ("LEFT_GRIPPER", "RIGHT_GRIPPER")
CAMERA_KEYS = {
    "observation.images.cam_high_right": "head_right",
    "observation.images.cam_hand_left": "hand_left",
    "observation.images.cam_hand_right": "hand_right",
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def nearest_index(sorted_ts: np.ndarray, target: float) -> int:
    i = int(np.searchsorted(sorted_ts, target, side="left"))
    if i <= 0:
        return 0
    if i >= len(sorted_ts):
        return len(sorted_ts) - 1
    return i - 1 if target - sorted_ts[i - 1] <= sorted_ts[i] - target else i


def interpolate_pose(frames: list[dict], target_ts: np.ndarray) -> np.ndarray:
    frames = sorted(frames, key=lambda x: float(x["timestamp"]))
    ts = np.asarray([float(x["timestamp"]) for x in frames], dtype=np.float64)
    values = np.asarray(
        [[float(x["data"][name]) for name in STATE_NAMES] for x in frames], dtype=np.float32
    )
    result = np.empty((len(target_ts), len(STATE_NAMES)), dtype=np.float32)
    for dim in range(values.shape[1]):
        result[:, dim] = np.interp(target_ts, ts, values[:, dim]).astype(np.float32)
    return result


def load_image(path: Path, camera_type: str) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if camera_type == "head_right":
            # Preserve the square input contract used by the W1 visual model:
            # center the 1920x1080 eye image in a black 1920x1920 canvas.
            canvas = Image.new("RGB", (1920, 1920), (0, 0, 0))
            canvas.paste(image, (0, (1920 - 1080) // 2))
            image = canvas.resize((224, 224), Image.Resampling.LANCZOS)
        elif camera_type in ("hand_left", "hand_right"):
            # Follow the requested two-stage wrist preprocessing exactly.
            image = image.resize((360, 360), Image.Resampling.LANCZOS)
            image = image.resize((224, 224), Image.Resampling.LANCZOS)
        else:
            raise ValueError(f"unsupported camera type: {camera_type}")
        return np.asarray(image, dtype=np.uint8)


def _load_frame_spec(spec: dict) -> dict:
    item = {"state": spec["state"]}
    for feature_key, (image_path, camera_type) in spec["images"].items():
        item[feature_key] = load_image(image_path, camera_type)
    return item


def iter_episode_frames(episode_dir: Path, workers: int, batch_size: int):
    metadata_path = episode_dir / "metadata.jsonl"
    pose_paths = sorted(episode_dir.glob("pose_record_*.json"))
    if not metadata_path.is_file() or len(pose_paths) != 1:
        raise ValueError(f"incomplete episode: {episode_dir}")

    metadata = load_jsonl(metadata_path)
    streams: dict[str, list[dict]] = {}
    for camera_type in CAMERA_KEYS.values():
        rows = sorted((x for x in metadata if x["camera_type"] == camera_type), key=lambda x: float(x["timestamp"]))
        if not rows:
            raise ValueError(f"missing camera stream {camera_type} in {episode_dir}")
        streams[camera_type] = rows

    anchor = streams["head_right"]
    anchor_ts = np.asarray([float(x["timestamp"]) for x in anchor], dtype=np.float64)
    poses = json.loads(pose_paths[0].read_text(encoding="utf-8"))["frames"]
    state = interpolate_pose(poses, anchor_ts)
    stream_ts = {name: np.asarray([float(x["timestamp"]) for x in rows]) for name, rows in streams.items()}
    specs = []
    for row_idx, anchor_row in enumerate(anchor):
        images = {}
        for feature_key, camera_type in CAMERA_KEYS.items():
            rows = streams[camera_type]
            row = rows[nearest_index(stream_ts[camera_type], anchor_ts[row_idx])]
            image_path = episode_dir / row["image_path"]
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            images[feature_key] = (image_path, camera_type)
        specs.append({"state": state[row_idx], "images": images})

    if workers <= 1:
        for spec in specs:
            yield _load_frame_spec(spec)
        return

    # Submit bounded batches so multiple JPEGs are decoded/resized in parallel
    # without allowing a whole episode of image arrays to accumulate in RAM.
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(specs), batch_size):
            batch = specs[start : start + batch_size]
            yield from executor.map(_load_frame_spec, batch)


def create_dataset(
    output_root: Path, fps: int, image_writer_threads: int,
    codec: str, codec_preset: str, crf: int, resume: bool,
):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.configs.video import RGBEncoderConfig

    features = {
        "observation.state": {"dtype": "float32", "shape": (19,), "names": list(STATE_NAMES)},
        "action": {"dtype": "float32", "shape": (19,), "names": list(STATE_NAMES)},
        "observation.images.cam_high_right": {
            "dtype": "video", "shape": (3, 224, 224), "names": ["channels", "height", "width"]
        },
        "observation.images.cam_hand_left": {
            "dtype": "video", "shape": (3, 224, 224), "names": ["channels", "height", "width"]
        },
        "observation.images.cam_hand_right": {
            "dtype": "video", "shape": (3, 224, 224), "names": ["channels", "height", "width"]
        },
    }
    encoder = RGBEncoderConfig(vcodec=codec, preset=codec_preset, crf=crf)
    if resume:
        return LeRobotDataset.resume(
            repo_id="popcorn_0827_w1_v30",
            root=output_root,
            tolerance_s=0.05,
            image_writer_threads=image_writer_threads,
            image_writer_processes=0,
            batch_encoding_size=1,
            rgb_encoder=encoder,
        )
    return LeRobotDataset.create(
        repo_id="popcorn_0827_w1_v30",
        root=output_root,
        fps=fps,
        robot_type="dexforce_w1",
        features=features,
        use_videos=True,
        tolerance_s=0.05,
        image_writer_threads=image_writer_threads,
        image_writer_processes=0,
        batch_encoding_size=1,
        rgb_encoder=encoder,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("/data/popcorn/0827"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--codec", default="h264", choices=["h264", "libsvtav1"])
    parser.add_argument("--codec-preset", default="fast")
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--preprocess-workers", type=int, default=8)
    parser.add_argument("--preprocess-batch-size", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--task", default="W1 Popcorn teleoperation")
    args = parser.parse_args()

    if args.output_root.exists() and not args.resume:
        raise SystemExit(f"Refusing existing output path: {args.output_root}")
    if args.resume and not args.output_root.exists():
        raise SystemExit(f"Cannot resume missing output path: {args.output_root}")
    episode_dirs = sorted(
        (p for p in args.source_root.glob("episode*") if p.is_dir() and p.name[7:].isdigit()),
        key=lambda p: int(p.name[7:]),
    )
    if args.limit_episodes is not None:
        episode_dirs = episode_dirs[: args.limit_episodes]
    if not episode_dirs:
        raise SystemExit("No episode directories found")

    dataset = create_dataset(
        args.output_root, args.fps, args.image_writer_threads,
        args.codec, args.codec_preset, args.crf, args.resume,
    )
    if args.resume:
        completed = int(dataset.meta.total_episodes)
        if dataset.writer is not None:
            dataset.writer.cleanup_interrupted_episode(completed)
        episode_dirs = episode_dirs[completed:]
        print(f"resuming after {completed} complete episodes", flush=True)
    total = 0
    try:
        for episode_dir in episode_dirs:
            frame_count = 0
            for record in iter_episode_frames(
                episode_dir, args.preprocess_workers, args.preprocess_batch_size
            ):
                frame = {"observation.state": record["state"], "action": record["state"], "task": args.task}
                frame.update({key: record[key] for key in CAMERA_KEYS})
                dataset.add_frame(frame)
                frame_count += 1
            # Keep encoding serial: each episode contains high-resolution 1080p
            # frames and process-parallel encoding can multiply memory usage.
            dataset.save_episode(parallel_encoding=True)
            total += frame_count
            print(f"saved {episode_dir.name}: {frame_count} frames, total={total}", flush=True)
    finally:
        dataset.finalize()
    print(f"conversion complete: episodes={len(episode_dirs)} frames={total} output={args.output_root}")


if __name__ == "__main__":
    main()
