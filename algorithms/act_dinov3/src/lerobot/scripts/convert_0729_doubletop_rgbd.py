"""Convert 0729 dual-top RGB and dual-gripper RGB-D captures to LeRobot v3."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.configs import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.convert_dual_arm_three_camera_to_lerobot_v30 import (
    JOINT_NAMES,
    _discover_episodes,
    _load_image,
    _load_joint,
    _load_paths,
    _nearest,
    _next_actions,
    align_episode,
)

TOP_LEFT = Path("camera/color/stereoLeft")
DEPTH = {
    "observation.images.gripper_left_depth": (Path("camera/depth/pikaGripperDepthCamera_l"), 70, 800),
    "observation.images.gripper_right_depth": (Path("camera/depth/pikaGripperDepthCamera_r"), 70, 600),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    args = parser.parse_args()
    build = args.output_root.parent / f".{args.output_root.name}.building"
    if args.output_root.exists() or build.exists():
        raise FileExistsError(args.output_root)
    episodes = _discover_episodes(args.input_root)
    first = align_episode(episodes[0], max_alignment_delta_sec=0.02, resample_to_fps=25)[0]
    left_top, _ = _load_paths(episodes[0] / TOP_LEFT, ".jpg")
    first_depth = np.asarray(Image.open(sorted((episodes[0] / DEPTH["observation.images.gripper_left_depth"][0]).glob("*.png"))[0]), dtype=np.uint16)
    features = {
        "observation.images.top_right": {"dtype": "video", "shape": _load_image(first.stereo_path).shape, "names": None},
        "observation.images.top_left": {"dtype": "video", "shape": _load_image(left_top[0]).shape, "names": None},
        "observation.images.gripper_right": {"dtype": "video", "shape": _load_image(first.gripper_right_path).shape, "names": None},
        "observation.images.gripper_left": {"dtype": "video", "shape": _load_image(first.gripper_left_path).shape, "names": None},
        "observation.state": {"dtype": "float32", "shape": (14,), "names": JOINT_NAMES},
        "action": {"dtype": "float32", "shape": (14,), "names": JOINT_NAMES},
    }
    for key, (_, low, high) in DEPTH.items():
        features[key] = {"dtype": "video", "shape": (1, *first_depth.shape), "names": None, "info": {"is_depth_map": True, "depth_unit": "mm", "depth_min": low / 1000, "depth_max": high / 1000}}
    dataset = LeRobotDataset.create(repo_id=args.repo_id, root=build, fps=25, features=features, use_videos=True, rgb_encoder=RGBEncoderConfig(vcodec="h264", pix_fmt="yuv444p"), streaming_encoding=True, encoder_threads=1, encoder_queue_maxsize=700)
    report = []
    try:
        for episode in episodes:
            frames = align_episode(episode, max_alignment_delta_sec=0.02, resample_to_fps=25)
            top_paths, top_times = _load_paths(episode / TOP_LEFT, ".jpg")
            depth_streams = {key: _load_paths(episode / path, ".png") for key, (path, _, _) in DEPTH.items()}
            samples = []
            for frame in frames:
                anchor = float(frame.stereo_path.stem)
                top_left, top_delta = _nearest(top_paths, top_times, anchor)
                if top_delta > 0.02:
                    continue
                payload = {"observation.images.top_right": _load_image(frame.stereo_path), "observation.images.top_left": _load_image(top_left), "observation.images.gripper_right": _load_image(frame.gripper_right_path), "observation.images.gripper_left": _load_image(frame.gripper_left_path), "task": "0729_doubletop_rgbd"}
                for key, (paths, times) in depth_streams.items():
                    path, delta = _nearest(paths, times, anchor)
                    raw = np.asarray(Image.open(path), dtype=np.uint16)
                    _, low, high = DEPTH[key]
                    value = np.where((raw >= low) & (raw <= high) & (delta <= 0.04), raw, 0).astype(np.uint16)
                    payload[key] = value[None]
                samples.append((np.concatenate((_load_joint(frame.left_joint_path), _load_joint(frame.right_joint_path))).astype(np.float32), payload))
            if samples:
                states = np.asarray([state for state, _ in samples], dtype=np.float32)
                for (state, payload), action in zip(samples, _next_actions(states), strict=True):
                    payload["observation.state"] = state
                    payload["action"] = action
                    dataset.add_frame(payload)
                dataset.save_episode(parallel_encoding=False)
                report.append({"source_episode": episode.name, "frames": len(samples)})
        dataset.finalize()
        (build / "conversion_summary.json").write_text(json.dumps({"episodes": report, "fps": 25, "depth_ranges_mm": {"left": [70, 800], "right": [70, 600]}}, indent=2) + "\n")
        build.rename(args.output_root)
    except BaseException:
        shutil.rmtree(build, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
