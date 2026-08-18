"""Convert 0806 dual-arm RGB captures to LeRobot v3 with 20D state and 14D action."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.configs import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Detected from 0806 stereoRight timestamps: 1 / median_dt ~= 25.636 FPS.
FPS = 26
MAX_DELTA = 0.02
TASK = "Pick up the bread with the right gripper, transfer it to the left gripper, and place it in the bowl."
CAMERAS = {
    "observation.images.top": Path("camera/color/stereoRight"),
    "observation.images.gripper_left": Path("camera/color/pikaGripperDepthCamera_l"),
    "observation.images.gripper_right": Path("camera/color/pikaGripperDepthCamera_r"),
}
JOINTS = {"left": Path("arm/jointState/puppetLeft"), "right": Path("arm/jointState/puppetRight")}
ENDPOINTS = {"left": Path("arm/endPose/puppetLeft"), "right": Path("arm/endPose/puppetRight")}
STATE_NAMES = (
    [f"left_joint_{i}" for i in range(6)]
    + ["left_endpoint_x", "left_endpoint_y", "left_endpoint_z", "left_gripper"]
    + [f"right_joint_{i}" for i in range(6)]
    + ["right_endpoint_x", "right_endpoint_y", "right_endpoint_z", "right_gripper"]
)
ACTION_NAMES = [f"left_joint_{i}" for i in range(6)] + ["left_gripper"] + [f"right_joint_{i}" for i in range(6)] + ["right_gripper"]


def paths(directory: Path, suffix: str):
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    result = sorted((p for p in directory.glob(f"*{suffix}") if _timestamp(p) is not None), key=_timestamp)
    if not result:
        raise ValueError(f"No timestamped {suffix}: {directory}")
    return result, np.array([_timestamp(p) for p in result], dtype=np.float64)


def _timestamp(path: Path):
    try:
        return float(path.stem)
    except ValueError:
        return None


def nearest(items, times, target):
    i = int(np.searchsorted(times, target))
    choices = [max(0, i - 1), min(len(items) - 1, i)]
    j = min(choices, key=lambda k: abs(float(times[k]) - target))
    return items[j], abs(float(times[j]) - target)


def json_value(path: Path, key: str):
    return json.loads(path.read_text(encoding="utf-8")).get(key)


def joint(path: Path) -> np.ndarray:
    value = np.asarray(json_value(path, "position"), dtype=np.float32)
    if value.shape != (7,) or not np.isfinite(value).all():
        raise ValueError(f"Invalid 7D joint: {path}")
    return value


def endpoint(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = np.asarray([payload.get("x"), payload.get("y"), payload.get("z")], dtype=np.float32)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f"Invalid endpoint xyz: {path}")
    return value


def image(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def episodes(root: Path):
    return sorted((p for p in root.iterdir() if p.is_dir() and p.name.startswith("episode")), key=lambda p: int(p.name[7:]))


def aligned_frames(ep: Path):
    top, top_t = paths(ep / CAMERAS["observation.images.top"], ".jpg")
    left_j, left_jt = paths(ep / JOINTS["left"], ".json")
    right_j, right_jt = paths(ep / JOINTS["right"], ".json")
    left_e, left_et = paths(ep / ENDPOINTS["left"], ".json")
    right_e, right_et = paths(ep / ENDPOINTS["right"], ".json")
    wrist_l, wrist_lt = paths(ep / CAMERAS["observation.images.wrist_left"], ".jpg")
    wrist_r, wrist_rt = paths(ep / CAMERAS["observation.images.wrist_right"], ".jpg")
    grid = np.arange(top_t[0], top_t[-1] + 0.5 / FPS, 1 / FPS)
    selected = []
    for target in grid:
        top_path, _ = nearest(top, top_t, target)
        if selected and top_path == selected[-1][0]:
            continue
        t = _timestamp(top_path)
        values = [nearest(items, times, t) for items, times in ((left_j, left_jt), (right_j, right_jt), (left_e, left_et), (right_e, right_et), (wrist_l, wrist_lt), (wrist_r, wrist_rt))]
        if max(delta for _, delta in values) <= MAX_DELTA:
            selected.append((top_path, values))
    if not selected:
        raise ValueError(f"No aligned frames: {ep}")
    return selected, len(top)


def convert(input_root: Path, output_root: Path):
    if output_root.exists():
        raise FileExistsError(output_root)
    build = output_root.parent / f".{output_root.name}.building"
    if build.exists():
        raise FileExistsError(build)
    valid = []
    skipped = []
    for ep in episodes(input_root):
        try:
            valid.append((ep, *aligned_frames(ep)))
        except (FileNotFoundError, ValueError) as exc:
            skipped.append({"episode": ep.name, "reason": str(exc)})
            print(f"skip {ep.name}: {exc}")
    if not valid:
        raise ValueError("No complete 0806 episodes")
    first = image(valid[0][1][0][0])
    first_frame, first_values = valid[0][1][0]
    first_images = {
        "observation.images.top": image(first_frame),
        "observation.images.gripper_left": image(first_values[4][0]),
        "observation.images.gripper_right": image(first_values[5][0]),
    }
    features = {key: {"dtype": "video", "shape": value.shape, "names": None} for key, value in first_images.items()}
    features.update({"observation.state": {"dtype": "float32", "shape": (20,), "names": STATE_NAMES}, "action": {"dtype": "float32", "shape": (14,), "names": ACTION_NAMES}})
    ds = LeRobotDataset.create(repo_id="local/0806swap", root=build, fps=FPS, features=features, use_videos=True, rgb_encoder=RGBEncoderConfig(vcodec="h264", pix_fmt="yuv444p"), streaming_encoding=True, encoder_threads=4, image_writer_threads=8)
    try:
        for ep, frames, raw_count in valid:
            joint_values = []
            for _, values in frames:
                lj, rj, le, re, _, _ = [item[0] for item in values]
                lv, rv = joint(lj), joint(rj)
                joint_values.append(np.concatenate((lv[:6], lv[6:7], rv[:6], rv[6:7])))
            actions = np.concatenate((np.asarray(joint_values[1:]), np.asarray(joint_values[-1:]))).astype(np.float32)
            states = []
            for _, values in frames:
                lj, rj, le, re, _, _ = [item[0] for item in values]
                lv, rv = joint(lj), joint(rj)
                left_xyz = endpoint(le)
                left_xyz[1] += 0.49
                states.append(np.concatenate((lv[:6], left_xyz, lv[6:7], rv[:6], endpoint(re), rv[6:7])).astype(np.float32))
            for (top_path, values), state, action in zip(frames, states, actions, strict=True):
                ds.add_frame({
                    "observation.state": state,
                    "action": action,
                    "observation.images.top": image(top_path),
                    "observation.images.gripper_left": image(values[4][0]),
                    "observation.images.gripper_right": image(values[5][0]),
                    "task": TASK,
                })
            ds.save_episode(parallel_encoding=False)
        ds.finalize()
        (build / "conversion_summary.json").write_text(json.dumps({"skipped": skipped, "fps": FPS, "state_dim": 20, "action_dim": 14}, indent=2) + "\n")
        build.rename(output_root)
    except BaseException:
        try: ds.finalize()
        except BaseException: pass
        shutil.rmtree(build, ignore_errors=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    convert(args.input_root, args.output_root)
