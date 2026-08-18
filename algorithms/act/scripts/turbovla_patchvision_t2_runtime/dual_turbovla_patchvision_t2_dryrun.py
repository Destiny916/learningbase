#!/usr/bin/env python3
"""Read-only dry-run client for the temporal PatchVision T2 checkpoint."""

from __future__ import annotations

import argparse
import signal
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np

from pi052_reference_client import DualActHardware
from temporal_image_cache import CameraProducer, TemporalImageCache, TemporalPair
from turbovla_patchvision_t2_image_layout import preprocess_patchvision_t2_views
from turbovla_endpoint20_protocol import (
    build_model_state,
    q99_normalize,
    unnormalize_and_anchor_action,
)

sys.path.insert(0, str(Path(__file__).parent))
import msgpack_numpy


STATE_NAMES = (
    "left_joint_0", "left_joint_1", "left_joint_2", "left_joint_3", "left_joint_4", "left_joint_5",
    "left_endpoint_x", "left_endpoint_y", "left_endpoint_z", "left_gripper",
    "right_joint_0", "right_joint_1", "right_joint_2", "right_joint_3", "right_joint_4", "right_joint_5",
    "right_endpoint_x", "right_endpoint_y", "right_endpoint_z", "right_gripper",
)


def load_stats(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import json

    with open(path) as stream:
        stats = json.load(stream)
    if "new_embodiment" in stats:
        stats = stats["new_embodiment"]
    state = stats["state"]
    action = stats["action"]
    return tuple(
        np.asarray(values, dtype=np.float32)
        for values in (state["q01"], state["q99"], action["q01"], action["q99"])
    )


def validate_dry_run_args(args: argparse.Namespace) -> None:
    if args.enable_arms or args.enable_grippers or args.execute_robot_actions:
        raise SystemExit("PatchVision T2 dry-run forbids all hardware enable/action flags")


def temporal_pair_to_model_images(pair: TemporalPair) -> list[list[np.ndarray]]:
    return [
        preprocess_patchvision_t2_views([frame.image for frame in sample.frames])
        for sample in (pair.previous, pair.current)
    ]


def build_camera_producers(
    hardware: DualActHardware,
    cache: TemporalImageCache,
) -> list[CameraProducer]:
    top_sequence = -1

    def read_top() -> tuple[float, np.ndarray]:
        nonlocal top_sequence
        top_sequence, timestamp, image = hardware.top_camera.read_right_after(
            top_sequence,
            timeout_s=1.0,
        )
        return timestamp, image

    return [
        CameraProducer("top", read_top, cache),
        CameraProducer("gripper_left", hardware.left_camera.read, cache),
        CameraProducer("gripper_right", hardware.right_camera.read, cache),
    ]


class TurboVLAClient:
    def __init__(self, uri: str) -> None:
        import websockets.sync.client

        self._packer = msgpack_numpy.Packer()
        self._connection = websockets.sync.client.connect(
            uri, compression=None, max_size=None, open_timeout=30, ping_interval=None
        )
        metadata = self._connection.recv()
        if isinstance(metadata, str):
            raise RuntimeError(metadata)
        print("server_metadata", msgpack_numpy.unpackb(metadata))

    def infer(self, example: dict[str, Any], request_id: str) -> np.ndarray:
        message = {
            "type": "infer",
            "request_id": request_id,
            "payload": {"examples": [example], "profile_latency": True},
        }
        self._connection.send(self._packer.pack(message))
        response = self._connection.recv()
        if isinstance(response, str):
            raise RuntimeError(response)
        decoded = msgpack_numpy.unpackb(response)
        if not decoded.get("ok"):
            raise RuntimeError(f"TurboVLA inference error: {decoded}")
        actions = np.asarray(decoded["data"]["normalized_actions"], dtype=np.float32)
        if actions.shape != (1, 50, 14):
            raise ValueError(f"Expected [1,50,14] normalized action, got {actions.shape}")
        latency = decoded["data"].get("latency_ms", {})
        print("server_latency_ms", latency)
        return actions[0]

    def close(self) -> None:
        self._connection.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-uri", default="ws://127.0.0.1:18061")
    parser.add_argument("--stats-path", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-chunks", type=int, default=1)
    parser.add_argument("--left-can", default="left_piper")
    parser.add_argument("--right-can", default="right_piper")
    parser.add_argument("--left-d405-serial", default="412622273326")
    parser.add_argument("--right-d405-serial", default="260622271788")
    parser.add_argument("--top-device", default="/dev/video26")
    parser.add_argument("--top-codec", choices=("raw", "opencv"), default="raw")
    parser.add_argument("--left-pika-port", default="/dev/serial/by-path/pci-0000:c4:00.3-usb-0:3.4:1.0-port0")
    parser.add_argument("--right-pika-port", default="/dev/serial/by-path/pci-0000:c6:00.4-usb-0:1.4:1.0-port0")
    parser.add_argument("--gripper-max-m", type=float, default=0.10)
    parser.add_argument("--enable-arms", action="store_true")
    parser.add_argument("--enable-grippers", action="store_true")
    parser.add_argument("--execute-robot-actions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_dry_run_args(args)
    state_q01, state_q99, action_q01, action_q99 = load_stats(args.stats_path)
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    hardware = DualActHardware(args)
    client: TurboVLAClient | None = None
    producers: list[CameraProducer] = []
    try:
        hardware.connect()
        cache = TemporalImageCache(
            max_camera_skew_s=0.05,
            min_pair_interval_s=0.015,
            max_pair_interval_s=0.060,
        )
        producers = build_camera_producers(hardware, cache)
        for producer in producers:
            producer.start()
        client = TurboVLAClient(args.server_uri)
        chunk_index = 0
        while not stop.is_set() and (args.max_chunks == 0 or chunk_index < args.max_chunks):
            observation = hardware.get_state_observation()
            previous = observation.pop("__relative_joint_previous_state")
            current = np.asarray([observation[name] for name in STATE_NAMES], dtype=np.float32)
            relative_state = build_model_state(previous, current)
            normalized_state = q99_normalize(relative_state, state_q01, state_q99)
            pair = cache.latest_pair(timeout_s=3.0)
            temporal_images = temporal_pair_to_model_images(pair)
            example = {"image": temporal_images, "lang": args.task, "state": normalized_state}
            normalized_action = client.infer(example, request_id=f"chunk-{chunk_index}")
            absolute_chunk = unnormalize_and_anchor_action(normalized_action, current, action_q01, action_q99)
            print(
                f"chunk={chunk_index} feedback_seq={observation['__relative_joint_previous_sequence']}"
                f"->{observation['__relative_joint_current_sequence']}"
                f" image_seq={pair.previous.sequence}->{pair.current.sequence}"
                f" image_dt_ms={pair.interval_s * 1000.0:.3f}"
                f" camera_skew_ms={pair.previous.max_camera_skew_s * 1000.0:.3f}"
                f"->{pair.current.max_camera_skew_s * 1000.0:.3f}"
                f" temporal_images={[[list(image.shape) for image in step] for step in temporal_images]}"
                f" state_joint_delta_max={float(np.abs(relative_state[[*range(6), *range(10,16)]]).max()):.6f}"
                f" gripper_m=left[{absolute_chunk[:,6].min():.5f},{absolute_chunk[:,6].max():.5f}]"
                f" right[{absolute_chunk[:,13].min():.5f},{absolute_chunk[:,13].max():.5f}]"
            )
            print("DRY_RUN: action chunk was not sent to hardware")
            chunk_index += 1
    finally:
        for producer in producers:
            producer.stop()
        for producer in producers:
            producer.join(timeout_s=2.0)
        if client is not None:
            client.close()
        hardware.disconnect()


if __name__ == "__main__":
    main()
