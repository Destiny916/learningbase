#!/usr/bin/env python3
"""Real-robot client for the temporal PatchVision T2 checkpoint."""

from __future__ import annotations

import argparse
import signal
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from dual_turbovla_patchvision_t2_dryrun import (
    STATE_NAMES,
    TurboVLAClient,
    build_camera_producers,
    load_stats,
    temporal_pair_to_model_images,
)
from pi052_reference_client import ACTION_NAMES, DualActHardware
from temporal_image_cache import CameraProducer, TemporalImageCache
from turbovla_endpoint20_protocol import (
    build_model_state,
    q99_normalize,
    unnormalize_and_anchor_action,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-uri", default="ws://127.0.0.1:18067")
    parser.add_argument("--stats-path", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-chunks", type=int, default=0, help="0 runs until interrupted")
    parser.add_argument("--left-can", default="can1")
    parser.add_argument("--right-can", default="can0")
    parser.add_argument("--left-d405-serial", default="412622273326")
    parser.add_argument("--right-d405-serial", default="260622271788")
    parser.add_argument("--top-device", default="/dev/video26")
    parser.add_argument("--top-codec", choices=("raw", "opencv"), default="raw")
    parser.add_argument(
        "--left-pika-port",
        default="/dev/serial/by-path/pci-0000:c4:00.3-usb-0:3.4:1.0-port0",
    )
    parser.add_argument(
        "--right-pika-port",
        default="/dev/serial/by-path/pci-0000:c6:00.4-usb-0:1.4:1.0-port0",
    )
    parser.add_argument("--gripper-max-m", type=float, default=0.10)
    parser.add_argument("--enable-arms", action="store_true")
    parser.add_argument("--enable-grippers", action="store_true")
    parser.add_argument("--execute-robot-actions", action="store_true")
    return parser.parse_args()


def validate_real_args(args: argparse.Namespace) -> None:
    if not (args.enable_arms and args.enable_grippers and args.execute_robot_actions):
        raise SystemExit(
            "PatchVision T2 real client requires --enable-arms --enable-grippers "
            "--execute-robot-actions"
        )
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")


def execute_action_chunk(
    hardware: Any,
    absolute_chunk: np.ndarray,
    *,
    fps: float,
    stop: Any,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    chunk = np.asarray(absolute_chunk, dtype=np.float32)
    if chunk.shape != (50, 14):
        raise ValueError(f"absolute action chunk must have shape [50,14], got {chunk.shape}")
    if fps <= 0:
        raise ValueError("fps must be positive")
    deadline = clock()
    for row in chunk:
        if stop.is_set():
            break
        hardware.send_action(
            {name: float(row[index]) for index, name in enumerate(ACTION_NAMES)}
        )
        deadline += 1.0 / fps
        sleep(max(0.0, deadline - clock()))


def main() -> None:
    args = parse_args()
    validate_real_args(args)
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
            normalized_action = client.infer(
                {"image": temporal_images, "lang": args.task, "state": normalized_state},
                request_id=f"chunk-{chunk_index}",
            )
            absolute_chunk = unnormalize_and_anchor_action(
                normalized_action,
                current,
                action_q01,
                action_q99,
            )
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
                f" right[{absolute_chunk[:,13].min():.5f},{absolute_chunk[:,13].max():.5f}]",
                flush=True,
            )
            execute_action_chunk(hardware, absolute_chunk, fps=args.fps, stop=stop)
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
