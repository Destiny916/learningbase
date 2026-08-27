#!/usr/bin/env python
"""Compute Popcorn relative-joint stats for the same-frame action contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from lerobot.datasets.relative_joint_stats import (
    compute_relative_joint_stats_from_episodes,
    save_relative_joint_stats,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=16)
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    state_feature = info["features"]["observation.state"]
    action_feature = info["features"]["action"]
    state_names = list(state_feature["names"])
    action_names = list(action_feature["names"])
    if state_names != action_names or state_feature["shape"] != [19] or action_feature["shape"] != [19]:
        raise ValueError("Popcorn stats require matching 19D state/action features")

    grouped: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    for parquet_path in sorted((root / "data").glob("**/*.parquet")):
        table = pq.read_table(parquet_path, columns=["observation.state", "action", "episode_index", "frame_index"])
        for state, action, episode, frame in zip(
            table["observation.state"].to_pylist(),
            table["action"].to_pylist(),
            table["episode_index"].to_pylist(),
            table["frame_index"].to_pylist(),
            strict=True,
        ):
            state_array = np.asarray(state, dtype=np.float64)
            action_array = np.asarray(action, dtype=np.float64)
            if state_array.shape != (19,) or action_array.shape != (19,):
                raise ValueError(f"unexpected state/action shape in {parquet_path}")
            if not np.all(np.isfinite(state_array)) or not np.all(np.isfinite(action_array)):
                raise ValueError(f"non-finite values in {parquet_path}")
            grouped.setdefault(int(episode), []).append((int(frame), state_array, action_array))

    episodes = []
    action_episodes = []
    for episode_index in sorted(grouped):
        rows = sorted(grouped[episode_index], key=lambda row: row[0])
        frame_indices = [row[0] for row in rows]
        if frame_indices != list(range(len(rows))):
            raise ValueError(f"episode {episode_index} frame indices are not contiguous")
        for row_index in range(len(rows) - 1):
            if not np.allclose(rows[row_index][2], rows[row_index + 1][1], rtol=1e-5, atol=1e-6):
                raise ValueError(f"Popcorn action contract violated in episode {episode_index}: expected action[t] == state[t+1]")
        episodes.append(torch.from_numpy(np.stack([row[1] for row in rows])))
        action_episodes.append(torch.from_numpy(np.stack([row[2] for row in rows])))

    manifest = {
        "format_version": 1,
        "mode": "all_dataset_episodes_same_frame_action",
        "dataset_root": str(root),
        "info_sha256": hashlib.sha256(info_path.read_bytes()).hexdigest(),
        "total_episodes": len(episodes),
        "total_frames": sum(len(episode) for episode in episodes),
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    source_sha = hashlib.sha256(manifest_bytes).hexdigest()
    command = shlex.join([
        sys.executable, "-m", "lerobot.scripts.compute_popcorn_relative_joint_stats",
        f"--dataset-root={root}", f"--output-dir={args.output_dir.resolve()}", f"--horizon={args.horizon}",
    ])
    stats = compute_relative_joint_stats_from_episodes(
        episodes,
        gripper_indices=[17, 18],
        horizons=[args.horizon],
        feature_names=action_names,
        state_feature_names=state_names,
        action_feature_names=action_names,
        state_gripper_indices=[17, 18],
        action_gripper_indices=[17, 18],
        action_state_indices=list(range(19)),
        action_episodes=action_episodes,
        state_absolute_indices=[0, 8, 9, 17, 18],
        action_absolute_indices=[0, 8, 9, 17, 18],
        source_manifest_sha256=source_sha,
        source_dataset_root=str(root),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "full_dataset_manifest.json").write_bytes(manifest_bytes)
    save_relative_joint_stats(stats, args.output_dir, generation_command=command)
    print(f"POPCORN_RELATIVE_STATS_OK episodes={len(episodes)} frames={sum(len(ep) for ep in episodes)} action_contract=action[t]==state[t+1] for non-tail frames source_manifest_sha256={source_sha}")


if __name__ == "__main__":
    main()
