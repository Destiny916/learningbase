#!/usr/bin/env python3
"""Compute independent 19D state/action q01-q99 for W1 JEPA-WAM training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from prismatic.vla.datasets.lerobot_w1 import (
    W1_ACTION_HORIZON,
    build_action_chunk,
    relative_action_representation,
    relative_state_representation,
    validate_w1_info,
)


def _read_episodes(root: Path) -> dict[int, dict[str, np.ndarray]]:
    grouped: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    for path in sorted((root / "data").glob("**/*.parquet")):
        table = pq.read_table(path, columns=["observation.state", "action", "episode_index", "frame_index"])
        for state, action, episode, frame in zip(
            table["observation.state"].to_pylist(),
            table["action"].to_pylist(),
            table["episode_index"].to_pylist(),
            table["frame_index"].to_pylist(),
            strict=True,
        ):
            grouped.setdefault(int(episode), []).append(
                (int(frame), np.asarray(state, dtype=np.float32), np.asarray(action, dtype=np.float32))
            )
    result = {}
    for episode, rows in grouped.items():
        rows.sort(key=lambda row: row[0])
        if [row[0] for row in rows] != list(range(len(rows))):
            raise ValueError(f"episode {episode} frame indices are not contiguous")
        result[episode] = {
            "state": np.stack([row[1] for row in rows]),
            "action": np.stack([row[2] for row in rows]),
        }
    return result


def compute(root: Path, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    episodes = _read_episodes(root)
    state_values, action_values = [], []
    for rows in episodes.values():
        state_values.append(relative_state_representation(rows["state"]))
        for start in range(rows["action"].shape[0]):
            chunk, valid = build_action_chunk(rows["action"], start, horizon)
            action_values.append(relative_action_representation(chunk, rows["state"][start])[valid])
    if not state_values or not action_values:
        raise ValueError(f"No training rows found under {root}")
    state = np.concatenate(state_values)
    action = np.concatenate(action_values)
    return np.quantile(state, [0.01, 0.99], axis=0), np.quantile(action, [0.01, 0.99], axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--horizon", type=int, default=W1_ACTION_HORIZON)
    args = parser.parse_args()
    root = args.dataset_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing existing output: {output}")
    with (root / "meta" / "info.json").open(encoding="utf-8") as handle:
        info = json.load(handle)
    validate_w1_info(info)
    state_q, action_q = compute(root, args.horizon)
    output.mkdir(parents=True)
    for name, quantiles in (("state", state_q), ("action", action_q)):
        (output / f"{name}_q01_q99.json").write_text(
            json.dumps({"q01": quantiles[0].tolist(), "q99": quantiles[1].tolist()}, indent=2) + "\n",
            encoding="utf-8",
        )
    manifest = {
        "format_version": 1,
        "source_dataset_root": str(root),
        "state_dim": 19,
        "action_dim": 19,
        "action_horizon": args.horizon,
        "relative_joint_indices": [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16],
        "absolute_indices": [0, 8, 9, 17, 18],
        "state_q01_q99_file": "state_q01_q99.json",
        "action_q01_q99_file": "action_q01_q99.json",
        "statistics_are_independent": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"W1_Q01_Q99_OK source={root} output={output} horizon={args.horizon}")


if __name__ == "__main__":
    main()
