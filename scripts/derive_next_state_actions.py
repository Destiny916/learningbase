#!/usr/bin/env python3
"""Create a non-destructive LeRobot copy with action[t] = state[t+1]."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing existing output: {output}")

    parquet_paths = sorted((source / "data").glob("**/*.parquet"))
    if not parquet_paths:
        raise SystemExit("no parquet files found")
    rows: dict[int, list[tuple[int, np.ndarray]]] = {}
    tables: list[tuple[Path, pa.Table]] = []
    for path in parquet_paths:
        table = pq.read_table(path)
        tables.append((path, table))
        states = table["observation.state"].to_pylist()
        episodes = table["episode_index"].to_pylist()
        frames = table["frame_index"].to_pylist()
        for state, episode, frame in zip(states, episodes, frames, strict=True):
            rows.setdefault(int(episode), []).append((int(frame), np.asarray(state, dtype=np.float32)))

    next_state: dict[tuple[int, int], np.ndarray] = {}
    for episode, episode_rows in rows.items():
        episode_rows.sort(key=lambda item: item[0])
        frames = [frame for frame, _ in episode_rows]
        if frames != list(range(len(frames))):
            raise ValueError(f"episode {episode} frame indices are not contiguous")
        for index, (frame, state) in enumerate(episode_rows):
            next_state[(episode, frame)] = episode_rows[min(index + 1, len(episode_rows) - 1)][1]

    output.mkdir(parents=True)
    for relative in ("meta", "videos"):
        source_dir = source / relative
        if source_dir.exists():
            shutil.copytree(source_dir, output / relative)
    for path, table in tables:
        episodes = table["episode_index"].to_pylist()
        frames = table["frame_index"].to_pylist()
        actions = [next_state[(int(ep), int(frame))].tolist() for ep, frame in zip(episodes, frames, strict=True)]
        action_type = table.schema.field("action").type
        replacement = pa.array(actions, type=action_type)
        updated = table.set_column(table.schema.get_field_index("action"), "action", replacement)
        destination = output / "data" / path.relative_to(source / "data")
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(updated, destination, compression="zstd")

    provenance = {
        "source_dataset": str(source),
        "action_contract": "action[t] == observation.state[t+1] for non-tail frames",
        "tail_action_contract": "last frame repeats its own state; future-chunk padding masks it",
        "state_and_images": "copied unchanged",
    }
    (output / "meta" / "derived_action_contract.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(f"DERIVED_DATASET_OK source={source} output={output} episodes={len(rows)} frames={sum(map(len, rows.values()))}")


if __name__ == "__main__":
    main()
