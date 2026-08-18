"""Create a PI052 subtask/memory annotated copy of the 0806 dual-arm dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.language import LANGUAGE_EVENTS, LANGUAGE_PERSISTENT, language_feature_info
from lerobot.datasets.language_render import active_at, render_sample
from lerobot.policies.pi052.processor_pi052 import _load_recipe

SUBTASK_PICKUP_RIGHT = "Pick up the bread with the right gripper"
SUBTASK_TRANSFER_LEFT_AND_PLACE = "transfer it to the left gripper, and place it in the bowl."
MEMORY_AFTER_RIGHT_PICKUP = "The bread has been picked up with the right gripper."

_GUIDE_LINE = re.compile(r"^- episode(?P<episode>\d+)[:：]\s*第\s*(?P<split>\d+)\s*帧后分开（总帧数\s*(?P<frames>\d+)\s*）$")


@dataclass(frozen=True)
class SplitSpec:
    split_after_frame: int
    total_frames: int


def parse_split_guide(path: Path) -> dict[int, SplitSpec]:
    """Parse the human-reviewed 0806 split guide and reject duplicate episodes."""
    splits: dict[int, SplitSpec] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        match = _GUIDE_LINE.match(raw_line.strip())
        if match is None:
            continue
        episode = int(match["episode"])
        if episode in splits:
            raise ValueError(f"duplicate split guide entry for episode {episode}")
        splits[episode] = SplitSpec(
            split_after_frame=int(match["split"]), total_frames=int(match["frames"])
        )
    if not splits:
        raise ValueError(f"no episode split entries found in {path}")
    return splits


def build_persistent_rows(
    *, frame_indices: list[int], timestamps: list[float], split_after_frame: int
) -> list[dict[str, object]]:
    """Return sparse subtask/memory transitions for one episode.

    The dataset writer broadcasts this three-row persistent list to each frame
    in the episode. ``active_at`` then yields subtask 1 through the reviewed
    split frame and subtask 2 plus the completion memory afterwards.
    """
    if len(frame_indices) != len(timestamps) or not frame_indices:
        raise ValueError("frame_indices and timestamps must be non-empty and have equal length")
    if frame_indices[0] != 0:
        raise ValueError(f"episode must start at frame_index=0, got {frame_indices[0]}")
    if frame_indices != sorted(frame_indices) or len(set(frame_indices)) != len(frame_indices):
        raise ValueError("frame_indices must be strictly increasing")

    next_frame = split_after_frame + 1
    try:
        next_position = frame_indices.index(next_frame)
    except ValueError as exc:
        raise ValueError(
            f"split_after_frame={split_after_frame} requires boundary frame_index={next_frame}"
        ) from exc

    start_timestamp = float(timestamps[0])
    boundary_timestamp = float(timestamps[next_position])
    return [
        _persistent_row(SUBTASK_PICKUP_RIGHT, "subtask", start_timestamp),
        _persistent_row(SUBTASK_TRANSFER_LEFT_AND_PLACE, "subtask", boundary_timestamp),
        _persistent_row(MEMORY_AFTER_RIGHT_PICKUP, "memory", boundary_timestamp),
    ]


def _persistent_row(content: str, style: str, timestamp: float) -> dict[str, object]:
    return {
        "role": "assistant",
        "content": content,
        "style": style,
        "timestamp": timestamp,
        "camera": None,
        "tool_calls": None,
    }


def _copy_without_videos(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    shutil.copytree(source, output, ignore=shutil.ignore_patterns("videos"))
    (output / "videos").symlink_to(source / "videos", target_is_directory=True)


def _rename_episode_metadata_columns(table: pa.Table) -> pa.Table:
    """Canonicalize legacy wrist metadata names to the dataset's gripper keys."""
    renames = {
        "observation.images.wrist_left": "observation.images.gripper_left",
        "observation.images.wrist_right": "observation.images.gripper_right",
    }
    return table.rename_columns(
        [
            next((name.replace(old, new) for old, new in renames.items() if old in name), name)
            for name in table.column_names
        ]
    )


def _canonicalize_episode_metadata(root: Path) -> None:
    for path in sorted((root / "meta/episodes").glob("chunk-*/*.parquet")):
        updated = _rename_episode_metadata_columns(pq.read_table(path))
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        pq.write_table(updated, tmp_path)
        tmp_path.replace(path)


def _add_language_features(root: Path) -> None:
    """Declare the parquet language columns so training selects lerobot_collate_fn."""
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = dict(info.get("features", {}))
    features.update(language_feature_info())
    info["features"] = features
    info_path.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _replace_language_columns(table: pa.Table, persistent: list[list[dict[str, object]]]) -> pa.Table:
    columns = []
    names = []
    for name in table.column_names:
        if name not in {LANGUAGE_PERSISTENT, LANGUAGE_EVENTS}:
            columns.append(table.column(name))
            names.append(name)

    # Every persistent list has the same struct shape. Include an explicit
    # non-empty event struct once so pyarrow never infers list<null> for the
    # otherwise empty ``language_events`` column.
    event_seed = {
        "role": "assistant",
        "content": "",
        "style": "trace",
        "camera": None,
        "tool_calls": None,
    }
    persistent_array = pa.array(persistent)
    event_type = pa.array([[event_seed]]).type
    events_array = pa.array([[]] * table.num_rows, type=event_type)
    return pa.Table.from_arrays(
        [*columns, persistent_array, events_array], names=[*names, LANGUAGE_PERSISTENT, LANGUAGE_EVENTS]
    )


def _annotate_table(table: pa.Table, splits: dict[int, SplitSpec]) -> tuple[pa.Table, dict[int, list[dict[str, object]]]]:
    required = {"episode_index", "frame_index", "timestamp"}
    missing = required - set(table.column_names)
    if missing:
        raise ValueError(f"dataset parquet is missing columns: {sorted(missing)}")

    per_episode: dict[int, list[tuple[int, float, int]]] = {}
    for row, (episode, frame, timestamp) in enumerate(
        zip(
            table["episode_index"].to_pylist(),
            table["frame_index"].to_pylist(),
            table["timestamp"].to_pylist(),
            strict=True,
        )
    ):
        per_episode.setdefault(int(episode), []).append((int(frame), float(timestamp), row))

    rows_by_episode: dict[int, list[dict[str, object]]] = {}
    for episode, values in per_episode.items():
        if episode not in splits:
            raise ValueError(f"no split guide entry for episode {episode}")
        spec = splits[episode]
        frames = [frame for frame, _, _ in values]
        timestamps = [timestamp for _, timestamp, _ in values]
        if len(frames) != spec.total_frames:
            raise ValueError(
                f"episode {episode} has {len(frames)} frames, guide expects {spec.total_frames}"
            )
        rows_by_episode[episode] = build_persistent_rows(
            frame_indices=frames,
            timestamps=timestamps,
            split_after_frame=spec.split_after_frame,
        )

    if set(rows_by_episode) != set(splits):
        missing = sorted(set(splits) - set(rows_by_episode))
        raise ValueError(f"split guide has episodes absent from dataset: {missing}")
    persistent = [rows_by_episode[int(episode)] for episode in table["episode_index"].to_pylist()]
    return _replace_language_columns(table, persistent), rows_by_episode


def _update_relative_manifests(root: Path) -> None:
    info_sha = hashlib.sha256((root / "meta/info.json").read_bytes()).hexdigest()
    for manifest_path in root.glob("normalization_*/**/*manifest.json"):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "source_dataset_root" in payload:
            payload["source_dataset_root"] = str(root.resolve())
            payload["source_manifest_sha256"] = info_sha
            manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate(root: Path, splits: dict[int, SplitSpec]) -> None:
    recipe = _load_recipe("recipes/subtask_mem.yaml")
    expected_episodes = set(splits)
    seen: set[int] = set()
    for path in sorted((root / "data").glob("chunk-*/*.parquet")):
        table = pq.read_table(path)
        if LANGUAGE_PERSISTENT not in table.column_names or LANGUAGE_EVENTS not in table.column_names:
            raise ValueError(f"language columns missing from {path}")
        episodes = table["episode_index"].to_pylist()
        timestamps = table["timestamp"].to_pylist()
        persistent = table[LANGUAGE_PERSISTENT].to_pylist()
        for episode in sorted(set(int(value) for value in episodes)):
            indices = [index for index, value in enumerate(episodes) if int(value) == episode]
            first_rows = persistent[indices[0]]
            if any(persistent[index] != first_rows for index in indices):
                raise ValueError(f"persistent annotations differ between frames in episode {episode}")
            if len(first_rows) != 3:
                raise ValueError(f"episode {episode} has {len(first_rows)} persistent rows, expected 3")
            first_t = float(timestamps[indices[0]])
            boundary_t = float(first_rows[1]["timestamp"])
            if active_at(first_t, persistent=first_rows, style="subtask")["content"] != SUBTASK_PICKUP_RIGHT:
                raise ValueError(f"episode {episode} does not activate pickup subtask at its first frame")
            if active_at(boundary_t, persistent=first_rows, style="subtask")["content"] != SUBTASK_TRANSFER_LEFT_AND_PLACE:
                raise ValueError(f"episode {episode} does not activate transfer subtask at its boundary")
            if active_at(boundary_t, persistent=first_rows, style="memory")["content"] != MEMORY_AFTER_RIGHT_PICKUP:
                raise ValueError(f"episode {episode} does not activate pickup memory at its boundary")
            # A rendered phase-2 sample must have a real subtask-conditioned recipe branch.
            if render_sample(
                recipe=recipe,
                persistent=first_rows,
                events=[],
                t=boundary_t,
                sample_idx=indices[0],
                task="task",
            ) is None:
                raise ValueError(f"episode {episode} does not render a PI052 recipe branch at its boundary")
            seen.add(episode)
    if seen != expected_episodes:
        raise ValueError(f"annotated episode mismatch: seen={sorted(seen)}, expected={sorted(expected_episodes)}")


def annotate(source: Path, output: Path, guide: Path) -> None:
    splits = parse_split_guide(guide)
    _copy_without_videos(source, output)
    try:
        _canonicalize_episode_metadata(output)
        for path in sorted((output / "data").glob("chunk-*/*.parquet")):
            updated, _ = _annotate_table(pq.read_table(path), splits)
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            pq.write_table(updated, tmp_path)
            tmp_path.replace(path)
        _add_language_features(output)
        _update_relative_manifests(output)
        _validate(output, splits)
        summary = {
            "source": str(source.resolve()),
            "guide": str(guide.resolve()),
            "guide_sha256": hashlib.sha256(guide.read_bytes()).hexdigest(),
            "episodes": len(splits),
            "subtask_1": SUBTASK_PICKUP_RIGHT,
            "subtask_2": SUBTASK_TRANSFER_LEFT_AND_PLACE,
            "memory_at_subtask_2": MEMORY_AFTER_RIGHT_PICKUP,
            "videos": "symlinked_to_source_without_frame_modification",
            "state_action_and_normalization": "copied_without_numeric_changes",
        }
        (output / "subtask_memory_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--guide", type=Path, required=True)
    args = parser.parse_args()
    annotate(args.source, args.output, args.guide)


if __name__ == "__main__":
    main()
