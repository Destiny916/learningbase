#!/usr/bin/env python3
"""Repair stale wrist video metadata keys in the 0806swap LeRobot v3 dataset."""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq


RENAMES = {
    "wrist_left": "gripper_left",
    "wrist_right": "gripper_right",
}


def repair(dataset_root: Path) -> None:
    episode_path = dataset_root / "meta/episodes/chunk-000/file-000.parquet"
    table = pq.read_table(episode_path)
    names = set(table.column_names)
    replacements = {}
    for old_camera, new_camera in RENAMES.items():
        for suffix in ("chunk_index", "file_index", "from_timestamp", "to_timestamp"):
            old_name = f"videos/observation.images.{old_camera}/{suffix}"
            new_name = f"videos/observation.images.{new_camera}/{suffix}"
            if new_name in names:
                continue
            if old_name not in names:
                raise ValueError(f"missing expected stale metadata column: {old_name}")
            replacements[old_name] = new_name

    if replacements:
        renamed = table.rename_columns([replacements.get(name, name) for name in table.column_names])
        backup = episode_path.with_suffix(episode_path.suffix + f".before_gripper_key_repair_{datetime.now():%Y%m%d_%H%M%S}")
        shutil.copy2(episode_path, backup)
        pq.write_table(renamed, episode_path)
        print(f"repaired {episode_path}")
        print(f"backup {backup}")

    stats_path = dataset_root / "meta/stats.json"
    stats = json.loads(stats_path.read_text())
    stats_changed = False
    for old_camera, new_camera in RENAMES.items():
        old_name = f"observation.images.{old_camera}"
        new_name = f"observation.images.{new_camera}"
        if new_name in stats:
            continue
        if old_name not in stats:
            raise ValueError(f"missing expected stale stats key: {old_name}")
        stats[new_name] = stats.pop(old_name)
        stats_changed = True
    if stats_changed:
        backup = stats_path.with_suffix(stats_path.suffix + f".before_gripper_key_repair_{datetime.now():%Y%m%d_%H%M%S}")
        shutil.copy2(stats_path, backup)
        stats_path.write_text(json.dumps(stats, indent=2) + "\n")
        print(f"repaired {stats_path}")
        print(f"backup {backup}")

    info_path = dataset_root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info_changed = False
    for camera in ("top", "gripper_left", "gripper_right"):
        feature = info["features"][f"observation.images.{camera}"]
        if "names" not in feature:
            feature["names"] = None
            info_changed = True
    if info_changed:
        backup = info_path.with_suffix(info_path.suffix + f".before_gripper_key_repair_{datetime.now():%Y%m%d_%H%M%S}")
        shutil.copy2(info_path, backup)
        info_path.write_text(json.dumps(info, indent=2) + "\n")
        print(f"repaired {info_path}")
        print(f"backup {backup}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    repair(parser.parse_args().dataset_root)
