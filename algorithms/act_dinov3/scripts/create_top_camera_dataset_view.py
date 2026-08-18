#!/usr/bin/env python
"""Create a LeRobot v3 dataset view that exposes only the top camera.

The view symlinks immutable frames and videos from its source, while writing
its own metadata so policies infer a single visual input.
"""

import argparse
import json
import shutil
from pathlib import Path


TOP_CAMERA = "observation.images.top"


def link(source: Path, destination: Path) -> None:
    destination.symlink_to(source, target_is_directory=source.is_dir())


def create_view(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    if not (source / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"not a LeRobot v3 dataset: {source}")

    source_meta = source / "meta"
    destination_meta = destination / "meta"
    destination_meta.mkdir(parents=True)

    with (source_meta / "info.json").open() as file:
        info = json.load(file)
    retained = {
        key
        for key in info["features"]
        if not key.startswith("observation.images.") or key == TOP_CAMERA
    }
    if TOP_CAMERA not in retained:
        raise ValueError(f"source dataset has no {TOP_CAMERA} feature")
    info["features"] = {key: value for key, value in info["features"].items() if key in retained}
    with (destination_meta / "info.json").open("w") as file:
        json.dump(info, file, indent=2)
        file.write("\n")

    with (source_meta / "stats.json").open() as file:
        stats = json.load(file)
    stats = {key: value for key, value in stats.items() if key in retained}
    with (destination_meta / "stats.json").open("w") as file:
        json.dump(stats, file, indent=2)
        file.write("\n")

    shutil.copy2(source_meta / "tasks.parquet", destination_meta / "tasks.parquet")
    link(source_meta / "episodes", destination_meta / "episodes")
    link(source / "data", destination / "data")
    (destination / "videos").mkdir()
    link(source / "videos" / TOP_CAMERA, destination / "videos" / TOP_CAMERA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    create_view(args.source.resolve(), args.destination)


if __name__ == "__main__":
    main()
