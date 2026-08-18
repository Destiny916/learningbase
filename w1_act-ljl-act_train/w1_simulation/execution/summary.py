from __future__ import annotations

from pathlib import Path

from w1_simulation.artifacts import write_json


def save_summary(path: Path, summary: dict[str, object]) -> None:
    write_json(path, summary)
