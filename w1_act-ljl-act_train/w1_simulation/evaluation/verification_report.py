from __future__ import annotations

from pathlib import Path

from w1_simulation.artifacts import write_json


def save_verification_report(path: Path, report: dict[str, object]) -> None:
    write_json(path, report)
