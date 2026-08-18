from __future__ import annotations

import subprocess
from collections.abc import Callable
from typing import Any


def collect_gpu_metrics(run_command: Callable[..., Any]) -> dict[str, float]:
    try:
        completed = run_command(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
        row = completed.stdout.strip().splitlines()[0]
        utilization, memory, temperature = (value.strip() for value in row.split(",")[:3])
        return {
            "gpu/utilization_percent": float(utilization),
            "gpu/memory_used_mb": float(memory),
            "gpu/temperature_c": float(temperature),
        }
    except (IndexError, OSError, ValueError, subprocess.SubprocessError):
        return {}


def gpu_metrics() -> dict[str, float]:
    return collect_gpu_metrics(subprocess.run)
