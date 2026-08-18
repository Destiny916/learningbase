#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

docker compose ps --status running --services | rg --quiet '^dexchain$' || {
    printf 'dexchain container is not running\n' >&2
    exit 1
}

docker compose exec -T dexchain nvidia-smi \
    --query-gpu=index,name,memory.total --format=csv,noheader

docker compose exec -T dexchain python3 - <<'PY'
from importlib import metadata

import dexechain
import embodichain

print("python imports: OK")
for package in ("dexechain", "embodichain"):
    try:
        version = metadata.version(package)
    except metadata.PackageNotFoundError:
        version = getattr(globals()[package], "__version__", "unknown")
    print(f"{package}={version}")

try:
    import torch
except ImportError:
    print("torch=not-installed")
else:
    assert torch.cuda.is_available(), "PyTorch cannot access CUDA"
    assert torch.cuda.device_count() == 1, torch.cuda.device_count()
    value = (torch.ones(1, device="cuda") + 1).item()
    assert value == 2.0, value
    print(f"torch={torch.__version__} cuda={torch.cuda.get_device_name(0)} SMOKE_TEST_OK")
PY
