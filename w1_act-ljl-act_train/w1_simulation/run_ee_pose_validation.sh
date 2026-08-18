#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
PROJECT_PYTHONPATH="$WORKSPACE_ROOT/w1_lerobot/src:$WORKSPACE_ROOT"
if [[ -n "${PYTHONPATH:-}" ]]; then
  PROJECT_PYTHONPATH="$PROJECT_PYTHONPATH:$PYTHONPATH"
fi
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-}}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x /home/dex/miniconda3/envs/lerobot/bin/python ]]; then
    PYTHON_BIN=/home/dex/miniconda3/envs/lerobot/bin/python
  else
    PYTHON_BIN=python3
  fi
fi

cd "$WORKSPACE_ROOT"
exec env "PYTHONPATH=$PROJECT_PYTHONPATH" "$PYTHON_BIN" -m w1_simulation.evaluation.ee_pose "$@"
