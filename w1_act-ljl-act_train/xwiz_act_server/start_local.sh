#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
w1_root="$(cd "${script_dir}/.." && pwd)"
runtime_deps="${XWIZ_ACT_RUNTIME_DEPS:-/home/wengyikun/.local/share/popcorn-xwiz-act/runtime-deps}"
policy_path="${XWIZ_ACT_POLICY_PATH:-/home/wengyikun/workplace/popcorn/act_popcorn_45w}"
python_bin="${XWIZ_ACT_PYTHON:-/home/wengyikun/workplace/joint_songling/lerobot/.venv/bin/python}"

export PYTHONPATH="${runtime_deps}:${w1_root}/w1_lerobot/src:${w1_root}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${python_bin}" -m xwiz_act_server.server \
  --host "${XWIZ_ACT_HOST:-0.0.0.0}" \
  --port "${XWIZ_ACT_PORT:-8889}" \
  --policy-path "${policy_path}" \
  --device "${XWIZ_ACT_DEVICE:-cuda}"
