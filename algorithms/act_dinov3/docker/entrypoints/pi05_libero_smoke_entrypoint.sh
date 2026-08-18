#!/usr/bin/env bash
set -euo pipefail

mkdir -p "$HOME/.libero"
if [[ ! -f "$HOME/.libero/config.yaml" ]]; then
  cat > "$HOME/.libero/config.yaml" <<EOF
assets: ${LIBERO_ASSETS_DIR}
bddl_files: ${LIBERO_PACKAGE_DIR}/bddl_files
datasets: ${LIBERO_PACKAGE_DIR}/../datasets
init_states: ${LIBERO_PACKAGE_DIR}/init_files
EOF
fi

if [[ -x /opt/nvidia/nvidia_entrypoint.sh ]]; then
  exec /opt/nvidia/nvidia_entrypoint.sh "$@"
fi

exec "$@"
