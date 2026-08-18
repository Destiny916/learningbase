#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="$(pwd)"
OUTPUT_DIR="${WORKSPACE_DIR}/release"

echo "Running w1_act build preparation..."

mkdir -p "${OUTPUT_DIR}"

if [ ! -f "${WORKSPACE_DIR}/packaging/w1-act.default" ]; then
    echo "Error: packaging/w1-act.default not found!"
    exit 1
fi

echo "No compile step is required for w1_act; source files will be packaged by install.sh."
