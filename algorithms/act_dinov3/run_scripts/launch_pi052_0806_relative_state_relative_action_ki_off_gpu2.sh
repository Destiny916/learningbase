#!/usr/bin/env bash
set -euo pipefail

export KNOWLEDGE_INSULATION=false
export OUTPUT_DIR="${OUTPUT_DIR:-/data/wengyikun/outputs/pi052_0806_relative_state_relative_action_ki_off_chunk50_b4_400k_gpu2/train_out}"
export JOB_NAME="${JOB_NAME:-pi052_0806_relative_state_relative_action_ki_off_chunk50_b4_400k_gpu2}"

exec bash "$(dirname "$0")/launch_pi052_0806_relative_state_relative_action_gpu0.sh"
