#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
ASSET_ROOT="${W1_SIMULATION_ASSET_ROOT:-$WORKSPACE_ROOT}"
PROJECT_PYTHONPATH="$WORKSPACE_ROOT/w1_lerobot/src:$WORKSPACE_ROOT"
if [[ -n "${PYTHONPATH:-}" ]]; then
  PROJECT_PYTHONPATH="$PROJECT_PYTHONPATH:$PYTHONPATH"
fi

PROFILE="${W1_SIMULATION_PROFILE:-$PROJECT_ROOT/configs/w1_popcorn_v1.json}"
POLICY_BACKEND="${POLICY_BACKEND:-script}"
QUALITY_POSE="${QUALITY_POSE:-1}"
QUALITY_END_EFFECTOR="${QUALITY_END_EFFECTOR:-1}"
QUALITY_MOTION_DIRECTION="${QUALITY_MOTION_DIRECTION:-1}"
QUALITY_AMPLITUDE="${QUALITY_AMPLITUDE:-1}"
SCORE_SMOOTHNESS="${SCORE_SMOOTHNESS:-1}"
SCORE_REALTIME="${SCORE_REALTIME:-1}"
SAVE_ARTIFACTS="${SAVE_ARTIFACTS:-false}"

START_FRAME="${START_FRAME:-0}"
MAX_FRAMES="${MAX_FRAMES:-0}"
DEVICE="${DEVICE:-cuda:0}"
RERUN_PORT="${RERUN_PORT:-0}"
TENSORBOARD_PORT="${TENSORBOARD_PORT:-0}"
REALTIME="${REALTIME:-1}"
KEEP_OPEN="${KEEP_OPEN:-1}"
STRICT_VERIFICATION="${STRICT_VERIFICATION:-0}"
RUN_NAME="${RUN_NAME:-act_raw}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x /home/dex/miniconda3/envs/lerobot/bin/python ]]; then
    PYTHON_BIN=/home/dex/miniconda3/envs/lerobot/bin/python
  else
    PYTHON_BIN=python3
  fi
fi

resolve_asset_path() {
  if [[ "$1" = /* ]]; then
    printf '%s\n' "$1"
  else
    printf '%s/%s\n' "$ASSET_ROOT" "$1"
  fi
}

resolve_project_path() {
  if [[ "$1" = /* ]]; then
    printf '%s\n' "$1"
  else
    printf '%s/%s\n' "$PROJECT_ROOT" "$1"
  fi
}

ARGS=(
  -m w1_simulation.launch
  --profile "$PROFILE"
  --run-name "$RUN_NAME"
  --start-frame "$START_FRAME"
  --max-frames "$MAX_FRAMES"
  --device "$DEVICE"
  --policy-backend "$POLICY_BACKEND"
  --rerun-port "$RERUN_PORT"
  --tensorboard-port "$TENSORBOARD_PORT"
)

if [[ -n "${CHECKPOINT:-}" ]]; then ARGS+=(--checkpoint "$(resolve_asset_path "$CHECKPOINT")"); fi
if [[ -n "${ORIGIN_ROOT:-}" ]]; then ARGS+=(--origin "$(resolve_asset_path "$ORIGIN_ROOT")"); fi
if [[ -n "${ARTIFACT_ROOT:-}" ]]; then ARGS+=(--artifacts "$(resolve_project_path "$ARTIFACT_ROOT")"); fi
if [[ -n "${POLICY_SCRIPT:-}" ]]; then ARGS+=(--policy-script "$(resolve_project_path "$POLICY_SCRIPT")"); fi
if [[ -n "${CONTROL_MODE:-}" ]]; then ARGS+=(--control-mode "$CONTROL_MODE"); fi
if [[ -n "${RAW_EXECUTION_HORIZON:-}" ]]; then ARGS+=(--execution-horizon "$RAW_EXECUTION_HORIZON"); fi
if [[ -n "${RAW_REPLAN_INTERVAL:-}" ]]; then ARGS+=(--replan-interval "$RAW_REPLAN_INTERVAL"); fi
if [[ -n "${RERUN_VIEW_MODE:-}" ]]; then ARGS+=(--rerun-view-mode "$RERUN_VIEW_MODE"); fi
if [[ -n "${EYE_CAMERA_WIDTH:-}" ]]; then ARGS+=(--eye-camera-width "$EYE_CAMERA_WIDTH"); fi
if [[ -n "${EYE_CAMERA_HEIGHT:-}" ]]; then ARGS+=(--eye-camera-height "$EYE_CAMERA_HEIGHT"); fi
if [[ -n "${EYE_CAMERA_FPS:-}" ]]; then ARGS+=(--eye-camera-fps "$EYE_CAMERA_FPS"); fi
if [[ -n "${EYE_CAMERA_FOVY:-}" ]]; then ARGS+=(--eye-camera-fovy "$EYE_CAMERA_FOVY"); fi
if [[ -n "${EYE_CAMERA_SCENE:-}" ]]; then ARGS+=(--eye-camera-scene "$EYE_CAMERA_SCENE"); fi

append_quality_metric() {
  case "$1" in
    1) ARGS+=(--quality-metric "$2") ;;
    0) ;;
    *) printf 'Quality metric switch must be 0 or 1: %s=%s\n' "$2" "$1"; exit 2 ;;
  esac
}

append_quality_metric "$QUALITY_POSE" pose
append_quality_metric "$QUALITY_END_EFFECTOR" end_effector
append_quality_metric "$QUALITY_MOTION_DIRECTION" motion_direction
append_quality_metric "$QUALITY_AMPLITUDE" amplitude

append_boolean_switch() {
  case "$1" in
    1) ARGS+=("--$2") ;;
    0) ARGS+=("--no-$2") ;;
    *) printf 'Score switch must be 0 or 1: %s=%s\n' "$2" "$1"; exit 2 ;;
  esac
}

append_boolean_switch "$SCORE_SMOOTHNESS" score-smoothness
append_boolean_switch "$SCORE_REALTIME" score-realtime

case "$SAVE_ARTIFACTS" in
  true) ARGS+=(--save-artifacts) ;;
  false) ARGS+=(--no-save-artifacts) ;;
  *) printf 'SAVE_ARTIFACTS must be true or false: %s\n' "$SAVE_ARTIFACTS"; exit 2 ;;
esac

if [[ "$REALTIME" == "1" ]]; then
  ARGS+=(--realtime)
else
  ARGS+=(--no-realtime)
fi
if [[ "$KEEP_OPEN" == "1" ]]; then
  ARGS+=(--keep-open)
else
  ARGS+=(--no-keep-open)
fi
if [[ "$STRICT_VERIFICATION" == "1" ]]; then
  ARGS+=(--strict-verification)
else
  ARGS+=(--no-strict-verification)
fi

ARGS+=("$@")
ARGS+=(--action-pipeline raw)

cd "$WORKSPACE_ROOT"
exec env "PYTHONPATH=$PROJECT_PYTHONPATH" "$PYTHON_BIN" "${ARGS[@]}"
