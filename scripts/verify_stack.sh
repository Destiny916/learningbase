#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

fail() {
  echo "Popcorn training stack verification: FAIL: $*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || fail "missing file: $1"
}

require_submodule() {
  local name="$1"
  local expected_branch="$2"
  local expected_url="$3"
  local actual_branch actual_url

  actual_branch="$(git config -f .gitmodules --get "submodule.${name}.branch")"
  actual_url="$(git config -f .gitmodules --get "submodule.${name}.url")"
  [[ "$actual_branch" == "$expected_branch" ]] || fail "$name branch is $actual_branch, expected $expected_branch"
  [[ "$actual_url" == "$expected_url" ]] || fail "$name URL is $actual_url, expected $expected_url"
  [[ -e "$name/.git" ]] || fail "$name is not initialized; run git submodule update --init --recursive"
}

require_file .gitmodules
require_file docs/POPCORN_19D_CONTRACT.md
require_file checkpoint_metadata/act_popcorn_45w/config.json
require_file dexchain/.env.example
require_file dexchain/tests/test_docker_config.sh

lerobot_url="http://192.168.10.28:3000/chengdu/lerobot_joint.git"
turbovla_url="http://192.168.10.28:3000/chengdu/turboVLA.git"
require_submodule algorithms/act main "$lerobot_url"
require_submodule algorithms/act_dinov3 feat/act-dinov3 "$lerobot_url"
require_submodule algorithms/turbovla main "$turbovla_url"
require_submodule algorithms/turbovla_patchvision feature/turbovla-patchvision-t2-act "$turbovla_url"

while IFS= read -r line; do
  case "${line:0:1}" in
    -) fail "uninitialized submodule: ${line:1}" ;;
    +) fail "submodule differs from the pinned commit: ${line:1}" ;;
    U) fail "submodule has a merge conflict: ${line:1}" ;;
  esac
done < <(git submodule status)

python3 - <<'PY'
import json
from pathlib import Path

config = json.loads(Path("checkpoint_metadata/act_popcorn_45w/config.json").read_text())
expected_cameras = {
    "observation.images.cam_high_left",
    "observation.images.cam_hand_left",
    "observation.images.cam_hand_right",
}
inputs = config["input_features"]
assert inputs["observation.state"]["shape"] == [19]
assert expected_cameras.issubset(inputs)
assert config["output_features"]["action"]["shape"] == [19]
assert config["chunk_size"] == 100
assert config["n_action_steps"] == 100
PY

candidate_files="$(git ls-files --cached --others --exclude-standard)"
if grep -Eq '(^|/)\.env$|\.(safetensors|pt|pth|ckpt)$' <<<"$candidate_files"; then
  fail "a secret environment file or model/training weight is eligible for commit"
fi

echo "Pinned algorithm commits:"
git submodule status
echo "Popcorn training stack verification: PASS"
