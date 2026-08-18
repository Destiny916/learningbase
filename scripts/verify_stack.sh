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

require_file UPSTREAM_SOURCES.md
require_file docs/POPCORN_19D_CONTRACT.md
require_file checkpoint_metadata/act_popcorn_45w/config.json
require_file dexchain/.env.example
require_file dexchain/tests/test_docker_config.sh
require_file algorithms/act/pyproject.toml
require_file algorithms/act/src/lerobot/policies/act/modeling_act.py
require_file algorithms/act_dinov3/src/lerobot/policies/act_dinov3/dinov3_backbone.py
require_file algorithms/act_dinov3/src/lerobot/policies/act_dinov3/modeling_act_dinov3.py
require_file algorithms/turbovla/pyproject.toml
require_file algorithms/turbovla/scripts/joint_songling/train_0812_closed_patchvision_t2_gpu7.sh
require_file algorithms/turbovla_patchvision/pyproject.toml
require_file algorithms/turbovla_patchvision/tests/test_turbovla_patchvision_t2.py
require_file w1_act-ljl-act_train/README.md
require_file w1_act-ljl-act_train/w1_lerobot/src/lerobot/policies/act/kinematics.py

[[ ! -e .gitmodules ]] || fail "top-level .gitmodules still exists"
if git ls-files --stage | grep -q '^160000 '; then
  fail "repository still contains a Git submodule entry"
fi
for source_dir in algorithms/act algorithms/act_dinov3 algorithms/turbovla algorithms/turbovla_patchvision; do
  [[ ! -e "$source_dir/.git" ]] || fail "$source_dir contains nested Git metadata"
done

for revision in \
  8296f39e0db0d3f6a2da4eaf68d51da0505fe4de \
  c26cd5eec99d5978fd9ee374e5950e182586843a \
  7a9b9c948d0de6298133ad5f7df58b6df0b7339f \
  4d7a5935b40ffdb26e1a721f892cd83b215ed43a; do
  grep -Fq "$revision" UPSTREAM_SOURCES.md || fail "missing upstream revision $revision"
done

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
if grep -Eq '(^|/)\.env$' <<<"$candidate_files"; then
  fail "a secret environment file is eligible for commit"
fi
weight_files="$(grep -E '\.(safetensors|pt|pth|ckpt)$' <<<"$candidate_files" || true)"
unexpected_weights="$(grep -Ev '^algorithms/(act|act_dinov3)/tests/artifacts/.*\.safetensors$' <<<"$weight_files" || true)"
if [[ -n "$unexpected_weights" ]]; then
  fail "a model/training weight outside the upstream test fixtures is eligible for commit: $unexpected_weights"
fi

echo "Vendored algorithm sources: PASS"
echo "Popcorn training stack verification: PASS"
