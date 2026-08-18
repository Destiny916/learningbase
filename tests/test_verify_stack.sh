#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
verifier="$repo_root/scripts/verify_stack.sh"

if [[ ! -x "$verifier" ]]; then
  echo "expected executable verifier at scripts/verify_stack.sh" >&2
  exit 1
fi

output="$(bash "$verifier")"
grep -Fq "Popcorn training stack verification: PASS" <<<"$output"
printf '%s\n' "$output"
