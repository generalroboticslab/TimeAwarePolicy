#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: initializer_quality.sh --task TASK --producer FOLDER --output-bank FOLDER [extra arguments]"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "${1:-}" == "--task" ]]; then shift; fi
task="${1:-}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ "${1:-}" == "--producer" ]]; then shift; fi
producer="${1:-}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ "${1:-}" == "--output-bank" ]]; then shift; fi
output_bank="${1:-}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ -z "$task" || -z "$producer" || -z "$output_bank" ]]; then
  usage >&2
  exit 2
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
python -m projects.TimeAwarePolicy.initializer_quality.select_checkpoints \
  --task_name "$task" \
  --producer "$producer" \
  --output_bank "$output_bank" \
  --num_envs 2000 \
  "$@"
