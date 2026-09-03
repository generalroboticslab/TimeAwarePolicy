#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: temporal_bank.sh --checkpoint TEMPORAL_STUDENT [extra evaluation arguments]"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "${1:-}" == "--checkpoint" ]]; then shift; fi
checkpoint="${1:-}"
if [[ -z "$checkpoint" ]]; then
  usage >&2
  exit 2
fi
shift
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
python -m projects.TimeAwarePolicy.eval \
  --saving --graphics_device_id -1 \
  --num_envs 10000 --target_success_eps 10000 \
  --target_record_eps 1000 --save_threshold 10 \
  --record_init_configs --use_par_checkpoint \
  --checkpoint "$checkpoint" --index_episode best \
  "$@"
