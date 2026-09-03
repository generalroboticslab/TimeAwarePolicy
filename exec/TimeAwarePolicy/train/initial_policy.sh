#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: initial_policy.sh --task TASK [extra training arguments]"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "${1:-}" == "--task" ]]; then
  shift
fi
task="${1:-}"
if [[ -z "$task" ]]; then
  usage >&2
  exit 2
fi
shift
case "$task" in
  FrankaCubeStack) updates=2500; horizon=500 ;;
  FrankaGmPour) updates=6000; horizon=500 ;;
  FrankaCabinet) updates=2500; horizon=800 ;;
  *) echo "Unknown task: $task" >&2; exit 2 ;;
esac

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
python -m projects.TimeAwarePolicy.train \
  --saving --task_name "$task" \
  --num_updates "$updates" --episodeLength "$horizon" \
  --fix_priv --gamma 0.995 --value_bootstrap false \
  --successRewardScale 1000 \
  --quality_candidate_interval 5 \
  --quality_candidate_start_success 0.90 \
  "$@"
