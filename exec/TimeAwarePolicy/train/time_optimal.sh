#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: time_optimal.sh --task TASK --checkpoint CHECKPOINT --index INDEX [extra training arguments]"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "${1:-}" == "--task" ]]; then shift; fi
task="${1:-}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ "${1:-}" == "--checkpoint" ]]; then shift; fi
checkpoint="${1:-}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ "${1:-}" == "--index" ]]; then shift; fi
index="${1:-}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ -z "$task" || -z "$checkpoint" || -z "$index" ]]; then
  usage >&2
  exit 2
fi
case "$task" in
  FrankaCubeStack) updates=1500; horizon=500 ;;
  FrankaGmPour) updates=2500; horizon=500 ;;
  FrankaCabinet) updates=1500; horizon=800 ;;
  *) echo "Unknown task: $task" >&2; exit 2 ;;
esac

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
python -m projects.TimeAwarePolicy.train \
  --saving --task_name "$task" \
  --checkpoint "$checkpoint" --index_episode "$index" \
  --num_updates "$updates" --episodeLength "$horizon" \
  --fix_priv --reset_critic --warmup_iters 50 --no_dense \
  --gamma 0.995 --value_bootstrap false --target_kl 2.5 \
  --successRewardScale 1000 --epstimeRewardScale "[100, 100]" \
  "$@"
