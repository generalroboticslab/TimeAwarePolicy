#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: temporal_student.sh --task TASK --checkpoint TIME_OPTIMAL_CHECKPOINT [extra training arguments]"
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
if [[ -z "$task" || -z "$checkpoint" ]]; then
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
  --checkpoint "$checkpoint" --index_episode best_rew \
  --num_updates "$updates" --episodeLength "$horizon" \
  --stu_train --warmup_rand --time2end --time_ratio \
  --lr 5e-4 --gamma 0.995 --value_bootstrap false --wandb false \
  "$@"
