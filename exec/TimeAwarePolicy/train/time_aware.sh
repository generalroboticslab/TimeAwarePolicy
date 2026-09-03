#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: time_aware.sh --task TASK --checkpoint TEMPORAL_STUDENT [extra training arguments]"
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
  FrankaCubeStack) updates=1500; horizon=2000 ;;
  FrankaGmPour) updates=2500; horizon=1600 ;;
  FrankaCabinet) updates=1500; horizon=2600 ;;
  *) echo "Unknown task: $task" >&2; exit 2 ;;
esac

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
python -m projects.TimeAwarePolicy.train \
  --saving --task_name "$task" \
  --checkpoint "$checkpoint" --index_episode best \
  --num_updates "$updates" --episodeLength "$horizon" \
  --reset_critic --warmup_iters 50 --no_dense \
  --time2end --time_ratio --ratio_range "[0.2, 1]" \
  --fixed_configs --use_cost --cmdp_method np3o \
  --lr 2e-4 --gamma 1.0 --value_bootstrap false \
  --c_gamma "[1, 0.99]" --c_scale "[0, 1]" \
  --successRewardScale 1000 --epstimeRewardScale "[100, 100]" \
  "$@"
