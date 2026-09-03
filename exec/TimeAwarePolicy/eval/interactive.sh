#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: interactive.sh --task TASK [extra evaluation arguments]"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "${1:-}" == "--task" ]]; then shift; fi
task="${1:-}"
if [[ -z "$task" ]]; then
  usage >&2
  exit 2
fi
shift

case "$task" in
  FrankaCubeStack) checkpoint=20250717_162724_tw_FrankaCubeStack ;;
  FrankaGmPour) checkpoint=20250715_123940_tw_FrankaGmPour ;;
  FrankaCabinet) checkpoint=20250730_151924_tw_FrankaCabinet ;;
  *) echo "Unknown task: $task" >&2; exit 2 ;;
esac

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
python -m projects.TimeAwarePolicy.eval \
  --rendering --graphics_device_id 0 --num_envs 1 \
  --checkpoint "$checkpoint" --index_episode best_rew \
  --par_configs_eval true \
  --goal_speed 0.6 --keyboard_ctrl --simple_layout --draw_scevel \
  "$@"
