#!/bin/bash
# Explicit candidate launcher only. Does not submit a queue job.
set -euo pipefail
source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
profile="${1:?Choose 00_chunks or 01_imagined}"
shift
case "$profile" in
    00_chunks) world_config="$source_root/examples/jit_03_fused_ema.json" ;;
    01_imagined) world_config="$source_root/examples/trajectory_01_imagined.json" ;;
    *) echo "Unknown trajectory profile: $profile" >&2; exit 2 ;;
esac
: "${ACCERL_EXPERIMENT_ROOT:?Set a new isolated experiment root}"
for argument in "$@"; do
    case "$argument" in
        --world-backend-config|--world-backend-config=*) echo "Select the sampling profile explicitly" >&2; exit 2 ;;
    esac
done
export ACCERL_WORLD_BACKEND_CONFIG="$world_config"
export ACCERL_COST_SCHEDULER_CONFIG="${ACCERL_COST_SCHEDULER_CONFIG:-$source_root/examples/cost_03_adaptive.json}"
export ACCERL_PIPELINE_CONFIG="${ACCERL_PIPELINE_CONFIG:-$source_root/examples/pipeline_lookahead_01_two_updates.json}"
exec bash "$source_root/run.sh" "$@"
