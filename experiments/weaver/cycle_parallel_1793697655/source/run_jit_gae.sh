#!/bin/bash
# Explicit candidate launcher; this file does not submit a queue job.
set -euo pipefail
source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
profile="${1:?Choose 00_shared_exact, 01_trainer_exact, 02_trainer_ema, 03_fused_ema, or 04_fused_rms}"
shift
case "$profile" in
    00_shared_exact|01_trainer_exact|02_trainer_ema|03_fused_ema|04_fused_rms) ;;
    *) echo "Unknown JIT profile: $profile" >&2; exit 2 ;;
esac
: "${ACCERL_EXPERIMENT_ROOT:?Set a new isolated experiment root}"
for argument in "$@"; do
    case "$argument" in
        --world-backend-config|--world-backend-config=*) echo "Select the JIT profile explicitly" >&2; exit 2 ;;
    esac
done
export ACCERL_WORLD_BACKEND_CONFIG="$source_root/examples/jit_$profile.json"
export ACCERL_COST_SCHEDULER_CONFIG="${ACCERL_COST_SCHEDULER_CONFIG:-$source_root/examples/cost_03_adaptive.json}"
export ACCERL_PIPELINE_CONFIG="${ACCERL_PIPELINE_CONFIG:-$source_root/examples/pipeline_lookahead_01_two_updates.json}"
exec bash "$source_root/run.sh" "$@"
