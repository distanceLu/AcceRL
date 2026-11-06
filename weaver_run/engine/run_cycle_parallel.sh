#!/bin/bash
# Isolated candidate entry. No queue submission or changes to an existing run.
set -euo pipefail
source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
profile="${1:?Choose 00_fixed_serial, 01_feedback_serial, 02_parallel_fixed or 03_feedback_parallel}"
shift
case "$profile" in
    00_parent|00_fixed_serial|01_feedback_serial|02_parallel_fixed|03_feedback_parallel) ;;
    *) echo "Unknown cycle scheduling profile: $profile" >&2; exit 2 ;;
esac
: "${ACCERL_EXPERIMENT_ROOT:?Set a new isolated experiment root}"
for argument in "$@"; do
    case "$argument" in
        --world-backend-config|--world-backend-config=*|--cost-scheduler-config|--cost-scheduler-config=*|--pipeline-config|--pipeline-config=*)
            echo "Use the selected candidate's explicit configuration" >&2; exit 2 ;;
    esac
done
export ACCERL_COST_SCHEDULER_CONFIG="$source_root/examples/cycle_$profile.json"
export ACCERL_PIPELINE_CONFIG="$source_root/examples/pipeline_lookahead_01_two_updates.json"
exec bash "$source_root/run_trajectory.sh" 01_imagined "$@"
