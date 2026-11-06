#!/bin/bash
# Candidate launcher only. No queue submission, GPU diagnostic or production edit.
set -euo pipefail
source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
profile="${1:?Choose 00_disabled, 01_fixed, 02_measured, or 03_adaptive}"
shift
case "$profile" in
    00_disabled|01_fixed|02_measured|03_adaptive) ;;
    *) echo "Unknown cost scheduler profile: $profile" >&2; exit 2 ;;
esac
: "${ACCERL_EXPERIMENT_ROOT:?Set a new isolated experiment root}"
if [[ -e "$ACCERL_EXPERIMENT_ROOT/formal_v3/runnerstatus.json" ]]; then
    echo "Cost scheduler candidates require a new experiment root" >&2
    exit 2
fi
export ACCERL_COST_SCHEDULER_CONFIG="$source_root/examples/cost_$profile.json"
exec bash "$source_root/run_fresh.sh" 04_real "$@"
