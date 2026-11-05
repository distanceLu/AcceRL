#!/bin/bash
# Isolated candidate launch only; this script does not submit a queue job.
set -euo pipefail
source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
profile="${1:?Choose reactive_fixed, lookahead_fixed, reactive_adaptive, or lookahead_adaptive}"
shift
case "$profile" in
    reactive_fixed) inventory=00_off; cost=01_fixed ;;
    lookahead_fixed) inventory=01_two_updates; cost=01_fixed ;;
    reactive_adaptive) inventory=00_off; cost=03_adaptive ;;
    lookahead_adaptive) inventory=01_two_updates; cost=03_adaptive ;;
    *) echo "Unknown inventory profile: $profile" >&2; exit 2 ;;
esac
export ACCERL_PIPELINE_CONFIG="$source_root/examples/pipeline_lookahead_$inventory.json"
exec bash "$source_root/run_cost_scheduled.sh" "$cost" "$@"
