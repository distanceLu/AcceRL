#!/bin/bash
# Explicit profile selection. This file is a candidate launcher, not a submission.
set -euo pipefail
source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
profile="${1:?Choose 00_baseline, 01_dense_signal, 02_replay, 03_seeds, 04_real, or an ablation}"
shift
case "$profile" in
    00_baseline|01_dense_signal|02_replay|03_seeds|04_real|ablate_replay_only|ablate_seeds_only|ablate_real_only) ;;
    *) echo "Unknown FRESH profile: $profile" >&2; exit 2 ;;
esac
: "${ACCERL_EXPERIMENT_ROOT:?Set a new isolated experiment root for this profile}"
export ACCERL_WORLD_BACKEND_CONFIG="$source_root/examples/fresh_$profile.json"
export ACCERL_FRESH_PROFILE="$profile"
exec bash "$source_root/run.sh" "$@"
