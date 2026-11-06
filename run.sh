#!/usr/bin/env bash
set -euo pipefail
umask 077
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ -f "$repo/.weaver.env" ]]; then
    set -a
    source "$repo/.weaver.env"
    set +a
fi
case "${1:-start}" in
    setup) exec bash "$repo/weaver_run/setup.sh" ;;
    -h|--help|status|stop) ;;
    *)
        if [[ "$(hostname)" != h20-xyff52ab-0001 && -z "${WEAVER_CONTROL_PYTHON:-}" && ! -f "$repo/.weaver-deps/ready" ]]; then
            bash "$repo/weaver_run/setup.sh"
        fi
        ;;
esac
if [[ -z "${WEAVER_CONTROL_PYTHON:-}" && -x "$repo/.venv-control/bin/python" ]]; then
    WEAVER_CONTROL_PYTHON="$repo/.venv-control/bin/python"
elif [[ -z "${WEAVER_CONTROL_PYTHON:-}" && "$(hostname)" == h20-xyff52ab-0001 ]]; then
    WEAVER_CONTROL_PYTHON=/mnt/data/lcx3/envs/merged-env/bin/python
fi
if [[ -z "${WEAVER_LIBERO_ROOT:-}" && -d "$repo/.weaver-deps/LIBERO" ]]; then
    export WEAVER_LIBERO_ROOT="$repo/.weaver-deps/LIBERO"
fi
exec "${WEAVER_CONTROL_PYTHON:-python3}" -B "$repo/weaver_run/launch.py" "$@"
