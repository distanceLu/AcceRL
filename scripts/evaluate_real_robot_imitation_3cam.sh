#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/../data_collect}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-/mnt/data/lcx2/yanjieworkspace/models/finetune_im/openvla-7b+libero_object_no_noops+b40+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--discrete_acts--proprio_state--100000_chkpt}"
AGENT_CHECKPOINT="${AGENT_CHECKPOINT:-}"
DEVICE="${DEVICE:-cuda:0}"
TASK_LABEL="${TASK_LABEL:-brush the paper surface}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-0}"
PRINT_EVERY="${PRINT_EVERY:-1}"
PRINT_ACTIONS="${PRINT_ACTIONS:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
DEFAULT_PYTHON="/mnt/data/lcx2/yanjieworkspace/clone_env_smoke_test/rlinf_env/bin/python"
if [[ ! -x "${DEFAULT_PYTHON}" ]]; then
  DEFAULT_PYTHON="python"
fi
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"

# The checkpoint may be the first positional argument or AGENT_CHECKPOINT.
if [[ -z "${AGENT_CHECKPOINT}" && $# -gt 0 && "${1}" != --* ]]; then
  AGENT_CHECKPOINT="${1}"
  shift
fi
if [[ -z "${AGENT_CHECKPOINT}" ]]; then
  echo "Usage: AGENT_CHECKPOINT=/path/to/agent_checkpoint_epoch_N $0 [extra evaluator flags]" >&2
  echo "   or: $0 /path/to/agent_checkpoint_epoch_N [extra evaluator flags]" >&2
  exit 2
fi

args=(
  --data-root "${DATA_ROOT}"
  --pretrained-checkpoint "${PRETRAINED_CHECKPOINT}"
  --agent-checkpoint "${AGENT_CHECKPOINT}"
  --device "${DEVICE}"
  --task-label "${TASK_LABEL}"
  --batch-size "${BATCH_SIZE}"
  --dataloader-workers "${DATALOADER_WORKERS}"
  --print-every "${PRINT_EVERY}"
  --max-samples "${MAX_SAMPLES}"
)

if [[ "${PRINT_ACTIONS}" == "1" ]]; then
  args+=(--print-actions)
fi

exec "${PYTHON_BIN}" -m rl.evaluate_real_robot_imitation "${args[@]}" "$@"
