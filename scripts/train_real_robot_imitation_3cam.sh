#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data_collect}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-/mnt/data/lcx2/yanjieworkspace/models/finetune_im/openvla-7b+libero_object_no_noops+b40+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--discrete_acts--proprio_state--100000_chkpt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/runs/real_robot_imitation}"
DEVICE="${DEVICE:-cuda:1}"
TASK_LABEL="${TASK_LABEL:-brush the paper surface}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
MAX_STEPS="${MAX_STEPS:-10000}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-2}"
LORA_RANK="${LORA_RANK:-32}"
USE_TENSORBOARD="${USE_TENSORBOARD:-1}"
DEFAULT_PYTHON="/mnt/data/lcx2/yanjieworkspace/clone_env_smoke_test/rlinf_env/bin/python"
if [[ ! -x "${DEFAULT_PYTHON}" ]]; then
  DEFAULT_PYTHON="python"
fi
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"

args=(
  --data-root "${DATA_ROOT}"
  --pretrained-checkpoint "${PRETRAINED_CHECKPOINT}"
  --output-root "${OUTPUT_ROOT}"
  --device "${DEVICE}"
  --task-label "${TASK_LABEL}"
  --num-actions-chunk 8
  --batch-size "${BATCH_SIZE}"
  --grad-accumulation-steps "${GRAD_ACCUMULATION_STEPS}"
  --learning-rate "${LEARNING_RATE}"
  --max-steps "${MAX_STEPS}"
  --save-freq "${SAVE_FREQ}"
  --dataloader-workers "${DATALOADER_WORKERS}"
  --lora-rank "${LORA_RANK}"
)

if [[ "${USE_TENSORBOARD}" == "1" ]]; then
  args+=(--use-tensorboard)
fi

# Additional CLI flags can be passed directly, for example:
#   bash scripts/train_real_robot_imitation_3cam.sh --prepare-only --run-name alignment_check
exec "${PYTHON_BIN}" -m rl.train_real_robot_imitation "${args[@]}" "$@"
