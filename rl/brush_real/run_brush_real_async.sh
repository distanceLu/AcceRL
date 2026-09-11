#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/mnt/data/lcx3/envs/merged-env/bin/python}"
CTRL_WORLD_ROOT="${CTRL_WORLD_ROOT:-/mnt/data/lcx3/Ctrl-World}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-6,7,5}"
cd "${WORKSPACE_ROOT}"

export TMPDIR="${TMPDIR:-/dev/shm}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
export PYTHONPATH="${CTRL_WORLD_ROOT}:${WORKSPACE_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"

args=(
  rl/brush_real/ds_wm_discrete_brush_real.py
  --cuda-visible-devices "${GPU_LIST}"
  --use-bf16
  --real-data-root /mnt/data/lcx/data/brush/data_collect
  --openvla-checkpoint /mnt/data/lcx3/AcceRL/runs/brush_openvla_discrete/openvla-7b+brush_realworld+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--discrete_acts--3_cams--proprio--10hz--26000_chkpt
  --ctrl-world-root "${CTRL_WORLD_ROOT}"
  --world-checkpoint /mnt/data/lcx3/Ctrl-World/model_ckpt/brush_pen_3cam_delta_5hz_20260901/exp4_bs32_4gpu_lr1e5_warmup150_cosine_fixedval_continue_seed20260902_20260909_161127
  --wm-dataset-root /mnt/data/lcx3/Ctrl-World/output_brush_pen_3cam_delta_5hz_20260901/dataset
  --wm-meta-root /mnt/data/lcx3/Ctrl-World/output_brush_pen_3cam_delta_5hz_20260901/meta
  --wm-dataset-name brush_pen_3cam_delta_5hz_20260901
  --svd-model-path /mnt/data/lcx/models/stabilityai/stable-video-diffusion-img2vid
  --clip-model-path /mnt/data/lcx/models/openai/clip-vit-base-patch32
  --num-trainer-gpus 1
  --num-inference-actors 1
  --num-ctrl-inference-actors 1
  --num-rollout-workers 10
  --num-eval-workers 10
  --train-iters 30000
  --train-batch-size 8
  --accumulation-steps 72
  --inference-batch 8
  --inference-timeout-ms 300
  --imagine-horizon 32
  --num-inference-steps 10
  --gamma 0.99
  --lambda 0.95
  --clip-eps 0.2
  --vf-coef 0.5
  --ent-coef 0.0
  --kl-coef 0.1
  --clip-mode gipo
  --value-lr 1e-4
  --policy-lr 1e-5
  --value-warmup-steps 500
  --policy-warmup-steps 500
  --policy-train-start-step 0
  --reward-scale 1.0
  --smoke-reward-value 1.0
  --replay-capacity 10000
  --ckpt-dir /mnt/data/lcx3/AcceRL/runs/brush_real_async/checkpoints
  --ckpt-every-steps 500
  --exp-name brush_real_frozen_ctrl_world_smoke_reward
)

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  args+=(--preflight-only)
fi

exec "${PYTHON_BIN}" "${args[@]}" "$@"
