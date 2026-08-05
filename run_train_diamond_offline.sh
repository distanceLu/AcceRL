#!/bin/bash
# Offline DIAMOND (denoiser-only) training on episode .pt dataset.
# Extracted from run_oft_diamond.sh / rl/ds_wm_discrete_diffusion.py denoiser path.
#
# Data: tests_dsj/dataset_episode/*.pt
# Entry: tests_dsj/train_diamond_offline.py

set -euo pipefail
cd /mnt/data/lcx3/AcceRL

export TMPDIR=/dev/shm
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export NUMBA_CACHE_DIR=/dev/shm/numba_cache
export MPLCONFIGDIR=/dev/shm/mpl
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

/mnt/data/lcx3/envs/merged-env/bin/python tests_dsj/train_diamond_offline.py \
    --dataset /mnt/data/lcx3/AcceRL/tests_dsj/dataset_episode \
    --agent-config envs/config/agent.yaml \
    --trainer-config envs/config/trainer.yaml \
    --denoiser-checkpoint /mnt/data/lcx2/yanjieworkspace/openvla_oft_rl/runs/wm_reward_denoiser_distill_named/denoiser_smoke_test/denoiser_smoke_test.pt \
    --framework-checkpoint /mnt/data/lcx3/checkpoint/dsj/openvla-7b+libero_spatial_no_noops+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--discrete_acts--proprio_state--100000_chkpt \
    --normalization-key libero_spatial_no_noops \
    --image-size 224 \
    --num-step-cond 4 \
    --device cuda:0 \
    --seed 0 \
    --train-iters 10000 \
    --batch-size 8 \
    --grad-accum 128 \
    --lr 1e-4 \
    --weight-decay 1e-2 \
    --warmup-steps 500 \
    --max-grad-norm 1.0 \
    --num-workers 4 \
    --val-ratio 0.1 \
    --eval-every 200 \
    --ckpt-every 1000 \
    --log-every 10 \
    --output-dir /mnt/data/lcx3/AcceRL/runs/diamond_offline \
    --exp-name diamond_offline_dataset_episode
