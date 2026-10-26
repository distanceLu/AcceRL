#!/bin/bash
# OpenVLA RL + trainable Ctrl-World (train + broadcast), mirroring Diamond train_wm.
# Physical GPUs 4-7: Policy/World ZeRO-2 and Reward FSDP2, independent world4 groups
# (Policy, Ctrl-World, Reward), plus one inference actor per model per card.
# Each card therefore hosts six model processes, using the existing MPS server.
# The three four-rank training groups run concurrently; each owner prepares
# GPU batches only while consuming them and prefetches CPU data in the background.
# Global update batches stay Policy=576, World=1024, Reward=1024.
# Does NOT modify the frozen Ctrl-World script tests_dsj/ds_wm_discrete_ctrl.py

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export TMPDIR=/dev/shm
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
export RAY_DEDUP_LOGS=0
set -euo pipefail
FSDP_SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ACCERL_REPO_ROOT="${ACCERL_REPO_ROOT:-/mnt/data/lcx1/dong/AcceRL-scheduler}"
export PYTHONPATH="$FSDP_SOURCE:$ACCERL_REPO_ROOT:$ACCERL_REPO_ROOT/rl/ctrl_world:${PYTHONPATH:-}"
cd "$ACCERL_REPO_ROOT"
export ACCERL_POLICY_BACKEND=zero2
export ACCERL_REWARD_BACKEND=fsdp2
export ACCERL_REWARD_FEATURES_ONLY=1
export ACCERL_CPU_REPLAY_PREFETCH=1
export ACCERL_CPU_THREADS="${ACCERL_CPU_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$ACCERL_CPU_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$ACCERL_CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$ACCERL_CPU_THREADS}"
export ACCERL_RESOURCE_GOVERNOR=0
export ACCERL_SPLIT_TRAIN_SERIAL=0
export ACCERL_NSIGHT_NVTX=0
export ACCERL_NSIGHT_STEPS=
export ACCERL_TRIM_IDLE_CACHE="${ACCERL_TRIM_IDLE_CACHE:-1}"
export ACCERL_TRIM_MICRO_CACHE="${ACCERL_TRIM_MICRO_CACHE:-1}"
export ACCERL_SNAPSHOT_INTERVAL="${ACCERL_SNAPSHOT_INTERVAL:-25}"
export ACCERL_CAPTURE_ONLY="${ACCERL_CAPTURE_ONLY:-1}"
export ACCERL_TENSORBOARD="${ACCERL_TENSORBOARD:-1}"
export ACCERL_EVAL_TARGET="${ACCERL_EVAL_TARGET:-0.85}"
export ACCERL_EVAL_CONSECUTIVE="${ACCERL_EVAL_CONSECUTIVE:-2}"
export ACCERL_EVAL_MIN_EPISODES="${ACCERL_EVAL_MIN_EPISODES:-1000}"
export ACCERL_USE_MPS="${ACCERL_USE_MPS:-1}"
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/dev/shm/mps_lcx1_gpu47_pzero2_20261020/pipe}"
if [[ "$ACCERL_USE_MPS" != 1 || "$CUDA_MPS_PIPE_DIRECTORY" != /* || ! -d "$CUDA_MPS_PIPE_DIRECTORY" ]]; then
    echo "Select an existing MPS pipe directory before launching split training." >&2
    exit 1
fi
# CUDA_MPS_LOG_DIRECTORY, when explicitly supplied, is inherited unchanged.
# This launcher never starts, stops, or changes the MPS server.
export ACCERL_RUN_ID="${ACCERL_RUN_ID:-mps_split_$(date +%Y%m%d_%H%M%S_%N)_$$}"
[[ "$ACCERL_RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Invalid ACCERL_RUN_ID." >&2; exit 1; }
export ACCERL_DATA_DIR="${ACCERL_DATA_DIR:-/mnt/data/lcx1/dong/need_data/mps_split_train/$ACCERL_RUN_ID}"
export ACCERL_RAY_TMP="${ACCERL_RAY_TMP:-/dev/shm/aspt_${UID}_$$_${RANDOM}}"
export ACCERL_RAY_SPILL="${ACCERL_RAY_SPILL:-$ACCERL_DATA_DIR/ray_spill}"
export ACCERL_TELEMETRY_DB="${ACCERL_TELEMETRY_DB:-$ACCERL_DATA_DIR/collection.sqlite}"
export ACCERL_FINAL_CHECKPOINT="${ACCERL_FINAL_CHECKPOINT:-$ACCERL_DATA_DIR/checkpoints}"
export ACCERL_STOP_FILE="${ACCERL_STOP_FILE:-$ACCERL_DATA_DIR/stop.flag}"
# Guard only this invocation; no global Ray/process termination.
exec 9>"$PWD/.four_rank_fsdp_policy_train.lock"
flock -n 9 || { echo "This training launcher is already running." >&2; exit 1; }
# Require a fresh output directory. Existing training artifacts are not resumed
# or overwritten, including the archived 85% baseline and adaptive experiment.
mkdir -p -- "$(dirname "$ACCERL_DATA_DIR")"
mkdir -- "$ACCERL_DATA_DIR" || { echo "ACCERL_DATA_DIR must be a new directory." >&2; exit 1; }
for output in "$ACCERL_TELEMETRY_DB" "$ACCERL_FINAL_CHECKPOINT" "$ACCERL_STOP_FILE" "$ACCERL_RAY_SPILL"; do
    case "$(realpath -m -- "$output")" in
        "$(realpath -- "$ACCERL_DATA_DIR")"/*) ;;
        *) echo "Run outputs must remain inside ACCERL_DATA_DIR: $output" >&2; exit 1 ;;
    esac
done
echo "Policy/World ZeRO-2 + Reward FSDP2 + existing MPS: run=$ACCERL_RUN_ID serial=$ACCERL_SPLIT_TRAIN_SERIAL data=$ACCERL_DATA_DIR pipe=$CUDA_MPS_PIPE_DIRECTORY"

/mnt/data/lcx3/envs/merged-env/bin/python "$FSDP_SOURCE/train.py" \
    --cuda-visible-devices 4,5,6,7 \
    --gpu-layout four_rank_split_train \
    --use-bf16 \
    --benchmark libero_spatial \
    --num-images-in-input 2 \
    --pretrained-checkpoint /mnt/data/lcx3/checkpoint/dsj/openvla-7b+libero_spatial_no_noops+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--discrete_acts--proprio_state--100000_chkpt \
    --checkpoint2 /mnt/data/lcx3/checkpoint/dsj/20251225_113851_distill_checkpoint_latest.pt \
    --agent-config-path envs/config/agent.yaml \
    --trainer-config-path envs/config/trainer.yaml \
    --svd-model-path /mnt/data/lcx3/checkpoint/ctrl_world/svd/svd_model \
    --clip-model-path /mnt/data/lcx3/checkpoint/ctrl_world/clip/clip_model \
    --ctrl-world-ckpt /mnt/data/lcx3/Ctrl-World/model_ckpt/libero_spatial/2026-08-07T10-39-47_libero_spatial_newdecoder_fromlastbest/best_val_loss.pt \
    --vae-decoder-checkpoint /mnt/data/lcx3/Ctrl-World/model_ckpt/vae_decoder_libero/deterministic_full_stage2/best_decoder.pt \
    --condition-stat-path /mnt/data/lcx3/Ctrl-World/model_ckpt/libero_spatial/2026-07-21T16-40-56_libero_vla_delta_finetune/condition_stat.json \
    --num-cams 2 \
    --num-history 6 \
    --num-step-cond 7 \
    --num-frames-pred 5 \
    --num-inference-steps 10 \
    --num-trainer-gpus 4 \
    --num-inference-actors 4 \
    --num-rollout-workers 10 \
    --num-eval-workers 10 \
    --num-reward-inference-actors 4 \
    --num-ctrl-inference-actors 4 \
    --train-iters 30000 \
    --train-batch-size 8 \
    --accumulation-steps 18 \
    --inference-batch 8 \
    --inference-timeout-ms 300 \
    --gamma 0.99 \
    --lambda 0.95 \
    --clip-eps 0.2 \
    --vf-coef 0.5 \
    --ent-coef 0.00 \
    --kl-coef 0.1 \
    --clip-mode gipo \
    --value-lr 1e-4 \
    --policy-lr 1e-5 \
    --value-warmup-steps 100 \
    --policy-warmup-steps 100 \
    --policy-train-start-step 0 \
    --imagine-horizon 32 \
    --reward-scale 1.0 \
    --wm-replay-capacity 50000 \
    --real-traj-collect-interval 1 \
    --reward-batch-size 16 \
    --reward-accumulation-steps 16 \
    --reward-lr 1e-4 \
    --reward-warmup-steps 20 \
    --reward-train-interval 5 \
    --ctrl-batch-size 4 \
    --ctrl-accumulation-steps 64 \
    --ctrl-lr 1e-4 \
    --ctrl-warmup-steps 20 \
    --ctrl-train-interval 5 \
    --ctrl-ema-decay 0 \
    --ctrl-eval-interval 50 \
    --ctrl-eval-batch-size 64 \
    --ctrl-eval-micro-batch-size 4 \
    --ctrl-eval-seed 12345 \
    --ctrl-eval-episode-ratio 0.05 \
    --ctrl-eval-windows-per-episode 8 \
    --ctrl-eval-max-horizon 4 \
    --ctrl-eval-lpips-net alex \
    --replay-capacity 10000 \
    --ckpt-dir "$ACCERL_FINAL_CHECKPOINT" \
    --ckpt-every-steps 500 \
    --moving-avg-window 1000 \
    --log-interval-seconds 10 \
    --exp-name OpenVLA_DS_gipo_DISCRETE_task0_ctrl_wm_four_zero2_mps_split_train \
    --reward-checkpoint /mnt/data/lcx3/checkpoint/reward/reward.pt \
    "$@"
