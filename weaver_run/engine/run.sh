#!/bin/bash
# OpenVLA Policy + native online WEAVER with its jointly adapted internal Reward.
# GPU4-7: Policy ZeRO2 + native World/RM DDP; four models/card, no external Reward service.
# WEAVER uses its isolated verified environment with native loss and per-rank IPC publication.
# Global batches Policy576/World+RM1024; 5:1. Real1/imag4/budget1M.

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export TMPDIR=/dev/shm
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
export RAY_DEDUP_LOGS=0
set -euo pipefail
FSDP_SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ACCERL_REPO_ROOT="${ACCERL_REPO_ROOT:-/mnt/data/lcx1/dong/AcceRL-scheduler}"
export PYTHONPATH="$FSDP_SOURCE:$ACCERL_REPO_ROOT:$ACCERL_REPO_ROOT/rl/ctrl_world:${PYTHONPATH:-}"
export ACCERL_PIPELINE_CONFIG="${ACCERL_PIPELINE_CONFIG:-$FSDP_SOURCE/pipeline_default.json}"
export ACCERL_WORLD_BACKEND_CONFIG="${ACCERL_WORLD_BACKEND_CONFIG:-$FSDP_SOURCE/examples/weaver_integrated_optimized.json}"
export ACCERL_PIPELINE_CONFIG="$(realpath -- "$ACCERL_PIPELINE_CONFIG")"
CUDA_VISIBLE_DEVICES='' /mnt/data/lcx3/envs/merged-env/bin/python -c 'import os; from pipeline_budget import PipelineBudgetConfig; PipelineBudgetConfig.from_json(os.environ["ACCERL_PIPELINE_CONFIG"])'
# There is no inherited live-experiment output root in this isolated candidate.
: "${ACCERL_EXPERIMENT_ROOT:?Set an explicit new absolute candidate experiment root}"
CUDA_VISIBLE_DEVICES='' /mnt/data/lcx3/envs/merged-env/bin/python -c 'from candidate_paths import launch_paths; launch_paths()'
export ACCERL_EXPERIMENT_ROOT="$(realpath -m -- "$ACCERL_EXPERIMENT_ROOT")"
# Optional backend admission runs on CPU before any MPS checks or model launch.
# The empty array preserves the original default argv exactly.
WORLD_BACKEND_ARGS=()
WORLD_BACKEND_CONFIG="${ACCERL_WORLD_BACKEND_CONFIG:-}"
WORLD_CONFIG_NEXT=0
for argument in "$@"; do
    if [[ "$WORLD_CONFIG_NEXT" == 1 ]]; then
        WORLD_BACKEND_CONFIG="$argument"
        WORLD_CONFIG_NEXT=0
    elif [[ "$argument" == --world-backend-config ]]; then
        WORLD_CONFIG_NEXT=1
    elif [[ "$argument" == --world-backend-config=* ]]; then
        WORLD_BACKEND_CONFIG="${argument#*=}"
    fi
done
[[ "$WORLD_CONFIG_NEXT" == 0 ]] || { echo "--world-backend-config requires a path." >&2; exit 1; }
if [[ -n "$WORLD_BACKEND_CONFIG" || -n "${ACCERL_WORLD_BACKEND:-}" || -n "${ACCERL_REWARD_PROVIDER:-}" ]]; then
    if [[ -n "$WORLD_BACKEND_CONFIG" ]]; then
        WORLD_BACKEND_CONFIG="$(realpath -- "$WORLD_BACKEND_CONFIG")"
        export ACCERL_WORLD_BACKEND_CONFIG="$WORLD_BACKEND_CONFIG"
        WORLD_BACKEND_ARGS+=(--world-backend-config "$WORLD_BACKEND_CONFIG")
    fi
    BACKEND_GEOMETRY="$(CUDA_VISIBLE_DEVICES='' /mnt/data/lcx3/envs/merged-env/bin/python -c '
from world_backends.runtime import load_backend_config, prepare_runtime, context_geometry
config = load_backend_config()
plan = prepare_runtime(config, world_size=4)
geometry = context_geometry(config)
print(0 if plan.integrated_reward else 4, geometry["required_context_frames"] if config.backend == "weaver" else 7)
')"
    read -r REWARD_INFERENCE_COUNT WORLD_CONTEXT_FRAMES <<< "$BACKEND_GEOMETRY"
    WORLD_BACKEND_ARGS+=(--num-reward-inference-actors "$REWARD_INFERENCE_COUNT" --num-step-cond "$WORLD_CONTEXT_FRAMES")
fi
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
export ACCERL_RUN_ID="${ACCERL_RUN_ID:-weaverintegrated_$(date +%Y%m%d_%H%M%S_%N)_$$}"
[[ "$ACCERL_RUN_ID" =~ ^weaverintegrated_[A-Za-z0-9_-]+$ ]] || { echo "Invalid candidate ACCERL_RUN_ID." >&2; exit 1; }
export ACCERL_DATA_DIR="${ACCERL_DATA_DIR:-$ACCERL_EXPERIMENT_ROOT/formal_v3/training}"
export ACCERL_RAY_TMP="${ACCERL_RAY_TMP:-/dev/shm/wi_${UID}_$$_${RANDOM}}"
export ACCERL_RAY_SPILL="${ACCERL_RAY_SPILL:-$ACCERL_DATA_DIR/ray_spill}"
export ACCERL_TELEMETRY_DB="${ACCERL_TELEMETRY_DB:-$ACCERL_DATA_DIR/collection.sqlite}"
export ACCERL_FINAL_CHECKPOINT="${ACCERL_FINAL_CHECKPOINT:-$ACCERL_DATA_DIR/checkpoints}"
export ACCERL_STOP_FILE="${ACCERL_STOP_FILE:-$ACCERL_DATA_DIR/stop.flag}"
# Guard only this invocation; no global Ray/process termination.
mkdir -p -- "$ACCERL_EXPERIMENT_ROOT"
exec 9>"$ACCERL_EXPERIMENT_ROOT/.weaver_integrated_v1.lock"
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
echo "Online WEAVER initialized3200 + Policy ZeRO2 + Reward FSDP2 + existing MPS: run=$ACCERL_RUN_ID serial=$ACCERL_SPLIT_TRAIN_SERIAL data=$ACCERL_DATA_DIR pipe=$CUDA_MPS_PIPE_DIRECTORY"

# Use the same checked-in argv and base imports as the portable entry.
export PYTHONPATH="$FSDP_SOURCE/..:$PYTHONPATH:$FSDP_SOURCE/../vendor/ctrl"
/mnt/data/lcx3/envs/merged-env/bin/python -B -c 'import sys; from launch import exec_managed_driver; exec_managed_driver(sys.argv[1:])' "$@"
