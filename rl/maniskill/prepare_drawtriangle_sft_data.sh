#!/usr/bin/env bash
set -eo pipefail

MANISKILL_ENV=/mnt/data/lcx4/miniforge3/envs/why_maniskill
PROJECT_ROOT=/mnt/data/lcx/AcceRL
DRAW_ROOT=/mnt/data/lcx/data/maniskill/DrawTriangle-v1
SOURCE_H5="$DRAW_ROOT/motionplanning/trajectory.h5"
REPLAY_H5="$DRAW_ROOT/motionplanning/trajectory.rgbd.pd_ee_delta_pose.physx_cpu.h5"
RLDS_ROOT="$DRAW_ROOT/rlds"
PT_ROOT="$DRAW_ROOT/preprocessed_pt"

source /mnt/data/lcx4/miniforge3/etc/profile.d/conda.sh
conda activate why_maniskill
cd "$PROJECT_ROOT"

export MPLCONFIGDIR="$PROJECT_ROOT/.runtime/matplotlib-why-maniskill"
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$MPLCONFIGDIR"

if [[ ! -f "$REPLAY_H5" ]]; then
  TRAJ_PATH="$SOURCE_H5" python rl/maniskill/replay_drawtriangle_224.py
else
  echo "Using existing replay: $REPLAY_H5"
fi

python -m rl.maniskill.maniskill_drawtriangle_dataset_builder \
  --h5-path "$REPLAY_H5" \
  --data-dir "$RLDS_ROOT"

python rl/maniskill/preprocess_rlds_to_pt.py \
  --vla_path /mnt/data/lcx4/hf_cache/openvla-7b \
  --data_root_dir "$RLDS_ROOT" \
  --dataset_name maniskill_drawtriangle \
  --output_dir "$PT_ROOT" \
  --num_images_in_input 1 \
  --use_proprio False \
  --image_aug False
