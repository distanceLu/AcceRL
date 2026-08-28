#!/usr/bin/env bash
set -eo pipefail

MANISKILL_ENV=/mnt/data/lcx4/miniforge3/envs/why_maniskill
PROJECT_ROOT=/mnt/data/lcx/AcceRL
DRAW_DATA_ROOT=/mnt/data/lcx/data/maniskill/DrawTriangle-v1
# Physical GPU selection. PyTorch sees this device as logical cuda:0.
GPU_ID=0

source /mnt/data/lcx4/miniforge3/etc/profile.d/conda.sh
conda activate why_maniskill

set -u

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export VULKAN_VISIBLE_DEVICES="$GPU_ID"
export CUDA_HOME="$MANISKILL_ENV"
export PATH="$MANISKILL_ENV/bin:$MANISKILL_ENV/targets/x86_64-linux/bin:$PATH"
export LD_LIBRARY_PATH="$MANISKILL_ENV/lib:$MANISKILL_ENV/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export TORCH_EXTENSIONS_DIR=/mnt/data/lcx4/.cache/torch_extensions/why_maniskill_py310_cu124

export LIBERO_CONFIG_PATH="$MANISKILL_ENV/libero_config"
export MPLCONFIGDIR=/tmp/matplotlib-why-maniskill
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT_ROOT"

echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "CUDA_HOME=$CUDA_HOME"
echo "python=$(command -v python)"
echo "nvcc=$(command -v nvcc)"

python -c "import numpy, torch; print('numpy:', numpy.__version__); print('torch:', torch.__version__); print('torch CUDA:', torch.version.cuda)"
nvcc --version

python rl/maniskill/finetune_maniskill.py \
  --vla_path /mnt/data/lcx4/hf_cache/openvla-7b \
  --dataset_name maniskill_drawtriangle \
  --preprocessed_data_dir "$DRAW_DATA_ROOT/preprocessed_pt" \
  --num_images_in_input 1 \
  --use_preprocessed_data True \
  --use_proprio False \
  --image_aug True \
  --batch_size 16 \
  --grad_accumulation_steps 8 \
  --learning_rate 5e-4 \
  --max_steps 50000 \
  --save_freq 200 \
  --run_id_note drawtriangle_1cam_delta_pose \
  --use_maniskill_env_eval False
