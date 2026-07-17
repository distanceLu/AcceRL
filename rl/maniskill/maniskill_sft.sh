set -eo pipefail

MANISKILL_ENV=/mnt/data/lcx4/miniforge3/envs/why_maniskill

source /mnt/data/lcx4/miniforge3/etc/profile.d/conda.sh
conda activate why_maniskill

set -u

export CUDA_HOME="$MANISKILL_ENV"
export PATH="$MANISKILL_ENV/bin:$MANISKILL_ENV/targets/x86_64-linux/bin:$PATH"
export LD_LIBRARY_PATH="$MANISKILL_ENV/lib:$MANISKILL_ENV/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export TORCH_EXTENSIONS_DIR=/mnt/data/lcx4/.cache/torch_extensions/why_maniskill_py310_cu124

export LIBERO_CONFIG_PATH="$MANISKILL_ENV/libero_config"
export MPLCONFIGDIR=/tmp/matplotlib-why-maniskill

cd /mnt/data/lcx4/openvla_oft_rl

echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "CUDA_HOME=$CUDA_HOME"
echo "python=$(command -v python)"
echo "nvcc=$(command -v nvcc)"

python -c "import numpy, torch; print('numpy:', numpy.__version__); print('torch:', torch.__version__); print('torch CUDA:', torch.version.cuda)"
nvcc --version

python rl/maniskill/finetune_maniskill.py \
  --vla_path /mnt/data/lcx4/hf_cache/openvla-7b \
  --dataset_name maniskill_three_tasks \
  --preprocessed_data_dir /mnt/data/lcx4/openvla_oft_rl/rl/maniskill/sft_data \
  --num_images_in_input 2 \
  --use_preprocessed_data True \
  --use_proprio False \
  --image_aug True \
  --batch_size 16 \
  --grad_accumulation_steps 8 \
  --learning_rate 5e-4 \
  --max_steps 100000 \
  --save_freq 200 \
  --run_id_note three_tasks_2cam_preprocessed \
  --use_maniskill_env_eval False
