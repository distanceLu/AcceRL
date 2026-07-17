#!/usr/bin/env bash
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


python rl/maniskill/ds_maniskill_ppo_discrete.py \
  --cuda-visible-devices "0,1,2,3" \
  --maniskill-tasks PickCube-v1,StackCube-v1 \
  --camera-name base_camera \
  --wrist-camera-name hand_camera \
  --robot-uids panda_wristcam \
  --camera-res 224 \
  --num-images-in-input 2 \
  --num-trainer-gpus 3 \
  --num-inference-actors 1 \
  --num-rollout-workers 30 \
  --num-eval-workers 2 \
  --inference-batch 4 \
  --replay-capacity 3000 \
  --train-batch-size 8 \
  --accumulation-steps 8 \
  --train-iters 60000 \
  --object-store-memory-gb 256 \
  --ckpt-dir "/mnt/data/lcx4/openvla_oft_rl/runs/rl_maniskill" \
  --ckpt-every-steps 2000000 \
  --gamma 0.99 \
  --lambda 0.95 \
  --vf-coef 0.5 \
  --ent-coef 0.00 \
  --kl-coef 0.1 \
  --value-lr 3e-5 \
  --policy-lr 3e-6 \
  --use-bf16 \
  --pretrained-checkpoint "/mnt/data/lcx4/openvla_oft_rl/rl/maniskill/sft_model" \
  --clip-mode gipo \
  --exp-name "ManiSkill_PickCube_StackCube_dual_cam_gipo_60k_potential_reward_cpu_96" \
  --sigma 0.5