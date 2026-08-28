#!/usr/bin/env bash
set -eo pipefail

# ManiSkill 训练使用的 Conda 环境；换环境时修改此路径。
MANISKILL_ENV=/mnt/data/lcx4/miniforge3/envs/why_maniskill
# 根据脚本位置解析当前仓库根目录，避免依赖固定工作树路径。
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
RUNTIME_ROOT="$PROJECT_ROOT/.runtime/why_maniskill_drawtriangle"
DEBUG_LOG_DIR="$RUNTIME_ROOT/debug"
RAY_TEMP_DIR="/tmp/ray-lcx-drawtriangle"
PRETRAINED_CHECKPOINT="$PROJECT_ROOT/runs/imitation/20260826_202524_openvla-7b+maniskill_drawtriangle+b128+lr-0.0005+lora-r32+dropout-0.0--image_aug--drawtriangle_1cam_delta_pose"

source /mnt/data/lcx4/miniforge3/etc/profile.d/conda.sh
conda activate why_maniskill

set -u

mkdir -p \
  "$RUNTIME_ROOT/matplotlib" \
  "$RUNTIME_ROOT/numba" \
  "$RUNTIME_ROOT/torch_extensions" \
  "$RUNTIME_ROOT/tmp" \
  "$DEBUG_LOG_DIR" \
  "$RAY_TEMP_DIR" \
  "$PROJECT_ROOT/runs/ManiSkill" \
  "$PROJECT_ROOT/runs/rl_maniskill_drawtriangle"

export CUDA_HOME="$MANISKILL_ENV"
export PATH="$MANISKILL_ENV/bin:$MANISKILL_ENV/targets/x86_64-linux/bin:$PATH"
export LD_LIBRARY_PATH="$MANISKILL_ENV/lib:$MANISKILL_ENV/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export TORCH_EXTENSIONS_DIR="$RUNTIME_ROOT/torch_extensions"

export LIBERO_CONFIG_PATH="$MANISKILL_ENV/libero_config"
export MPLCONFIGDIR="$RUNTIME_ROOT/matplotlib"
export NUMBA_CACHE_DIR="$RUNTIME_ROOT/numba"
export TMPDIR="$RUNTIME_ROOT/tmp"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT_ROOT"

echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "CUDA_HOME=$CUDA_HOME"
echo "python=$(command -v python)"
echo "nvcc=$(command -v nvcc)"

python -c "import numpy, torch; print('numpy:', numpy.__version__); print('torch:', torch.__version__); print('torch CUDA:', torch.version.cuda)"
nvcc --version

python rl/maniskill/ds_maniskill_ppo_discrete.py \
  --cuda-visible-devices "0,1" \
  --maniskill-tasks DrawTriangle-v1 \
  --camera-name base_camera \
  --robot-uids panda_stick \
  --camera-res 224 \
  --max-episode-steps 300 \
  --language-instruction "draw the outlined triangle on the canvas" \
  --unnorm-key maniskill_drawtriangle \
  --sim-backend gpu \
  --num-images-in-input 1 \
  --num-trainer-gpus 1 \
  --num-inference-actors 1 \
  --num-rollout-workers 10 \
  --num-eval-workers 1 \
  --inference-batch 4 \
  --replay-capacity 3000 \
  --train-batch-size 16 \
  --accumulation-steps 8 \
  --train-iters 60000 \
  --object-store-memory-gb 256 \
  --ray-temp-dir "$RAY_TEMP_DIR" \
  --log-root "$PROJECT_ROOT/runs/ManiSkill" \
  --ckpt-dir "$PROJECT_ROOT/runs/rl_maniskill_drawtriangle" \
  --debug-log-dir "$DEBUG_LOG_DIR" \
  --ckpt-every-steps 2000000000 \
  --gamma 0.99 \
  --lambda 0.95 \
  --vf-coef 0.5 \
  --ent-coef 0.00 \
  --kl-coef 0.1 \
  --value-lr 3e-5 \
  --policy-lr 3e-6 \
  --use-bf16 \
  --pretrained-checkpoint "$PRETRAINED_CHECKPOINT" \
  --no-center-crop \
  --clip-mode gipo \
  --exp-name "ManiSkill_DrawTriangle_1cam_gipo_60k_no_center_crop" \
  --sigma 0.5
