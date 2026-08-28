#!/usr/bin/env bash
set -eo pipefail

MANISKILL_ENV=/mnt/data/lcx4/miniforge3/envs/why_maniskill
PROJECT_ROOT=/mnt/data/lcx/AcceRL

source /mnt/data/lcx4/miniforge3/etc/profile.d/conda.sh
conda activate why_maniskill

set -u

export CUDA_HOME="$MANISKILL_ENV"
export PATH="$MANISKILL_ENV/bin:$MANISKILL_ENV/targets/x86_64-linux/bin:$PATH"
export LD_LIBRARY_PATH="$MANISKILL_ENV/lib:$MANISKILL_ENV/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export TORCH_EXTENSIONS_DIR=/mnt/data/lcx4/.cache/torch_extensions/why_maniskill_py310_cu124
export LIBERO_CONFIG_PATH="$MANISKILL_ENV/libero_config"
export MPLCONFIGDIR=/tmp/matplotlib-why-maniskill-eval-drawtriangle
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT_ROOT"

python rl/maniskill/maniskill_actor_critic_eval.py \
  --gpu-id 1 \
  --checkpoint /mnt/data/lcx/AcceRL/runs/imitation/20260826_202524_openvla-7b+maniskill_drawtriangle+b128+lr-0.0005+lora-r32+dropout-0.0--image_aug--drawtriangle_1cam_delta_pose \
  --eval-output-dir /mnt/data/lcx/AcceRL/runs \
  --task-id DrawTriangle-v1 \
  --unnorm-key maniskill_drawtriangle \
  --language-instruction "draw the outlined triangle on the canvas" \
  --max-steps 300 \
  --num-eval-episodes 50 \
  --num-envs 1 \
  --base-seed 0 \
  --camera-name base_camera \
  --camera-res 224 \
  --num-images-in-input 1 \
  --robot-uids panda_stick \
  --env-action-dim 6 \
  --control-mode pd_ee_delta_pose \
  --sim-backend gpu \
  --reward-mode sparse \
  --render-mode rgb_array \
  --exec-actions-per-inference 8 \
  --use-bf16 \
  --no-use-proprio \
  --no-use-lora \
  --no-center-crop \
  --lora-rank 32 \
  --lora-dropout 0.0 \
  --no-load-in-8bit \
  --no-load-in-4bit \
  --no-use-film \
  --enable-pmvt \
  --checkpoint2 "" \
  --record-eval-video \
  --record-video-num-episodes 5 \
  --no-show-pickcube-goal-in-policy-obs
