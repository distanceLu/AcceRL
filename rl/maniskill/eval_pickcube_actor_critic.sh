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
export MPLCONFIGDIR=/tmp/matplotlib-why-maniskill-eval-pickcube
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT_ROOT"

python rl/maniskill/maniskill_actor_critic_eval.py \
  --gpu-id 2 \
  --checkpoint /mnt/data/lcx4/openvla_oft_rl/rl/maniskill/imitation_model \
  --eval-output-dir /mnt/data/lcx/AcceRL/runs \
  --task-id PickCube-v1 \
  --unnorm-key maniskill_pickcube \
  --language-instruction "pick up the red cube and place it at the green target" \
  --max-steps 200 \
  --num-eval-episodes 50 \
  --num-envs 1 \
  --base-seed 0 \
  --camera-name base_camera \
  --wrist-camera-name hand_camera \
  --camera-res 224 \
  --num-images-in-input 2 \
  --robot-uids panda_wristcam \
  --env-action-dim 7 \
  --control-mode pd_ee_delta_pose \
  --sim-backend gpu \
  --reward-mode sparse \
  --render-mode rgb_array \
  --exec-actions-per-inference 8 \
  --use-bf16 \
  --no-use-proprio \
  --center-crop \
  --use-lora \
  --lora-rank 32 \
  --lora-dropout 0.0 \
  --no-load-in-8bit \
  --no-load-in-4bit \
  --no-use-film \
  --enable-pmvt \
  --checkpoint2 "" \
  --record-eval-video \
  --record-video-num-episodes 5 \
  --show-pickcube-goal-in-policy-obs
