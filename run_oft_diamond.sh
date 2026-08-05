#!/bin/bash
# OpenVLA RL训练脚本示例
# 使用命令行参数启动训练

ACCERL_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${ACCERL_SCRIPT_DIR}" || exit 1
export PYTHONPATH="${ACCERL_SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
ACCERL_PYTHON="/mnt/data/lcx3/envs/merged-env/bin/python"

if [[ ! -x "${ACCERL_PYTHON}" ]]; then
    echo "Python environment not found: ${ACCERL_PYTHON}" >&2
    exit 1
fi

# python rl/ds_wm_discrete_diffusion.py \
#   --cuda-visible-devices "0,5" \
#   --benchmark libero_spatial \
#   --num-trainer-gpus 1 \
#   --num-inference-actors 1 \
#   --num-rollout-workers 20 \
#   --num-eval-workers 10 \
#   --rollout-local-buf 64 \
#   --inference-batch 12 \
#   --inference-timeout-ms 300 \
#   --replay-capacity 10000 \
#   --train-batch-size 16 \
#   --accumulation-steps 16 \
#   --train-iters 30000 \
#   --ckpt-dir "/cpfs01/lcx_workspace/models/finetune_rl" \
#   --ckpt-every-steps 2000000 \
#   --gamma 0.99 \
#   --lambda 0.95 \
#   --clip-eps 0.2 \
#   --vf-coef 0.5 \
#   --ent-coef 0.00 \
#   --kl-coef 0.1 \
#   --reward-scale 1.0 \
#   --value-lr 1e-4 \
#   --policy-lr 1e-5 \
#   --value-warmup-steps 500 \
#   --policy-warmup-steps 500 \
#   --policy-train-start-step 0 \
#   --imagine-horizon 8 \
#   --num-step-cond 4 \
#   --num-reward-inference-actors 1 \
#   --num-denoiser-inference-actors 1 \
#   --agent-config-path "envs/config/agent.yaml" \
#   --trainer-config-path "envs/config/trainer.yaml" \
#   --moving-avg-window 1000 \
#   --log-interval-seconds 10 \
#   --broadcast-group-name "trainer_to_inference_broadcast" \
#   --use-bf16 \
#   --pretrained-checkpoint "/cpfs01/liuwei_workspace/models/finetune_im/openvla-7b+libero_spatial_no_noops+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--discrete_acts--proprio_state--100000_chkpt" \
#   --checkpoint2 "runs/distill/20251225_113851_distill/checkpoints/checkpoint_latest.pt" \
#   --clip-mode sapo \
#   --exp-name "OpenVLA_DS_sapo_DISCRETE_task0_wm_args"


"${ACCERL_PYTHON}" rl/ds_wm_discrete_diffusion.py \
    --cuda-visible-devices 0,1,2,3 \
    --use-bf16 \
    --benchmark libero_spatial \
    --num-images-in-input 1 \
    --pretrained-checkpoint /mnt/data/lcx3/checkpoint/dsj/openvla-7b+libero_spatial_no_noops+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--discrete_acts--proprio_state--100000_chkpt \
    --checkpoint2 /mnt/data/lcx3/checkpoint/dsj/20251225_113851_distill_checkpoint_latest.pt \
    --agent-config-path envs/config/agent.yaml \
    --trainer-config-path envs/config/trainer.yaml \
    --num-trainer-gpus 1 \
    --num-inference-actors 1 \
    --num-rollout-workers 10 \
    --num-eval-workers 10 \
    --num-reward-inference-actors 1 \
    --num-denoiser-inference-actors 1 \
    --train-iters 30000 \
    --train-batch-size 16 \
    --accumulation-steps 36 \
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
    --value-warmup-steps 500 \
    --policy-warmup-steps 500 \
    --policy-train-start-step 0 \
    --imagine-horizon 4 \
    --num-step-cond 4 \
    --reward-scale 1.0 \
    --wm-replay-capacity 50000 \
    --real-traj-collect-interval 1 \
    --denoiser-batch-size 8 \
    --denoiser-accumulation-steps 128 \
    --denoiser-lr 1e-4 \
    --denoiser-warmup-steps 500 \
    --reward-batch-size 32 \
    --reward-accumulation-steps 32 \
    --reward-lr 1e-4 \
    --reward-warmup-steps 500 \
    --denoiser-train-interval 5 \
    --reward-train-interval 5 \
    --wm-validation-fraction 0.1 \
    --wm-eval-interval 50 \
    --wm-eval-batch-size 8 \
    --reward-eval-batch-size 512 \
    --reward-trajectory-eval-window 100 \
    --replay-capacity 10000 \
    --ckpt-dir /cpfs01/liuwei_workspace/models/finetune_rl \
    --ckpt-every-steps 2000000 \
    --moving-avg-window 1000 \
    --log-interval-seconds 10 \
    --exp-name OpenVLA_DS_gipo_DISCRETE_task0_train_wm \
    --denoiser-checkpoint /mnt/data/lcx2/yanjieworkspace/openvla_oft_rl/runs/wm_reward_denoiser_distill_named/denoiser_smoke_test/denoiser_smoke_test.pt \
    --reward-checkpoint /mnt/data/lcx3/checkpoint/reward/reward.pt
