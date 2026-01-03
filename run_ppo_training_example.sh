#!/bin/bash
# OpenVLA PPO RL训练脚本示例 (无世界模型)
# 使用命令行参数启动训练

python rl/ds_libero_ppo_discrete.py \
  --cuda-visible-devices "6,7" \
  --benchmark libero_spatial \
  --num-trainer-gpus 1 \
  --num-inference-actors 1 \
  --num-rollout-workers 2 \
  --num-eval-workers 20 \
  --rollout-local-buf 64 \
  --inference-batch 8 \
  --inference-timeout-ms 300 \
  --replay-capacity 10000 \
  --train-batch-size 12 \
  --accumulation-steps 21 \
  --train-iters 30000 \
  --object-store-memory-gb 256 \
  --ckpt-dir "/cpfs01/liuwei_workspace/models/finetune_rl" \
  --ckpt-every-steps 2000000 \
  --gamma 0.99 \
  --lambda 0.95 \
  --clip-eps 0.2 \
  --vf-coef 0.5 \
  --ent-coef 0.00 \
  --kl-coef 0.1 \
  --reward-scale 1.0 \
  --value-lr 1e-4 \
  --policy-lr 1e-5 \
  --value-warmup-steps 500 \
  --policy-warmup-steps 500 \
  --policy-train-start-step 0 \
  --moving-avg-window 1000 \
  --log-interval-seconds 10 \
  --broadcast-group-name "trainer_to_inference_broadcast" \
  --use-bf16 \
  --use-proprio \
  --pretrained-checkpoint "/cpfs01/liuwei_workspace/models/finetune_im/openvla-7b+libero_spatial_no_noops+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--discrete_acts--proprio_state--100000_chkpt" \
  --checkpoint2 "runs/distill/20251225_113851_distill/checkpoints/checkpoint_latest.pt" \
  --clip-mode sapo \
  --exp-name "OpenVLA_DS_sapo_DISCRETE_task0_10k_buffer"

