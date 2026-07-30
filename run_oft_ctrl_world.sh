#!/bin/bash
# OpenVLA RL训练脚本示例：Ctrl-World 世界模型版本
# 使用命令行参数启动 tests_dsj/ds_wm_discrete_ctrl.py 训练

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export TMPDIR=/dev/shm
#export CUDA_VISIBLE_DEVICES=3,4,5,6
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
export RAY_DEDUP_LOGS=0

/mnt/data/lcx3/envs/merged-env/bin/python tests_dsj/ds_wm_discrete_ctrl.py \
    --cuda-visible-devices 3,4,5,6 \
    --use-bf16 \
    --benchmark libero_spatial \
    --num-images-in-input 2 \
    --pretrained-checkpoint /mnt/data/lcx3/checkpoint/dsj/openvla-7b+libero_spatial_no_noops+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--discrete_acts--proprio_state--100000_chkpt \
    --checkpoint2 /mnt/data/lcx3/checkpoint/dsj/20251225_113851_distill_checkpoint_latest.pt \
    --agent-config-path envs/config/agent.yaml \
    --trainer-config-path envs/config/trainer.yaml \
    --svd-model-path /mnt/data/lcx3/checkpoint/ctrl_world/svd/svd_model \
    --clip-model-path /mnt/data/lcx3/checkpoint/ctrl_world/clip/clip_model \
    --ctrl-world-ckpt /mnt/data/lcx3/Ctrl-World/model_ckpt/libero_vla_delta_finetune/2026-07-21T16-40-56_libero_vla_delta_finetune/best_val_loss.pt \
    --condition-stat-path /mnt/data/lcx3/Ctrl-World/model_ckpt/libero_vla_delta_finetune/2026-07-21T16-40-56_libero_vla_delta_finetune/condition_stat.json \
    --num-cams 2 \
    --num-history 6 \
    --num-step-cond 7 \
    --num-frames-pred 5 \
    --num-inference-steps 10 \
    --num-trainer-gpus 1 \
    --num-inference-actors 1 \
    --num-rollout-workers 4 \
    --num-eval-workers 2 \
    --num-reward-inference-actors 1 \
    --num-ctrl-inference-actors 1 \
    --train-iters 30000 \
    --train-batch-size 8 \
    --accumulation-steps 8 \
    --inference-batch 4 \
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
    --imagine-horizon 8 \
    --reward-scale 1.0 \
    --wm-replay-capacity 50000 \
    --real-traj-collect-interval 1 \
    --reward-batch-size 16 \
    --reward-accumulation-steps 8 \
    --reward-lr 1e-4 \
    --reward-warmup-steps 500 \
    --reward-train-interval 5 \
    --replay-capacity 10000 \
    --ckpt-dir /mnt/data/lcx3/AcceRL/runs/ctrl_wm_checkpoints \
    --ckpt-every-steps 5000 \
    --moving-avg-window 1000 \
    --log-interval-seconds 10 \
    --exp-name OpenVLA_DS_gipo_DISCRETE_task0_ctrl_wm \
    --reward-checkpoint /mnt/data/lcx3/checkpoint/reward/reward.pt
