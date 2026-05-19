#!/bin/bash


/cpfs01/lcx_stu4_workspace/envs/why_maniskill/bin/python rl/ManiSkill/ds_maniskill_ppo_discrete.py \
  --cuda-visible-devices "4,5,6,7" \
  --maniskill-task PickCube-v1 \
  --camera-name base_camera \
  --wrist-camera-name hand_camera \
  --robot-uids panda_wristcam \
  --camera-res 224 \
  --num-images-in-input 2 \
  --num-trainer-gpus 3 \
  --num-inference-actors 1 \
  --num-rollout-workers 30 \
  --num-eval-workers 1 \
  --inference-batch 4 \
  --replay-capacity 3000 \
  --train-batch-size 16 \
  --accumulation-steps 4 \
  --train-iters 60000 \
  --object-store-memory-gb 256 \
  --ckpt-dir "/cpfs01/lcx_stu4_workspace/openvla_oft_rl/runs/rl_maniskill" \
  --ckpt-every-steps 2000000 \
  --gamma 0.99 \
  --lambda 0.95 \
  --vf-coef 0.5 \
  --ent-coef 0.00 \
  --kl-coef 0.1 \
  --value-lr 3e-5 \
  --policy-lr 3e-6 \
  --use-bf16 \
  --pretrained-checkpoint "/cpfs01/lcx_stu4_workspace/openvla_oft_rl/runs/imitation/20260511_203047_openvla-7b+maniskill_pickcube+b64+lr-0.0005+lora-r32+dropout-0.0--image_aug_2images" \
  --clip-mode gipo \
  --exp-name "ManiSkill_PickCube_dual_cam_gipo_60k" \
  --sigma 0.5
