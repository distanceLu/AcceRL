#!/bin/bash

python rl/maniskill/finetune_maniskill.py \
    --use_preprocessed_data True \
    --dataloader_num_workers 4 \
    --use_maniskill_env_eval False \
    --maniskill_eval_freq 200 \
    --maniskill_eval_num_episodes 50 \
    --maniskill_eval_num_envs 10