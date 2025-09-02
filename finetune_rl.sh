export CUDA_VISIBLE_DEVICES=1,3

torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune_rl.py \
  --vla_path /cpfs01/lcx_workspace/models/openvla-7b-oft-finetuned-libero-spatial-object-goal-10/ \
  --data_root_dir /cpfs01/lcx_workspace/data/openvla/modified_libero_rlds/ \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir /cpfs01/liuwei_workspace/openvla_oft_rl/ckpt/finetune_nll_test \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 16 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 100000 \
  --max_steps 150005 \
  --save_freq 10000 \
  --save_latest_checkpoint_only True \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity "YOUR_WANDB_ENTITY" \
  --wandb_project "YOUR_WANDB_PROJECT" \
  --run_id_note parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state