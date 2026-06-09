# Launch LIBERO-Spatial evals
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint /cpfs01/lcx_workspace/models/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
  --task_suite_name libero_spatial

python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint /cpfs01/lcx_workspace/models/openvla-oft-libero10 \
  --task_suite_name libero_10

# python experiments/robot/libero/run_libero_eval.py \
#   --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
#   --task_suite_name libero_spatial
  
# # Launch LIBERO-Object evals
# python experiments/robot/libero/run_libero_eval.py \
#   --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-object \
#   --task_suite_name libero_object