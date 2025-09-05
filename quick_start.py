import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import pickle
from experiments.robot.libero.run_libero_eval import GenerateConfig
from experiments.robot.openvla_utils import get_action_head, get_processor, get_proprio_projector, get_vla, get_vla_action
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM

# Instantiate config (see class GenerateConfig in experiments/robot/libero/run_libero_eval.py for definitions)
cfg = GenerateConfig(
    pretrained_checkpoint = "moojink/openvla-7b-oft-finetuned-libero-spatial",
    use_l1_regression = True,
    use_diffusion = False,
    use_film = False,
    num_images_in_input = 2,
    use_proprio = True,
    load_in_8bit = False,
    load_in_4bit = False,
    center_crop = True,
    num_open_loop_steps = NUM_ACTIONS_CHUNK,
    unnorm_key = "libero_spatial_no_noops",
)

# Load OpenVLA-OFT policy and inputs processor
vla = get_vla(cfg)
processor = get_processor(cfg)

# Load MLP action head to generate continuous actions (via L1 regression)
action_head = get_action_head(cfg, llm_dim=vla.llm_dim)

# Load proprio projector to map proprio to language embedding spacec
proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)

# Load sample observation:
#   observation (dict): {
#     "full_image": primary third-person image,
#     "wrist_image": wrist-mounted camera image,
#     "state": robot proprioceptive state,
#     "task_description": task description,
#   }

with open("experiments/robot/libero/sample_libero_spatial_observation.pkl", "rb") as file:
    observation = pickle.load(file)

# Generate robot action chunk (sequence of future actions)
actions = get_vla_action(cfg, vla, processor, observation, observation["task_description"], action_head, proprio_projector)
# print("Generated action chunk:")
# for act in actions:
#     print(act)


import torch
import torch.nn as nn

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Collect all input images
all_images = [observation["full_image"]]
if cfg.num_images_in_input > 1:
    all_images.extend([observation[k] for k in observation.keys() if "wrist" in k])

# Process images
from experiments.robot.openvla_utils import prepare_images_for_vla
all_images = prepare_images_for_vla(all_images, cfg)

# Extract primary image and additional images
primary_image = all_images.pop(0)

# Build VLA prompt
task_label = observation["task_description"]
prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"


# # Process primary image
# inputs = processor(prompt, primary_image).to(DEVICE, dtype=torch.bfloat16)

# # Process additional wrist images if any
# if all_images:
#     all_wrist_inputs = [
#         processor(prompt, image_wrist).to(DEVICE, dtype=torch.bfloat16) for image_wrist in all_images
#     ]
#     # Concatenate all images
#     primary_pixel_values = inputs["pixel_values"]
#     all_wrist_pixel_values = [wrist_inputs["pixel_values"] for wrist_inputs in all_wrist_inputs]
#     inputs["pixel_values"] = torch.cat([primary_pixel_values] + all_wrist_pixel_values, dim=1)


encoder = vla.vision_backbone
projector = vla.projector
language_model = vla.language_model

encoder_params = sum(p.numel() for p in encoder.parameters())
projector_params = sum(p.numel() for p in projector.parameters())
language_model_params = sum(p.numel() for p in language_model.parameters())
print(f"encoder_params: {encoder_params}") # 730911680
print(f"projector_params: {projector_params}") # 71385600
print(f"language_model_params: {language_model_params}") # 6738939904


from storm.vit_decoder import ViTDecoder
decoder = ViTDecoder(embed_dim=2176, depth=12).to(DEVICE, dtype=torch.bfloat16)
decoder_params = sum(p.numel() for p in decoder.parameters()) # 1138243843
print(f"decoder_params: {decoder_params}") 

# pixel_values = inputs["pixel_values"]
pixel_values = torch.randn((1, 12, 224, 224)).to(DEVICE, dtype=torch.bfloat16)
# print(pixel_values[:, :3, :]==pixel_values[:, 3:6, :])
# print(pixel_values[:, 6:9, :]==pixel_values[:, 9:, :])
patch_features = encoder(pixel_values)
projected_patch_embeddings = projector(patch_features)
projected_patch_embeddings = projected_patch_embeddings.unsqueeze(1).expand(-1, 2, -1, -1)
# patch_features = patch_features.to(DEVICE, dtype=torch.bfloat16)
pixel_values_hat = decoder(projected_patch_embeddings)

print('patch_features', patch_features.shape)

print('pixel_values_hat', pixel_values_hat.shape)



def create_patch_causal_mask(seq_len, num_patches, device):
    """
    创建patch级别的因果掩码
    Args:
        seq_len: 序列长度 (T)
        num_patches: 每个位置的patch数量 (N)
    Returns:
        mask: [T*N, T*N] 的布尔掩码
    """
    # 创建块对角矩阵
    mask = torch.tril(torch.ones((seq_len, seq_len), device=device))  # [T, T] 标准下三角
    
    # 扩展为patch级别
    mask = mask.repeat_interleave(num_patches, dim=0)  # 行扩展
    mask = mask.repeat_interleave(num_patches, dim=1)  # 列扩展
    
    return mask.bool()

B, L, N, D = projected_patch_embeddings.shape
projected_patch_embeddings = projected_patch_embeddings.reshape(B, L*N, D)
attention_mask = create_patch_causal_mask(L, N, DEVICE)
attention_mask = attention_mask.unsqueeze(0).unsqueeze(1).expand(B, 1, -1, -1)
language_model_output = language_model(
    input_ids=None,
    attention_mask=attention_mask,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=projected_patch_embeddings,
    labels=None,
    use_cache=False,
    output_attentions=False,
    output_hidden_states=True,
    return_dict=True,
)

print(language_model_output)
print("Done")