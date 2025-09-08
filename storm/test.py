

from experiments.robot.libero.run_libero_eval import GenerateConfig
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
import torch


# Instantiate config
cfg = GenerateConfig(
    pretrained_checkpoint="/cpfs01/lcx_workspace/models/openvla-7b-oft-finetuned-libero-spatial-object-goal-10/",
    use_l1_regression=True,
    use_diffusion=False,
    use_film=False,
    num_images_in_input=2,
    use_proprio=True,
    load_in_8bit=False,
    load_in_4bit=False,
    center_crop=True,
    num_open_loop_steps=8,
    unnorm_key="libero_spatial_no_noops",
)

# 1) 显式加载 Config（不会触发 auto_map 也不会写文件）
vla_cfg = OpenVLAConfig.from_pretrained(
    cfg.pretrained_checkpoint,
    trust_remote_code=True,   # 允许自定义类
)

# 2) 显式加载模型（不走 Auto*，不需要 auto_map）
vla = OpenVLAForActionPrediction.from_pretrained(
    cfg.pretrained_checkpoint,
    config=vla_cfg,
    torch_dtype=torch.bfloat16,
    load_in_8bit=cfg.load_in_8bit,
    load_in_4bit=cfg.load_in_4bit,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
# config.timm_model_ids = ['vit_large_patch14_reg4_dinov2.lvd142m', 'vit_so400m_patch14_siglip_224']

encoder = vla.vision_backbone

language_model = vla.language_model

encoder_params = sum(p.numel() for p in encoder.parameters())
language_model_params = sum(p.numel() for p in language_model.parameters())
print(f"encoder_params: {encoder_params}") # 303230976 + 427680704
print(f"language_model_params: {language_model_params}")

pixel_values = torch.randn(2, 6, 224, 224).to(torch.bfloat16)

patch_features = encoder(pixel_values)

print('patch_features', patch_features.shape)
