
from argparse import Namespace
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
import json
from pathlib import Path
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from omegaconf import OmegaConf
import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
import wandb
from rl.models.reward_model import RewardModel
from experiments.robot.libero.libero_utils import GenerateConfig
from PIL import Image
LossAndLogs = Tuple[Tensor, Dict[str, Any]]


def image_to_tensor(img, device, image_size=224):
    img_pil = Image.fromarray(img)
    img_resized = img_pil.resize((image_size, image_size), Image.BILINEAR)
    img_array = np.array(img_resized)  # shape: (image_size, image_size, 3), dtype: uint8
    img_tensor = torch.from_numpy(img_array).to(device).permute(2, 0, 1)  # (3, image_size, image_size)
    img_float = img_tensor.float() / 255.0  # [0, 1]
    img_normalized = img_float * 2.0 - 1.0  # [-1, 1]
    return img_normalized


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    if tensor.ndim == 4:    #[B, C, H, W]
        tensor = tensor[0]  
    img = tensor.cpu().clone()
    img = (img + 1) / 2  
    img = img.clamp(0, 1)
    img = img.permute(1, 2, 0)  
    img_np = (img.numpy() * 255).astype(np.uint8)
    
    return img_np


def tensor_to_image_batch(tensor: torch.Tensor) -> List[np.ndarray]:
    """
    Batch version of tensor_to_image.

    Args:
        tensor: [B, C, H, W] - batch of tensors in [-1, 1] range

    Returns:
        List[np.ndarray]: List of [H, W, C] uint8 images
    """
    B = tensor.shape[0]
    # Process all images in batch at once using vectorized operations
    img = tensor.cpu().clone()  # [B, C, H, W]
    img = (img + 1) / 2  # Normalize to [0, 1]
    img = img.clamp(0, 1)
    img = img.permute(0, 2, 3, 1)  # [B, H, W, C]
    img_np = (img.numpy() * 255).astype(np.uint8)

    # Return as list of individual images
    return [img_np[i] for i in range(B)]

@torch.no_grad()
def load_reward_model(
    model_path: str,
    device: str,
    pretrained_checkpoint: str,
    focal_alpha: float = 0.9,
) -> Tuple[RewardModel, Any]:
    """
    加载 RewardModel 及其 cfg（基本复用 rl/models/reward_model.py 中的 __main__ 逻辑）
    """
    torch_dtype = torch.bfloat16
    cfg = GenerateConfig(
        pretrained_checkpoint=pretrained_checkpoint,
        use_l1_regression=False,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=1,
        use_proprio=False,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=True,
        num_open_loop_steps=1,
        unnorm_key="libero_spatial_no_noops",
        device=torch.device(device),
        lora_rank=0,
    )

    model = RewardModel(cfg, torch_dtype, keep_num=4, focal_alpha=focal_alpha)
    state = torch.load(model_path, map_location=model.device)
    if "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg