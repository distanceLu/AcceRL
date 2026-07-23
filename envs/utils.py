
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
from experiments.robot.openvla_utils import get_processor

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

def load_reward_model_from_checkpoint(
    agent_cfg: Any,
    trainer_cfg: Any,
    device: torch.device,
) -> Tuple[RewardModel, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, int, Any, Any]:
    """
    加载 Reward Model 及其训练状态
    
    Args:
        agent_cfg: agent 配置
        trainer_cfg: trainer 配置
        device: 设备
    
    Returns:
        reward_model: RewardModel 模型
        reward_optimizer: 优化器
        reward_lr_scheduler: 学习率调度器
        reward_start_step: 训练起始步数（从 epoch 推算）
        processor: 数据预处理器
        reward_cfg: Reward Model 配置
    """
    torch_dtype = torch.bfloat16
    cfg = GenerateConfig(
        pretrained_checkpoint=agent_cfg.openvla_path,
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
        device=device,
        lora_rank=0,
    )

    reward_model = RewardModel(cfg, torch_dtype, keep_num=4, focal_alpha=agent_cfg.reward_model.focal_alpha)
    checkpoint = torch.load(agent_cfg.reward_model_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model"]
    reward_model.load_state_dict(state_dict, strict=True)

    reward_opt_cfg = trainer_cfg.reward_model.optimizer
    reward_optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, reward_model.parameters()),
        lr=reward_opt_cfg.lr,
    )
    
    reward_optimizer.load_state_dict(checkpoint["optimizer"])
    reward_start_step = checkpoint["epoch"]
    
    for param_group in reward_optimizer.param_groups:
        param_group['initial_lr'] = reward_opt_cfg.lr
    
    reward_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        reward_optimizer, 
        T_max=reward_opt_cfg.T_max,
        eta_min=reward_opt_cfg.min_lr,
        last_epoch=reward_start_step - 1
    )
    
    processor = get_processor(cfg)
    
    return reward_model, reward_optimizer, reward_lr_scheduler, processor, cfg, reward_start_step

def configure_opt(model: nn.Module, lr: float, weight_decay: float, eps: float, *blacklist_module_names: str) -> AdamW:
    """Credits to https://github.com/karpathy/minGPT"""
    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.LSTMCell, nn.LSTM, 
                                 nn.TransformerEncoderLayer, nn.TransformerEncoder, 
                                 nn.TransformerDecoderLayer, nn.TransformerDecoder,
                                 nn.MultiheadAttention)
    blacklist_weight_modules = (nn.LayerNorm, nn.Embedding, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d)
    module_dict = dict(model.named_modules())
    
    for pn, p in model.named_parameters():
        parts = pn.rsplit('.', 1)
        if len(parts) == 2:
            module_name, param_name = parts
            m = module_dict.get(module_name, None)
        else:
            param_name = pn
            m = model
        
        if any([pn.startswith(module_name) for module_name in blacklist_module_names]):
            no_decay.add(pn)
        elif "bias" in param_name:
            no_decay.add(pn)
        elif (param_name.endswith("weight") or param_name.startswith("weight_")) and m is not None and isinstance(m, blacklist_weight_modules):
            no_decay.add(pn)
        elif (param_name.endswith("weight") or param_name.startswith("weight_")) and m is not None and isinstance(m, whitelist_weight_modules):
            decay.add(pn)
        elif "embedding" in pn.lower() or "pos_embedding" in pn.lower():
            no_decay.add(pn)

    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    missing_params = param_dict.keys() - union_params
    
    for pn in missing_params:
        if "bias" in pn or "embedding" in pn.lower():
            no_decay.add(pn)
        elif "weight" in pn:
            decay.add(pn)
        else:
            decay.add(pn)
    
    union_params = decay | no_decay
    inter_params = decay & no_decay
    assert len(inter_params) == 0, f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
    assert (
        len(param_dict.keys() - union_params) == 0
    ), f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"

    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optim_groups, lr=lr, eps=eps)
    return optimizer

def get_lr_sched(opt: torch.optim.Optimizer, num_warmup_steps: int) -> LambdaLR:
    def lr_lambda(current_step: int):
        return 1 if current_step >= num_warmup_steps else current_step / max(1, num_warmup_steps)

    return LambdaLR(opt, lr_lambda, last_epoch=-1)


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
    state = torch.load(model_path, map_location=model.device, weights_only=False)
    if "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg

def save_checkpoint(
    save_dir,
    iteration,
    denoiser,
    denoiser_optimizer,
    denoiser_lr_scheduler,
    denoiser_step,
    reward_model,
    reward_optimizer,
    reward_lr_scheduler,
    reward_step,
    reward_cfg,
    eval_metrics,
    is_best_denoiser,
    is_best_reward,
):
    save_dir.mkdir(parents=True, exist_ok=True)

    denoiser_checkpoint = {
        "effective_step": denoiser_step,
        "denoiser_state_dict": denoiser.state_dict(),
        "optimizer_state_dict": denoiser_optimizer.state_dict(),
    }
    denoiser_checkpoint_path = save_dir / f"denoiser_checkpoint_iter_{iteration}.pt"
    torch.save(denoiser_checkpoint, denoiser_checkpoint_path)
    
    reward_checkpoint = {
        "model": reward_model.state_dict(),
        "optimizer": reward_optimizer.state_dict(),
        "cfg": reward_cfg.__dict__ if hasattr(reward_cfg, '__dict__') else reward_cfg,
        "epoch": reward_step,
        "val_loss": eval_metrics['reward_loss'],
        "val_pr_auc": eval_metrics['reward_pos_acc'],
    }
    reward_checkpoint_path = save_dir / f"reward_model_checkpoint_iter_{iteration}.pt"
    torch.save(reward_checkpoint, reward_checkpoint_path)
    
    if is_best_denoiser:
        best_checkpoint = {
            "effective_step": denoiser_step,
            "denoiser_state_dict": denoiser.state_dict(),
            "optimizer_state_dict": denoiser_optimizer.state_dict(),
        }
        best_denoiser_path = save_dir / "best_denoiser.pt"
        torch.save(best_checkpoint, best_denoiser_path)
    
    if is_best_reward:
        best_reward_checkpoint = {
            "model": reward_model.state_dict(),
            "optimizer": reward_optimizer.state_dict(),
            "cfg": reward_cfg.__dict__ if hasattr(reward_cfg, '__dict__') else reward_cfg,
            "epoch": reward_step,
            "val_loss": eval_metrics['reward_loss'],
            "val_pr_auc": eval_metrics['reward_pos_acc'],
        }
        best_reward_path = save_dir / "best_reward_model.pt"
        torch.save(best_reward_checkpoint, best_reward_path)
    
    return denoiser_checkpoint_path, reward_checkpoint_path


def manage_checkpoints(save_dir: Path, num_to_keep: int):
    denoiser_checkpoints = list(save_dir.glob("denoiser_checkpoint_iter_*.pt"))
    if len(denoiser_checkpoints) > num_to_keep:
        denoiser_checkpoints.sort(key=lambda p: p.stat().st_mtime)
        for old_checkpoint in denoiser_checkpoints[:-num_to_keep]:
            old_checkpoint.unlink()
            print(f"Delete old Denoiser checkpoint: {old_checkpoint}")
    
    reward_checkpoints = list(save_dir.glob("reward_model_checkpoint_iter_*.pt"))
    if len(reward_checkpoints) > num_to_keep:
        reward_checkpoints.sort(key=lambda p: p.stat().st_mtime)
        for old_checkpoint in reward_checkpoints[:-num_to_keep]:
            old_checkpoint.unlink()
            print(f"Delete old Reward Model checkpoint: {old_checkpoint}")