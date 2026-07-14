from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any, Dict, List

from .inner_model import InnerModel, InnerModelConfig
from ..utils import LossAndLogs, configure_opt, get_lr_sched
import sys
from pathlib import Path
from hydra.utils import instantiate
from omegaconf import OmegaConf


def add_dims(input: Tensor, n: int) -> Tensor:
    return input.reshape(input.shape + (1,) * (n - input.ndim))

class SimpleBatch:
    def __init__(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        mask_padding: torch.Tensor,
        rew: torch.Tensor = None,
        end: torch.Tensor = None,
        trunc: torch.Tensor = None,
        info: List[List[Dict]] = None
    ):
        self.obs = obs
        self.act = act
        self.mask_padding = mask_padding
        self.rew = rew
        self.end = end
        self.trunc = trunc
        self.info = info

    def pin_memory(self) -> "SimpleBatch":
        """Pin memory for all tensor attributes"""
        for attr in ['obs', 'act', 'mask_padding', 'rew', 'end', 'trunc']:
            val = getattr(self, attr)
            if val is not None and isinstance(val, torch.Tensor):
                setattr(self, attr, val.pin_memory())
        return self

    def to(self, device: torch.device, non_blocking: bool = False) -> 'SimpleBatch':
        """Move all tensors to device"""
        return SimpleBatch(
            obs=self.obs.to(device, non_blocking=non_blocking) if isinstance(self.obs, torch.Tensor) else self.obs,
            act=self.act.to(device, non_blocking=non_blocking) if isinstance(self.act, torch.Tensor) else self.act,
            mask_padding=self.mask_padding.to(device, non_blocking=non_blocking) if isinstance(self.mask_padding, torch.Tensor) else self.mask_padding,
            rew=self.rew.to(device, non_blocking=non_blocking) if self.rew is not None and isinstance(self.rew, torch.Tensor) else self.rew,
            end=self.end.to(device, non_blocking=non_blocking) if self.end is not None and isinstance(self.end, torch.Tensor) else self.end,
            trunc=self.trunc.to(device, non_blocking=non_blocking) if self.trunc is not None and isinstance(self.trunc, torch.Tensor) else self.trunc,
            info=self.info
        )

    def save(self, filepath: str):
        """Save batch to file"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'SimpleBatch':
        """Load batch from file"""
        batch = torch.load(filepath, map_location='cpu', weights_only=False)
        
        # Extract attributes and create new instance
        obs = getattr(batch, 'obs', None)
        act = getattr(batch, 'act', None)
        mask_padding = getattr(batch, 'mask_padding', None)
        rew = getattr(batch, 'rew', None)
        end = getattr(batch, 'end', None)
        trunc = getattr(batch, 'trunc', None)
        info = getattr(batch, 'info', None)
        
        # Create default mask_padding if missing
        if mask_padding is None and obs is not None:
            batch_size, seq_len = obs.shape[:2]
            mask_padding = torch.ones(batch_size, seq_len, dtype=torch.bool)

        return cls(
            obs=obs,
            act=act,
            mask_padding=mask_padding,
            rew=rew,
            end=end,
            trunc=trunc,
            info=info,
        )


@dataclass
class Conditioners:
    c_in: Tensor
    c_out: Tensor
    c_skip: Tensor
    c_noise: Tensor


@dataclass
class SigmaDistributionConfig:
    loc: float
    scale: float
    sigma_min: float
    sigma_max: float


@dataclass
class DenoiserConfig:
    inner_model: InnerModelConfig
    sigma_data: float
    sigma_offset_noise: float


class Denoiser(nn.Module):
    def __init__(self, cfg: DenoiserConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.inner_model = InnerModel(cfg.inner_model)
        self.sample_sigma_training = None

    @property
    def device(self) -> torch.device:
        return self.inner_model.noise_emb.weight.device

    def setup_training(self, cfg: SigmaDistributionConfig) -> None:
        assert self.sample_sigma_training is None

        def sample_sigma(n: int, device: torch.device):
            s = torch.randn(n, device=device) * cfg.scale + cfg.loc
            return s.exp().clip(cfg.sigma_min, cfg.sigma_max)

        self.sample_sigma_training = sample_sigma
    
    def apply_noise(self, x: Tensor, sigma: Tensor, sigma_offset_noise: float) -> Tensor:
        b, c, _, _ = x.shape 
        offset_noise = sigma_offset_noise * torch.randn(b, c, 1, 1, device=self.device)
        return x + offset_noise + torch.randn_like(x) * add_dims(sigma, x.ndim)

    def compute_conditioners(self, sigma: Tensor) -> Conditioners:
        sigma = (sigma**2 + self.cfg.sigma_offset_noise**2).sqrt()
        c_in = 1 / (sigma**2 + self.cfg.sigma_data**2).sqrt()
        c_skip = self.cfg.sigma_data**2 / (sigma**2 + self.cfg.sigma_data**2)
        c_out = sigma * c_skip.sqrt()
        c_noise = sigma.log() / 4
        return Conditioners(*(add_dims(c, n) for c, n in zip((c_in, c_out, c_skip, c_noise), (4, 4, 4, 1, 1))))

    def compute_model_output(self, noisy_next_obs: Tensor, obs: Tensor, act: Tensor, cs: Conditioners) -> Tensor:
        rescaled_obs = obs / self.cfg.sigma_data
        rescaled_noise = noisy_next_obs * cs.c_in
        return self.inner_model(rescaled_noise, cs.c_noise, rescaled_obs, act)
    
    @torch.no_grad()
    def wrap_model_output(self, noisy_next_obs: Tensor, model_output: Tensor, cs: Conditioners) -> Tensor:
        d = cs.c_skip * noisy_next_obs + cs.c_out * model_output
        # Quantize to {0, ..., 255}, then back to [-1, 1]
        d = d.clamp(-1, 1).add(1).div(2).mul(255).byte().div(255).mul(2).sub(1)
        return d
    
    @torch.no_grad()
    def denoise(self, noisy_next_obs: Tensor, sigma: Tensor, obs: Tensor, act: Tensor) -> Tensor:
        cs = self.compute_conditioners(sigma)
        model_output = self.compute_model_output(noisy_next_obs, obs, act, cs)
        denoised = self.wrap_model_output(noisy_next_obs, model_output, cs)
        return denoised

    def forward(self, *args, **kwargs) -> LossAndLogs:
        """
        Forward accepts either (obs, act, mask_padding) or a batch dict/object with attributes .obs, .act, .mask_padding
        """
        # Check if called as (obs, act, mask_padding)
        if len(args) == 3:
            obs, act, mask_padding = args
        elif len(args) == 1:
            batch = args[0]
            obs = batch.obs
            act = batch.act
            mask_padding = batch.mask_padding
        else:
            raise ValueError(f"Invalid arguments: {args}")
        
        # obs:[bs, 5, 3, h, w] act:[bs, 4, act_dim] n=4
        n = self.cfg.inner_model.num_steps_conditioning
        seq_length = obs.size(1) - n

        all_obs = obs.clone()
        all_act = act.clone()
        loss = 0

        for i in range(seq_length):
            obs = all_obs[:, i : n + i]
            next_obs = all_obs[:, n + i]
            act = all_act[:, i : n + i]
            mask = mask_padding[:, n + i]

            b, t, c, h, w = obs.shape
            obs = obs.reshape(b, t * c, h, w)
            sigma = self.sample_sigma_training(b, self.device)
            noisy_next_obs = self.apply_noise(next_obs, sigma, self.cfg.sigma_offset_noise)

            cs = self.compute_conditioners(sigma)
            model_output = self.compute_model_output(noisy_next_obs, obs, act, cs)

            target = (next_obs - cs.c_skip * noisy_next_obs) / cs.c_out
            loss += F.mse_loss(model_output[mask], target[mask])

            denoised = self.wrap_model_output(noisy_next_obs, model_output, cs)
            all_obs[:, n + i] = denoised

        loss /= seq_length
        return loss, {"loss_denoising": loss.detach()}

# def load_checkpoint_denoiser(checkpoint_path: Path, denoiser, optimizer, lr_scheduler, device):       
#     checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

#     state_dict = checkpoint["denoiser_state_dict"]
#     act_emb_float_key = "inner_model.act_emb_float.0.weight"
    
#     if act_emb_float_key in state_dict:
#         act_emb_float_weight = state_dict[act_emb_float_key]
#         act_dim = act_emb_float_weight.shape[1]
#         _ = denoiser.inner_model._get_act_emb_float(act_dim)
    
#     denoiser.load_state_dict(state_dict, strict=False)
#     optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
#     start_step = checkpoint["effective_step"]
#     lr_scheduler.last_epoch = -1

#     return start_step

def load_denoiser_from_checkpoint(
    agent_cfg: Any,
    trainer_cfg: Any,
    device: torch.device,
):
    """加载 Denoiser 模型"""
    denoiser_cfg = instantiate(agent_cfg.denoiser)
    if denoiser_cfg.inner_model.num_actions is None:
        denoiser_cfg.inner_model.num_actions = 256

    denoiser = Denoiser(denoiser_cfg).to(device)
    sigma_distribution_cfg = instantiate(trainer_cfg.denoiser.sigma_distribution)
    denoiser.setup_training(sigma_distribution_cfg)

    denoiser_opt_cfg = trainer_cfg.denoiser.optimizer
    optimizer = configure_opt(
        denoiser,
        lr=denoiser_opt_cfg.lr,
        weight_decay=denoiser_opt_cfg.weight_decay,
        eps=denoiser_opt_cfg.eps
    )
    lr_scheduler = get_lr_sched(optimizer, trainer_cfg.denoiser.training.lr_warmup_steps)

    checkpoint = torch.load(agent_cfg.denoiser_path, map_location=device, weights_only=False)

    state_dict = checkpoint["denoiser_state_dict"]
    act_emb_float_key = "inner_model.act_emb_float.0.weight"
    if act_emb_float_key in state_dict:
        act_emb_float_weight = state_dict[act_emb_float_key]
        act_dim = act_emb_float_weight.shape[1]
        _ = denoiser.inner_model._get_act_emb_float(act_dim)
    
    denoiser.load_state_dict(state_dict, strict=False)
    try:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    except (ValueError, KeyError) as e:
        print(f"load_denoiser_from_checkpoint: 跳过 optimizer state 加载 ({e})")
    start_step = checkpoint["effective_step"]
    lr_scheduler.last_epoch = -1
    # start_step = load_checkpoint_denoiser(agent_cfg.denoiser_path, denoiser, optimizer, lr_scheduler, device)
    
    return denoiser, optimizer, lr_scheduler, start_step