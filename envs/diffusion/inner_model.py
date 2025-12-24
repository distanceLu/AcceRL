from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Conv3x3, FourierFeatures, GroupNorm, UNet


@dataclass
class InnerModelConfig:
    img_channels: int
    num_steps_conditioning: int
    cond_channels: int
    depths: List[int]
    channels: List[int]
    attn_depths: List[bool]
    num_actions: Optional[int] = None


class InnerModel(nn.Module):
    def __init__(self, cfg: InnerModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.noise_emb = FourierFeatures(cfg.cond_channels)
        
        # 用于 long 类型 act 的 embedding
        self.act_emb_long = nn.Sequential(
            nn.Embedding(cfg.num_actions, cfg.cond_channels // cfg.num_steps_conditioning),
            nn.Flatten(),  # b t e -> b (t e)
        )
        
        # 用于 float 类型 act 的投影层（可选，延迟初始化）
        self.act_emb_float = None
        
        self.cond_proj = nn.Sequential(
            nn.Linear(cfg.cond_channels, cfg.cond_channels),
            nn.SiLU(),
            nn.Linear(cfg.cond_channels, cfg.cond_channels),
        )
        self.conv_in = Conv3x3((cfg.num_steps_conditioning + 1) * cfg.img_channels, cfg.channels[0])

        self.unet = UNet(cfg.cond_channels, cfg.depths, cfg.channels, cfg.attn_depths)

        self.norm_out = GroupNorm(cfg.channels[0])
        self.conv_out = Conv3x3(cfg.channels[0], cfg.img_channels)
        nn.init.zeros_(self.conv_out.weight)
    
    def _get_act_emb_float(self, act_dim: int):
        """延迟初始化 float act 的投影层"""
        if self.act_emb_float is None:
            # 计算目标维度：cond_channels // num_steps_conditioning
            target_dim = self.cfg.cond_channels // self.cfg.num_steps_conditioning
            # 创建投影层：将 [b, t, act_dim] -> [b, t, target_dim] -> [b, t * target_dim]
            self.act_emb_float = nn.Sequential(
                nn.Linear(act_dim, target_dim),
                nn.Flatten(start_dim=1),  # b t e -> b (t e)
            ).to(next(self.parameters()).device)
        return self.act_emb_float

    def forward(self, noisy_next_obs: Tensor, c_noise: Tensor, obs: Tensor, act: Tensor) -> Tensor:
        # 根据 act 的类型选择不同的处理方式
        if act.dtype == torch.long or act.dtype == torch.int or act.dtype == torch.int64:
            # Long 类型：使用 embedding
            act_emb = self.act_emb_long(act)
        else:
            # Float 类型：使用线性投影
            # act 形状应该是 [b, t, act_dim] 或 [b, t]
            if act.ndim == 2:
                # [b, t] -> [b, t, 1] 假设是 1 维动作
                act = act.unsqueeze(-1)
            elif act.ndim == 3:
                # [b, t, act_dim] 已经是正确的形状
                pass
            else:
                raise ValueError(f"不支持的 act 形状: {act.shape}")
            
            act_dim = act.shape[-1]
            act_emb_float = self._get_act_emb_float(act_dim)
            act_emb = act_emb_float(act)
        
        cond = self.cond_proj(self.noise_emb(c_noise) + act_emb)
        x = self.conv_in(torch.cat((obs, noisy_next_obs), dim=1))
        x, _, _ = self.unet(x, cond)
        x = self.conv_out(F.silu(self.norm_out(x)))
        return x
