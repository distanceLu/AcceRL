from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn


NUM_ACTIONS_CHUNK = 4
ACTION_DIM = 3
ACTION_BINS = 11
OBS_SHAPE = (3, 16, 16)


@dataclass
class ModelConfig:
    obs_shape: tuple[int, int, int] = OBS_SHAPE
    hidden_dim: int = 128
    action_dim: int = ACTION_DIM
    num_actions_chunk: int = NUM_ACTIONS_CHUNK
    action_bins: int = ACTION_BINS
    obs_key: str = "image"


def _to_writable_tensor(x, dtype=torch.float32) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.to(dtype=dtype)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(np.array(x, copy=True)).to(dtype=dtype)
    return torch.tensor(x, dtype=dtype)


class FakeActorCritic(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        flat_dim = int(np.prod(self.config.obs_shape))
        hidden = self.config.hidden_dim
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(
            hidden,
            self.config.num_actions_chunk * self.config.action_dim * self.config.action_bins,
        )
        self.value_head = nn.Linear(hidden, 1)

    def prepare_inputs_batch(self, obs_batch: Iterable) -> dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        images = []
        for obs in obs_batch:
            if isinstance(obs, dict):
                obs = obs[self.config.obs_key]
            images.append(_to_writable_tensor(obs))
        image_batch = torch.stack(images, dim=0).to(device)
        return {self.config.obs_key: image_batch}

    def forward(self, inputs) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(inputs, dict):
            obs = inputs[self.config.obs_key]
        else:
            obs = inputs
        device = next(self.parameters()).device
        obs = _to_writable_tensor(obs).to(device)
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)
        hidden = self.encoder(obs)
        logits = self.policy_head(hidden).view(
            -1,
            self.config.num_actions_chunk,
            self.config.action_dim,
            self.config.action_bins,
        )
        values = self.value_head(hidden).squeeze(-1)
        return logits, values

    def post_process(
        self,
        action_logits: torch.Tensor,
        deterministic: bool | list[bool] = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(deterministic, list):
            deterministic_flags = deterministic
        else:
            deterministic_flags = [bool(deterministic)] * action_logits.shape[0]

        greedy = action_logits.argmax(dim=-1)
        flat_logits = action_logits.reshape(-1, self.config.action_bins)
        dist = torch.distributions.Categorical(logits=flat_logits)
        sampled = dist.sample().view(
            -1, self.config.num_actions_chunk, self.config.action_dim
        )

        selected = []
        for batch_idx, is_det in enumerate(deterministic_flags):
            selected.append(greedy[batch_idx] if is_det else sampled[batch_idx])
        action_tokens = torch.stack(selected, dim=0)

        action_env = action_tokens.float() / max(self.config.action_bins - 1, 1)
        action_env = action_env * 2.0 - 1.0
        return action_tokens, action_env


class FakeRewardModel(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        flat_dim = int(np.prod(self.config.obs_shape))
        hidden = max(32, self.config.hidden_dim // 2)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def prepare_inputs_batch(self, obs_batch: Iterable) -> dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        images = []
        for obs in obs_batch:
            if isinstance(obs, dict):
                obs = obs[self.config.obs_key]
            images.append(_to_writable_tensor(obs))
        return {self.config.obs_key: torch.stack(images, dim=0).to(device)}

    def forward(self, inputs) -> torch.Tensor:
        if isinstance(inputs, dict):
            obs = inputs[self.config.obs_key]
        else:
            obs = inputs
        device = next(self.parameters()).device
        obs = _to_writable_tensor(obs).to(device)
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)
        return self.net(obs)


class FakeDenoiser(nn.Module):
    """A tiny transition model that predicts the next observation."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        c, h, w = self.config.obs_shape
        self.obs_channels = c
        self.obs_height = h
        self.obs_width = w
        hidden = self.config.hidden_dim
        self.obs_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * h * w, hidden),
            nn.Tanh(),
        )
        self.act_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.config.action_dim, hidden),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden * 2, c * h * w),
            nn.Tanh(),
        )

    def forward(self, obs_batch: torch.Tensor, act_batch: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        obs_batch = _to_writable_tensor(obs_batch).to(device)
        act_batch = _to_writable_tensor(act_batch).to(device)

        if obs_batch.ndim == 4:
            obs_batch = obs_batch.unsqueeze(0)
        if act_batch.ndim == 2:
            act_batch = act_batch.unsqueeze(0)

        last_obs = obs_batch[:, -1]
        last_act = act_batch[:, -1]
        obs_feat = self.obs_proj(last_obs)
        act_feat = self.act_proj(last_act)
        fused = torch.cat([obs_feat, act_feat], dim=-1)
        next_obs = self.decoder(fused).view(
            -1, self.obs_channels, self.obs_height, self.obs_width
        )
        return next_obs
