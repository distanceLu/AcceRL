from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn


NUM_ACTIONS_CHUNK = 4
ACTION_DIM = 3
ACTION_BINS = 11
DEFAULT_OBS_KEY = "state"


@dataclass
class FakeModelConfig:
    obs_dim: int = 16
    hidden_dim: int = 128
    num_actions_chunk: int = NUM_ACTIONS_CHUNK
    action_dim: int = ACTION_DIM
    action_bins: int = ACTION_BINS
    obs_key: str = DEFAULT_OBS_KEY


class FakeActorCritic(nn.Module):
    """A minimal MLP policy/value network for discrete PPO."""

    def __init__(self, config: FakeModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or FakeModelConfig()
        self.encoder = nn.Sequential(
            nn.Linear(self.config.obs_dim, self.config.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.Tanh(),
        )
        policy_out = (
            self.config.num_actions_chunk
            * self.config.action_dim
            * self.config.action_bins
        )
        self.policy_head = nn.Linear(self.config.hidden_dim, policy_out)
        self.value_head = nn.Linear(self.config.hidden_dim, 1)

    def _extract_obs_tensor(self, obs, device: torch.device | None = None) -> torch.Tensor:
        if isinstance(obs, dict):
            obs = obs[self.config.obs_key]
        if isinstance(obs, np.ndarray):
            tensor = torch.tensor(np.array(obs, copy=True), dtype=torch.float32)
        elif torch.is_tensor(obs):
            tensor = obs.float()
        else:
            tensor = torch.tensor(obs, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    def prepare_inputs_batch(self, obs_batch: Iterable) -> dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        stacked = [self._extract_obs_tensor(obs).squeeze(0) for obs in obs_batch]
        obs_tensor = torch.stack(stacked, dim=0).to(device)
        return {self.config.obs_key: obs_tensor}

    def forward(self, obs) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        if isinstance(obs, dict):
            obs_tensor = self._extract_obs_tensor(obs[self.config.obs_key], device=device)
        else:
            obs_tensor = self._extract_obs_tensor(obs, device=device)
        hidden = self.encoder(obs_tensor)
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

        flat_logits = action_logits.view(-1, self.config.action_bins)
        dist = torch.distributions.Categorical(logits=flat_logits)
        sampled = dist.sample().view(
            -1, self.config.num_actions_chunk, self.config.action_dim
        )
        greedy = action_logits.argmax(dim=-1)

        selected = []
        for batch_idx, is_det in enumerate(deterministic_flags):
            selected.append(greedy[batch_idx] if is_det else sampled[batch_idx])
        action_tokens = torch.stack(selected, dim=0)

        continuous = action_tokens.float() / max(self.config.action_bins - 1, 1)
        continuous = continuous * 2.0 - 1.0
        return action_tokens, continuous

    def get_parameter_groups(self, policy_lr: float, value_lr: float):
        return [
            {
                "name": "policy",
                "params": list(self.encoder.parameters()) + list(self.policy_head.parameters()),
                "lr": policy_lr,
            },
            {
                "name": "value",
                "params": list(self.value_head.parameters()),
                "lr": value_lr,
            },
        ]
