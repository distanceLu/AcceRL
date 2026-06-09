import argparse
import json
import os
import random
import sys
import time
import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("TMPDIR", "/dev/shm")

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from torch.utils.tensorboard import SummaryWriter


class MLPActorCriticDiscrete(nn.Module):
    """
    A simple MLP-based Actor-Critic model for low-dimensional state inputs.

    The interface is compatible with ActorCritic (discrete), so it can be swapped in easily.
    """
    
    def __init__(self, torch_dtype: torch.dtype = torch.float32, 
                 state_dim: int = 39, action_dim: int = 4, hidden_dim=512, n_action_bins=256):
        super().__init__()
       
        self.model_dtype = torch_dtype
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Action discretization parameters
        self.n_action_bins = n_action_bins  # default: 256 bins
        self.hidden_dim = hidden_dim
        
        # Shared MLP encoder
        self.shared_encoder = nn.Sequential(
            nn.Linear(state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
        )
        
        # Policy head: output discrete-action logits
        # Output shape: action_dim * n_action_bins
        self.policy_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, action_dim * self.n_action_bins),
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.n_action_bins),
            nn.ReLU(),
            nn.LayerNorm(self.n_action_bins),
            nn.Linear(self.n_action_bins, 1),
        )
        
        self.to(self.device).to(dtype=self.model_dtype)
        
    def get_parameter_groups(self) -> List[Dict[str, Any]]:
        """
        Split trainable parameters into 'policy' and 'value' groups.
        """
        policy_params = []
        value_params = []
        
        # Policy includes: shared_encoder, policy_head
        policy_params.extend(list(self.shared_encoder.parameters()))
        policy_params.extend(list(self.policy_head.parameters()))
        
        # Value includes only: value_head
        value_params.extend(list(self.value_head.parameters()))
        
        return [
            {"name": "policy", "params": policy_params},
            {"name": "value", "params": value_params},
        ]
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            state: (B, state_dim) state tensor
        
        Returns:
            action_logits: (B, action_dim, n_action_bins)
            value: (B,) state-value estimate
        """
        
        # Ensure state is on the correct device and dtype
        state = state.to(self.device).to(self.model_dtype)
        
        # Shared encoder
        features = self.shared_encoder(state)  # (B, 512)
        
        # Policy head: output logits
        policy_out = self.policy_head(features)  # (B, action_dim * n_action_bins)
        B = policy_out.shape[0]
        action_logits = policy_out.view(B, self.action_dim, self.n_action_bins)
        
        # Value head
        value = self.value_head(features.detach()).squeeze(-1)  # (B,)
     
        return action_logits, value.to(torch.float32)
    
    def post_process(self, logits: torch.Tensor, deterministic: List[bool]) -> Tuple[torch.distributions.Categorical, torch.Tensor, np.ndarray]:
        """
        Post-process logits to produce discrete actions.
        
        Args:
            logits: (B, action_dim, n_action_bins)
            deterministic: List[bool] whether each sample uses deterministic policy
        
        Returns:
            dist: Categorical distribution
            action_token_ids: (B, action_dim) sampled token IDs (torch.Tensor)
            discrete_actions: (B, action_dim) discrete action values [0, n_action_bins-1] (np.ndarray)
        """
        # 1. Build distribution
        dist = torch.distributions.Categorical(logits=logits)
        
        # 2. Sample actions
        stochastic_tokens = dist.sample()
        deterministic_tokens = torch.argmax(logits, dim=-1)
        
        is_deterministic_tensor = torch.tensor(
            deterministic, dtype=torch.bool, device=logits.device
        )
        is_deterministic_tensor = is_deterministic_tensor.unsqueeze(1)
        
        action_token_ids = torch.where(
            is_deterministic_tensor, deterministic_tokens, stochastic_tokens
        )
        
        # 3. Return discrete actions (token IDs directly)
        # token_id range: [0, n_action_bins-1]
        discrete_actions = action_token_ids.cpu().numpy().astype(np.int32)  # (B, action_dim)
        
        return dist, action_token_ids, discrete_actions
    
    def prepare_inputs_batch(self, obs_list: List):
        """
        Prepare a batch input from a list of observations.
        
        Args:
            obs_list: List, each item is:
                - np.ndarray: (state_dim,) state vector
        
        Returns:
            states: (B, state_dim)
        """
        obs_list = [torch.from_numpy(obs.astype(np.float32)) for obs in obs_list]
        states = torch.stack(obs_list, dim=0)  # (B, state_dim)
        return states
    
    def save_model(self, save_path: str, epoch: int | None = None):
        """Save model weights."""
        os.makedirs(save_path, exist_ok=True)
        
        if epoch is not None:
            ckpt_path = Path(save_path) / f"mlp_actor_critic_epoch_{epoch}.pt"
        else:
            ckpt_path = Path(save_path) / "mlp_actor_critic.pt"
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'n_action_bins': self.n_action_bins,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
        }, ckpt_path)
        
        print(f"[MLPActorCritic] Model saved to: {ckpt_path}")
    
    def load_model(self, load_path: str, epoch: int | None = None):
        """Load model weights."""
        if epoch is not None:
            ckpt_path = Path(load_path) / f"mlp_actor_critic_epoch_{epoch}.pt"
        else:
            ckpt_path = Path(load_path) / "mlp_actor_critic.pt"
        
        if not ckpt_path.exists():
            print(f"[MLPActorCritic] Warning: checkpoint file not found: {ckpt_path}")
            return
        
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"[MLPActorCritic] Model loaded from {ckpt_path}")
    
    def get_norm_stats(self):
        """
        Return normalization stats (placeholder for interface compatibility).
        The MLP model does not require special normalization statistics.
        """
        # Return a simple identity normalization
        return {
            "mean": np.zeros(self.action_dim),
            "std": np.ones(self.action_dim),
            "min": np.full(self.action_dim, -1.0),
            "max": np.full(self.action_dim, 1.0),
        }


if not hasattr(gym.vector, "AutoresetMode"):
    class _CompatAutoresetMode(str, Enum):
        NEXT_STEP = "next_step"
        SAME_STEP = "same_step"
        DISABLED = "disabled"

    gym.vector.AutoresetMode = _CompatAutoresetMode


class MetaWorldWrapperDiscrete(gym.Env):
    """
    MetaWorld single-task discrete-action wrapper.

    - Externally takes discrete token actions (each dim in [0, bins-1]).
    - Internally maps them to continuous actions required by MetaWorld (each dim in [-1, 1]).
    - Observations directly use the low-dimensional state vector from the base env (typically 39D).
    """

    metadata = {"render_modes": ["rgb_array", None]}

    def __init__(
        self,
        env_name: str = "reach-v3",
        bins: int = 256,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        max_episode_steps: Optional[int] = None,
    ) -> None:
        super().__init__()
        if bins < 2:
            raise ValueError(f"bins must be >= 2, got {bins}")

        self.env_name = env_name
        self.bins = int(bins)
        self._base_seed = seed
        self._episode_steps = 0

        import metaworld  # noqa: F401

        self.env = gym.make(
            "Meta-World/MT1",
            env_name=self.env_name,
            seed=seed,
            render_mode=render_mode,
        )

        self.max_episode_steps = int(
            max_episode_steps
            if max_episode_steps is not None
            else getattr(getattr(self.env, "spec", None), "max_episode_steps", 500)
        )

        self.action_dim = int(np.prod(self.env.action_space.shape))
        self.action_space = spaces.MultiDiscrete([self.bins] * self.action_dim)

        init_obs, _ = self.env.reset(seed=seed)
        init_obs = np.asarray(init_obs, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=init_obs.shape,
            dtype=np.float32,
        )
        self._episode_steps = 0

    def _token_to_continuous(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action)
        if action.shape != (self.action_dim,):
            action = action.reshape(self.action_dim)

        if np.issubdtype(action.dtype, np.floating):
            if np.all(action >= -1.01) and np.all(action <= 1.01):
                return np.clip(action, -1.0, 1.0).astype(np.float32)

        token = np.clip(np.rint(action), 0, self.bins - 1).astype(np.float32)
        continuous = -1.0 + 2.0 * token / float(self.bins - 1)
        return continuous.astype(np.float32)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._episode_steps = 0
        obs = np.asarray(obs, dtype=np.float32)
        info = dict(info or {})
        info.setdefault("success", float(info.get("success", 0.0)))
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        continuous_action = self._token_to_continuous(action)
        obs, reward, terminated, truncated, info = self.env.step(continuous_action)
        self._episode_steps += 1

        if not truncated and self._episode_steps >= self.max_episode_steps:
            truncated = True

        obs = np.asarray(obs, dtype=np.float32)
        info = dict(info or {})
        info["success"] = float(info.get("success", 0.0))
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        if hasattr(self.env, "render"):
            return self.env.render()
        return None

    def close(self) -> None:
        if hasattr(self.env, "close"):
            self.env.close()

STATE_DIM = 39
ACTION_DIM = 4
N_ACTION_BINS = 256
METAWORLD_TASK_NAMES = [
    "assembly-v3",
    "basketball-v3",
    "bin-picking-v3",
    "box-close-v3",
    "button-press-topdown-v3",
    "button-press-topdown-wall-v3",
    "button-press-v3",
    "button-press-wall-v3",
    "coffee-button-v3",
    "coffee-pull-v3",
    "coffee-push-v3",
    "dial-turn-v3",
    "disassemble-v3",
    "door-close-v3",
    "door-lock-v3",
    "door-open-v3",
    "door-unlock-v3",
    "drawer-close-v3",
    "drawer-open-v3",
    "faucet-close-v3",
    "faucet-open-v3",
    "hammer-v3",
    "hand-insert-v3",
    "handle-press-side-v3",
    "handle-press-v3",
    "handle-pull-side-v3",
    "handle-pull-v3",
    "lever-pull-v3",
    "peg-insert-side-v3",
    "peg-unplug-side-v3",
    "pick-out-of-hole-v3",
    "pick-place-v3",
    "pick-place-wall-v3",
    "plate-slide-back-side-v3",
    "plate-slide-back-v3",
    "plate-slide-side-v3",
    "plate-slide-v3",
    "push-back-v3",
    "push-v3",
    "push-wall-v3",
    "reach-v3",
    "reach-wall-v3",
    "shelf-place-v3",
    "soccer-v3",
    "stick-pull-v3",
    "stick-push-v3",
    "sweep-into-v3",
    "sweep-v3",
    "window-close-v3",
    "window-open-v3",
]


@dataclass
class Transition:
    obs: np.ndarray
    action_token: np.ndarray
    old_logits: np.ndarray
    old_value: float
    reward: float
    done: bool
    advantage: float
    value_target: float


@dataclass
class EpisodeResult:
    episode_return: float
    episode_length: int
    success: float


class TransitionBuffer:
    def __init__(self, max_steps: int) -> None:
        self.transitions: Deque[Transition] = deque(maxlen=max_steps)

    def extend(self, transitions: List[Transition]) -> None:
        self.transitions.extend(transitions)

    def __len__(self) -> int:
        return len(self.transitions)

    def as_tensors(
        self, device: torch.device, sample_size: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.transitions:
            raise ValueError("TransitionBuffer is empty; cannot sample.")

        transitions = list(self.transitions)
        if sample_size is not None and sample_size < len(transitions):
            indices = np.random.choice(len(transitions), size=sample_size, replace=False)
            transitions = [transitions[idx] for idx in indices]

        obs = torch.tensor(
            np.stack([item.obs for item in transitions]),
            dtype=torch.float32,
            device=device,
        )
        action_token = torch.tensor(
            np.stack([item.action_token for item in transitions]),
            dtype=torch.long,
            device=device,
        )
        old_logits = torch.tensor(
            np.stack([item.old_logits for item in transitions]),
            dtype=torch.float32,
            device=device,
        )
        advantage = torch.tensor(
            np.array([item.advantage for item in transitions], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        value_target = torch.tensor(
            np.array([item.value_target for item in transitions], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        return obs, action_token, old_logits, advantage, value_target


class SingleEnvRunner:
    def __init__(
        self,
        env: MetaWorldWrapperDiscrete,
        model: MLPActorCriticDiscrete,
        base_seed: int,
        reward_scale: float,
    ) -> None:
        self.env = env
        self.model = model
        self.base_seed = base_seed
        self.reward_scale = reward_scale
        self.reset_count = 0
        self.current_obs, _ = self._reset()
        self.episode_return = 0.0
        self.episode_length = 0

    def _reset(self) -> Tuple[np.ndarray, Dict]:
        seed = self.base_seed + self.reset_count
        self.reset_count += 1
        return self.env.reset(seed=seed)

    @torch.no_grad()
    def _infer_step(self, deterministic: bool) -> Tuple[np.ndarray, np.ndarray, float]:
        inputs_batch = self.model.prepare_inputs_batch([self.current_obs])
        action_logits, value = self.model(inputs_batch)
        dist, action_tokens, _ = self.model.post_process(
            action_logits, deterministic=[deterministic]
        )
        action_token = action_tokens[0].detach().cpu().numpy().astype(np.int64)
        logits = action_logits[0].float().cpu().numpy()
        return action_token, logits, float(value[0].item())

    @torch.no_grad()
    def _bootstrap_value(self) -> float:
        inputs_batch = self.model.prepare_inputs_batch([self.current_obs])
        _, value = self.model(inputs_batch)
        return float(value[0].item())

    def collect(
        self, rollout_steps: int, gamma: float, gae_lambda: float
    ) -> Tuple[List[Transition], List[EpisodeResult]]:
        obs_list: List[np.ndarray] = []
        action_tokens: List[np.ndarray] = []
        old_logits: List[np.ndarray] = []
        old_values: List[float] = []
        rewards: List[float] = []
        dones: List[bool] = []
        episode_results: List[EpisodeResult] = []

        for _ in range(rollout_steps):
            obs_list.append(self.current_obs.copy())
            action_token, logits, old_value = self._infer_step(deterministic=False)
            next_obs, reward, terminated, truncated, info = self.env.step(action_token)
            done = bool(terminated or truncated)

            action_tokens.append(action_token)
            old_logits.append(logits.astype(np.float32))
            old_values.append(old_value)
            rewards.append(float(reward) * self.reward_scale)
            dones.append(done)

            self.episode_return += float(reward)
            self.episode_length += 1
            self.current_obs = next_obs

            if done:
                episode_results.append(
                    EpisodeResult(
                        episode_return=self.episode_return,
                        episode_length=self.episode_length,
                        success=float(info.get("success", 0.0)),
                    )
                )
                self.current_obs, _ = self._reset()
                self.episode_return = 0.0
                self.episode_length = 0

        last_value = 0.0 if dones and dones[-1] else self._bootstrap_value()
        advantages, returns = compute_gae(
            rewards=np.asarray(rewards, dtype=np.float32),
            values=np.asarray(old_values, dtype=np.float32),
            dones=np.asarray(dones, dtype=np.bool_),
            last_value=last_value,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        transitions = []
        for idx in range(len(obs_list)):
            transitions.append(
                Transition(
                    obs=obs_list[idx],
                    action_token=action_tokens[idx],
                    old_logits=old_logits[idx],
                    old_value=float(old_values[idx]),
                    reward=float(rewards[idx]),
                    done=bool(dones[idx]),
                    advantage=float(advantages[idx]),
                    value_target=float(returns[idx]),
                )
            )
        return transitions, episode_results


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but it is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for step in reversed(range(len(rewards))):
        next_value = last_value if step == len(rewards) - 1 else values[step + 1]
        next_non_terminal = 1.0 - float(dones[step])
        delta = rewards[step] + gamma * next_value * next_non_terminal - values[step]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages[step] = gae
    returns = advantages + values
    return advantages.astype(np.float32), returns.astype(np.float32)


def build_optimizer(
    model: MLPActorCriticDiscrete, policy_lr: float, value_lr: float, weight_decay: float
) -> torch.optim.Optimizer:
    param_groups = model.get_parameter_groups()
    optimizer_groups = []
    for group in param_groups:
        lr = policy_lr if group["name"] == "policy" else value_lr
        optimizer_groups.append(
            {
                "params": group["params"],
                "lr": lr,
                "weight_decay": weight_decay,
                "name": group["name"],
            }
        )
    return torch.optim.AdamW(optimizer_groups)


def get_current_lr(current_step: int, peak_lr: float, warmup_steps: int, total_steps: int, start_step: int = 0) -> float:
    if current_step < start_step: return 0.0
    effective_step = current_step - start_step
    if effective_step < warmup_steps: return peak_lr * (effective_step / warmup_steps)
    progress = (effective_step - warmup_steps) / (total_steps - start_step - warmup_steps)
    progress = min(progress, 1.0)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * cosine_decay

def update_learning_rates(
    optimizer: torch.optim.Optimizer, policy_lr: float, value_lr: float
) -> None:
    for group in optimizer.param_groups:
        if group.get("name") == "policy":
            group["lr"] = policy_lr
        elif group.get("name") == "value":
            group["lr"] = value_lr


def build_policy_prob_pairs(
    old_pi_chunks: List[np.ndarray], new_pi_chunks: List[np.ndarray]
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not old_pi_chunks or not new_pi_chunks:
        return None

    old_pi = np.concatenate(old_pi_chunks, axis=0).astype(np.float64, copy=False)
    new_pi = np.concatenate(new_pi_chunks, axis=0).astype(np.float64, copy=False)
    finite_mask = np.isfinite(old_pi) & np.isfinite(new_pi)
    old_pi = old_pi[finite_mask]
    new_pi = new_pi[finite_mask]
    if old_pi.size == 0:
        return None
    return old_pi, new_pi


def save_latest_policy_prob_pairs(
    output_path: Path,
    old_pi: np.ndarray,
    new_pi: np.ndarray,
) -> None:
    old_pi_line = ",".join(map(repr, old_pi.tolist()))
    new_pi_line = ",".join(map(repr, new_pi.tolist()))
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(f"old_pi,{old_pi_line}\n")
        file.write(f"new_pi,{new_pi_line}\n")


def run_ppo_updates(
    model: MLPActorCriticDiscrete,
    optimizer: torch.optim.Optimizer,
    buffer: TransitionBuffer,
    train_batch_size: int,
    sample_rounds: int,
    reuse_per_batch: int,
    actor_every: int,
    clip_eps: float,
    ent_coef: float,
    kl_coef: float,
    vf_coef: float,
    max_grad_norm: float,
    clip_mode: str,
    sigma_pos: float,
    sigma_neg: float,
    kernel_type: str,
) -> Tuple[Dict[str, float], Optional[Tuple[np.ndarray, np.ndarray]]]:
    device = model.device
    actual_buffer_size = len(buffer)
    if actual_buffer_size == 0:
        return {}, None
    sample_rounds = max(1, int(sample_rounds))
    reuse_per_batch = max(1, int(reuse_per_batch))
    actor_every = max(1, int(actor_every))
    num_samples = min(train_batch_size, actual_buffer_size)

    metrics = {
        "loss": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "kl_loss": [],
        "approx_kl": [],
        "clip_frac": [],
        "grad_norm": [],
        "explained_variance": [],
    }
    optimizer_steps = 0
    old_pi_chunks: List[np.ndarray] = []
    new_pi_chunks: List[np.ndarray] = []

    for sample_idx in range(sample_rounds):
        obs_t, action_token_t, old_logits_t, advantage_t, value_target_t = buffer.as_tensors(
            device, sample_size=train_batch_size
        )
        if obs_t.shape[0] == 0:
            continue

        adv_mean = advantage_t.mean()
        adv_std = advantage_t.std(unbiased=False).clamp_min(1e-8)
        batch_adv = ((advantage_t - adv_mean) / adv_std).to(torch.float32)
        batch_obs = obs_t
        batch_action = action_token_t
        batch_old_logits = old_logits_t.to(torch.float32)
        batch_value_target = value_target_t.to(torch.float32)

        with torch.no_grad():
            dist_old = torch.distributions.Categorical(logits=batch_old_logits)
            logp_old = dist_old.log_prob(batch_action)

        for reuse_idx in range(reuse_per_batch):
            action_logits, value = model(batch_obs)
            action_logits = action_logits.to(torch.float32)
            value = value.to(torch.float32)
            dist = torch.distributions.Categorical(logits=action_logits)
            logp = dist.log_prob(batch_action)
            entropy = dist.entropy().mean()

            ratio = torch.exp(logp - logp_old)
            old_pi_chunks.append(torch.exp(logp_old).detach().reshape(-1).cpu().numpy())
            new_pi_chunks.append(torch.exp(logp).detach().reshape(-1).cpu().numpy())
            adv_expanded = batch_adv.unsqueeze(-1)
            surr1 = ratio * adv_expanded

            do_actor_update = (reuse_idx % actor_every) == 0
            actor_scale = 1.0 / math.sqrt(float(reuse_idx + 1))

            if clip_mode == "gipo":
                sigma_pos_t = torch.full_like(adv_expanded, sigma_pos)
                sigma_neg_t = torch.full_like(adv_expanded, sigma_neg)
                sigma = torch.where(adv_expanded > 0, sigma_pos_t, sigma_neg_t)
                eps = 1e-9
                r_detach = ratio.clamp_min(eps).detach()
                if kernel_type == "gaussian":
                    coeff = torch.exp(-0.5 * (torch.log(r_detach) / sigma) ** 2)
                elif kernel_type == "laplacian":
                    coeff = torch.exp(-torch.abs(torch.log(r_detach) / sigma))
                elif kernel_type == "cauchy":
                    coeff = 1.0 / (1.0 + (torch.log(r_detach) / sigma) ** 2)
                else:
                    raise ValueError(f"Invalid KERNEL_TYPE: {kernel_type}")
                # coeff = torch.exp(-0.5 * (torch.log(r_detach) / sigma) ** 2)
                surr_soft = surr1 * coeff
                base_policy_loss = -torch.mean(surr_soft)
            elif clip_mode == "ppo":
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_expanded
                base_policy_loss = -torch.min(surr1, surr2).mean()
            elif clip_mode == "sapo":
                tau_pos = 1.0
                tau_neg = 2.0
                ratio_min = 1e-6
                ratio_max = 1e6
                r = ratio.clamp(ratio_min, ratio_max)

                tau_pos_t = torch.full_like(adv_expanded, tau_pos)
                tau_neg_t = torch.full_like(adv_expanded, tau_neg)
                tau = torch.where(adv_expanded > 0, tau_pos_t, tau_neg_t)

                x = tau * (r - 1.0)
                gate = torch.sigmoid(x) * (4.0 / tau)
                surr_sapo = gate * adv_expanded
                base_policy_loss = -torch.mean(surr_sapo)
            else:
                raise ValueError(f"Invalid CLIP_MODE: {clip_mode}")

            if do_actor_update:
                policy_loss = base_policy_loss * actor_scale
            else:
                policy_loss = torch.zeros((), device=device, dtype=torch.float32)

            approx_kl = (logp_old - logp).mean()
            if do_actor_update:
                kl_div_tensor = torch.distributions.kl.kl_divergence(dist_old, dist)
                kl_loss = kl_coef * torch.mean(kl_div_tensor)
                ent_loss = -ent_coef * entropy
            else:
                kl_loss = torch.zeros((), device=device, dtype=torch.float32)
                ent_loss = torch.zeros((), device=device, dtype=torch.float32)

            value_loss = vf_coef * torch.mean((value - batch_value_target) ** 2)
            loss = policy_loss + value_loss + ent_loss + kl_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer_steps += 1

            clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean()
            var_target = torch.var(batch_value_target, unbiased=False)
            if var_target < 1e-12:
                explained_variance = torch.tensor(0.0, device=device)
            else:
                explained_variance = 1.0 - torch.var(
                    batch_value_target - value, unbiased=False
                ) / (var_target + 1e-12)

            metrics["loss"].append(float(loss.item()))
            metrics["policy_loss"].append(float(policy_loss.item()))
            metrics["value_loss"].append(float(value_loss.item()))
            metrics["entropy"].append(float(entropy.item()))
            metrics["kl_loss"].append(float(kl_loss.item()))
            metrics["approx_kl"].append(float(approx_kl.item()))
            metrics["clip_frac"].append(float(clip_frac.item()))
            metrics["grad_norm"].append(float(grad_norm))
            metrics["explained_variance"].append(float(explained_variance.item()))


    result = {
        key: float(np.mean(values)) if values else 0.0 for key, values in metrics.items()
    }
    result["optimizer_steps"] = float(optimizer_steps)
    result["buffer_size"] = float(actual_buffer_size)
    result["sample_size"] = float(num_samples)
    result["utd_estimate"] = float(sample_rounds * reuse_per_batch)
    policy_prob_pairs = build_policy_prob_pairs(old_pi_chunks, new_pi_chunks)
    if policy_prob_pairs is not None:
        old_pi_np, new_pi_np = policy_prob_pairs
        ratio_np = new_pi_np / np.clip(old_pi_np, 1e-12, None)
        result["ratio_mean"] = float(np.mean(ratio_np))
        result["ratio_std"] = float(np.std(ratio_np))
        result["ratio_min"] = float(np.min(ratio_np))
        result["ratio_max"] = float(np.max(ratio_np))
    return result, policy_prob_pairs


@torch.no_grad()
def run_evaluation(
    model: MLPActorCriticDiscrete,
    task_name: str,
    seed: int,
    eval_episodes: int,
    deterministic: bool,
) -> Dict[str, float]:
    env = MetaWorldWrapperDiscrete(env_name=task_name, bins=N_ACTION_BINS)
    returns = []
    lengths = []
    successes = []
    try:
        for episode_idx in range(eval_episodes):
            obs, _ = env.reset(seed=seed + 10_000 + episode_idx)
            episode_return = 0.0
            episode_length = 0
            done = False
            last_info: Dict = {}
            while not done:
                inputs_batch = model.prepare_inputs_batch([obs])
                action_logits, _ = model(inputs_batch)
                _, action_tokens, _ = model.post_process(
                    action_logits, deterministic=[deterministic]
                )
                obs, reward, terminated, truncated, last_info = env.step(
                    action_tokens[0].cpu().numpy()
                )
                episode_return += float(reward)
                episode_length += 1
                done = bool(terminated or truncated)

            returns.append(episode_return)
            lengths.append(episode_length)
            successes.append(float(last_info.get("success", 0.0)))
    finally:
        env.close()

    return {
        "return_mean": float(np.mean(returns)) if returns else 0.0,
        "return_std": float(np.std(returns)) if returns else 0.0,
        "length_mean": float(np.mean(lengths)) if lengths else 0.0,
        "success_mean": float(np.mean(successes)) if successes else 0.0,
    }


def save_checkpoint(
    ckpt_dir: Path,
    iteration: int,
    model: MLPActorCriticDiscrete,
    optimizer: torch.optim.Optimizer,
    global_env_steps: int,
    global_update_steps: int,
    args: argparse.Namespace,
) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"simple_state_iter_{iteration}.pt"
    state = {
        "iteration": iteration,
        "global_env_steps": global_env_steps,
        "global_update_steps": global_update_steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "random_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    torch.save(state, ckpt_path)
    return ckpt_path


def load_checkpoint(
    checkpoint_path: Path,
    model: MLPActorCriticDiscrete,
    optimizer: torch.optim.Optimizer,
) -> Dict:
    checkpoint = torch.load(checkpoint_path, map_location=model.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    torch.set_rng_state(checkpoint["random_state"]["torch"])
    if checkpoint["random_state"]["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint["random_state"]["cuda"])
    np.random.set_state(checkpoint["random_state"]["numpy"])
    random.setstate(checkpoint["random_state"]["python"])
    return checkpoint


def resolve_resume_path(resume_from: str) -> Path:
    resume_path = Path(resume_from)
    if resume_path.is_file():
        return resume_path
    candidates = sorted(resume_path.glob("simple_state_iter_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No simple_state_iter_*.pt found under {resume_path}")
    return candidates[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lightweight single-machine single-process PPO training script for MetaWorld"
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default="reach-v3",
        choices=METAWORLD_TASK_NAMES,
        help=f"MetaWorld task name (all tasks): {', '.join(METAWORLD_TASK_NAMES)}",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        type=str,
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        help="CUDA visible devices",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["auto", "cpu", "cuda"],
        help="Training device",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train-iters", type=int, default=1000, help="Outer iterations")
    parser.add_argument(
        "--rollout-steps-per-iter",
        type=int,
        default=500,
        help="Number of environment transitions to collect before each update phase",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=10,
        help="Collect at least this many env steps before updating",
    )
    parser.add_argument(
        "--buffer-horizon-steps",
        type=int,
        default=20000,
        help="Keep only the latest N transitions in local buffer",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=512,
        help="Mini-batch size for PPO updates",
    )
    parser.add_argument(
        "--sample-rounds",
        type=int,
        default=10,
        help="Number of re-sampling rounds from replay buffer per update",
    )
    parser.add_argument(
        "--reuse-per-batch",
        type=int,
        default=10,
        help="Number of times to reuse each sampled batch for training",
    )
    parser.add_argument(
        "--actor-every",
        type=int,
        default=10,
        help="Run one actor update every N reuse steps (others focus on value updates)",
    )
    parser.add_argument("--policy-lr", type=float, default=3e-4, help="Policy learning rate")
    parser.add_argument("--value-lr", type=float, default=3e-3, help="Value learning rate")
    parser.add_argument("--policy-warmup-steps", type=int, default=10, help="Policy network warmup steps")
    parser.add_argument("--value-warmup-steps", type=int, default=10, help="Value network warmup steps")
    parser.add_argument("--policy-train-start-step", type=int, default=0, help="Start training policy network at step N")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lambda", type=float, default=0.95, dest="lambda_", help="GAE lambda")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--ent-coef", type=float, default=0.00, help="Entropy coefficient")
    parser.add_argument("--kl-coef", type=float, default=0.1, help="KL divergence coefficient")
    parser.add_argument("--sigma", type=float, default=1.0, help="Sigma parameter for GIPO")
    parser.add_argument("--sigma-neg-ratio", type=float, default=0.5, help="Sigma negative ratio for GIPO")
    parser.add_argument("--kernel-type", type=str, default="gaussian", choices=["gaussian", "laplacian", "cauchy"], help="Kernel type for GIPO")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value loss coefficient")
    parser.add_argument(
        "--max-grad-norm", type=float, default=1.0, help="Gradient clipping norm"
    )
    parser.add_argument("--reward-scale", type=float, default=0.001, help="Reward scaling")
    parser.add_argument(
        "--clip-mode",
        type=str,
        default="gipo",
        choices=["ppo", "sapo", "gipo"],
        help="Clipping mode (default: gipo)",
    )
    parser.add_argument(
        "--moving-avg-window",
        type=int,
        default=100,
        help="Window size for rollout statistics",
    )
    parser.add_argument(
        "--log-interval-seconds",
        type=int,
        default=10,
        help="Print/log interval in seconds",
    )
    parser.add_argument(
        "--eval-every-iters",
        type=int,
        default=1000,
        help="Run evaluation every N outer iterations",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--eval-deterministic",
        action="store_true",
        default=True,
        help="Use deterministic policy during evaluation",
    )
    parser.add_argument(
        "--no-eval-deterministic",
        action="store_false",
        dest="eval_deterministic",
        help="Use stochastic policy during evaluation",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Optional experiment name",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default=None,
        help="Checkpoint directory, default to run_dir/checkpoints",
    )
    parser.add_argument(
        "--ckpt-every-iters",
        type=int,
        default=5000000000,
        help="Save checkpoint every N outer iterations",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume from checkpoint file or directory",
    )
    parser.add_argument(
        "--use-bf16",
        action="store_true",
        default=True,
        help="Run model in bfloat16 when supported",
    )
    parser.add_argument(
        "--no-bf16",
        action="store_false",
        dest="use_bf16",
        help="Disable bfloat16",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Log directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    device = resolve_device(args.device)
    torch_dtype = (
        torch.bfloat16 if args.use_bf16 and device.type == "cuda" else torch.float32
    )
    set_seed(args.seed)

    if args.exp_name is None:
        args.exp_name = f"debug_{args.task_name.replace('-', '_')}"

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.log_dir is None:
        log_dir = Path("runs") / "MetaWorldSimple" / args.task_name / f"{args.exp_name}_{timestamp}"
    else:
        log_dir = Path(args.log_dir) / f"{args.exp_name}_{timestamp}"
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else log_dir / "checkpoints"
    policy_prob_pairs_path = log_dir / "policy_prob_pairs_latest.csv"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with open(log_dir / "args.json", "w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2, ensure_ascii=False)
    with open(ckpt_dir / "args.json", "w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2, ensure_ascii=False)

    writer = SummaryWriter(str(log_dir))
    train_env = MetaWorldWrapperDiscrete(
        env_name=args.task_name,
        bins=N_ACTION_BINS,
        seed=args.seed,
    )

    model = MLPActorCriticDiscrete(
        torch_dtype=torch_dtype,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        n_action_bins=N_ACTION_BINS,
    )
    model.device = device
    model.to(device=device, dtype=torch_dtype)
    optimizer = build_optimizer(model, args.policy_lr, args.value_lr, args.weight_decay)

    start_iter = 0
    global_env_steps = 0
    global_update_steps = 0
    if args.resume_from:
        checkpoint_path = resolve_resume_path(args.resume_from)
        checkpoint = load_checkpoint(checkpoint_path, model, optimizer)
        start_iter = int(checkpoint.get("iteration", -1)) + 1
        global_env_steps = int(checkpoint.get("global_env_steps", 0))
        global_update_steps = int(checkpoint.get("global_update_steps", 0))
        print(
            f"[Resume] loaded={checkpoint_path} start_iter={start_iter} "
            f"global_env_steps={global_env_steps} global_update_steps={global_update_steps}"
        )

    buffer = TransitionBuffer(max_steps=args.buffer_horizon_steps)
    runner = SingleEnvRunner(
        env=train_env,
        model=model,
        base_seed=args.seed,
        reward_scale=args.reward_scale,
    )
    rollout_returns: Deque[float] = deque(maxlen=args.moving_avg_window)
    rollout_lengths: Deque[int] = deque(maxlen=args.moving_avg_window)
    rollout_successes: Deque[float] = deque(maxlen=args.moving_avg_window)

    print(
        f"[Start] task={args.task_name} device={device} dtype={torch_dtype} "
        f"rollout_steps_per_iter={args.rollout_steps_per_iter} "
    )

    last_log_time = time.time()
    try:
        for iteration in range(start_iter, args.train_iters):
            t_rollout_start = time.time()
            transitions, episode_results = runner.collect(
                rollout_steps=args.rollout_steps_per_iter,
                gamma=args.gamma,
                gae_lambda=args.lambda_,
            )
            rollout_time = time.time() - t_rollout_start
            buffer.extend(transitions)
            global_env_steps += len(transitions)

            for item in episode_results:
                rollout_returns.append(item.episode_return)
                rollout_lengths.append(item.episode_length)
                rollout_successes.append(item.success)

            update_metrics: Dict[str, float] = {}
            policy_prob_pairs: Optional[Tuple[np.ndarray, np.ndarray]] = None
            t_update_start = time.time()
            if global_env_steps >= args.warmup_steps and len(buffer) >= args.train_batch_size:
                current_value_lr = get_current_lr(
                    current_step=iteration,
                    peak_lr=args.value_lr,
                    warmup_steps=args.value_warmup_steps,
                    total_steps=args.train_iters,
                )
                current_policy_lr = get_current_lr(
                    current_step=iteration,
                    peak_lr=args.policy_lr,
                    warmup_steps=args.policy_warmup_steps,
                    total_steps=args.train_iters,
                    start_step=args.policy_train_start_step,
                )
                update_learning_rates(optimizer, current_policy_lr, current_value_lr)
                update_metrics, policy_prob_pairs = run_ppo_updates(
                    model=model,
                    optimizer=optimizer,
                    buffer=buffer,
                    train_batch_size=args.train_batch_size,
                    sample_rounds=args.sample_rounds,
                    reuse_per_batch=args.reuse_per_batch,
                    actor_every=args.actor_every,
                    clip_eps=args.clip_eps,
                    ent_coef=args.ent_coef,
                    kl_coef=args.kl_coef,
                    vf_coef=args.vf_coef,
                    max_grad_norm=args.max_grad_norm,
                    clip_mode=args.clip_mode,
                    sigma_pos=args.sigma,
                    sigma_neg=args.sigma * args.sigma_neg_ratio,
                    kernel_type=args.kernel_type,
                )
                global_update_steps += int(update_metrics.get("optimizer_steps", 0.0))
                if policy_prob_pairs is not None:
                    old_pi_np, new_pi_np = policy_prob_pairs
                    if iteration % 100 == 0:
                        save_latest_policy_prob_pairs(
                            output_path=policy_prob_pairs_path,
                            old_pi=old_pi_np,
                            new_pi=new_pi_np,
                        )
            update_time = time.time() - t_update_start

            current_time = time.time()
            should_log = (
                iteration == start_iter
                or current_time - last_log_time >= args.log_interval_seconds
                or iteration == args.train_iters - 1
            )
            if should_log:
                avg_return = float(np.mean(rollout_returns)) if rollout_returns else 0.0
                avg_length = float(np.mean(rollout_lengths)) if rollout_lengths else 0.0
                avg_success = float(np.mean(rollout_successes)) if rollout_successes else 0.0
                print(
                    f"[Iter {iteration + 1}/{args.train_iters}] "
                    f"env_steps={global_env_steps} update_steps={global_update_steps} "
                    f"buffer={len(buffer)} avg_return={avg_return:.3f} "
                    f"avg_success={avg_success:.3f} avg_len={avg_length:.1f} "
                    f"rollout_time={rollout_time:.2f}s update_time={update_time:.2f}s"
                )
                last_log_time = current_time

            writer.add_scalar("System/EnvSteps", global_env_steps, iteration)
            writer.add_scalar("System/UpdateSteps", global_update_steps, iteration)
            writer.add_scalar("System/BufferSize", len(buffer), iteration)
            writer.add_scalar("Timing/RolloutTimeSec", rollout_time, iteration)
            writer.add_scalar("Timing/UpdateTimeSec", update_time, iteration)

            if rollout_returns:
                writer.add_scalar("Rollout/ReturnMean", float(np.mean(rollout_returns)), iteration)
                writer.add_scalar("Rollout/LengthMean", float(np.mean(rollout_lengths)), iteration)
                writer.add_scalar(
                    "Rollout/SuccessMean", float(np.mean(rollout_successes)), iteration
                )

            for key, value in update_metrics.items():
                writer.add_scalar(f"Train/{key}", value, iteration)
            
            if update_metrics:
                writer.add_scalar("Train/policy_lr", current_policy_lr, iteration)
                writer.add_scalar("Train/value_lr", current_value_lr, iteration)

            if args.eval_every_iters > 0 and (iteration + 1) % args.eval_every_iters == 0:
                eval_metrics = run_evaluation(
                    model=model,
                    task_name=args.task_name,
                    seed=args.seed,
                    eval_episodes=args.eval_episodes,
                    deterministic=args.eval_deterministic,
                )
                print(
                    f"[Eval] iter={iteration + 1} "
                    f"return_mean={eval_metrics['return_mean']:.3f} "
                    f"success_mean={eval_metrics['success_mean']:.3f} "
                    f"length_mean={eval_metrics['length_mean']:.1f}"
                )
                for key, value in eval_metrics.items():
                    writer.add_scalar(f"Eval/{key}", value, iteration)

            if args.ckpt_every_iters > 0 and (iteration + 1) % args.ckpt_every_iters == 0:
                ckpt_path = save_checkpoint(
                    ckpt_dir=ckpt_dir,
                    iteration=iteration,
                    model=model,
                    optimizer=optimizer,
                    global_env_steps=global_env_steps,
                    global_update_steps=global_update_steps,
                    args=args,
                )
                print(f"[Checkpoint] saved={ckpt_path}")

        final_ckpt = save_checkpoint(
            ckpt_dir=ckpt_dir,
            iteration=max(args.train_iters - 1, 0),
            model=model,
            optimizer=optimizer,
            global_env_steps=global_env_steps,
            global_update_steps=global_update_steps,
            args=args,
        )
        print(f"[Done] final_checkpoint={final_ckpt}")
    finally:
        writer.close()
        train_env.close()


if __name__ == "__main__":
    main()
