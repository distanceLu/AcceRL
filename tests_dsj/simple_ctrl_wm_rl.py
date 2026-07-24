"""Single-process AcceRL PPO with a frozen, pretrained Ctrl-World model.

The training loop mirrors ``rl/simple_diffusion_wm_rl.py``:
LIBERO collection -> world-model imagination -> PPO update.  The only model
that is not trained online is Ctrl-World; it is loaded once and used under
``torch.no_grad`` for all imagined rollouts.
"""
import os
import json
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["TMPDIR"] = "/dev/shm"

import sys
import argparse
import time
from datetime import datetime
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# ---- AcceRL imports ----
from rl.libero_env import LiberoEnvWrapper
from rl.actor_critic_model_discrete import ActorCritic
from rl.utils import prepare_inputs_batch, prepare_one_obs
from experiments.robot.libero.libero_utils import GenerateConfig, TaskSuite
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM

# ---- AcceRL sys.path 设置 ----
ACCE_RL_ROOT = Path(__file__).resolve().parent.parent
if str(ACCE_RL_ROOT) not in sys.path:
    sys.path.insert(0, str(ACCE_RL_ROOT))

from tests_dsj.ctrl_world_env_batch import CtrlWorldEnvBatch
from envs.utils import image_to_tensor, load_reward_model_from_checkpoint, tensor_to_image
from ctrl_world.config import wm_args
from ctrl_world.models.ctrl_world import CrtlWorld


def _obs_to_views(obs: Dict[str, np.ndarray], device: torch.device) -> torch.Tensor:
    """Convert a LIBERO observation to [agentview, wrist] in [-1, 1]."""
    return torch.stack(
        [
            image_to_tensor(obs["full_image"], device),
            image_to_tensor(obs["wrist_image"], device),
        ],
        dim=0,
    )


def collect_initial_obs_act_from_libero(
    env: LiberoEnvWrapper,
    actor: ActorCritic,
    num_steps_conditioning: int,
    num_trajectories: int,
    device: torch.device,
    deterministic: bool = False,
    writer: Optional[SummaryWriter] = None,
    global_step: int = 0,
    success_window: Optional[deque] = None,
    length_window: Optional[deque] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[int], List[str]]:
    """Collect frame-aligned, two-view Ctrl-World conditioning windows.

    Each sample contains one extra target frame for reward-model training.
    The first ``num_steps_conditioning`` observations and the equally sized
    frame-aligned action window initialize Ctrl-World.
    """
    obs_list, act_list, rew_list, step_counts, instructions = [], [], [], [], []
    success_window = success_window if success_window is not None else deque(maxlen=100)
    length_window = length_window if length_window is not None else deque(maxlen=100)
    batch_successes, batch_lengths = [], []
    storage_device = torch.device("cpu")

    for traj_idx in range(num_trajectories):
        obs_dict, info = env.reset()
        instruction = env.task_description

        traj_obs = [_obs_to_views(obs_dict, storage_device)]
        traj_act_raw, traj_rew = [], []
        action_queue = deque()

        step_count = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            if len(action_queue) == 0:
                inputs = prepare_one_obs(
                    actor.cfg, actor.processor, obs_dict,
                    instruction, actor.model_dtype,
                )
                inputs_batch = actor.prepare_inputs_batch([inputs])
                with torch.no_grad():
                    action_logits, _ = actor.forward(inputs_batch)
                _, _, normalized_actions = actor.post_process(action_logits, [deterministic])
                action_sequence = normalized_actions[0]
                for j in range(action_sequence.shape[0]):
                    action_queue.append(action_sequence[j])

            action_norm = action_queue.popleft()
            if isinstance(action_norm, list):
                action_norm = np.array(action_norm)
            action_env_np = actor.vla._unnormalize_actions(action_norm, actor.cfg.unnorm_key)

            obs_dict, reward, terminated, truncated, info = env.step(action_env_np)
            step_count += 1

            traj_obs.append(_obs_to_views(obs_dict, storage_device))
            traj_act_raw.append(
                torch.as_tensor(action_env_np, dtype=torch.float32, device=storage_device)
            )
            traj_rew.append(
                torch.tensor(reward, dtype=torch.float32, device=storage_device)
            )

        T = len(traj_obs)
        is_success = float(terminated)
        batch_successes.append(is_success)
        batch_lengths.append(step_count)
        success_window.append(is_success)
        length_window.append(step_count)

        if T < num_steps_conditioning + 1:
            print(f"  轨迹 {traj_idx + 1}: T={T}，不足以构造窗口，跳过")
            continue

        num_valid_windows = T - num_steps_conditioning
        for window_idx in range(num_valid_windows):
            current_index = window_idx + num_steps_conditioning - 1
            window_obs = torch.stack(
                traj_obs[
                    window_idx : window_idx + num_steps_conditioning + 1
                ],
                dim=0,
            )
            first_action = (
                traj_act_raw[window_idx - 1]
                if window_idx > 0
                else torch.zeros(
                    ACTION_DIM, dtype=torch.float32, device=storage_device
                )
            )
            remaining_actions = traj_act_raw[
                window_idx : window_idx + num_steps_conditioning - 1
            ]
            window_act = torch.stack([first_action, *remaining_actions], dim=0)
            window_rew = traj_rew[current_index]

            obs_list.append(window_obs)
            act_list.append(window_act)
            rew_list.append(window_rew)
            step_counts.append(current_index)
            instructions.append(instruction)

        print(f"  轨迹 {traj_idx+1}/{num_trajectories}: T={T}, steps={step_count}, "
              f"windows={num_valid_windows}, success={terminated}")

    if writer is not None and batch_successes:
        writer.add_scalar("collect/batch_success_rate", np.mean(batch_successes), global_step)
        writer.add_scalar("collect/batch_avg_trajectory_length", np.mean(batch_lengths), global_step)
        writer.add_scalar("collect/window_success_rate", np.mean(success_window), global_step)
        writer.add_scalar("collect/window_avg_trajectory_length", np.mean(length_window), global_step)

    return obs_list, act_list, rew_list, step_counts, instructions


@torch.no_grad()
def _predict_reward_end(
    env_batch: CtrlWorldEnvBatch,
    observations: torch.Tensor,
    instructions: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Predict success probabilities from agent-view images."""
    batch_size = observations.shape[0]
    if env_batch.reward_model is None:
        return (
            torch.zeros(batch_size, device=env_batch.device),
            torch.zeros(batch_size, dtype=torch.long, device=env_batch.device),
        )

    inputs_list = []
    for index in range(batch_size):
        reward_obs = {
            "full_image": tensor_to_image(observations[index, 0].float())
        }
        inputs_list.append(
            prepare_one_obs(
                env_batch.reward_cfg,
                env_batch.processor,
                reward_obs,
                instructions[index],
                env_batch.torch_dtype,
            )
        )
    batch_inputs = prepare_inputs_batch(env_batch.reward_model, inputs_list)
    logits = env_batch.reward_model.forward(batch_inputs)
    probabilities = torch.softmax(logits, dim=-1)
    return probabilities[:, 1], logits.argmax(dim=-1)


def _policy_observation(observation: torch.Tensor) -> Dict[str, np.ndarray]:
    """Convert [M,C,H,W] in [-1,1] to the VLA observation dictionary."""
    result = {"full_image": tensor_to_image(observation[0].float())}
    if observation.shape[0] > 1:
        result["wrist_image"] = tensor_to_image(observation[1].float())
    return result


def rollout_with_ctrl_world(
    env_batch: CtrlWorldEnvBatch,
    actor: ActorCritic,
    initial_obs: torch.Tensor,
    initial_act: torch.Tensor,
    instructions: List[str],
    initial_step_counts: Optional[torch.Tensor] = None,
    max_steps: int = 8,
    deterministic: bool = False,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Dict[str, Any]:
    """Run PPO imagination with Ctrl-World's frame-aligned chunk protocol."""
    assert max_steps % NUM_ACTIONS_CHUNK == 0
    stride = env_batch.num_frames_pred - 1
    if NUM_ACTIONS_CHUNK % stride != 0:
        raise ValueError(
            f"NUM_ACTIONS_CHUNK={NUM_ACTIONS_CHUNK} must be divisible by "
            f"Ctrl-World stride={stride}"
        )
    if initial_obs.ndim != 6:
        raise ValueError(
            f"Expected initial observations [B,T,M,C,H,W], got {tuple(initial_obs.shape)}"
        )
    if initial_act.shape[1:] != (env_batch.num_history + 1, ACTION_DIM):
        raise ValueError(
            "Initial actions must be frame-aligned with history plus current: "
            f"expected [B,{env_batch.num_history + 1},{ACTION_DIM}], "
            f"got {tuple(initial_act.shape)}"
        )

    device = env_batch.device
    horizon = env_batch.horizon
    initial_obs = initial_obs.to(device)
    initial_act = initial_act.to(device)
    if initial_step_counts is not None:
        initial_step_counts = initial_step_counts.to(device)
        valid_mask = initial_step_counts < horizon
        valid_indices = torch.where(valid_mask)[0]
        initial_obs = initial_obs[valid_indices]
        initial_act = initial_act[valid_indices]
        instructions = [instructions[i] for i in valid_indices.cpu().tolist()]
        initial_step_counts = initial_step_counts[valid_indices]
    batch_size = initial_obs.shape[0]
    if batch_size == 0:
        return _empty_rollout_result(device)

    latent_history, current_latent = env_batch.init_latent_state(initial_obs)
    current_obs = initial_obs[:, -1]
    action_window = initial_act.clone()
    ep_len = (
        initial_step_counts.clone()
        if initial_step_counts is not None
        else torch.full(
            (batch_size,), env_batch.num_history, dtype=torch.long, device=device
        )
    )
    alive_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    last_success_prob, _ = _predict_reward_end(env_batch, current_obs, instructions)

    obs_list, act_logits_list, act_tokens_list = [], [], []
    rew_list, val_list, end_list, trunc_list, mask_list = [], [], [], [], []
    for _ in range(max_steps // NUM_ACTIONS_CHUNK):
        if not alive_mask.any():
            break
        chunk_start_alive = alive_mask.clone()
        inputs_list = [
            prepare_one_obs(
                actor.cfg,
                actor.processor,
                _policy_observation(current_obs[index]),
                instructions[index],
                actor.model_dtype,
            )
            for index in range(batch_size)
        ]
        inputs_batch = actor.prepare_inputs_batch(inputs_list)
        with torch.no_grad():
            action_logits, values = actor.forward(inputs_batch)
        _, action_token_ids, normalized_actions = actor.post_process(
            action_logits, [deterministic] * batch_size
        )
        raw_actions = np.stack(
            [
                actor.vla._unnormalize_actions(actions, actor.cfg.unnorm_key)
                for actions in normalized_actions
            ]
        )
        raw_actions = torch.as_tensor(raw_actions, dtype=torch.float32, device=device)
        raw_actions = raw_actions * chunk_start_alive[:, None, None]

        obs_list.append(current_obs[:, 0].clone())
        val_list.append(values)
        act_logits_list.append(action_logits)
        act_tokens_list.append(action_token_ids)
        chunk_rewards = torch.zeros(batch_size, device=device)
        chunk_ended = torch.zeros(batch_size, dtype=torch.long, device=device)
        chunk_truncated = torch.zeros(batch_size, dtype=torch.long, device=device)

        for offset in range(0, NUM_ACTIONS_CHUNK, stride):
            future_actions = raw_actions[:, offset : offset + stride]
            action_condition = torch.cat(
                [action_window[:, :-1], action_window[:, -1:], future_actions], dim=1
            )
            future_obs, future_latents = env_batch.predict_chunk_stateless(
                current_latent=current_latent,
                latent_history=latent_history,
                action_condition=action_condition,
                instructions=instructions,
                output_size=(current_obs.shape[-2], current_obs.shape[-1]),
            )

            for relative_index in range(stride):
                old_current = current_latent
                current_latent = future_latents[:, relative_index]
                latent_history = env_batch.update_latent_history(
                    latent_history, old_current
                )
                current_obs = future_obs[:, relative_index]
                action_window = torch.cat(
                    [
                        action_window[:, 1:],
                        future_actions[:, relative_index : relative_index + 1],
                    ],
                    dim=1,
                )

                alive_before = alive_mask.clone()
                success_prob, end_pred = _predict_reward_end(
                    env_batch, current_obs, instructions
                )
                success_prob = torch.where(
                    alive_before, success_prob, last_success_prob
                )
                end_pred = end_pred.long() * alive_before.long()
                reward = (success_prob - last_success_prob) * alive_before.float()
                last_success_prob = success_prob
                chunk_rewards += reward

                ep_len += alive_before.long()
                trunc = (ep_len >= horizon).long() * alive_before.long()
                chunk_ended = torch.maximum(chunk_ended, end_pred)
                chunk_truncated = torch.maximum(chunk_truncated, trunc)
                alive_mask &= ~(end_pred.bool() | trunc.bool())

        rew_list.append(chunk_rewards)
        end_list.append(chunk_ended)
        trunc_list.append(chunk_truncated)
        mask_list.append(chunk_start_alive)

    if not obs_list:
        return _empty_rollout_result(device)

    obs_tensor = torch.stack(obs_list, dim=1)
    act_logits_tensor = torch.stack(act_logits_list, dim=1)
    act_tokens_tensor = torch.stack(act_tokens_list, dim=1)
    rew_tensor = torch.stack(rew_list, dim=1)
    val_tensor = torch.stack(val_list, dim=1)
    end_tensor = torch.stack(end_list, dim=1)
    trunc_tensor = torch.stack(trunc_list, dim=1)
    mask_tensor = torch.stack(mask_list, dim=1)

    # 计算 GAE
    next_values = torch.cat(
        [val_tensor[:, 1:], torch.zeros(batch_size, 1, device=device)], dim=1
    )
    dones = (end_tensor | trunc_tensor).float()
    next_values = next_values * (1 - dones)

    advantages, returns = compute_gae(rew_tensor, val_tensor, next_values, dones, gamma, gae_lambda)

    return {
        "obs": obs_tensor,
        "act_logits": act_logits_tensor,
        "act_tokens": act_tokens_tensor,
        "rew": rew_tensor,
        "val": val_tensor,
        "end": end_tensor,
        "trunc": trunc_tensor,
        "mask": mask_tensor,
        "advantages": advantages,
        "returns": returns,
        "instructions": instructions,
    }


def _empty_rollout_result(device):
    return {
        "obs": torch.zeros(0, 0, 3, 224, 224, device=device),
        "act_logits": torch.zeros(0, 0, 8, 1, device=device),
        "act_tokens": torch.zeros(0, 0, 8, dtype=torch.long, device=device),
        "rew": torch.zeros(0, 0, device=device),
        "val": torch.zeros(0, 0, device=device),
        "end": torch.zeros(0, 0, dtype=torch.long, device=device),
        "trunc": torch.zeros(0, 0, dtype=torch.long, device=device),
        "mask": torch.zeros(0, 0, dtype=torch.bool, device=device),
        "advantages": torch.zeros(0, 0, device=device),
        "returns": torch.zeros(0, 0, device=device),
        "instructions": [],
    }


def rollout_with_ctrl_world_batched(
    env_batch: CtrlWorldEnvBatch,
    actor: ActorCritic,
    initial_obs: torch.Tensor,
    initial_act: torch.Tensor,
    instructions: List[str],
    initial_step_counts: torch.Tensor,
    max_steps: int,
    deterministic: bool,
    batch_size: int,
) -> Dict[str, Any]:
    """Run Ctrl-World in micro-batches and concatenate PPO trajectories."""
    results = []
    for start in range(0, initial_obs.shape[0], batch_size):
        end = min(start + batch_size, initial_obs.shape[0])
        result = rollout_with_ctrl_world(
            env_batch=env_batch,
            actor=actor,
            initial_obs=initial_obs[start:end],
            initial_act=initial_act[start:end],
            instructions=instructions[start:end],
            initial_step_counts=initial_step_counts[start:end],
            max_steps=max_steps,
            deterministic=deterministic,
        )
        if result["obs"].shape[0] > 0:
            results.append(result)
    if not results:
        return _empty_rollout_result(env_batch.device)
    tensor_keys = (
        "obs", "act_logits", "act_tokens", "rew", "val", "end", "trunc",
        "mask", "advantages", "returns",
    )
    max_time = max(result["obs"].shape[1] for result in results)
    padded_results = []
    for result in results:
        padded = dict(result)
        missing = max_time - result["obs"].shape[1]
        if missing > 0:
            for key in tensor_keys:
                value = result[key]
                if key == "obs":
                    padding = value[:, -1:].expand(
                        value.shape[0], missing, *value.shape[2:]
                    )
                else:
                    padding = torch.zeros(
                        value.shape[0],
                        missing,
                        *value.shape[2:],
                        dtype=value.dtype,
                        device=value.device,
                    )
                padded[key] = torch.cat([value, padding], dim=1)
        padded_results.append(padded)
    merged = {
        key: torch.cat([result[key] for result in padded_results], dim=0)
        for key in tensor_keys
    }
    merged["instructions"] = [
        instruction
        for result in results
        for instruction in result["instructions"]
    ]
    return merged


def compute_gae(rewards, values, next_values, dones, gamma=0.99, gae_lambda=0.95):
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    deltas = rewards + gamma * next_values * (1 - dones) - values
    gae = torch.zeros(B, device=rewards.device)
    for t in reversed(range(T)):
        gae = deltas[:, t] + gamma * gae_lambda * (1 - dones[:, t]) * gae
        advantages[:, t] = gae
        returns[:, t] = advantages[:, t] + values[:, t]
    return advantages, returns


# ================================================================
# 3. PPO 更新（与 simple_diffusion_wm_rl.py 一致，简化版）
# ================================================================
def ppo_update(
    actor_critic: ActorCritic,
    rollout_data: Dict[str, torch.Tensor],
    optimizer: optim.Optimizer,
    clip_ratio: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    num_epochs: int = 1,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 1,
) -> Dict[str, float]:
    obs = rollout_data["obs"]
    old_action_logits = rollout_data["act_logits"]
    old_action_tokens = rollout_data["act_tokens"]
    advantages = rollout_data["advantages"]
    returns = rollout_data["returns"]
    old_values = rollout_data["val"]
    mask = rollout_data["mask"]
    instructions = rollout_data.get("instructions", [""] * obs.shape[0])

    B, T_ac, C, H, W = obs.shape
    device = actor_critic.device
    print(f"  PPO: batch={obs.shape}, valid_steps={mask.sum().item()}")

    if B == 0 or T_ac == 0:
        return {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy_loss": 0.0,
            "total_loss": 0.0,
            "mean_reward": 0.0,
            "mean_episode_length": 0.0,
        }

    obs = obs.to(device)
    old_action_logits = old_action_logits.to(device)
    old_action_tokens = old_action_tokens.to(device)
    advantages = advantages.to(device)
    returns = returns.to(device)
    old_values = old_values.to(device)

    mask_flat = mask.reshape(-1)
    valid_indices = torch.where(mask_flat)[0].to(device)
    if valid_indices.numel() == 0:
        return {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy_loss": 0.0,
            "total_loss": 0.0,
            "mean_reward": 0.0,
            "mean_episode_length": 0.0,
        }

    obs_flat = obs.reshape(B * T_ac, C, H, W)[valid_indices]
    advantages_flat = advantages.reshape(B * T_ac)[valid_indices]
    returns_flat = returns.reshape(B * T_ac)[valid_indices]
    old_action_tokens_flat = old_action_tokens.reshape(B * T_ac, -1)[valid_indices]
    old_logits_flat = old_action_logits.reshape(B * T_ac, *old_action_logits.shape[2:])[valid_indices]

    advantages_flat = (advantages_flat - advantages_flat.mean()) / (
        advantages_flat.std(unbiased=False) + 1e-8
    )

    instructions_expanded = []
    for i in range(B):
        instructions_expanded.extend([instructions[i]] * T_ac)

    total_policy_loss, total_value_loss, total_entropy_loss, total_loss = 0, 0, 0, 0
    num_forward = 0

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        N_valid = len(obs_flat)
        indices = torch.randperm(N_valid, device=device)

        for start_idx in range(0, N_valid, batch_size):
            end_idx = min(start_idx + batch_size, N_valid)
            batch_indices = indices[start_idx:end_idx]

            batch_obs = obs_flat[batch_indices]
            batch_advantages = advantages_flat[batch_indices]
            batch_returns = returns_flat[batch_indices]
            batch_old_action_tokens = old_action_tokens_flat[batch_indices]
            batch_old_logits = old_logits_flat[batch_indices]
            batch_original_indices = valid_indices[batch_indices]
            batch_instructions = [instructions_expanded[i] for i in batch_original_indices.cpu().tolist()]

            inputs_list = []
            for i in range(len(batch_obs)):
                obs_tensor = batch_obs[i]
                obs_np = obs_tensor.cpu().numpy().transpose(1, 2, 0)
                obs_img = ((obs_np + 1) / 2 * 255).astype(np.uint8)
                obs_dict = {"full_image": obs_img}
                inputs = prepare_one_obs(
                    actor_critic.cfg, actor_critic.processor, obs_dict,
                    batch_instructions[i], actor_critic.model_dtype,
                )
                inputs_list.append(inputs)

            inputs_batch = actor_critic.prepare_inputs_batch(inputs_list)
            action_logits, values = actor_critic.forward(inputs_batch)

            new_dist = torch.distributions.Categorical(logits=action_logits)
            old_dist = torch.distributions.Categorical(logits=batch_old_logits)
            old_log_probs = old_dist.log_prob(batch_old_action_tokens)
            new_log_probs = new_dist.log_prob(batch_old_action_tokens)

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * batch_advantages.unsqueeze(-1)
            surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * batch_advantages.unsqueeze(-1)
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values, batch_returns)
            entropy_loss = -new_dist.entropy().mean()

            loss = (policy_loss + value_coef * value_loss + entropy_coef * entropy_loss) / gradient_accumulation_steps
            loss.backward()
            num_forward += 1

            is_last_batch = (start_idx + batch_size >= N_valid)
            if num_forward % gradient_accumulation_steps == 0 or is_last_batch:
                torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()
            total_loss += loss.item() * gradient_accumulation_steps

    n = max(num_forward, 1)
    return {
        "policy_loss": total_policy_loss / n,
        "value_loss": total_value_loss / n,
        "entropy_loss": total_entropy_loss / n,
        "total_loss": total_loss / n,
        "mean_reward": rollout_data["rew"].sum(dim=1).mean().item(),
        "mean_episode_length": float(T_ac),
    }


def _load_condition_bounds(
    checkpoint_path: str,
    condition_stat_path: Optional[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    path = Path(condition_stat_path) if condition_stat_path else (
        Path(checkpoint_path).resolve().parent / "condition_stat.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Ctrl-World condition statistics not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        stats = json.load(file)
    if stats.get("condition_mode") != "action":
        raise ValueError(f"Expected action condition stats, got {stats.get('condition_mode')!r}")
    if stats.get("normalize_condition") != "bounds":
        raise ValueError("Ctrl-World checkpoint must use bounds-normalized actions")
    if (
        set(stats.get("skip_choices", [])) != {1}
        or int(stats.get("skip_his_multiplier", -1)) != 1
        or float(stats.get("skip_his_zero_prob", -1.0)) != 0.0
    ):
        raise ValueError("Ctrl-World action statistics are not frame-aligned")
    low = torch.tensor(stats["condition_p01"], dtype=torch.float32)
    high = torch.tensor(stats["condition_p99"], dtype=torch.float32)
    if low.shape != (ACTION_DIM,) or high.shape != (ACTION_DIM,):
        raise ValueError(
            f"Expected {ACTION_DIM}-D action bounds, got {tuple(low.shape)} and {tuple(high.shape)}"
        )
    if not torch.all(high > low):
        raise ValueError("Every Ctrl-World p99 action bound must exceed p01")
    print(f"  Ctrl-World action statistics: {path}")
    return low, high


def train_reward_model_only(
    obs: torch.Tensor,
    rew: torch.Tensor,
    instructions: List[str],
    reward_model: Any,
    reward_optimizer: optim.Optimizer,
    reward_lr_scheduler: Any,
    reward_cfg: Any,
    processor: Any,
    trainer_cfg: Any,
    reward_step: int,
    writer: SummaryWriter,
) -> int:
    """Keep the reference reward-model update while leaving Ctrl-World frozen."""
    if reward_model is None:
        return reward_step
    device = reward_model.device
    last_agentview = obs[:, -1, 0]
    inputs_list = []
    for index in range(last_agentview.shape[0]):
        reward_obs = {
            "full_image": tensor_to_image(last_agentview[index].float())
        }
        inputs_list.append(
            prepare_one_obs(
                reward_cfg,
                processor,
                reward_obs,
                instructions[index],
                reward_model.model_dtype,
            )
        )
    inputs_batch = prepare_inputs_batch(reward_model, inputs_list)
    labels = (rew.to(device) > 0).long()
    batch_size = int(trainer_cfg.trainer.batch_size)
    grad_accum = int(trainer_cfg.trainer.grad_accum)

    reward_model.train()
    reward_optimizer.zero_grad()
    total_loss = 0.0
    num_batches = (labels.shape[0] + batch_size - 1) // batch_size
    for batch_index, start in enumerate(range(0, labels.shape[0], batch_size)):
        end = min(start + batch_size, labels.shape[0])
        batch_inputs = {key: value[start:end] for key, value in inputs_batch.items()}
        loss, _ = reward_model.compute_loss_and_metrics(batch_inputs, labels[start:end])
        (loss / grad_accum).backward()
        total_loss += loss.item()
        if (batch_index + 1) % grad_accum == 0 or end == labels.shape[0]:
            torch.nn.utils.clip_grad_norm_(
                reward_model.parameters(),
                trainer_cfg.reward_model.training.clip_grad_norm,
            )
            reward_optimizer.step()
            reward_lr_scheduler.step()
            reward_optimizer.zero_grad()
    average_loss = total_loss / max(num_batches, 1)
    writer.add_scalar("train/reward_model_loss", average_loss, reward_step)
    writer.add_scalar(
        "train/reward_model_lr", reward_optimizer.param_groups[0]["lr"], reward_step
    )
    reward_model.eval()
    return reward_step + 1


def main():
    parser = argparse.ArgumentParser(
        description="AcceRL PPO using a frozen pretrained Ctrl-World"
    )
    parser.add_argument(
        "--svd-model-path",
        default="/mnt/data/lcx3/checkpoint/ctrl_world/svd/svd_model",
    )
    parser.add_argument(
        "--clip-model-path",
        default="/mnt/data/lcx3/checkpoint/ctrl_world/clip/clip_model",
    )
    parser.add_argument(
        "--ctrl-world-ckpt",
        default=(
            "/mnt/data/lcx3/Ctrl-World/model_ckpt/libero_vla_delta_finetune/"
            "2026-07-21T16-40-56_libero_vla_delta_finetune/checkpoint-20000.pt"
        ),
        help="Pretrained Ctrl-World checkpoint; it is never optimized by this script",
    )
    parser.add_argument("--condition-stat-path", default=None)
    parser.add_argument("--agent-config", type=str,
                        default=str(ACCE_RL_ROOT / "envs/config/agent.yaml"),
                        help="Path to agent.yaml")
    parser.add_argument("--trainer-config", type=str,
                        default=str(ACCE_RL_ROOT / "envs/config/trainer.yaml"))
    parser.add_argument("--reward-ckpt", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-cams", type=int, default=2)
    parser.add_argument("--num-history", type=int, default=6)
    parser.add_argument("--num-frames-pred", type=int, default=5)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--target-height", type=int, default=192)
    parser.add_argument("--target-width", type=int, default=320)
    parser.add_argument("--num-iterations", type=int, default=1000)
    parser.add_argument("--num-trajectories", type=int, default=5)
    parser.add_argument("--num-samples-for-rollout", type=int, default=1024)
    parser.add_argument("--rollout-max-steps", type=int, default=8)
    parser.add_argument("--rollout-batch-size", type=int, default=8)
    parser.add_argument("--ppo-batch-size", type=int, default=8)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--ppo-lr", type=float, default=3e-6)
    parser.add_argument("--gradient-accumulation", type=int, default=64)
    parser.add_argument("--benchmark", type=str, default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--no-reward-model", action="store_true")
    parser.add_argument("--no-train-reward-model", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("Ctrl-World + VLA PPO Single-Process Test")
    print("=" * 80)

    log_dir = ACCE_RL_ROOT / "runs/simple_ctrl_wm_rl" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard: {log_dir}")

    print("\n[1/4] Loading Ctrl-World model...")
    cfg = wm_args()
    cfg.svd_model_path = args.svd_model_path
    cfg.clip_model_path = args.clip_model_path
    cfg.num_cams = args.num_cams
    cfg.num_history = args.num_history
    cfg.num_frames = args.num_frames_pred
    cfg.action_dim = ACTION_DIM
    cfg.text_cond = True
    cfg.frame_level_cond = True
    cfg.his_cond_zero = False
    cfg.height = args.target_height
    cfg.width = args.target_width
    cfg.num_inference_steps = args.num_inference_steps

    ctrl_world = CrtlWorld(cfg).to(device).to(torch.bfloat16)
    if not Path(args.ctrl_world_ckpt).is_file():
        raise FileNotFoundError(f"Ctrl-World checkpoint not found: {args.ctrl_world_ckpt}")
    checkpoint = torch.load(args.ctrl_world_ckpt, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    state_dict = {
        key.removeprefix("module."): value for key, value in state_dict.items()
    }
    ctrl_world.load_state_dict(state_dict, strict=True)
    ctrl_world.eval().requires_grad_(False)
    condition_low, condition_high = _load_condition_bounds(
        args.ctrl_world_ckpt, args.condition_stat_path
    )
    print("  Frozen Ctrl-World checkpoint loaded.")

    print("\n[2/4] Loading ActorCritic (OpenVLA-OFT discrete)...")
    from omegaconf import OmegaConf
    agent_cfg = OmegaConf.load(args.agent_config)
    trainer_cfg = OmegaConf.load(args.trainer_config)

    actor_cfg = GenerateConfig(
        pretrained_checkpoint=agent_cfg.openvla_path,
        use_l1_regression=False,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=1,
        use_proprio=False,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=True,
        num_open_loop_steps=NUM_ACTIONS_CHUNK,
        unnorm_key=f"{args.benchmark}_no_noops",
        device=device,
        checkpoint2=agent_cfg.checkpoint2_path,
    )
    actor = ActorCritic(actor_cfg, torch.bfloat16)
    actor.train()
    optimizer = optim.Adam(actor.parameters(), lr=args.ppo_lr)
    print("  ActorCritic loaded.")

    print("\n[3/4] Loading Reward Model...")
    reward_model = None
    reward_optimizer = None
    reward_lr_scheduler = None
    reward_cfg = None
    processor = None
    reward_step = 0
    torch_dtype = torch.bfloat16

    if not args.no_reward_model:
        if args.reward_ckpt:
            agent_cfg.reward_model_path = args.reward_ckpt
        (
            reward_model,
            reward_optimizer,
            reward_lr_scheduler,
            processor,
            reward_cfg,
            reward_step,
        ) = load_reward_model_from_checkpoint(
            agent_cfg, trainer_cfg, device
        )
        reward_model.eval()
        print("  Reward model loaded.")
    else:
        print("  Skipping reward model (dummy rewards will be used).")

    print("\n[4/4] Loading LIBERO environment...")
    libero_env = LiberoEnvWrapper(
        benchmark_name=args.benchmark,
        task_id=args.task_id,
        image_size=224,
        render_mode="rgb_array",
    )

    @dataclass
    class CtrlWorldRolloutConfig:
        horizon: int

    env_batch = CtrlWorldEnvBatch(
        ctrl_world_model=ctrl_world,
        cfg=CtrlWorldRolloutConfig(horizon=int(trainer_cfg.world_model_env.horizon)),
        reward_model=reward_model,
        reward_cfg=reward_cfg,
        processor=processor,
        torch_dtype=torch_dtype,
        instructions=None,
        num_cams=args.num_cams,
        target_height=args.target_height,
        target_width=args.target_width,
        num_frames_pred=args.num_frames_pred,
        num_inference_steps=args.num_inference_steps,
        condition_low=condition_low,
        condition_high=condition_high,
    )
    num_steps_conditioning = args.num_history + 1

    print(f"\n开始训练，共 {args.num_iterations} 次迭代")
    print(f"  num_steps_conditioning={num_steps_conditioning}")
    print(f"  rollout_max_steps={args.rollout_max_steps}")
    print(f"  num_cams={args.num_cams}, inference_steps={args.num_inference_steps}")

    obs_buffer = deque(maxlen=50000)
    act_buffer = deque(maxlen=50000)
    rew_buffer = deque(maxlen=50000)
    step_count_buffer = deque(maxlen=50000)
    instr_buffer = deque(maxlen=50000)
    success_window = deque(maxlen=100)
    length_window = deque(maxlen=100)

    for iteration in range(args.num_iterations):
        print(f"\n=== 迭代 {iteration+1}/{args.num_iterations} ===")
        t_start = time.time()

        if (
            len(obs_buffer) == 0
            or iteration % 5 == 0
            or len(obs_buffer) < args.num_samples_for_rollout
        ):
            print(f"  采集 {args.num_trajectories} 条轨迹...")
            obs_t, act_t, rew_t, sc_t, instr_t = collect_initial_obs_act_from_libero(
                env=libero_env, actor=actor,
                num_steps_conditioning=num_steps_conditioning,
                num_trajectories=args.num_trajectories,
                device=device, deterministic=False,
                writer=writer,
                global_step=iteration,
                success_window=success_window,
                length_window=length_window,
            )
            obs_buffer.extend(obs_t)
            act_buffer.extend(act_t)
            rew_buffer.extend(rew_t)
            step_count_buffer.extend(sc_t)
            instr_buffer.extend(instr_t)
            print(f"  缓冲区大小: {len(obs_buffer)}")

        if len(obs_buffer) < args.num_samples_for_rollout:
            print(
                f"  缓冲区不足 ({len(obs_buffer)} < "
                f"{args.num_samples_for_rollout})，跳过"
            )
            continue

        indices = np.random.choice(
            len(obs_buffer), args.num_samples_for_rollout, replace=False
        )
        sampled_obs = torch.stack([obs_buffer[i] for i in indices], dim=0)
        initial_obs = sampled_obs[:, :num_steps_conditioning]
        initial_act = torch.stack([act_buffer[i] for i in indices], dim=0)
        rew_sampled = torch.stack([rew_buffer[i] for i in indices], dim=0)
        instructions_sampled = [instr_buffer[i] for i in indices]
        step_counts_sampled = torch.tensor([step_count_buffer[i] for i in indices],
                                           device=device, dtype=torch.long)

        if reward_model is not None and not args.no_train_reward_model:
            reward_step = train_reward_model_only(
                obs=sampled_obs,
                rew=rew_sampled,
                instructions=instructions_sampled,
                reward_model=reward_model,
                reward_optimizer=reward_optimizer,
                reward_lr_scheduler=reward_lr_scheduler,
                reward_cfg=reward_cfg,
                processor=processor,
                trainer_cfg=trainer_cfg,
                reward_step=reward_step,
                writer=writer,
            )

        print("  执行 imagination rollout...")
        t_rollout = time.time()
        rollout_data = rollout_with_ctrl_world_batched(
            env_batch=env_batch,
            actor=actor,
            initial_obs=initial_obs,
            initial_act=initial_act,
            instructions=instructions_sampled,
            initial_step_counts=step_counts_sampled,
            max_steps=args.rollout_max_steps,
            deterministic=False,
            batch_size=args.rollout_batch_size,
        )
        rollout_time = time.time() - t_rollout
        print(f"  rollout 完成: {rollout_time:.1f}s, shape={rollout_data['obs'].shape}")

        if rollout_data["obs"].shape[0] == 0:
            print("  rollout 为空，跳过 PPO")
            continue

        # 4. PPO 更新
        print("  PPO 更新...")
        t_ppo = time.time()
        metrics = ppo_update(
            actor_critic=actor,
            rollout_data=rollout_data,
            optimizer=optimizer,
            clip_ratio=0.2,
            value_coef=0.5,
            entropy_coef=0.0,
            max_grad_norm=0.5,
            num_epochs=args.ppo_epochs,
            batch_size=args.ppo_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
        )
        ppo_time = time.time() - t_ppo

        # 5. 记录日志
        iter_time = time.time() - t_start
        writer.add_scalar("train/policy_loss", metrics["policy_loss"], iteration)
        writer.add_scalar("train/value_loss", metrics["value_loss"], iteration)
        writer.add_scalar("train/entropy_loss", metrics["entropy_loss"], iteration)
        writer.add_scalar("train/total_loss", metrics["total_loss"], iteration)
        writer.add_scalar("train/mean_reward", metrics["mean_reward"], iteration)
        writer.add_scalar(
            "train/mean_episode_length", metrics["mean_episode_length"], iteration
        )
        writer.add_scalar("train/rollout_time", rollout_time, iteration)
        writer.add_scalar("train/ppo_time", ppo_time, iteration)
        writer.add_scalar("train/iteration_time", iter_time, iteration)
        writer.add_scalar("train/total_reward", rollout_data["rew"].sum().item(), iteration)
        writer.add_scalar("train/buffer_size", len(obs_buffer), iteration)

        print(f"  iter={iteration+1}, policy_loss={metrics['policy_loss']:.4f}, "
              f"value_loss={metrics['value_loss']:.4f}, "
              f"reward={metrics['mean_reward']:.4f}, "
              f"time={iter_time:.1f}s (rollout={rollout_time:.1f}s, ppo={ppo_time:.1f}s)")

    libero_env.close()
    writer.close()
    print(f"\n训练完成！TensorBoard: {log_dir}")


if __name__ == "__main__":
    main()
