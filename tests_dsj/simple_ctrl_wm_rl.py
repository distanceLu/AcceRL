"""
单进程 Ctrl-World 世界模型 + VLA 策略 + PPO 训练测试脚本。

与 simple_diffusion_wm_rl.py 对齐，但把 Denoiser 替换为 Ctrl-World (SVD) 世界模型。
用于验证 Ctrl-World 接入 AcceRL 框架的数据流和接口兼容性。

用法:
    python rl/simple_ctrl_wm_rl.py \
        --svd-model-path /mnt/data/lcx3/checkpoint/ctrl_world/svd/svd_model \
        --clip-model-path /mnt/data/lcx3/checkpoint/ctrl_world/clip/clip_model \
        --ctrl-world-ckpt /path/to/ctrl_world_checkpoint.pt \
        --device cuda:0
"""
import os
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
from rl.utils import prepare_one_obs
from experiments.robot.libero.libero_utils import GenerateConfig, TaskSuite
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM

# ---- Ctrl-World imports (需要把 Ctrl-World 加入 sys.path) ----
ACCE_RL_ROOT = Path(__file__).resolve().parent.parent
CTRL_WORLD_ROOT = ACCE_RL_ROOT.parent / "Ctrl-World"
if str(CTRL_WORLD_ROOT) not in sys.path:
    sys.path.insert(0, str(CTRL_WORLD_ROOT))
if str(ACCE_RL_ROOT) not in sys.path:
    sys.path.insert(0, str(ACCE_RL_ROOT))

from envs.ctrl_world_env_batch import CtrlWorldEnvBatch
from envs.utils import image_to_tensor, load_reward_model_from_checkpoint, tensor_to_image
from config import wm_args
from models.ctrl_world import CrtlWorld


# ================================================================
# 1. 数据采集（与 simple_diffusion_wm_rl.py 完全一致）
# ================================================================
def collect_initial_obs_act_from_libero(
    env: LiberoEnvWrapper,
    actor: ActorCritic,
    num_steps_conditioning: int,
    num_trajectories: int,
    device: torch.device,
    deterministic: bool = False,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[int], List[str]]:
    """
    从 LIBERO 真实环境采集完整轨迹，提取 (obs_window, act_window) 滑窗样本。
    返回:
        obs_list:  List of [T, C, H, W]
        act_list:  List of [T-1, act_dim]
        rew_list:  List of scalar
        step_counts: List of int
        instructions: List of str
    """
    obs_list, act_list, rew_list, step_counts, instructions = [], [], [], [], []

    for traj_idx in range(num_trajectories):
        obs_dict, info = env.reset()
        instruction = env.task_description

        traj_obs, traj_act, traj_rew = [], [], []
        action_queue = deque()

        img = obs_dict["full_image"]
        obs_tensor = image_to_tensor(img, device)
        traj_obs.append(obs_tensor)

        step_count = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            if len(action_queue) == 0:
                obs_dict_for_actor = {"full_image": obs_dict["full_image"]}
                inputs = prepare_one_obs(
                    actor.cfg, actor.processor, obs_dict_for_actor,
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

            img = obs_dict["full_image"]
            obs_tensor = image_to_tensor(img, device)
            traj_obs.append(obs_tensor)

            action_tensor = torch.from_numpy(action_norm).to(device).float()
            traj_act.append(action_tensor)

            reward_tensor = torch.tensor(reward, dtype=torch.float32, device=device)
            traj_rew.append(reward_tensor)

        T = len(traj_obs)
        if T < num_steps_conditioning + 1:
            print(f"  轨迹 {traj_idx+1}: T={T} < {num_steps_conditioning+1}，跳过")
            continue

        num_valid_windows = T - num_steps_conditioning
        for window_idx in range(num_valid_windows):
            obs_start = window_idx
            obs_end = obs_start + num_steps_conditioning + 1
            act_start = obs_start
            act_end = act_start + num_steps_conditioning

            window_obs = torch.stack(traj_obs[obs_start:obs_end], dim=0)
            window_act = torch.stack(traj_act[act_start:act_end], dim=0)
            window_rew = traj_rew[act_end - 1]

            obs_list.append(window_obs)
            act_list.append(window_act)
            rew_list.append(window_rew)
            step_counts.append(obs_start)
            instructions.append(instruction)

        print(f"  轨迹 {traj_idx+1}/{num_trajectories}: T={T}, steps={step_count}, "
              f"windows={num_valid_windows}, success={terminated}")

    return obs_list, act_list, rew_list, step_counts, instructions


# ================================================================
# 2. Rollout（使用 CtrlWorldEnvBatch 无状态接口，Worker 端维护 latent_history）
# ================================================================
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
    """
    在 CtrlWorldEnvBatch 中执行 imagination rollout（无状态版本）。

    Worker 端自行维护 latent_history、obs 滑窗、act 滑窗，
    每步调用 env_batch.predict_next_stateless() 获取 (next_obs, next_latent)，
    再用 env_batch.update_latent_history() 更新 latent_history。

    Args:
        env_batch: CtrlWorldEnvBatch 实例（仅用其模型权重和编解码工具，不依赖内部状态）
        actor: ActorCritic 策略网络
        initial_obs: [B, T, C, H, W] in [-1, 1]
        initial_act: [B, T-1, act_dim]
        instructions: List[str]
        initial_step_counts: [B] optional
        max_steps: 最大环境步数
    """
    assert max_steps % NUM_ACTIONS_CHUNK == 0
    B = initial_obs.shape[0]
    device = env_batch.device
    horizon = env_batch.horizon

    # 过滤已到达 horizon 的轨迹
    if initial_step_counts is not None:
        valid_mask = initial_step_counts < horizon
        valid_indices = torch.where(valid_mask)[0]
        initial_obs = initial_obs[valid_indices]
        initial_act = initial_act[valid_indices]
        instructions = [instructions[i] for i in valid_indices.cpu().tolist()]
        initial_step_counts = initial_step_counts[valid_indices]
        B = initial_obs.shape[0]
        if B == 0:
            print("  所有轨迹已到达 horizon，跳过 rollout")
            return _empty_rollout_result(device)

    # ================================================================
    # ★ Worker 端维护状态
    # ================================================================
    # 1. obs 滑窗: [B, T_cond, C, H, W]
    obs_window = initial_obs.clone()  # [B, T, C, H, W]
    # 2. act 滑窗: [B, T-1, act_dim]
    act_window = initial_act.clone()  # [B, T-1, act_dim]
    # 3. latent_history: [B, num_history, 4, latent_h*num_cams, latent_w]
    latent_history = env_batch.init_latent_history(obs_window)
    # 4. 其他状态
    ep_len = initial_step_counts.clone() if initial_step_counts is not None else torch.zeros(B, dtype=torch.long, device=device)
    alive_mask = torch.ones(B, dtype=torch.bool, device=device)
    last_success_prob = torch.zeros(B, device=device)

    # 初始化 last_success_prob（用第一帧预测一次 reward）
    current_obs = obs_window[:, -1]  # [B, C, H, W]
    if env_batch.reward_model is not None:
        last_success_prob, _, _ = env_batch.predict_rew_end(current_obs)
    else:
        last_success_prob = torch.zeros(B, device=device)

    # 存储 ActorCritic 级别的数据
    obs_list, act_logits_list, act_tokens_list = [], [], []
    rew_list, val_list, end_list, trunc_list, mask_list = [], [], [], [], []

    action_queues = [deque() for _ in range(B)]
    active_envs = list(range(B))

    env_step = 0
    while env_step < max_steps and active_envs:
        # 策略采样动作
        need_actions = [i for i in active_envs if len(action_queues[i]) == 0]

        if need_actions:
            inputs_list = []
            for i in need_actions:
                obs_tensor = current_obs[i]
                obs_np = obs_tensor.cpu().numpy().transpose(1, 2, 0)
                obs_img = ((obs_np + 1) / 2 * 255).astype(np.uint8)
                obs_dict = {"full_image": obs_img}
                inputs = prepare_one_obs(
                    actor.cfg, actor.processor, obs_dict,
                    instructions[i], actor.model_dtype,
                )
                inputs_list.append(inputs)

            inputs_batch = actor.prepare_inputs_batch(inputs_list)
            with torch.no_grad():
                action_logits, values = actor.forward(inputs_batch)

            # 存储（对齐 B 维度）
            obs_list.append(current_obs.clone())
            val_list.append(torch.zeros(B, device=device))
            val_list[-1][need_actions] = values
            act_logits_list.append(torch.zeros(B, *action_logits.shape[1:], device=device))
            act_tokens_list.append(torch.zeros(B, action_logits.shape[1], dtype=torch.long, device=device))

            _, action_token_ids, normalized_actions = actor.post_process(action_logits, [deterministic] * len(inputs_list))
            act_logits_list[-1][need_actions] = action_logits
            act_tokens_list[-1][need_actions] = action_token_ids

            for idx, env_idx in enumerate(need_actions):
                action_sequence = normalized_actions[idx]
                for j in range(action_sequence.shape[0]):
                    action_queues[env_idx].append(action_sequence[j])

        # 执行 NUM_ACTIONS_CHUNK 步
        chunk_rewards = torch.zeros(B, device=device)
        chunk_ended = torch.zeros(B, dtype=torch.long, device=device)
        chunk_truncated = torch.zeros(B, dtype=torch.long, device=device)

        for chunk_step in range(NUM_ACTIONS_CHUNK):
            if not active_envs:
                break

            # 构建动作 batch
            actions_step = []
            for i in range(B):
                if i in active_envs and len(action_queues[i]) > 0:
                    action_norm = action_queues[i].popleft()
                    if isinstance(action_norm, list):
                        action_norm = np.array(action_norm)
                    act_norm_torch = torch.from_numpy(action_norm).to(device).float()
                    actions_step.append(act_norm_torch)
                else:
                    actions_step.append(torch.zeros(ACTION_DIM, device=device))

            actions_batch = torch.stack(actions_step, dim=0)  # [B, act_dim]

            # ★ 无状态调用：Worker 把完整状态传给世界模型
            # 拼接 act_window + 当前动作 → act_history
            act_full = torch.cat([act_window, actions_batch.unsqueeze(1)], dim=1)  # [B, T, act_dim]

            next_obs, next_latent = env_batch.predict_next_stateless(
                current_obs=current_obs,
                latent_history=latent_history,
                act_history=act_full,
                instructions=instructions,
            )

            # ★ Worker 端更新状态
            # 1. 更新 latent_history（用 pipeline 返回的 latent，无需重复 VAE encode）
            latent_history = env_batch.update_latent_history(latent_history, next_latent)
            # 2. 滑窗更新 obs_window
            obs_window = obs_window.roll(-1, dims=1)
            obs_window[:, -1] = next_obs
            # 3. 滑窗更新 act_window
            act_window = torch.cat([act_window[:, 1:], actions_batch.unsqueeze(1)], dim=1)
            # 4. 更新 current_obs
            current_obs = next_obs

            # 计算 reward（势能奖励）
            if env_batch.reward_model is not None:
                success_prob, end_pred, _ = env_batch.predict_rew_end(next_obs)
                # mask dead envs
                success_prob = torch.where(alive_mask, success_prob, last_success_prob)
                end_pred = end_pred * alive_mask.long()
                rew = success_prob - last_success_prob
                rew = rew * alive_mask.float()
                last_success_prob = success_prob
            else:
                rew = torch.zeros(B, device=device)
                end_pred = torch.zeros(B, dtype=torch.long, device=device)

            ep_len += 1
            trunc = (ep_len >= horizon).long() * alive_mask.long()
            dead = torch.logical_or(end_pred.bool(), trunc.bool()) if env_batch.reward_model is not None else trunc.bool()
            dead = dead & alive_mask
            alive_mask = alive_mask & (~dead)

            for i in active_envs:
                chunk_rewards[i] += rew[i]

            terminated_envs = []
            for i in range(B):
                if (i in active_envs) and (end_pred[i] or trunc[i]):
                    chunk_ended[i] = end_pred[i]
                    chunk_truncated[i] = trunc[i]
                    terminated_envs.append(i)

            for env_idx in terminated_envs:
                if env_idx in active_envs:
                    active_envs.remove(env_idx)

            env_step += 1
            if env_step >= max_steps:
                break

        if need_actions:
            rew_list.append(chunk_rewards.clone())
            end_list.append(chunk_ended.clone())
            trunc_list.append(chunk_truncated.clone())
            mask_list.append(alive_mask.clone())

        if not any(alive_mask.cpu().tolist()):
            break

    # 转换为张量
    if not obs_list:
        raise ValueError(f"No observations collected at step {env_step}")

    obs_tensor = torch.stack(obs_list, dim=1)
    act_logits_tensor = torch.stack(act_logits_list, dim=1)
    act_tokens_tensor = torch.stack(act_tokens_list, dim=1)
    rew_tensor = torch.stack(rew_list, dim=1)
    val_tensor = torch.stack(val_list, dim=1)
    end_tensor = torch.stack(end_list, dim=1)
    trunc_tensor = torch.stack(trunc_list, dim=1)
    mask_tensor = torch.stack(mask_list, dim=1)

    # 计算 GAE
    T_ac = obs_tensor.shape[1]
    next_values = torch.cat([val_tensor[:, 1:], torch.zeros(B, 1, device=device)], dim=1)
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
        return {"policy_loss": 0, "value_loss": 0, "entropy_loss": 0, "total_loss": 0}

    obs = obs.to(device)
    old_action_logits = old_action_logits.to(device)
    old_action_tokens = old_action_tokens.to(device)
    advantages = advantages.to(device)
    returns = returns.to(device)
    old_values = old_values.to(device)

    mask_flat = mask.reshape(-1)
    valid_indices = torch.where(mask_flat)[0].to(device)

    obs_flat = obs.reshape(B * T_ac, C, H, W)[valid_indices]
    advantages_flat = advantages.reshape(B * T_ac)[valid_indices]
    returns_flat = returns.reshape(B * T_ac)[valid_indices]
    old_action_tokens_flat = old_action_tokens.reshape(B * T_ac, -1)[valid_indices]
    old_logits_flat = old_action_logits.reshape(B * T_ac, *old_action_logits.shape[2:])[valid_indices]

    advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)

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
    }


# ================================================================
# 4. 主函数
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="Single-process Ctrl-World + VLA PPO test")
    parser.add_argument("--svd-model-path", type=str, required=True,
                        help="Path to SVD model directory")
    parser.add_argument("--clip-model-path", type=str, required=True,
                        help="Path to CLIP model directory")
    parser.add_argument("--ctrl-world-ckpt", type=str, default=None,
                        help="Path to Ctrl-World checkpoint .pt file (optional)")
    parser.add_argument("--agent-config", type=str,
                        default=str(ACCE_RL_ROOT / "envs/config/agent.yaml"),
                        help="Path to agent.yaml")
    parser.add_argument("--reward-ckpt", type=str, default=None,
                        help="Path to reward model checkpoint (optional, skip reward if not provided)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-cams", type=int, default=2,
                        help="Number of camera views for Ctrl-World")
    parser.add_argument("--num-inference-steps", type=int, default=4,
                        help="SVD denoising steps (fewer = faster)")
    parser.add_argument("--target-height", type=int, default=192)
    parser.add_argument("--target-width", type=int, default=320)
    parser.add_argument("--num-iterations", type=int, default=50)
    parser.add_argument("--num-trajectories", type=int, default=3,
                        help="Number of LIBERO trajectories to collect per iteration")
    parser.add_argument("--rollout-max-steps", type=int, default=8)
    parser.add_argument("--rollout-batch-size", type=int, default=4)
    parser.add_argument("--ppo-batch-size", type=int, default=4)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--ppo-lr", type=float, default=3e-6)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--benchmark", type=str, default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--no-reward-model", action="store_true",
                        help="Skip reward model (use dummy rewards)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("Ctrl-World + VLA PPO Single-Process Test")
    print("=" * 80)

    # ---- TensorBoard ----
    log_dir = ACCE_RL_ROOT / "runs/simple_ctrl_wm_rl" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard: {log_dir}")

    # ---- 加载 Ctrl-World 模型 ----
    print("\n[1/4] Loading Ctrl-World model...")
    cfg = wm_args()
    cfg.svd_model_path = args.svd_model_path
    cfg.clip_model_path = args.clip_model_path
    cfg.num_cams = args.num_cams
    cfg.num_history = 6  # Ctrl-World 默认 6 帧历史
    cfg.action_dim = 7
    cfg.text_cond = True
    cfg.frame_level_cond = True

    ctrl_world = CrtlWorld(cfg).to(device).to(torch.bfloat16)
    ctrl_world.eval()

    # 加载 Ctrl-World checkpoint（如果有）
    if args.ctrl_world_ckpt and os.path.exists(args.ctrl_world_ckpt):
        print(f"  Loading checkpoint: {args.ctrl_world_ckpt}")
        ckpt = torch.load(args.ctrl_world_ckpt, map_location=device, weights_only=False)
        state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
        ctrl_world.load_state_dict(state_dict, strict=False)
        print("  Checkpoint loaded.")
    else:
        print("  No checkpoint provided, using randomly initialized weights (smoke test only).")

    # ---- 加载 ActorCritic ----
    print("\n[2/4] Loading ActorCritic (OpenVLA-OFT discrete)...")
    from omegaconf import OmegaConf
    agent_cfg = OmegaConf.load(args.agent_config)

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

    # ---- 加载 Reward Model（可选）----
    print("\n[3/4] Loading Reward Model...")
    reward_model = None
    reward_cfg = None
    processor = None
    torch_dtype = torch.bfloat16

    if not args.no_reward_model and args.reward_ckpt:
        trainer_cfg_path = ACCE_RL_ROOT / "envs/config/trainer.yaml"
        from omegaconf import OmegaConf
        trainer_cfg = OmegaConf.load(trainer_cfg_path)
        reward_model, _, _, processor, reward_cfg, _ = load_reward_model_from_checkpoint(
            agent_cfg, trainer_cfg, device
        )
        print("  Reward model loaded.")
    else:
        print("  Skipping reward model (dummy rewards will be used).")

    # ---- 加载 LIBERO 环境 ----
    print("\n[4/4] Loading LIBERO environment...")
    libero_env = LiberoEnvWrapper(
        benchmark_name=args.benchmark,
        task_id=args.task_id,
        image_size=224,
        render_mode="rgb_array",
    )

    # ---- 训练循环 ----
    num_steps_conditioning = cfg.num_history  # Ctrl-World 需要的观测序列长度

    print(f"\n开始训练，共 {args.num_iterations} 次迭代")
    print(f"  num_steps_conditioning={num_steps_conditioning}")
    print(f"  rollout_max_steps={args.rollout_max_steps}")
    print(f"  num_cams={args.num_cams}, inference_steps={args.num_inference_steps}")

    obs_buffer = deque(maxlen=10000)
    act_buffer = deque(maxlen=10000)
    rew_buffer = deque(maxlen=10000)
    step_count_buffer = deque(maxlen=10000)
    instr_buffer = deque(maxlen=10000)

    for iteration in range(args.num_iterations):
        print(f"\n=== 迭代 {iteration+1}/{args.num_iterations} ===")
        t_start = time.time()

        # 1. 采集真实数据
        if iteration % 3 == 0 or len(obs_buffer) < args.rollout_batch_size:
            print(f"  采集 {args.num_trajectories} 条轨迹...")
            obs_t, act_t, rew_t, sc_t, instr_t = collect_initial_obs_act_from_libero(
                env=libero_env, actor=actor,
                num_steps_conditioning=num_steps_conditioning,
                num_trajectories=args.num_trajectories,
                device=device, deterministic=False,
            )
            obs_buffer.extend(obs_t)
            act_buffer.extend(act_t)
            rew_buffer.extend(rew_t)
            step_count_buffer.extend(sc_t)
            instr_buffer.extend(instr_t)
            print(f"  缓冲区大小: {len(obs_buffer)}")

        if len(obs_buffer) < args.rollout_batch_size:
            print(f"  缓冲区不足 ({len(obs_buffer)} < {args.rollout_batch_size})，跳过")
            continue

        # 2. 随机采样
        indices = np.random.choice(len(obs_buffer), args.rollout_batch_size, replace=False)
        initial_obs = torch.stack([obs_buffer[i] for i in indices], dim=0).to(device)
        initial_act = torch.stack([act_buffer[i] for i in indices], dim=0).to(device)
        instructions_sampled = [instr_buffer[i] for i in indices]
        step_counts_sampled = torch.tensor([step_count_buffer[i] for i in indices],
                                           device=device, dtype=torch.long)

        # 3. 创建 CtrlWorldEnvBatch 并 rollout
        print("  创建 CtrlWorldEnvBatch...")
        @dataclass
        class FakeCfg:
            horizon: int = 220

        env_batch = CtrlWorldEnvBatch(
            ctrl_world_model=ctrl_world,
            cfg=FakeCfg(horizon=220),
            reward_model=reward_model,
            reward_cfg=reward_cfg,
            processor=processor,
            torch_dtype=torch_dtype,
            instructions=instructions_sampled,
            num_cams=args.num_cams,
            target_height=args.target_height,
            target_width=args.target_width,
            num_frames_pred=1,
            num_inference_steps=args.num_inference_steps,
        )

        print("  执行 imagination rollout...")
        t_rollout = time.time()
        rollout_data = rollout_with_ctrl_world(
            env_batch=env_batch,
            actor=actor,
            initial_obs=initial_obs,
            initial_act=initial_act,
            instructions=instructions_sampled,
            initial_step_counts=step_counts_sampled,
            max_steps=args.rollout_max_steps,
            deterministic=False,
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
        writer.add_scalar("train/total_loss", metrics["total_loss"], iteration)
        writer.add_scalar("train/rollout_time", rollout_time, iteration)
        writer.add_scalar("train/ppo_time", ppo_time, iteration)
        writer.add_scalar("train/iteration_time", iter_time, iteration)
        writer.add_scalar("train/total_reward", rollout_data["rew"].sum().item(), iteration)
        writer.add_scalar("train/buffer_size", len(obs_buffer), iteration)

        print(f"  iter={iteration+1}, policy_loss={metrics['policy_loss']:.4f}, "
              f"value_loss={metrics['value_loss']:.4f}, "
              f"reward={rollout_data['rew'].sum().item():.4f}, "
              f"time={iter_time:.1f}s (rollout={rollout_time:.1f}s, ppo={ppo_time:.1f}s)")

    libero_env.close()
    writer.close()
    print(f"\n训练完成！TensorBoard: {log_dir}")


if __name__ == "__main__":
    main()
