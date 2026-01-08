"""
简单的 WorldModelEnvBatch rollout 实现
使用 LiberoEnvWrapper 生成初始数据，ActorCritic 生成动作
"""
from datetime import datetime
import os
os.environ["MUJOCO_GL"] = "osmesa"           # 强制软件渲染
os.environ["PYOPENGL_PLATFORM"] = "osmesa" 
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import deque
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from hydra.utils import instantiate

from envs.world_model_env_batch import WorldModelEnvBatch, WorldModelEnvConfig
from envs.utils import load_reward_model_from_checkpoint, image_to_tensor, save_checkpoint, manage_checkpoints
from envs.diffusion.denoiser import Denoiser, load_denoiser_from_checkpoint
from experiments.robot.openvla_utils import get_processor
from rl.libero_env import LiberoEnvWrapper
from rl.actor_critic_model_discrete import ActorCritic
from rl.utils import prepare_one_obs
from experiments.robot.libero.libero_utils import GenerateConfig
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM

# export PYTHONPATH=/cpfs01/jinshiji_workspace/openvla_oft_rl:$PYTHONPATH


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
    window_size: int = 100,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[int], List[str]]:
    """
    从 LiberoEnvWrapper 收集多条完整轨迹的初始观测和动作序列
    
    Args:
        env: LiberoEnvWrapper 环境
        actor: ActorCritic 模型用于生成动作
        num_steps_conditioning: 需要的观测步数
        num_trajectories: 收集的轨迹数量
        device: torch device
        deterministic: 是否使用确定性动作
        writer: TensorBoard SummaryWriter (可选)
        global_step: 全局步数，用于tensorboard记录
        success_window: 成功率滑动窗口 (可选，如果None会创建新的)
        length_window: 轨迹长度滑动窗口 (可选，如果None会创建新的)
        window_size: 滑动窗口大小，默认为100
    
    Returns:
        obs_list: List of [T, C, H, W] tensors
        act_list: List of [T-1, act_dim] tensors
        rew_list: List of [T-1] tensors (rewards for each action step)
        step_counts: List of step counts for each trajectory
        instructions: List of task descriptions
    """
    obs_list = []
    act_list = []
    rew_list = []
    step_counts = []
    instructions = []
    
    # 初始化滑动窗口用于统计
    if success_window is None:
        success_window = deque(maxlen=window_size)
    if length_window is None:
        length_window = deque(maxlen=window_size)
    
    # 当前批次的统计
    batch_successes = []
    batch_lengths = []
    
    for traj_idx in range(num_trajectories):
        # Reset environment
        obs_dict, info = env.reset()
        instruction = env.task_description
        
        # Collect observations and actions
        traj_obs_list = []
        traj_act_list = []
        traj_rew_list = [] # rew for initial obs
        action_queue = deque()
        
        # First observation
        img = obs_dict["full_image"]  # [H, W, C] uint8
        obs_tensor = image_to_tensor(img, device)  # [C, H, W]
        traj_obs_list.append(obs_tensor)
        
        step_count = 0
        terminated = False
        truncated = False
        
        # Run episode until done
        while not (terminated or truncated):
            # Generate actions if queue is empty
            if len(action_queue) == 0:
                # Prepare input for actor
                obs_dict_for_actor = {"full_image": obs_dict["full_image"]}
                inputs = prepare_one_obs(
                    actor.cfg,
                    actor.processor,
                    obs_dict_for_actor,
                    instruction,
                    actor.model_dtype,
                )
                inputs_batch = actor.prepare_inputs_batch([inputs])
                
                with torch.no_grad():
                    action_logits, _ = actor.forward(inputs_batch)
                
                deterministic_flags = [deterministic]
                _, _, normalized_actions = actor.post_process(action_logits, deterministic_flags)
                # normalized_actions: [1, 8, 7]
                
                # Add actions to queue
                action_sequence = normalized_actions[0]  # [8, 7]
                for j in range(action_sequence.shape[0]):
                    action_queue.append(action_sequence[j])  # [7] numpy array
            
            # Get action from queue
            action_norm = action_queue.popleft()
            # Convert to numpy array for _unnormalize_actions
            if isinstance(action_norm, list):
                action_norm = np.array(action_norm)
            # Unnormalize action
            action_env_np = actor.vla._unnormalize_actions(action_norm, actor.cfg.unnorm_key)
            action_env = action_env_np  # numpy array
            
            # Step environment
            obs_dict, reward, terminated, truncated, info = env.step(action_env)
            step_count += 1
            # if step_count >= 9:
            #     truncated = True
            
            # Store observation
            img = obs_dict["full_image"]
            obs_tensor = image_to_tensor(img, device)
            traj_obs_list.append(obs_tensor)
            
            # Store action (convert to tensor)
            action_tensor = torch.from_numpy(action_norm).to(device).float()  # 由于训练denoiser的时候用的是action_norm，需要统一一下，TODO
            traj_act_list.append(action_tensor)
            
            # Store reward (convert to tensor)
            reward_tensor = torch.tensor(reward, dtype=torch.float32, device=device)
            traj_rew_list.append(reward_tensor)
            
            # Check if we have enough observations for conditioning
            if len(traj_obs_list) >= num_steps_conditioning + 1:
                # We have enough observations, can stop collecting
                # But continue to collect full trajectory for step count
                pass
        
        # 统计当前轨迹的成功与否和长度
        is_success = float(terminated)  # terminated=True 表示成功完成任务
        traj_length = step_count
        batch_successes.append(is_success)
        batch_lengths.append(traj_length)
        
        # 添加到滑动窗口
        success_window.append(is_success)
        length_window.append(traj_length)
        
        # Extract valid conditioning windows using sliding window
        # Note: traj_obs_list has T observations (indices 0 to T-1)
        #       traj_act_list has T-1 actions (indices 0 to T-2)
        # Invalid: first (num_steps_conditioning-1) observations (not enough length)
        #          last observation (T-1, already ended)
        # Valid windows: start from index (num_steps_conditioning-1), 
        #                end before index (T-1) so that window doesn't include last obs
        T = len(traj_obs_list)
        
        if T < num_steps_conditioning + 1:
            # Not enough observations, skip this trajectory
            print(f"轨迹 {traj_idx + 1}/{num_trajectories}: 观测数量 {T} < {num_steps_conditioning + 1}，跳过")
            continue
        
        # Calculate number of valid windows
        num_valid_windows = T - num_steps_conditioning
        
        if num_valid_windows <= 0:
            print(f"轨迹 {traj_idx + 1}/{num_trajectories}: 没有有效窗口，跳过")
            continue
        
        # Extract all valid windows
        for window_idx in range(num_valid_windows):
            # Window start index for observations
            # Start from 0, increment by 1 for each window
            obs_start_idx = window_idx
            obs_end_idx = obs_start_idx + num_steps_conditioning + 1
            
            # Corresponding action indices (one less than obs)
            act_start_idx = obs_start_idx
            act_end_idx = act_start_idx + num_steps_conditioning
            
            # Extract window
            window_obs = traj_obs_list[obs_start_idx:obs_end_idx]
            window_act = traj_act_list[act_start_idx:act_end_idx]
            window_rew = traj_rew_list[act_end_idx - 1]  # Rewards correspond to actions
            
            # Stack observations: [n_condition+1, C, H, W]
            obs = torch.stack(window_obs, dim=0)
            # Stack actions: [n_condition, act_dim]
            act = torch.stack(window_act, dim=0)
            # Stack rewards: [n_condition-1]
            rew = window_rew
            assert obs.shape[0] == num_steps_conditioning + 1
            assert act.shape[0] == num_steps_conditioning
            
            obs_list.append(obs)
            act_list.append(act)
            rew_list.append(rew)
            # For step count, use the step count at the start of this window
            # step_count is the total steps, so at window start it's (obs_start_idx)
            step_counts.append(obs_start_idx)
            instructions.append(instruction)

        print(f"轨迹 {traj_idx + 1}/{num_trajectories}: T={T}, step_count={step_count}, "
              f"有效窗口数={num_valid_windows}, terminated={terminated}, truncated={truncated}")
    
    # 记录统计信息到 TensorBoard
    if writer is not None and len(batch_successes) > 0:
        # 当前批次的统计
        batch_success_rate = np.mean(batch_successes)
        batch_avg_length = np.mean(batch_lengths)
        
        # 滑动窗口的统计
        window_success_rate = np.mean(list(success_window))
        window_avg_length = np.mean(list(length_window))
        
        # 记录到 TensorBoard
        writer.add_scalar("collect/batch_success_rate", batch_success_rate, global_step)
        writer.add_scalar("collect/batch_avg_trajectory_length", batch_avg_length, global_step)
        writer.add_scalar("collect/window_success_rate", window_success_rate, global_step)
        writer.add_scalar("collect/window_avg_trajectory_length", window_avg_length, global_step)
        writer.add_scalar("collect/num_trajectories", len(batch_successes), global_step)
        
        print(f"\n=== 轨迹收集统计 (Step {global_step}) ===")
        print(f"当前批次: 成功率={batch_success_rate:.2%}, 平均长度={batch_avg_length:.1f}")
        print(f"滑动窗口 (size={len(success_window)}): 成功率={window_success_rate:.2%}, 平均长度={window_avg_length:.1f}")
    
    assert len(obs_list) == len(act_list) == len(rew_list) == len(step_counts) == len(instructions)
    return obs_list, act_list, rew_list, step_counts, instructions


def rollout_with_world_model_batched(
    env_batch: WorldModelEnvBatch,
    actor: ActorCritic,
    initial_obs: torch.Tensor,
    initial_act: torch.Tensor,
    instructions: List[str],
    initial_step_counts: Optional[torch.Tensor] = None,
    max_steps: int = 100,
    deterministic: bool = False,
    batch_size: int = 8,
) -> Dict[str, Any]:
    """
    分批执行 rollout，避免 OOM
    
    Args:
        env_batch: WorldModelEnvBatch 环境
        actor: ActorCritic 模型
        initial_obs: [B, T, C, H, W] 初始观测序列
        initial_act: [B, T-1, act_dim] 初始动作序列
        instructions: List[str] 任务指令列表
        initial_step_counts: Optional [B] tensor of initial step counts
        max_steps: 最大步数
        deterministic: 是否使用确定性动作
        batch_size: 每个批次的样本数量
    
    Returns:
        Dict containing rollout data (concatenated from all batches)
    """
    B = initial_obs.shape[0]
    device = env_batch.device
    
    if B <= batch_size:
        # 如果 batch 大小小于等于 batch_size，直接调用
        return rollout_with_world_model(
            env_batch=env_batch,
            actor=actor,
            initial_obs=initial_obs,
            initial_act=initial_act,
            instructions=instructions,
            initial_step_counts=initial_step_counts,
            max_steps=max_steps,
            deterministic=deterministic,
        )
    
    # 切分成多个小批次
    num_batches = (B + batch_size - 1) // batch_size
    print(f"将 {B} 个样本切分成 {num_batches} 个批次，每批 {batch_size} 个样本")
    
    all_obs_list = []
    all_act_list = []
    all_act_tokens_list = []
    all_rew_list = []
    all_val_list = []
    all_end_list = []
    all_trunc_list = []
    all_mask_list = []
    all_advantages_list = []
    all_returns_list = []
    all_instructions_list = []
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, B)
        
        print(f"处理批次 {batch_idx + 1}/{num_batches} (样本 {start_idx} 到 {end_idx - 1})")
        
        # 提取当前批次的数据
        batch_obs = initial_obs[start_idx:end_idx]
        batch_act = initial_act[start_idx:end_idx]
        batch_instructions = instructions[start_idx:end_idx]
        batch_step_counts = initial_step_counts[start_idx:end_idx] if initial_step_counts is not None else None
        
        # 执行 rollout
        batch_result = rollout_with_world_model(
            env_batch=env_batch,
            actor=actor,
            initial_obs=batch_obs,
            initial_act=batch_act,
            instructions=batch_instructions,
            initial_step_counts=batch_step_counts,
            max_steps=max_steps,
            deterministic=deterministic,
        )
        
        if batch_result["obs"].shape[0] > 0:
            # print(f"batch_idx: {batch_idx}, reward sum: {batch_result['rew'].sum(dim=1)}")
            # 收集结果
            all_obs_list.append(batch_result["obs"])
            all_act_list.append(batch_result["act_logits"])
            all_act_tokens_list.append(batch_result["act_tokens"])
            all_rew_list.append(batch_result["rew"])
            all_val_list.append(batch_result["val"])
            all_end_list.append(batch_result["end"])
            all_trunc_list.append(batch_result["trunc"])
            all_mask_list.append(batch_result["mask"])
            all_advantages_list.append(batch_result["advantages"])
            all_returns_list.append(batch_result["returns"])
            
            if "instructions" in batch_result:
                all_instructions_list.extend(batch_result["instructions"])
    
    if len(all_obs_list) == 0:
        return {
            "obs": torch.zeros(0, 0, *initial_obs.shape[2:], device=device),
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

    # 合并所有批次的结果
    # 注意：不同批次的 rollout 长度可能不同，需要处理
    # 找到最大长度
    max_obs_len = max(obs.shape[1] for obs in all_obs_list)
    max_act_len = max(act.shape[1] for act in all_act_list)
    max_act_tokens_len = max(act_tokens.shape[1] for act_tokens in all_act_tokens_list)
    max_val_len = max(val.shape[1] for val in all_val_list)
    max_mask_len = max(mask.shape[1] for mask in all_mask_list)
    max_adv_len = max(adv.shape[1] for adv in all_advantages_list)
    max_ret_len = max(ret.shape[1] for ret in all_returns_list)
    
    # 获取形状信息
    _, T_obs, C, H, W = all_obs_list[0].shape
    _, T_act, num_actions, vocab_size = all_act_list[0].shape
    
    # 填充到相同长度并合并
    padded_obs_list = []
    padded_act_list = []
    padded_act_tokens_list = []
    padded_rew_list = []
    padded_val_list = []
    padded_end_list = []
    padded_trunc_list = []
    padded_mask_list = []
    padded_advantages_list = []
    padded_returns_list = []
    
    for i in range(len(all_obs_list)):
        obs = all_obs_list[i]  # [B_i, T_i, C, H, W]
        act = all_act_list[i]  # [B_i, T_i, num_actions, act_dim]
        act_tokens = all_act_tokens_list[i]  # [B_i, T_i, num_actions]
        rew = all_rew_list[i]  # [B_i, T_i]
        val = all_val_list[i]  # [B_i, T_i]
        end = all_end_list[i]  # [B_i, T_i]
        trunc = all_trunc_list[i]  # [B_i, T_i]
        advantages = all_advantages_list[i]  # [B_i, T_i]
        returns = all_returns_list[i]  # [B_i, T_i]

        B_i = obs.shape[0]
        T_i_obs = obs.shape[1]
        T_i_act = act.shape[1]
        T_i_act_tokens = act_tokens.shape[1]
        T_i_val = val.shape[1]
        T_i_adv = advantages.shape[1]
        T_i_ret = returns.shape[1]
        
        # 填充观测（使用最后一个观测填充）
        if T_i_obs < max_obs_len:
            padding_obs = obs[:, -1:].expand(-1, max_obs_len - T_i_obs, -1, -1, -1)
            obs = torch.cat([obs, padding_obs], dim=1)
        
        # 填充动作、奖励、结束信号（使用零填充）
        if T_i_act < max_act_len:
            padding_act = torch.zeros(B_i, max_act_len - T_i_act, num_actions, vocab_size, device=device, dtype=act.dtype)
            padding_act_tokens = torch.zeros(B_i, max_act_len - T_i_act, num_actions, device=device, dtype=torch.long)
            padding_rew = torch.zeros(B_i, max_act_len - T_i_act, device=device, dtype=rew.dtype)
            padding_val = torch.zeros(B_i, max_act_len - T_i_act, device=device, dtype=val.dtype)
            padding_end = torch.zeros(B_i, max_act_len - T_i_act, device=device, dtype=end.dtype)
            padding_trunc = torch.zeros(B_i, max_act_len - T_i_act, device=device, dtype=trunc.dtype)
            padding_mask = torch.zeros(B_i, max_act_len - T_i_act, device=device, dtype=torch.bool)

            act = torch.cat([act, padding_act], dim=1)
            rew = torch.cat([rew, padding_rew], dim=1)
            val = torch.cat([val, padding_val], dim=1)
            end = torch.cat([end, padding_end], dim=1)
            trunc = torch.cat([trunc, padding_trunc], dim=1)

        # 填充act_tokens、val、advantages、returns、mask（使用零填充）
        if T_i_act_tokens < max_act_tokens_len:
            padding_act_tokens = torch.zeros(B_i, max_act_tokens_len - T_i_act_tokens, num_actions, device=device, dtype=act_tokens.dtype)
            act_tokens = torch.cat([act_tokens, padding_act_tokens], dim=1)

        if T_i_val < max_val_len:
            padding_val = torch.zeros(B_i, max_val_len - T_i_val, device=device, dtype=val.dtype)
            val = torch.cat([val, padding_val], dim=1)

        if T_i_adv < max_adv_len:
            padding_adv = torch.zeros(B_i, max_adv_len - T_i_adv, device=device, dtype=advantages.dtype)
            advantages = torch.cat([advantages, padding_adv], dim=1)

        if T_i_ret < max_ret_len:
            padding_ret = torch.zeros(B_i, max_ret_len - T_i_ret, device=device, dtype=returns.dtype)
            returns = torch.cat([returns, padding_ret], dim=1)

        # Get mask and pad it
        mask = all_mask_list[i]
        T_i_mask = mask.shape[1]
        if T_i_mask < max_mask_len:
            padding_mask = torch.zeros(B_i, max_mask_len - T_i_mask, device=device, dtype=torch.bool)
            mask = torch.cat([mask, padding_mask], dim=1)

        padded_obs_list.append(obs)
        padded_act_list.append(act)
        padded_act_tokens_list.append(act_tokens)
        padded_rew_list.append(rew)
        padded_val_list.append(val)
        padded_end_list.append(end)
        padded_trunc_list.append(trunc)
        padded_mask_list.append(mask)
        padded_advantages_list.append(advantages)
        padded_returns_list.append(returns)
    
    # 合并所有批次
    final_obs = torch.cat(padded_obs_list, dim=0)  # [B, max_obs_len, C, H, W]
    final_act = torch.cat(padded_act_list, dim=0)  # [B, max_act_len, num_actions, vocab_size]
    final_act_tokens = torch.cat(padded_act_tokens_list, dim=0)  # [B, max_act_tokens_len, num_actions]
    final_rew = torch.cat(padded_rew_list, dim=0)  # [B, max_act_len]
    final_val = torch.cat(padded_val_list, dim=0)  # [B, max_val_len]
    final_end = torch.cat(padded_end_list, dim=0)  # [B, max_act_len]
    final_trunc = torch.cat(padded_trunc_list, dim=0)  # [B, max_act_len]
    final_mask = torch.cat(padded_mask_list, dim=0)  # [B, max_mask_len]
    final_advantages = torch.cat(padded_advantages_list, dim=0)  # [B, max_adv_len]
    final_returns = torch.cat(padded_returns_list, dim=0)  # [B, max_ret_len]

    return {
        "obs": final_obs,
        "act_logits": final_act,
        "act_tokens": final_act_tokens,
        "rew": final_rew,
        "val": final_val,
        "end": final_end,
        "trunc": final_trunc,
        "mask": final_mask,
        "advantages": final_advantages,
        "returns": final_returns,
        "instructions": all_instructions_list,
    }


def rollout_with_world_model(
    env_batch: WorldModelEnvBatch,
    actor: ActorCritic,
    initial_obs: torch.Tensor,
    initial_act: torch.Tensor,
    instructions: List[str],
    initial_step_counts: Optional[torch.Tensor] = None,
    max_steps: int = 100,
    deterministic: bool = False,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Dict[str, Any]:
    """
    在 WorldModelEnvBatch 中执行 rollout，返回ActorCritic视角的步骤数据和GAE

    Args:
        env_batch: WorldModelEnvBatch 环境
        actor: ActorCritic 模型
        initial_obs: [B, T, C, H, W] 初始观测序列
        initial_act: [B, T-1, act_dim] 初始动作序列
        instructions: List[str] 任务指令列表
        initial_step_counts: Optional [B] tensor of initial step counts
        max_steps: 最大环境步数
        deterministic: 是否使用确定性动作
        gamma: 折扣因子
        gae_lambda: GAE lambda参数

    Returns:
        Dict containing rollout data with GAE
    """
    assert max_steps % NUM_ACTIONS_CHUNK == 0
    B = initial_obs.shape[0]
    device = env_batch.device
    horizon = env_batch.horizon

    # Filter out trajectories that have already reached horizon
    if initial_step_counts is not None:
        valid_mask = initial_step_counts < horizon
        
        # Filter valid trajectories
        valid_indices = torch.where(valid_mask)[0]
        
        # Keep track of original count for logging
        original_B = initial_obs.shape[0]
        
        initial_obs = initial_obs[valid_indices]
        initial_act = initial_act[valid_indices]
        instructions = [instructions[i] for i in valid_indices.cpu().tolist()]
        initial_step_counts = initial_step_counts[valid_indices]
        B = initial_obs.shape[0]
        
        if B < original_B:
            print(f"Filtered {B} valid trajectories out of {original_B} (removed {original_B - B} that reached horizon)")

    if B == 0:
        return {
            "obs": torch.zeros(0, 0, *initial_obs.shape[2:], device=device),
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

    # Reset environment with initial step counts
    current_obs, _ = env_batch.reset(initial_obs, initial_act, instructions, initial_step_counts)

    # Storage for ActorCritic-level steps (every NUM_ACTIONS_CHUNK env steps)
    # Fixed shape [B, T_ac, ...] - don't use need_actions slicing to keep B constant
    obs_list = []  # Observations at the start of each ActorCritic step [B, C, H, W]
    act_logits_list = []  # Action logits from ActorCritic [B, 8, vocab_size]
    act_tokens_list = []  # Action tokens actually executed [B, 8]
    rew_list = []  # Accumulated rewards over NUM_ACTIONS_CHUNK steps [B]
    val_list = []  # Values from ActorCritic [B]
    end_list = []  # Whether episode ended in this chunk [B]
    trunc_list = []  # Whether episode was truncated in this chunk [B]
    mask_list = []  # Mask indicating which environments are active [B]

    # Action queue for each environment (ActorCritic generates 8 actions at once)
    action_queues = [deque() for _ in range(B)]
    active_envs = list(range(B))  # Currently active environments
    alive_mask = torch.ones(B, dtype=torch.bool, device=device)  # Track alive environments

    env_step = 0
    while env_step < max_steps and active_envs:
        # Check which environments need new actions
        need_actions = []
        inputs_list = []
        env_indices = []

        for i in active_envs:
            if len(action_queues[i]) == 0:
                need_actions.append(i)

        if need_actions:
            # Prepare inputs for ActorCritic forward
            for i in need_actions:
                obs_tensor = current_obs[i]  # [C, H, W] in [-1, 1]
                obs_np = obs_tensor.cpu().numpy().transpose(1, 2, 0)  # [H, W, C]
                obs_img = ((obs_np + 1) / 2 * 255).astype(np.uint8)
                obs_dict = {"full_image": obs_img}

                inputs = prepare_one_obs(
                    actor.cfg,
                    actor.processor,
                    obs_dict,
                    instructions[i],
                    actor.model_dtype,
                )
                inputs_list.append(inputs)
                env_indices.append(i)

            # ActorCritic forward
            inputs_batch = actor.prepare_inputs_batch(inputs_list)
            with torch.no_grad():
                action_logits, values = actor.forward(inputs_batch)  # [b_s, 8, vocab_size], [b_s]

            # Store observation and value at this ActorCritic step - FIXED SHAPE [B, ...]
            obs_list.append(current_obs.clone())  # [B, C, H, W] - store for all envs
            val_list.append(torch.zeros(B, device=device))  # [B] - will fill only active ones
            val_list[-1][need_actions] = values  # Fill values only for active envs

            act_logits_list.append(torch.zeros(B, *action_logits.shape[1:], device=device))  # [B, 8, vocab_size]
            act_tokens_list.append(torch.zeros(B, action_logits.shape[1], dtype=torch.long, device=device))  # [B, 8]

            # Process actions
            deterministic_flags = [deterministic] * len(inputs_list)
            _, action_token_ids, normalized_actions = actor.post_process(action_logits, deterministic_flags)
            # action_token_ids: [B_active, 8] - 实际采样的动作tokens
            # normalized_actions: [B_active, 8, 7]

            # Store action logits and tokens for active environments only
            act_logits_list[-1][need_actions] = action_logits  # [B, 8, vocab_size]
            act_tokens_list[-1][need_actions] = action_token_ids  # [B, 8]

            # Add actions to queues
            for idx, env_idx in enumerate(env_indices):
                action_sequence = normalized_actions[idx]  # [8, 7] numpy array
                for j in range(action_sequence.shape[0]):
                    action_queues[env_idx].append(action_sequence[j])  # [7] numpy array

        # Execute NUM_ACTIONS_CHUNK steps or until all active envs are done
        chunk_rewards = torch.zeros(B, device=device)  # [B] accumulated rewards for this chunk
        chunk_ended = torch.zeros(B, dtype=torch.long, device=device)  # [B] whether ended in this chunk
        chunk_truncated = torch.zeros(B, dtype=torch.long, device=device)  # [B] whether truncated in this chunk

        for chunk_step in range(NUM_ACTIONS_CHUNK):
            if not active_envs:
                break

            # Execute one action for each environment (B), using zero actions for inactive ones
            actions_step = []
            for i in range(B):  # Always iterate over all B environments
                if i in active_envs and len(action_queues[i]) > 0:
                    action_norm = action_queues[i].popleft()
                    if isinstance(action_norm, list):
                        action_norm = np.array(action_norm)
                    action_env_np = actor.vla._unnormalize_actions(action_norm, actor.cfg.unnorm_key)
                    action_env = torch.from_numpy(action_env_np).to(device).float()
                    act_norm_torch = torch.from_numpy(action_norm).to(device).float()
                    actions_step.append(act_norm_torch)  # 由于训练denoiser的时候用的是action_norm，需要统一一下，TODO
                else:
                    # Use zero action for inactive environments
                    actions_step.append(torch.zeros(ACTION_DIM, device=device))

            # Prepare batched actions - always [B, act_dim]
            actions_batch = torch.stack(actions_step, dim=0)  # [B, act_dim]

            # Step environment
            next_obs, rew, end, trunc, info = env_batch.step(actions_batch)

            # Accumulate rewards only for active environments
            for i in active_envs:
                chunk_rewards[i] += rew[i]

            # Check for termination/truncation
            terminated_envs = []
            for i in range(B):
                if (i in active_envs) and (end[i] or trunc[i]):
                    chunk_ended[i] = end[i]
                    chunk_truncated[i] = trunc[i]
                    terminated_envs.append(i)

            # Remove terminated environments from active list and update alive_mask
            for env_idx in terminated_envs:
                if env_idx in active_envs:
                    active_envs.remove(env_idx)
                    alive_mask[env_idx] = False

            current_obs = next_obs
            env_step += 1

            if env_step >= max_steps:
                break

        # Store chunk results (only for environments that were active at the start of this chunk)
        if need_actions:  # Only store if we actually took an ActorCritic step
            rew_list.append(chunk_rewards.clone())  # [B] - store for all envs, inactive are 0
            end_list.append(chunk_ended.clone())    # [B] - store for all envs
            trunc_list.append(chunk_truncated.clone())  # [B] - store for all envs
            mask_list.append(alive_mask.clone())  # [B] - mask of active envs

        # Check if all environments are done
        if isinstance(info.get('alive', None), (list, np.ndarray)):
            alive = info['alive']
            if isinstance(alive, np.ndarray):
                alive = alive.tolist()
            if not any(alive):
                print(f"All environments done at step {env_step}")
                break
        elif not info.get('alive', True):
            break

    # Convert lists to tensors
    if obs_list:
        obs_tensor = torch.stack(obs_list, dim=1)  # [B, T_ac, C, H, W] where T_ac is ActorCritic steps
        act_logits_tensor = torch.stack(act_logits_list, dim=1)  # [B, T_ac, 8, vocab_size] - action logits
        act_tokens_tensor = torch.stack(act_tokens_list, dim=1)  # [B, T_ac, 8] - action tokens
        rew_tensor = torch.stack(rew_list, dim=1)  # [B, T_ac]
        val_tensor = torch.stack(val_list, dim=1)  # [B, T_ac]
        end_tensor = torch.stack(end_list, dim=1)  # [B, T_ac]
        trunc_tensor = torch.stack(trunc_list, dim=1)  # [B, T_ac]
        mask_tensor = torch.stack(mask_list, dim=1)  # [B, T_ac] - valid mask

        # Compute GAE
        T_ac = obs_tensor.shape[1]

        # Get next values for bootstrap (use 0 for terminal states)
        next_values = torch.cat([val_tensor[:, 1:], torch.zeros(B, 1, device=device)], dim=1)
        # For non-terminal states, bootstrap from the next value
        dones = (end_tensor | trunc_tensor).float()
        next_values = next_values * (1 - dones)

        advantages, returns = compute_gae(
            rewards=rew_tensor,  # [B, T_ac]
            values=val_tensor,  # [B, T_ac]
            next_values=next_values,  # [B, T_ac]
            dones=dones,  # [B, T_ac]
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        return {
            "obs": obs_tensor,  # [B, T_ac, C, H, W]
            "act_logits": act_logits_tensor,  # [B, T_ac, 8, vocab_size] - action logits
            "act_tokens": act_tokens_tensor,  # [B, T_ac, 8] - action tokens executed
            "rew": rew_tensor,  # [B, T_ac]
            "val": val_tensor,  # [B, T_ac]
            "end": end_tensor,  # [B, T_ac]
            "trunc": trunc_tensor,  # [B, T_ac]
            "mask": mask_tensor,  # [B, T_ac] - valid mask for training
            "advantages": advantages,  # [B, T_ac]
            "returns": returns,  # [B, T_ac]
            "instructions": instructions,
        }
    else:
        raise ValueError(f"No observations were collected at step {env_step}")


def evaluate_world_model(
    obs: torch.Tensor,
    act: torch.Tensor,
    rew: torch.Tensor,
    denoiser: Denoiser,
    reward_model: Any,
    reward_cfg: Any,
    processor: Any,
    instructions: List[str],
    device: torch.device,
    batch_size: int,
) -> Dict[str, float]:
    """
    评估 World Model（包括 Denoiser 和 Reward Model）
    
    Args:
        obs: [B, num_steps_conditioning+1, C, H, W] 观测序列（[-1, 1] 范围）
        act: [B, num_steps_conditioning, act_dim] 动作序列（归一化动作）
        rew: [B] 标量奖励，对应最后一个观测
        denoiser: Denoiser 模型
        reward_model: Reward Model 模型
        reward_cfg: Reward Model 配置
        processor: 图像预处理器
        instructions: 任务指令列表 [B]
        device: 设备
        batch_size: 分批大小
    
    Returns:
        评估指标字典
    """
    denoiser.eval()
    reward_model.eval()
    
    B, T = obs.shape[:2]
    mask_padding = torch.ones(obs.shape[:2], dtype=torch.bool, device=device)
    
    with torch.no_grad():
        # ========== 1. 评估 Denoiser（分批） ==========
        num_denoiser_samples = B
        num_denoiser_batches = (num_denoiser_samples + batch_size - 1) // batch_size
        
        total_denoiser_loss = 0.0
        for batch_idx in range(num_denoiser_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_denoiser_samples)
            
            obs_batch = obs[start_idx:end_idx]
            act_batch = act[start_idx:end_idx]
            mask_batch = mask_padding[start_idx:end_idx]
            
            loss_denoiser, _ = denoiser(obs_batch, act_batch, mask_batch)
            total_denoiser_loss += loss_denoiser.item() * (end_idx - start_idx)
        
        avg_denoiser_loss = total_denoiser_loss / num_denoiser_samples
        
        # ========== 2. 评估 Reward Model（分批） ==========
        # 提取最后一个观测，对应 reward
        last_obs = obs[:, -1]  # [B, C, H, W]
        
        # 预处理所有观测
        inputs_list = []
        for i in range(B):
            obs_tensor = last_obs[i]  # [C, H, W] in [-1, 1]
            obs_np = obs_tensor.cpu().numpy().transpose(1, 2, 0)  # [H, W, C]
            obs_img = ((obs_np + 1) / 2 * 255).astype(np.uint8)
            obs_dict = {"full_image": obs_img}
            
            inputs = prepare_one_obs(
                reward_cfg,
                processor,
                obs_dict,
                instructions[i],
                reward_model.model_dtype,
            )
            inputs_list.append(inputs)
        
        # 一次性批处理所有输入
        inputs_batch = reward_model.prepare_inputs_batch(inputs_list)
        labels = (rew > 0).long()  # [B]
        
        # 分批评估 Reward Model
        num_reward_samples = B
        num_reward_batches = (num_reward_samples + batch_size - 1) // batch_size
        
        total_reward_loss = 0.0
        total_tp = total_tn = total_fp = total_fn = 0
        
        for batch_idx in range(num_reward_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_reward_samples)
            
            # 切片批次数据
            batch_inputs = {k: v[start_idx:end_idx] for k, v in inputs_batch.items()}
            batch_labels = labels[start_idx:end_idx]
            
            # 前向传播
            logits = reward_model.forward(batch_inputs)
            loss_reward, metrics_reward = reward_model.compute_loss_and_metrics(batch_inputs, batch_labels)
            
            total_reward_loss += loss_reward.item() * (end_idx - start_idx)
            total_tp += metrics_reward["tp"].item()
            total_tn += metrics_reward["tn"].item()
            total_fp += metrics_reward["fp"].item()
            total_fn += metrics_reward["fn"].item()
        
        avg_reward_loss = total_reward_loss / num_reward_samples
        
        # 计算准确率
        pos_den = total_tp + total_fn
        neg_den = total_tn + total_fp
        pos_acc = float(total_tp) / pos_den if pos_den > 0 else 0.0
        neg_acc = float(total_tn) / neg_den if neg_den > 0 else 0.0
    
    return {
        "denoiser_loss": avg_denoiser_loss,
        "reward_loss": avg_reward_loss,
        "reward_pos_acc": pos_acc,
        "reward_neg_acc": neg_acc,
        "reward_tp": total_tp,
        "reward_tn": total_tn,
        "reward_fp": total_fp,
        "reward_fn": total_fn,
    }


def train_world_model(
    obs: torch.Tensor, 
    act: torch.Tensor,
    rew: torch.Tensor, 
    trainer_cfg: Dict[str, Any],
    denoiser: Denoiser,
    denoiser_optimizer: optim.Optimizer,
    denoiser_lr_scheduler: optim.lr_scheduler.LambdaLR,
    denoiser_start_step: int,
    reward_model: Any,
    reward_optimizer: optim.Optimizer,
    reward_lr_scheduler: optim.lr_scheduler._LRScheduler,
    reward_cfg: Any,
    processor: Any,
    instructions: List[str],
    reward_start_step: int,
    writer: SummaryWriter,
):
    """
    训练 World Model（包括 Denoiser 和 Reward Model）
    
    Args:
        obs: [B, num_steps_conditioning+1, C, H, W] 观测序列（[-1, 1] 范围）
        act: [B, num_steps_conditioning, act_dim] 动作序列（归一化动作）
        rew: [B] 标量奖励，对应最后一个观测
        trainer_cfg: 训练配置（包含 denoiser 和 reward_model 的训练参数）
        denoiser: Denoiser 模型
        denoiser_optimizer: Denoiser 优化器
        denoiser_lr_scheduler: Denoiser 学习率调度器
        denoiser_start_step: Denoiser 当前训练步数
        reward_model: Reward Model 模型
        reward_optimizer: Reward Model 优化器
        reward_lr_scheduler: Reward Model 学习率调度器
        reward_cfg: Reward Model 配置（用于数据预处理）
        processor: 图像预处理器（用于 reward model 输入处理）
        instructions: 任务指令列表 [B]
        reward_start_step: Reward Model 当前训练步数
        writer: TensorBoard SummaryWriter（用于记录训练日志）
    
    Returns:
        denoiser_step: 更新后的 Denoiser 训练步数
        reward_step: 更新后的 Reward Model 训练步数
    """
    device = obs.device
    B, T = obs.shape[:2]
    mask_padding = torch.ones(obs.shape[:2], dtype=torch.bool, device=device)
    
    # 获取训练步数
    steps_per_epoch = trainer_cfg.trainer.steps_per_epoch
    
    # 准备 Reward Model 的数据（只需准备一次）
    # 取最后一个观测，对应 reward
    last_obs = obs[:, -1]  # [B, C, H, W]
    
    # 将观测转换为 reward model 需要的格式（只需转换一次）
    inputs_list = []
    for i in range(B):
        obs_tensor = last_obs[i]  # [C, H, W] in [-1, 1]
        obs_np = obs_tensor.cpu().numpy().transpose(1, 2, 0)  # [H, W, C]
        obs_img = ((obs_np + 1) / 2 * 255).astype(np.uint8)
        obs_dict = {"full_image": obs_img}
        
        inputs = prepare_one_obs(
            reward_cfg,
            processor,
            obs_dict,
            instructions[i],
            reward_model.model_dtype,
        )
        inputs_list.append(inputs)
    
    inputs_batch = reward_model.prepare_inputs_batch(inputs_list)
    labels = (rew > 0).long()  # [B]
    
    # 获取批次配置
    batch_size = trainer_cfg.trainer.batch_size
    grad_accum = trainer_cfg.trainer.grad_accum
    
    # 多步训练循环
    for step_idx in range(steps_per_epoch):
        # ========== 1. 训练 Denoiser ==========
        denoiser.train()
        denoiser_optimizer.zero_grad()
        
        # 计算 Denoiser 的批次数量
        num_denoiser_samples = obs.shape[0]
        num_denoiser_batches = (num_denoiser_samples + batch_size - 1) // batch_size
        
        total_denoiser_loss = 0.0
        denoiser_update_count = 0
        for batch_idx in range(num_denoiser_batches):
            # 获取当前批次
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_denoiser_samples)
            
            obs_batch = obs[start_idx:end_idx]
            act_batch = act[start_idx:end_idx]
            mask_batch = mask_padding[start_idx:end_idx]
            
            # 前向传播
            loss_denoiser, logs = denoiser(obs_batch, act_batch, mask_batch)
            loss_denoiser = loss_denoiser / grad_accum  # 梯度累积缩放
            loss_denoiser.backward()
            
            total_denoiser_loss += loss_denoiser.item() * grad_accum
            
            # 每 grad_accum 个 batch 或最后一个 batch 更新梯度
            if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == num_denoiser_batches:
                torch.nn.utils.clip_grad_norm_(denoiser.parameters(), trainer_cfg.denoiser.training.max_grad_norm)
                denoiser_optimizer.step()
                denoiser_lr_scheduler.step()
                denoiser_optimizer.zero_grad()
                denoiser_update_count += 1
        
        avg_denoiser_loss = total_denoiser_loss / num_denoiser_batches
        writer.add_scalar("train/denoiser_loss", avg_denoiser_loss, denoiser_start_step)
        writer.add_scalar("train/denoiser_lr", denoiser_lr_scheduler.get_last_lr()[0], denoiser_start_step)
        
        # ========== 2. 训练 Reward Model ==========
        reward_model.train()
        reward_optimizer.zero_grad()
        
        # 计算 Reward Model 的批次数量
        num_reward_samples = B
        num_reward_batches = (num_reward_samples + batch_size - 1) // batch_size
        
        total_reward_loss = 0.0
        total_tp = total_tn = total_fp = total_fn = 0
        reward_update_count = 0
        
        for batch_idx in range(num_reward_batches):
            # 获取当前批次
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_reward_samples)
            
            # 直接对 inputs_batch 中的 tensor 切片（避免重复调用 prepare_inputs_batch）
            batch_inputs = {k: v[start_idx:end_idx] for k, v in inputs_batch.items()}
            batch_labels = labels[start_idx:end_idx]
            
            # 前向传播
            logits = reward_model.forward(batch_inputs)
            loss_reward, metrics_reward = reward_model.compute_loss_and_metrics(batch_inputs, batch_labels)
            loss_reward = loss_reward / grad_accum  # 梯度累积缩放
            loss_reward.backward()
            
            total_reward_loss += loss_reward.item() * grad_accum
            total_tp += metrics_reward["tp"].item()
            total_tn += metrics_reward["tn"].item()
            total_fp += metrics_reward["fp"].item()
            total_fn += metrics_reward["fn"].item()
            
            # 每 grad_accum 个 batch 或最后一个 batch 更新梯度
            if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == num_reward_batches:
                torch.nn.utils.clip_grad_norm_(reward_model.parameters(), trainer_cfg.reward_model.training.clip_grad_norm)
                reward_optimizer.step()
                reward_lr_scheduler.step()
                reward_optimizer.zero_grad()
                reward_update_count += 1
        
        avg_reward_loss = total_reward_loss / num_reward_batches
        
        # 记录训练日志
        writer.add_scalar("train/reward_model_loss", avg_reward_loss, reward_start_step)
        writer.add_scalar("train/reward_model_lr", reward_optimizer.param_groups[0]["lr"], reward_start_step)
        writer.add_scalar("train/reward_model_tp", total_tp, reward_start_step)
        writer.add_scalar("train/reward_model_tn", total_tn, reward_start_step)
        writer.add_scalar("train/reward_model_fp", total_fp, reward_start_step)
        writer.add_scalar("train/reward_model_fn", total_fn, reward_start_step)
        
        # 计算准确率
        pos_den = total_tp + total_fn
        neg_den = total_tn + total_fp
        pos_acc = float(total_tp) / pos_den if pos_den > 0 else 0.0
        neg_acc = float(total_tn) / neg_den if neg_den > 0 else 0.0
        writer.add_scalar("train/reward_model_pos_acc", pos_acc, reward_start_step)
        writer.add_scalar("train/reward_model_neg_acc", neg_acc, reward_start_step)
        
        # 更新 step 计数器
        denoiser_start_step += 1
        reward_start_step += 1
        
        print(f"  训练步骤 {step_idx + 1}/{steps_per_epoch}: "
              f"denoiser_loss={avg_denoiser_loss:.4f} (batches={num_denoiser_batches}, updates={denoiser_update_count}), "
              f"reward_loss={avg_reward_loss:.4f} (batches={num_reward_batches}, updates={reward_update_count}), "
              f"reward_pos_acc={pos_acc:.4f}, "
              f"reward_neg_acc={neg_acc:.4f}")
    
    return denoiser_start_step, reward_start_step

def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算 Generalized Advantage Estimation (GAE)

    Args:
        rewards: [B, T] 奖励
        values: [B, T] 状态价值
        next_values: [B, T] 下一状态价值
        dones: [B, T] 结束标志
        gamma: 折扣因子
        gae_lambda: GAE lambda参数

    Returns:
        advantages: [B, T] 优势函数
        returns: [B, T] 回报
    """
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)

    # 计算TD误差
    deltas = rewards + gamma * next_values * (1 - dones) - values

    # 从后往前计算GAE
    gae = torch.zeros(B, device=rewards.device)
    for t in reversed(range(T)):
        gae = deltas[:, t] + gamma * gae_lambda * (1 - dones[:, t]) * gae
        advantages[:, t] = gae
        returns[:, t] = advantages[:, t] + values[:, t]

    return advantages, returns


def ppo_update(
    actor_critic: ActorCritic,
    rollout_data: Dict[str, torch.Tensor],
    optimizer: optim.Optimizer,
    clip_ratio: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    num_epochs: int = 4,
    batch_size: int = 64,
    gradient_accumulation_steps: int = 1,
) -> Dict[str, float]:
    """
    执行PPO更新

    Args:
        actor_critic: ActorCritic模型
        rollout_data: rollout数据，包含obs, act, rew, advantages, returns等
        optimizer: 优化器
        clip_ratio: PPO clipping参数
        value_coef: 价值损失系数
        entropy_coef: 熵损失系数
        max_grad_norm: 最大梯度范数
        num_epochs: PPO更新轮数
        batch_size: mini batch大小
        gradient_accumulation_steps: 梯度累计步数

    Returns:
        训练指标字典
    """
    obs = rollout_data["obs"]  # [B, T_ac, C, H, W]
    old_action_logits = rollout_data["act_logits"]  # [B, T_ac, 8, vocab_size] - old action logits
    old_action_tokens = rollout_data["act_tokens"]  # [B, T_ac, 8] - action tokens actually executed
    advantages = rollout_data["advantages"]  # [B, T_ac]
    returns = rollout_data["returns"]  # [B, T_ac]
    old_values = rollout_data["val"]  # [B, T_ac] - old values
    mask = rollout_data["mask"]  # [B, T_ac] - valid mask
    instructions = rollout_data.get("instructions", [""] * obs.shape[0])

    B, T_ac, C, H, W = obs.shape
    print(f"PPO update batch shape: {obs.shape}")
    
    if B == 0 or T_ac == 0:
        print("Skipping PPO update due to empty batch")
        return {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy_loss": 0.0,
            "total_loss": 0.0,
            "mean_reward": 0.0,
            "mean_episode_length": 0.0,
        }

    # 确保所有张量在同一个设备上
    device = actor_critic.device
    obs = obs.to(device)
    old_action_logits = old_action_logits.to(device)
    old_action_tokens = old_action_tokens.to(device)
    advantages = advantages.to(device)
    returns = returns.to(device)
    old_values = old_values.to(device)

    # 展平批次和时间维度，只保留有效的样本
    mask_flat = mask.reshape(-1)  # [B*T_ac]
    valid_indices = torch.where(mask_flat)[0].to(device)  # Only train on valid steps

    obs_flat = obs.reshape(B * T_ac, C, H, W)[valid_indices]  # [N_valid, C, H, W]
    advantages_flat = advantages.reshape(B * T_ac)[valid_indices]  # [N_valid]
    returns_flat = returns.reshape(B * T_ac)[valid_indices]  # [N_valid]
    old_values_flat = old_values.reshape(B * T_ac)[valid_indices]  # [N_valid]
    old_action_tokens_flat = old_action_tokens.reshape(B * T_ac, -1)[valid_indices]  # [N_valid, 8]
    old_action_logits_flat = old_action_logits.reshape(B * T_ac, *old_action_logits.shape[2:])[valid_indices]  # [N_valid, 8, vocab_size]

    # 标准化优势函数
    advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)

    # 为每个批次重复指令
    instructions_expanded = []
    for i in range(B):
        instructions_expanded.extend([instructions[i]] * T_ac)
    instructions_flat = instructions_expanded

    total_policy_loss = 0
    total_value_loss = 0
    total_entropy_loss = 0
    total_loss = 0
    num_forward = 0

    # PPO更新循环
    for epoch in range(num_epochs):
        # 清零梯度
        optimizer.zero_grad()

        # 只对有效样本进行打乱
        N_valid = len(obs_flat)
        indices = torch.randperm(N_valid, device=device)

        obs_shuffled = obs_flat[indices]
        advantages_shuffled = advantages_flat[indices]
        returns_shuffled = returns_flat[indices]
        old_values_shuffled = old_values_flat[indices]
        old_action_tokens_shuffled = old_action_tokens_flat[indices]
        old_logits_shuffled = old_action_logits_flat[indices]

        # 分批处理（支持梯度累计）
        accumulation_step = 0
        for start_idx in range(0, N_valid, batch_size):
            end_idx = min(start_idx + batch_size, N_valid)
            batch_indices = indices[start_idx:end_idx]

            batch_obs = obs_shuffled[start_idx:end_idx]
            batch_advantages = advantages_shuffled[start_idx:end_idx]
            batch_returns = returns_shuffled[start_idx:end_idx]
            batch_old_values = old_values_shuffled[start_idx:end_idx]
            batch_old_action_tokens = old_action_tokens_shuffled[start_idx:end_idx]
            batch_old_logits = old_logits_shuffled[start_idx:end_idx]

            # 修正instructions索引：用batch_indices对应的原始位置
            batch_original_indices = valid_indices[batch_indices]  # 从valid_indices中取原始位置
            batch_instructions = [instructions_flat[i] for i in batch_original_indices.cpu().tolist()]

            # 准备输入
            inputs_list = []
            for i in range(len(batch_obs)):
                # Convert tensor [C, H, W] to numpy [H, W, C] uint8
                obs_tensor = batch_obs[i]
                obs_np = obs_tensor.cpu().numpy().transpose(1, 2, 0)
                obs_img = ((obs_np + 1) / 2 * 255).astype(np.uint8)
                obs_dict = {"full_image": obs_img}

                inputs = prepare_one_obs(
                    actor_critic.cfg,
                    actor_critic.processor,
                    obs_dict,
                    batch_instructions[i],
                    actor_critic.model_dtype,
                )
                inputs_list.append(inputs)

            inputs_batch = actor_critic.prepare_inputs_batch(inputs_list)

            # 前向传播
            action_logits, values = actor_critic.forward(inputs_batch)
            # action_logits: [batch_size, 8*7, 256] (NUM_ACTIONS_CHUNK * ACTION_DIM)
            # values: [batch_size]

            # 计算新旧策略的log概率
            # 我们使用重要性采样：使用rollout时实际执行的动作tokens

            # 从新策略和旧策略计算rollout时执行的动作的log概率
            new_dist = torch.distributions.Categorical(logits=action_logits)
            old_dist = torch.distributions.Categorical(logits=batch_old_logits)

            old_log_probs = old_dist.log_prob(batch_old_action_tokens)  # [batch_size, 8]
            new_log_probs = new_dist.log_prob(batch_old_action_tokens)  # [batch_size, 8]

            # PPO策略损失 
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * batch_advantages.unsqueeze(-1)  # 广播到action_dim维度
            surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * batch_advantages.unsqueeze(-1)
            policy_loss = -torch.min(surr1, surr2).mean()  # 求和后平均

            # 价值损失 (拟合returns，而不是旧value)
            value_loss = torch.nn.functional.mse_loss(values, batch_returns)

            # 熵损失 
            entropy_loss = -new_dist.entropy().mean()  # 求和后平均

            # 总损失
            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

            # 梯度累计
            loss = loss / gradient_accumulation_steps  # 缩放损失
            loss.backward()
            num_forward += 1
            accumulation_step += 1

            # 达到累计步数时更新参数，或者这是最后一个batch时也要更新
            is_last_batch = (start_idx + batch_size >= N_valid)  # Check if this is the last batch in valid samples
            if accumulation_step % gradient_accumulation_steps == 0 or is_last_batch:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), max_grad_norm)
                # 参数更新
                optimizer.step()
                optimizer.zero_grad()
                accumulation_step = 0  # 重置计数器

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()
            total_loss += loss.item() * gradient_accumulation_steps  # 恢复原始损失值

    # 计算平均指标
    if num_forward > 0:
        metrics = {
            "policy_loss": total_policy_loss / num_forward,
            "value_loss": total_value_loss / num_forward,
            "entropy_loss": total_entropy_loss / num_forward,
            "total_loss": total_loss / num_forward,
            "mean_reward": rollout_data["rew"].sum(dim=1).mean().item(),
            "mean_episode_length": T_ac,
        }
    else:
        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy_loss": 0.0,
            "total_loss": 0.0,
            "mean_reward": 0.0,
            "mean_episode_length": 0.0,
        }

    return metrics


def main():
    """主函数：执行PPO强化学习训练"""
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    # 配置路径
    current_dir = Path.cwd()
    agent_config_path = current_dir / "envs/config/agent.yaml"
    trainer_config_path = current_dir / "envs/config/trainer.yaml"

    print("=" * 80)
    print("WorldModelEnvBatch PPO Training")
    print("=" * 80)

    # 训练配置
    num_iterations = 100  # 训练迭代次数
    num_trajectories_per_iter = 5  # 每次迭代收集的轨迹数
    rollout_max_steps = 8  # 每次rollout的最大步数
    rollout_batch_size = 8  # rollout批大小

    # PPO配置
    learning_rate = 1e-5
    clip_ratio = 0.2
    value_coef = 0.5
    entropy_coef = 0.0
    max_grad_norm = 0.5
    ppo_epochs = 1
    ppo_batch_size = 8
    gradient_accumulation_steps = 32

    # 设置TensorBoard
    log_dir = current_dir / "runs/simple_diffusion_wm_rl" / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_wm_train_slide"
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard logs will be saved to: {log_dir}")

    # 加载模型
    print("\n加载模型...")

    # 加载 Denoiser
    agent_cfg = OmegaConf.load(agent_config_path)
    trainer_cfg = OmegaConf.load(trainer_config_path)
    denoiser, denoiser_optimizer, denoiser_lr_scheduler, denoiser_start_step = load_denoiser_from_checkpoint(
        agent_cfg, trainer_cfg, device
    )
    sampler_cfg = instantiate(trainer_cfg.world_model_env.diffusion_sampler)

    reward_model, reward_optimizer, reward_lr_scheduler, processor, reward_cfg, reward_start_step = load_reward_model_from_checkpoint(
        agent_cfg, trainer_cfg, device
    )

    # 创建 WorldModelEnvBatch配置
    env_cfg = WorldModelEnvConfig(
        horizon=trainer_cfg.world_model_env.horizon,
        num_batches_to_preload=trainer_cfg.world_model_env.num_batches_to_preload,
        diffusion_sampler=sampler_cfg,
    )

    # 加载 ActorCritic 模型
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
        unnorm_key="libero_spatial_no_noops",
        device=device,
        checkpoint2=agent_cfg.checkpoint2_path,
    )
    actor = ActorCritic(actor_cfg, torch.bfloat16)
    actor.train()  # 设置为训练模式

    # 设置PPO优化器
    optimizer = optim.Adam(actor.parameters(), lr=learning_rate)

    # 初始化 LiberoEnvWrapper
    BENCHMARK = "libero_spatial"
    TASK_ID = 0
    libero_env = LiberoEnvWrapper(
        benchmark_name=BENCHMARK,
        task_id=TASK_ID,
        image_size=224,
        render_mode="rgb_array",
    )

    num_steps_conditioning = agent_cfg.denoiser.inner_model.num_steps_conditioning
    eval_interval = trainer_cfg.trainer.eval_interval
    num_to_keep = trainer_cfg.trainer.num_to_keep
    checkpoint_dir = log_dir / "checkpoints"
    best_denoiser_loss = float('inf')
    best_reward_loss = float('inf')
    
    # 初始化统计滑动窗口
    success_window = deque(maxlen=100)
    length_window = deque(maxlen=100)

    # 训练循环
    print(f"\n开始训练，共 {num_iterations} 次迭代...")
    obs_list = deque(maxlen=50000)
    act_list = deque(maxlen=50000)
    rew_list = deque(maxlen=50000)
    step_counts = deque(maxlen=50000)
    instructions = deque(maxlen=50000)
    
    for iteration in range(num_iterations):
        print(f"\n=== 迭代 {iteration + 1}/{num_iterations} ===")
        start_time = time.time()

        # 1. 收集初始数据
        print(f"收集 {num_trajectories_per_iter} 条轨迹的初始数据...")
        if len(obs_list) == 0 or iteration % 5 == 0:
            obs_list_t, act_list_t, rew_list_t, step_counts_t, instructions_t = collect_initial_obs_act_from_libero(
                env=libero_env,
                actor=actor,
                num_steps_conditioning=num_steps_conditioning,
                num_trajectories=num_trajectories_per_iter,
                device=device,
                deterministic=False,
                writer=writer,
                global_step=iteration,
                success_window=success_window,
                length_window=length_window,
                window_size=100,
            )
            obs_list.extend(obs_list_t)
            act_list.extend(act_list_t)
            rew_list.extend(rew_list_t)
            step_counts.extend(step_counts_t)
            instructions.extend(instructions_t)

        if len(obs_list) == 0:
            print("警告：未收集到有效轨迹，跳过此次迭代")
            continue

        # 随机采样数据用于rollout
        num_available_samples = len(obs_list)
        num_samples_for_rollout = min(512, num_available_samples)
        sample_indices = torch.randperm(num_available_samples)[:num_samples_for_rollout].tolist()
        
        # 从列表中选择样本
        obs_sampled = [obs_list[i] for i in sample_indices]
        act_sampled = [act_list[i] for i in sample_indices]
        rew_sampled = [rew_list[i] for i in sample_indices]
        step_counts_sampled = [step_counts[i] for i in sample_indices]
        instructions_sampled = [instructions[i] for i in sample_indices]
        
        print(f"从 {num_available_samples} 个样本中随机选择了 {num_samples_for_rollout} 个样本用于rollout")

        # 转换为批格式
        obs = torch.stack(list(obs_list), dim=0)
        act = torch.stack(list(act_list), dim=0)
        rew = torch.stack(list(rew_list), dim=0)
        initial_obs = torch.stack(obs_sampled, dim=0)[:, :num_steps_conditioning]  # [B, T, C, H, W]
        initial_act = torch.stack(act_sampled, dim=0)[:, :num_steps_conditioning-1]  # [B, T-1, act_dim]
        initial_step_counts = torch.tensor(step_counts_sampled, device=device, dtype=torch.long)  # [B]

        print("训练 World Model...")
        wm_bs = 1024
        start_buffer_size = 4096
        if num_available_samples < start_buffer_size:
            print(f"样本数量不足512（当前={num_available_samples}），跳过本次 World Model 训练")
        else:
            sample_indices_wm = torch.randperm(num_available_samples)[:wm_bs].tolist()
            obs_wm = torch.stack([obs_list[i] for i in sample_indices_wm], dim=0)
            act_wm = torch.stack([act_list[i] for i in sample_indices_wm], dim=0)
            rew_wm = torch.stack([rew_list[i] for i in sample_indices_wm], dim=0)
            instructions_wm = [instructions[i] for i in sample_indices_wm]

            denoiser_start_step, reward_start_step = train_world_model(
                obs=obs_wm,
                act=act_wm,
                rew=rew_wm,
                trainer_cfg=trainer_cfg,
                denoiser=denoiser,
                denoiser_optimizer=denoiser_optimizer,
                denoiser_lr_scheduler=denoiser_lr_scheduler,
                denoiser_start_step=denoiser_start_step,
                reward_model=reward_model,
                reward_optimizer=reward_optimizer,
                reward_lr_scheduler=reward_lr_scheduler,
                reward_cfg=reward_cfg,
                processor=processor,
                instructions=instructions_wm,
                reward_start_step=reward_start_step,
                writer=writer,
            )

        if (iteration + 1) % eval_interval == 0 and False:
            print(f"\n=== 测试 World Model (Iteration {iteration + 1}) ===")
            # TODO 需重新收集测试数据 obs/act/rew 。此处省略，用训练集来测试。
            eval_metrics = evaluate_world_model(
                obs=obs,
                act=act,
                rew=rew,
                denoiser=denoiser,
                reward_model=reward_model,
                reward_cfg=reward_cfg,
                processor=processor,
                instructions=instructions,
                device=device,
                batch_size=trainer_cfg.trainer.batch_size,
            )
            
            print(f"Evaluation Results:")
            print(f"  Denoiser Loss: {eval_metrics['denoiser_loss']:.4f}")
            print(f"  Reward Model Loss: {eval_metrics['reward_loss']:.4f}")
            print(f"  Reward Pos Acc: {eval_metrics['reward_pos_acc']:.4f}")
            print(f"  Reward Neg Acc: {eval_metrics['reward_neg_acc']:.4f}")
            
            writer.add_scalar("eval/denoiser_loss", eval_metrics['denoiser_loss'], iteration)
            writer.add_scalar("eval/reward_loss", eval_metrics['reward_loss'], iteration)
            writer.add_scalar("eval/reward_pos_acc", eval_metrics['reward_pos_acc'], iteration)
            writer.add_scalar("eval/reward_neg_acc", eval_metrics['reward_neg_acc'], iteration)
            
            is_best_denoiser = eval_metrics['denoiser_loss'] < best_denoiser_loss
            is_best_reward = eval_metrics['reward_loss'] < best_reward_loss
            
            if is_best_denoiser:
                best_denoiser_loss = eval_metrics['denoiser_loss']
                print(f"  *** Best Denoiser! (loss={best_denoiser_loss:.4f}) ***")
            
            if is_best_reward:
                best_reward_loss = eval_metrics['reward_loss']
                print(f"  *** Best Reward Model! (loss={best_reward_loss:.4f}) ***")
            
            # save_checkpoint(
            #     save_dir=checkpoint_dir,
            #     iteration=iteration + 1,
            #     denoiser=denoiser,
            #     denoiser_optimizer=denoiser_optimizer,
            #     denoiser_lr_scheduler=denoiser_lr_scheduler,
            #     denoiser_step=denoiser_start_step,
            #     reward_model=reward_model,
            #     reward_optimizer=reward_optimizer,
            #     reward_lr_scheduler=reward_lr_scheduler,
            #     reward_step=reward_start_step,
            #     reward_cfg=reward_cfg,
            #     eval_metrics=eval_metrics,
            #     is_best_denoiser=is_best_denoiser,
            #     is_best_reward=is_best_reward,
            # )
            
            # manage_checkpoints(checkpoint_dir, num_to_keep)
            print("=" * 80 + "\n")

        # 创建 WorldModelEnvBatch
        env_batch = WorldModelEnvBatch(
            denoiser=denoiser,
            cfg=env_cfg,
            reward_model=reward_model,
            reward_cfg=reward_cfg,
            processor=processor,
            torch_dtype=torch.bfloat16,
            instructions=instructions_sampled,
            return_denoising_trajectory=False,
        )

        # 2. 执行rollout
        print("执行rollout...")
        rollout_data = rollout_with_world_model_batched(
            env_batch=env_batch,
            actor=actor,
            initial_obs=initial_obs,
            initial_act=initial_act,
            instructions=instructions_sampled,
            initial_step_counts=initial_step_counts,
            max_steps=rollout_max_steps,
            deterministic=False,
            batch_size=rollout_batch_size,
        )

        # 3. 执行PPO更新 (GAE已经在rollout_data中)
        print("执行PPO更新...")
        rollout_data_for_ppo = rollout_data.copy()
        # Use instructions returned from rollout (which are filtered and aligned)
        if "instructions" in rollout_data:
            rollout_data_for_ppo["instructions"] = rollout_data["instructions"]
        else:
            rollout_data_for_ppo["instructions"] = instructions_sampled

        metrics = ppo_update(
            actor_critic=actor,
            rollout_data=rollout_data_for_ppo,
            optimizer=optimizer,
            clip_ratio=clip_ratio,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
            num_epochs=ppo_epochs,
            batch_size=ppo_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        # 5. 记录指标到TensorBoard
        iteration_time = time.time() - start_time

        writer.add_scalar("train/policy_loss", metrics["policy_loss"], iteration)
        writer.add_scalar("train/value_loss", metrics["value_loss"], iteration)
        writer.add_scalar("train/entropy_loss", metrics["entropy_loss"], iteration)
        writer.add_scalar("train/total_loss", metrics["total_loss"], iteration)
        writer.add_scalar("train/mean_reward", metrics["mean_reward"], iteration)
        writer.add_scalar("train/mean_episode_length", metrics["mean_episode_length"], iteration)
        writer.add_scalar("train/iteration_time", iteration_time, iteration)
        writer.add_scalar("train/total_reward", rollout_data["rew"].sum().item(), iteration)
        writer.add_scalar("train/num_trajectories", len(obs_list), iteration)

        print(f"迭代 {iteration + 1} 完成:")
        print(f"策略损失: {metrics['policy_loss']:.4f}")
        print(f"价值损失: {metrics['value_loss']:.4f}")
        print(f"熵损失: {metrics['entropy_loss']:.4f}")
        print(f"总损失: {metrics['total_loss']:.4f}")
        print(f"平均奖励: {metrics['mean_reward']:.4f}")
        print(f"迭代时间: {iteration_time:.2f}秒")

        # 每10次迭代保存模型检查点
        if (iteration + 1) % 10 == 0:
            pass
            # checkpoint_path = log_dir / "actor_critic" / "checkpoint_latest.pt"
            # checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            # torch.save({
            #     'iteration': iteration + 1,
            #     'model_state_dict': actor.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'metrics': metrics,
            # }, checkpoint_path)
            # print(f"保存检查点到: {checkpoint_path}")

    libero_env.close()
    writer.close()
    print("\n训练完成！")
    print(f"TensorBoard日志保存在: {log_dir}")


if __name__ == "__main__":
    main()

