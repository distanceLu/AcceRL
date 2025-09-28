import os
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, List
from contextlib import nullcontext
import numpy as np
from collections import deque

from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform

from peft import LoraConfig, get_peft_model
from experiments.robot.openvla_utils import L1RegressionActionHead

# Core OpenVLA components
from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
)

# Masks used to extract action-related hidden states
from prismatic.training.train_utils import (
    get_current_action_mask,
    get_next_actions_mask,
)

# Constants
from prismatic.vla.constants import (
    NUM_ACTIONS_CHUNK,
    ACTION_DIM,
    PROPRIO_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
)
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType
from typing import Any
import torch

# 显式类：避免依赖 auto_map
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction

from rl.actor_critic_model import ActorCritic


if __name__ == "__main__":
    import numpy as np
    import random
    import time
    from experiments.robot.robot_utils import set_seed_everywhere
    

    # Libero env wrapper and helpers
    from rl.libero_env import LiberoEnvWrapper
    from rl.utils import prepare_one_obs, check_unnorm_key
    from experiments.robot.libero.libero_utils  import GenerateConfig, TaskSuite
    device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
    
    # Precision policy to match the example
    USE_BF16: bool = True
    TORCH_DTYPE = torch.bfloat16 if USE_BF16 else torch.float32

    # 在这里设置要并行处理的环境数量
    ENVS_ID = [5]
    envs_num = len(ENVS_ID)
    BENCHMARK = TaskSuite.LIBERO_SPATIAL

    unnorm_key = f"{BENCHMARK}_no_noops"
    # Instantiate config
    cfg = GenerateConfig(
        pretrained_checkpoint="/cpfs01/liuwei_workspace/openvla_oft_rl/ckpt/finetune_nll_std_param/openvla-7b-oft-finetuned-libero-spatial-object-goal-10+libero_spatial_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state--40000_chkpt", #/cpfs01/lcx_workspace/models/openvla-7b-oft-finetuned-libero-spatial-object-goal-10/
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=True,
        num_open_loop_steps=NUM_ACTIONS_CHUNK,
        unnorm_key=unnorm_key,
        device=device,
    )
    set_seed_everywhere(cfg.seed)
    # Create ActorCritic policy
    actor = ActorCritic(cfg, TORCH_DTYPE)
    # actor.load_log_std(cfg.pretrained_checkpoint, step="latest")

    check_unnorm_key(cfg, actor.vla)
    actor.get_parameter_groups()
    actor.eval()
    for key, value in actor.named_parameters():
        if value.dtype != TORCH_DTYPE:
            print(f"警告: 参数 {key} 的数据类型是 {value.dtype}, 但期望的是 {TORCH_DTYPE}.")
    print("策略初始化完成。")

    # --- 并行初始化多个环境 ---
    print(f"正在初始化 {len(ENVS_ID)} 个并行的 Libero 环境...")
    envs = [
        LiberoEnvWrapper(
            benchmark_name=BENCHMARK,
            task_id=env_id,  # 每个环境一个随机任务
            image_size=224,
            render_mode="rgb_array",
        )
        for env_id in ENVS_ID
    ]
    print("所有环境初始化完成。")

    # --- 初始化所有环境的状态 ---
    # 使用列表来独立跟踪每个环境的状态
    observations = []
    task_descriptions = []
    for i, env in enumerate(envs):
        # 为每个环境设置不同的随机种子以保证多样性
        obs, info = env.reset(seed=0)
        observations.append(obs)
        task_descriptions.append(env.task_description)
        print(f"环境 {i}: 任务 ID = {env.task_id}, 任务描述 = {env.task_description}")

    action_queues = [deque(maxlen=cfg.num_open_loop_steps) for _ in range(envs_num)]

    # 跟踪每个环境是否仍在活动、奖励和步数
    active_envs = [True] * envs_num
    total_rewards = [0.0] * envs_num
    episode_steps = [0] * envs_num
    success_info = [False] * envs_num

    # 用于统计最终成功率
    total_episodes_finished = 0
    total_successes = 0
    truncated_episodes = 0

    print("\n开始并行执行所有环境...")

    # --- 主循环：只要有任何一个环境在活动，就继续 ---
    while any(active_envs):
        # ============================ 修改后的核心逻辑: 按需推理 ============================

        # 1. 识别哪些环境的动作队列已空，需要进行新的推理
        inputs_for_inference = []
        indices_needing_inference = []
        for i in range(envs_num):
            if active_envs[i] and len(action_queues[i]) == 0:
                inputs_t = prepare_one_obs(cfg, actor.processor, observations[i], task_descriptions[i], TORCH_DTYPE)
                inputs_for_inference.append(inputs_t)
                indices_needing_inference.append(i)

        # 2. 如果有需要推理的环境，则执行一次批处理前向传播
        if inputs_for_inference:
            inputs_batch = actor.prepare_inputs_batch(inputs_for_inference)
            
            with torch.no_grad():
                sample_all, mu_all, _, _ = actor.forward(inputs_batch)
                action_all_norm = torch.clamp(sample_all, -1.0, 1.0)
                # action_all_norm = torch.clamp(mu_all, -1.0, 1.0)

            # 3. 将生成的动作块（chunks）填充到对应的队列中
            actions_unnorm = actor.vla._unnormalize_actions(action_all_norm.cpu().numpy(), cfg.unnorm_key)
            
            for i, env_idx in enumerate(indices_needing_inference):
                # actions_unnorm[i] 是一个形状为 (num_chunks, action_dim) 的数组
                action_queues[env_idx].extend(actions_unnorm[i])

        # ============================ 修改后的核心逻辑: 顺序执行 ============================
        
        # 4. 为所有活动环境执行队列中的一个动作
        for env_idx in range(envs_num):
            if not active_envs[env_idx]:
                continue
            
            # 从队列中取出一个动作
            # 此时可以保证队列非空，因为前面已经填充过了
            action_env = action_queues[env_idx].popleft()

            # 在对应的环境中执行动作
            obs, reward, terminated, truncated, info = envs[env_idx].step(action_env)

            # 更新该环境的状态
            observations[env_idx] = obs
            total_rewards[env_idx] += float(reward)
            episode_steps[env_idx] += 1

            if episode_steps[env_idx] % 50 == 0:
                print(f"环境 {env_idx}, Step: {episode_steps[env_idx]}, 奖励: {reward:.4f}, 终止: {terminated}, 截断: {truncated}")

            # 5. 检查环境是否完成
            if terminated or truncated:
                if truncated:
                    truncated_episodes += 1
                    print(f"环境 {env_idx} 因达到最大步数而截断。累计截断次数: {truncated_episodes}")
                active_envs[env_idx] = False # 标记环境为非活动
                is_success = info.get('is_success', False)
                total_successes += int(is_success)
                total_episodes_finished += 1
                success_info[env_idx] = is_success
                
                print("-" * 40)
                print(f"环境 {env_idx} 已完成 (任务: {envs[env_idx].task_description[:50]}...)")
                print(f"  总步数: {episode_steps[env_idx]}, 总奖励: {total_rewards[env_idx]:.4f}, 是否成功: {is_success}")
                if total_episodes_finished > 0:
                    print(f"当前成功率: {total_successes / total_episodes_finished:.2%}, 完成的回合数: {total_episodes_finished}")
                print("-" * 40)

                # # 如果需要自动重置并继续跑，可以取消下面的注释
                episode_steps[env_idx] = 0
                total_rewards[env_idx] = 0
                obs, info = envs[env_idx].reset(seed=0) # 可以用不同的seed增加多样性
                observations[env_idx] = obs
                active_envs[env_idx] = True
                action_queues[env_idx].clear() # 清空旧的动作队列
    
    print("\n所有环境执行完毕。")
    if total_episodes_finished > 0:
        final_rate = total_successes / total_episodes_finished
        print(f"最终成功率: {final_rate:.2%} ({total_successes}/{total_episodes_finished})")