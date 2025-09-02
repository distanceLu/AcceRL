import os
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, List
from contextlib import nullcontext
import numpy as np

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
    from experiments.robot.libero.run_libero_eval import GenerateConfig, TaskSuite
    device = torch.device("cuda:3") if torch.cuda.is_available() else torch.device("cpu")
    
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
        pretrained_checkpoint="/cpfs01/liuwei_workspace/openvla_oft_rl/ckpt/finetune_nll_16/openvla-7b-oft-finetuned-libero-spatial-object-goal-10+libero_spatial_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state", #/cpfs01/lcx_workspace/models/openvla-7b-oft-finetuned-libero-spatial-object-goal-10/
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

    # 从你的检查点目录名中提取步数
    checkpoint_step = 'latest'   # 或者设置为特定的步数，例如 10000 'latest'
    actor.load_weights_for_eval(cfg.pretrained_checkpoint, checkpoint_step)

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

    # 跟踪每个环境是否仍在活动、奖励和步数
    active_envs = [True] * envs_num
    total_rewards = [0.0] * envs_num
    episode_steps = [0] * envs_num
    success_info = [False] * envs_num

    # 用于统计最终成功率
    total_episodes_finished = 0
    total_successes = 0

    print("\n开始并行执行所有环境...")

    # --- 主循环：只要有任何一个环境在活动，就继续 ---
    while any(active_envs):
        # 1. 从所有【活动】的环境中收集输入数据
        inputs_t_list = []
        # 记录当前批次中数据对应的原始环境索引
        active_indices_this_step = []
        
        for i in range(envs_num):
            if active_envs[i]:
                inputs_t = prepare_one_obs(cfg, actor.processor, observations[i], task_descriptions[i], TORCH_DTYPE)
                inputs_t_list.append(inputs_t)
                active_indices_this_step.append(i)

        # 如果没有活动的输入，则退出循环
        if not inputs_t_list:
            break

        # 2. 使用类方法将输入列表批处理成一个大的张量
        #    这是实现并行处理的关键步骤
        inputs_batch = actor.prepare_inputs_batch(inputs_t_list)

        # 3. 执行一次前向传播，为批次中的所有环境获取动作
        with torch.no_grad():
            # actions_all 的形状是 (batch_size, num_chunks, action_dim)
            # 其中 batch_size 等于当前活动的任务数量 len(inputs_t_list)
            sample_all, mu_all, _, _, _ = actor.forward(inputs_batch)
            # action_all = torch.clamp(mu_all, -1.0, 1.0)
            action_all = torch.clamp(sample_all, -1.0, 1.0)

        # 4. 将批次动作分发回各自的环境并执行一步
        for i, env_idx in enumerate(active_indices_this_step):
            # i 是批次中的索引, env_idx 是原始环境列表中的索引
            action_norm = action_all[i, 0].cpu().numpy().astype(np.float32)
            action_env = actor.vla._unnormalize_actions(action_norm, cfg.unnorm_key)

            # 在对应的环境中执行动作
            obs, reward, terminated, truncated, info = envs[env_idx].step(action_env)

            # 更新该环境的状态
            observations[env_idx] = obs
            total_rewards[env_idx] += float(reward)
            episode_steps[env_idx] += 1

            # 使用确定性打印
            if episode_steps[env_idx] % 50 == 0:
                print(f"环境 {env_idx}, Step: {episode_steps[env_idx]}, 奖励: {reward:.4f}, 终止: {terminated}, 截断: {truncated}")

            # 5. 检查环境是否完成
            if terminated or truncated:
                is_success = info.get('is_success', False)
                total_successes += is_success
                total_episodes_finished += 1
                success_info[env_idx] = is_success
                
                # 打印单个环境完成的信息
                print("-" * 40)
                print(f"环境 {env_idx} 已完成 (任务: {envs[env_idx].task_description[:50]}...)")
                print(f"  总步数: {episode_steps[env_idx]}, 总奖励: {total_rewards[env_idx]:.4f}, 是否成功: {is_success}")
                print(f"Success rate: {total_successes / total_episodes_finished}, total_episodes_finished: {total_episodes_finished}")
                print("-" * 40)
                episode_steps[env_idx] = 0
                total_rewards[env_idx] = 0
                obs, info = envs[env_idx].reset(seed=0)
                observations[env_idx] = obs
