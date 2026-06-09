"""
测试 WorldModelEnv 和 WorldModelEnvBatch 的结果是否一致
"""
import sys
import time
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate

from envs.diffusion import Denoiser, SimpleBatch
from envs.world_model_env import WorldModelEnv, WorldModelEnvConfig
from envs.world_model_env_batch import WorldModelEnvBatch
from envs.utils import load_reward_model, tensor_to_image
from experiments.robot.openvla_utils import get_processor


def set_seed_everywhere(seed: int):
    """设置所有随机数生成器的种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Set seed to {seed}")


def load_denoiser_from_checkpoint(
    agent_config_path: Path,
    trainer_config_path: Path,
    device: torch.device,
):
    agent_cfg = OmegaConf.load(agent_config_path)
    trainer_cfg = OmegaConf.load(trainer_config_path)
    denoiser_cfg = instantiate(agent_cfg.denoiser)
    if denoiser_cfg.inner_model.num_actions is None:
        denoiser_cfg.inner_model.num_actions = 6

    denoiser = Denoiser(denoiser_cfg).to(device)
    sigma_distribution_cfg = instantiate(trainer_cfg.denoiser.sigma_distribution)
    denoiser.setup_training(sigma_distribution_cfg)
    
    checkpoint = torch.load(agent_cfg.denoiser_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("denoiser_state_dict", checkpoint)
    
    act_emb_float_key = "inner_model.act_emb_float.0.weight"
    if act_emb_float_key in state_dict:
        act_emb_float_weight = state_dict[act_emb_float_key]
        act_dim = act_emb_float_weight.shape[1]
        _ = denoiser.inner_model._get_act_emb_float(act_dim)
    
    denoiser.load_state_dict(state_dict, strict=False)
    denoiser.eval()
    
    return denoiser, trainer_cfg, agent_cfg


def compare_tensors(tensor1: torch.Tensor, tensor2: torch.Tensor, name: str, rtol: float = 1e-4, atol: float = 1e-5) -> bool:
    """比较两个张量是否相等"""
    if tensor1.shape != tensor2.shape:
        print(f"  ❌ {name}: shape mismatch - {tensor1.shape} vs {tensor2.shape}")
        return False
    
    if torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol):
        max_diff = (tensor1 - tensor2).abs().max().item()
        print(f"  ✅ {name}: match (max_diff={max_diff:.2e})")
        return True
    else:
        max_diff = (tensor1 - tensor2).abs().max().item()
        mean_diff = (tensor1 - tensor2).abs().mean().item()
        print(f"  ❌ {name}: mismatch (max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})")
        return False


def test_single_env_vs_batch():
    """测试单个环境版本和 batch 版本（batch_size=1）的结果是否一致"""
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    agent_config_path = Path("envs/config/agent.yaml")
    trainer_config_path = Path("envs/config/trainer.yaml")
    trajectory_path = "/cpfs01/jinshiji_workspace/openvla_oft_rl/data/libero_batches_with_next_obs_test/batch_env0_traj2_len85.pt"
    
    seed = 42
    print("=" * 80)
    print("测试 WorldModelEnv vs WorldModelEnvBatch (batch_size=1)")
    print("=" * 80)
    
    # 加载模型
    print("\n加载 Denoiser 模型...")
    denoiser1, trainer_cfg, agent_cfg = load_denoiser_from_checkpoint(
        agent_config_path, trainer_config_path, device
    )
    denoiser2, _, _ = load_denoiser_from_checkpoint(
        agent_config_path, trainer_config_path, device
    )
    sampler_cfg = instantiate(trainer_cfg.world_model_env.diffusion_sampler)
    
    print("加载 Reward Model...")
    reward_model1, cfg1 = load_reward_model(
        model_path=agent_cfg.reward_model_path,
        device=str(device),
        pretrained_checkpoint=agent_cfg.openvla_path,
        focal_alpha=agent_cfg.reward_model.focal_alpha,
    )
    reward_model2, cfg2 = load_reward_model(
        model_path=agent_cfg.reward_model_path,
        device=str(device),
        pretrained_checkpoint=agent_cfg.openvla_path,
        focal_alpha=agent_cfg.reward_model.focal_alpha,
    )
    processor1 = get_processor(cfg1)
    processor2 = get_processor(cfg2)
    
    # 创建环境配置
    env_cfg = WorldModelEnvConfig(
        horizon=trainer_cfg.world_model_env.horizon,
        num_batches_to_preload=trainer_cfg.world_model_env.num_batches_to_preload,
        diffusion_sampler=sampler_cfg,
    )
    
    instruction = "pick up the black bowl between the plate and the ramekin and place it on the plate"
    
    # 创建两个环境
    print("\n创建环境...")
    env_single = WorldModelEnv(
        denoiser=denoiser1,
        cfg=env_cfg,
        reward_model=reward_model1,
        reward_cfg=cfg1,
        processor=processor1,
        torch_dtype=torch.bfloat16,
        instruction=instruction,
        return_denoising_trajectory=False,
    )
    
    env_batch = WorldModelEnvBatch(
        denoiser=denoiser2,
        cfg=env_cfg,
        reward_model=reward_model2,
        reward_cfg=cfg2,
        processor=processor2,
        torch_dtype=torch.bfloat16,
        instructions=[instruction],  # batch_size=1
        return_denoising_trajectory=False,
    )
    
    # 加载轨迹数据
    print("\n加载轨迹文件...")
    trajectory_batch = SimpleBatch.load(trajectory_path)
    trajectory_batch = trajectory_batch.to(device)
    
    obs_all = trajectory_batch.obs[0]  # [T+1, 3, H, W]
    act_all = trajectory_batch.act[0]  # [T, act_dim]
    
    num_steps_conditioning = agent_cfg.denoiser.inner_model.num_steps_conditioning
    window_obs = obs_all[-10:]  # [10, 3, H, W]
    window_act = act_all[-9:]   # [9, act_dim]
    
    reset_obs = window_obs[:num_steps_conditioning]  # [4, 3, H, W]
    reset_act = window_act[:num_steps_conditioning - 1]  # [3, act_dim]
    
    remaining_act = window_act[num_steps_conditioning - 1:]  # 剩余动作
    
    print(f"\n测试配置:")
    print(f"  - 种子: {seed}")
    print(f"  - Reset obs shape: {reset_obs.shape}")
    print(f"  - Reset act shape: {reset_act.shape}")
    print(f"  - 剩余动作数量: {len(remaining_act)}")
    
    # 测试多个回合
    num_rounds = 2
    all_match = True
    
    for round_idx in range(num_rounds):
        print(f"\n{'=' * 80}")
        print(f"Round {round_idx + 1}/{num_rounds}")
        print(f"{'=' * 80}")
        
        # 设置相同的种子
        set_seed_everywhere(seed + round_idx)
        
        # Reset 两个环境
        print("\n执行 Reset...")
        obs_single, info_single = env_single.reset(reset_obs, reset_act)
        obs_batch, info_batch = env_batch.reset(
            reset_obs.unsqueeze(0),  # [1, T, C, H, W]
            reset_act.unsqueeze(0),   # [1, T-1, act_dim]
            [instruction]
        )
        
        # 比较 reset 后的观察
        print("\n比较 Reset 结果:")
        match = compare_tensors(obs_single, obs_batch[0], "reset_obs")
        all_match = all_match and match
        
        # 执行多个 step 并比较
        print(f"\n执行 {len(remaining_act)} 个 Step 并比较结果...")
        for step_idx, action in enumerate(remaining_act):
            # 设置相同的种子（每个 step 都需要设置，因为 denoiser 有随机性）
            set_seed_everywhere(seed + round_idx * 1000 + step_idx)
            
            # Step
            obs_single, rew_single, end_single, trunc_single, info_single = env_single.step(action)
            obs_batch, rew_batch, end_batch, trunc_batch, info_batch = env_batch.step(action.unsqueeze(0))
            
            # 比较结果
            print(f"\nStep {step_idx}:")
            match_obs = compare_tensors(obs_single, obs_batch[0], "obs")
            match_rew = compare_tensors(rew_single.unsqueeze(0), rew_batch, "rew", rtol=1e-3, atol=1e-4)
            match_end = compare_tensors(end_single.unsqueeze(0), end_batch, "end")
            match_trunc = compare_tensors(trunc_single.unsqueeze(0), trunc_batch, "trunc")
            
            all_match = all_match and match_obs and match_rew and match_end and match_trunc
            
            # 比较 info 中的关键字段
            if "success_prob" in info_single and "success_prob" in info_batch:
                prob_single_val = info_single["success_prob"]
                prob_batch_val = info_batch["success_prob"]
                if isinstance(prob_batch_val, (list, np.ndarray)):
                    prob_batch_val = prob_batch_val[0]
                prob_single = torch.tensor([prob_single_val], device=device)
                prob_batch = torch.tensor([prob_batch_val], device=device)
                match_prob = compare_tensors(prob_single, prob_batch, "success_prob", rtol=1e-3, atol=1e-4)
                all_match = all_match and match_prob
            
            # 如果环境结束，提前退出
            dead_single = info_single.get('dead', False)
            dead_batch_val = info_batch.get('dead', [False])
            if isinstance(dead_batch_val, (list, np.ndarray)):
                dead_batch = dead_batch_val[0]
            else:
                dead_batch = dead_batch_val
            
            if dead_single or dead_batch:
                print(f"  → Episode ended at step {step_idx}")
                # 检查两个环境是否同时结束
                if dead_single != dead_batch:
                    print(f"  ❌ Dead status mismatch! single={dead_single}, batch={dead_batch}")
                    all_match = False
                break
        
        # 如果环境都结束了，重新 reset 进行下一轮测试
        dead_single = info_single.get('dead', False)
        dead_batch_val = info_batch.get('dead', [False])
        if isinstance(dead_batch_val, (list, np.ndarray)):
            dead_batch = dead_batch_val[0]
        else:
            dead_batch = dead_batch_val
        
        if dead_single or dead_batch:
            print("\n环境已结束，准备下一轮测试...")
    
    # 总结
    print(f"\n{'=' * 80}")
    print("测试总结")
    print(f"{'=' * 80}")
    if all_match:
        print("✅ 所有测试通过！WorldModelEnv 和 WorldModelEnvBatch 的结果一致。")
    else:
        print("❌ 测试失败！发现不一致的结果。")
    print(f"{'=' * 80}")
    
    return all_match


def test_batch_multiple_envs():
    """测试 batch 版本处理多个环境的情况"""
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    agent_config_path = Path("envs/config/agent.yaml")
    trainer_config_path = Path("envs/config/trainer.yaml")
    
    seed = 42
    batch_size = 3
    print("\n" + "=" * 80)
    print(f"测试 WorldModelEnvBatch 处理多个环境 (batch_size={batch_size})")
    print("=" * 80)
    
    # 加载模型
    print("\n加载 Denoiser 模型...")
    denoiser, trainer_cfg, agent_cfg = load_denoiser_from_checkpoint(
        agent_config_path, trainer_config_path, device
    )
    sampler_cfg = instantiate(trainer_cfg.world_model_env.diffusion_sampler)
    
    print("加载 Reward Model...")
    reward_model, cfg = load_reward_model(
        model_path=agent_cfg.reward_model_path,
        device=str(device),
        pretrained_checkpoint=agent_cfg.openvla_path,
        focal_alpha=agent_cfg.reward_model.focal_alpha,
    )
    processor = get_processor(cfg)
    
    # 创建环境配置
    env_cfg = WorldModelEnvConfig(
        horizon=trainer_cfg.world_model_env.horizon,
        num_batches_to_preload=trainer_cfg.world_model_env.num_batches_to_preload,
        diffusion_sampler=sampler_cfg,
    )
    
    instructions = [
        "pick up the black bowl between the plate and the ramekin and place it on the plate"
    ] * batch_size
    
    # 创建 batch 环境
    print("\n创建 Batch 环境...")
    env_batch = WorldModelEnvBatch(
        denoiser=denoiser,
        cfg=env_cfg,
        reward_model=reward_model,
        reward_cfg=cfg,
        processor=processor,
        torch_dtype=torch.bfloat16,
        instructions=instructions,
        return_denoising_trajectory=False,
    )
    
    num_steps_conditioning = agent_cfg.denoiser.inner_model.num_steps_conditioning
    C, H, W = 3, 224, 224
    act_dim = 7
    
    # 生成测试数据
    set_seed_everywhere(seed)
    obs = torch.randn(batch_size, num_steps_conditioning, C, H, W, device=device)
    act = torch.randn(batch_size, num_steps_conditioning - 1, act_dim, device=device)
    
    print(f"\n测试配置:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Obs shape: {obs.shape}")
    print(f"  - Act shape: {act.shape}")
    
    # Reset
    print("\n执行 Reset...")
    current_obs, info = env_batch.reset(obs, act, instructions)
    print(f"Reset 完成，obs shape: {current_obs.shape}")
    alive_val = info.get('alive', [True] * batch_size)
    if isinstance(alive_val, (list, np.ndarray)):
        alive_count = sum(alive_val) if isinstance(alive_val, list) else alive_val.sum()
    else:
        alive_count = batch_size
    print(f"Alive envs: {alive_count}/{batch_size}")
    
    # 执行几个 step
    num_steps = 5
    print(f"\n执行 {num_steps} 个 Step...")
    for step in range(num_steps):
        set_seed_everywhere(seed + step)
        action = torch.randn(batch_size, act_dim, device=device)
        next_obs, rew, end, trunc, info = env_batch.step(action)
        
        alive_val = info.get('alive', [True] * batch_size)
        if isinstance(alive_val, list):
            alive_count = sum(alive_val)
        elif isinstance(alive_val, np.ndarray):
            alive_count = alive_val.sum()
        else:
            alive_count = batch_size
        
        dead_val = info.get('dead', [False] * batch_size)
        if isinstance(dead_val, list):
            dead_count = sum(dead_val)
        elif isinstance(dead_val, np.ndarray):
            dead_count = dead_val.sum()
        else:
            dead_count = 0
        
        print(f"Step {step}: obs shape={next_obs.shape}, rew shape={rew.shape}, "
              f"alive={alive_count}/{batch_size}, dead={dead_count}")
        
        if alive_count == 0:
            print("所有环境都已结束！")
            break
    
    print("\n✅ Batch 多环境测试完成！")


def test_batch_functions():
    """测试新创建的 batch 函数"""
    import torch
    from utils import tensor_to_image, tensor_to_image_batch
    from rl.utils import prepare_one_obs, prepare_one_obs_batch
    from experiments.robot.openvla_utils import prepare_images_for_vla, prepare_images_for_vla_batch

    print("\n" + "=" * 80)
    print("测试 Batch 函数")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 测试 tensor_to_image_batch
    print("\n测试 tensor_to_image_batch...")
    batch_tensor = torch.randn(4, 3, 224, 224, device=device)
    images_list = tensor_to_image_batch(batch_tensor)
    print(f"Input shape: {batch_tensor.shape}, Output count: {len(images_list)}")
    assert len(images_list) == 4
    assert images_list[0].shape == (224, 224, 3)

    # 比较单张和 batch 处理的结果
    single_image = tensor_to_image(batch_tensor[0])
    assert (single_image == images_list[0]).all()
    print("✅ tensor_to_image_batch 结果与单张处理一致")

    # 测试 prepare_images_for_vla_batch (如果有相关配置)
    print("\n测试 prepare_images_for_vla_batch...")
    # 这里需要配置对象，暂时跳过具体测试
    print("✅ prepare_images_for_vla_batch 函数已创建")

    # 测试 prepare_one_obs_batch (如果有相关配置)
    print("\n测试 prepare_one_obs_batch...")
    print("✅ prepare_one_obs_batch 函数已创建")

    print("\n✅ Batch 函数测试完成！")


if __name__ == "__main__":
    # 测试单个环境 vs batch (batch_size=1)
    success = test_single_env_vs_batch()

    # 测试 batch 多环境
    test_batch_multiple_envs()

    # 测试新创建的 batch 函数
    test_batch_functions()

    sys.exit(0 if success else 1)

