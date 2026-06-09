"""
测试 WorldModelEnv 使用真实轨迹数据
"""
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate

from envs.diffusion import Denoiser, SimpleBatch
from world_model_env import WorldModelEnv, WorldModelEnvConfig
from utils import load_reward_model, tensor_to_image
from experiments.robot.openvla_utils import get_processor
from PIL import Image

# OmegaConf.register_new_resolver("eval", eval)

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

def main():
    # 配置路径
    device = torch.device("cuda:7")
    agent_config_path = Path("envs/config/agent.yaml")
    trainer_config_path = Path("envs/config/trainer.yaml")
    trajectory_path = "/cpfs01/jinshiji_workspace/openvla_oft_rl/data/libero_batches_with_next_obs_test/batch_env0_traj2_len85.pt"

    print("=" * 80)
    print("加载 Denoiser 模型...")
    print("=" * 80)
    denoiser, trainer_cfg, agent_cfg = load_denoiser_from_checkpoint(
        agent_config_path, trainer_config_path, device
    )
    sampler_cfg = instantiate(trainer_cfg.world_model_env.diffusion_sampler)

    print("\n" + "=" * 80)
    print("加载 Reward Model...")
    print("=" * 80)
    reward_model, cfg = load_reward_model(
        model_path=agent_cfg.reward_model_path,
        device=str(device),
        pretrained_checkpoint=agent_cfg.openvla_path,
        focal_alpha=agent_cfg.reward_model.focal_alpha,
    )
    processor = get_processor(cfg)

    print("\n" + "=" * 80)
    print("创建 WorldModelEnv...")
    print("=" * 80)
    env_cfg = WorldModelEnvConfig(
        horizon=trainer_cfg.world_model_env.horizon,
        num_batches_to_preload=trainer_cfg.world_model_env.num_batches_to_preload,
        diffusion_sampler=sampler_cfg,
    )

    instruction = "pick up the black bowl between the plate and the ramekin and place it on the plate"
    env = WorldModelEnv(
        denoiser=denoiser,
        cfg=env_cfg,
        reward_model=reward_model,
        reward_cfg=cfg,
        processor=processor,
        torch_dtype=torch.bfloat16,
        instruction=instruction,
        return_denoising_trajectory=False,
    )

    print("\n" + "=" * 80)
    print("加载轨迹文件...")
    print("=" * 80)
    trajectory_batch = SimpleBatch.load(trajectory_path)
    trajectory_batch = trajectory_batch.to(device)

    # 获取第一个轨迹（batch_size=1的情况）
    obs_all = trajectory_batch.obs[0]  # [T+1, 3, H, W]
    act_all = trajectory_batch.act[0]  # [T, act_dim]

    print(f"轨迹总长度: obs={obs_all.shape[0]}, act={act_all.shape[0]}")
    print(f"obs shape: {obs_all.shape}, dtype: {obs_all.dtype}")
    print(f"act shape: {act_all.shape}, dtype: {act_all.dtype}")

    # 取最后一个窗口：obs[-10:], act[-9:]
    window_obs = obs_all[-10:]  # [10, 3, H, W]
    window_act = act_all[-9:]   # [9, act_dim]

    print(f"\n最后一个窗口: obs={window_obs.shape}, act={window_act.shape}")

    # 用前4帧obs和前3帧act来reset
    num_steps_conditioning = agent_cfg.denoiser.inner_model.num_steps_conditioning
    reset_obs = window_obs[:num_steps_conditioning]  # [4, 3, H, W] (假设 num_steps_conditioning=4)
    reset_act = window_act[:num_steps_conditioning - 1]  # [3, act_dim] (T-1 actions for T observations)

    print(f"\nReset 数据:")
    print(f"  reset_obs: {reset_obs.shape}")
    print(f"  reset_act: {reset_act.shape}")

    for i in range(2):
        print("\n" + "=" * 80)
        print("执行 Reset...")
        print("=" * 80)
        current_obs, info = env.reset(reset_obs, reset_act)
        print(f"Reset 完成，当前 obs shape: {current_obs.shape}")

        print("\n" + "=" * 80)
        print("执行 Step...")
        print("=" * 80)

        remaining_act = window_act[num_steps_conditioning - 1:]  # 从第 num_steps_conditioning-1 个动作开始

        print(f"剩余动作数量: {len(remaining_act)}")
        print(f"剩余动作 shape: {remaining_act.shape}")
        print(f"将执行 {len(remaining_act)} 次 step")

        for step_idx, action in enumerate(remaining_act):
            step_start_time = time.time()
            next_obs, rew, end, trunc, info = env.step(action)
            step_end_time = time.time()
            step_duration = step_end_time - step_start_time
            
            Image.fromarray(tensor_to_image(next_obs)).save(f"rollouts/tmp/{step_idx}.png")
            predict_obs_time = info.get('predict_obs_time', 0)
            predict_rew_time = info.get('predict_rew_time', 0)
            print(
                f"Step {step_idx}: "
                f"time={step_duration:.4f}s "
                f"(predict_obs={predict_obs_time:.4f}s, predict_rew={predict_rew_time:.4f}s), "
                f"obs shape={next_obs.shape}, "
                f"rew={rew.item():.4f}, "
                f"end={end.item()}, "
                f"trunc={trunc.item()}, "
                f"ep_len={info.get('ep_len', 'N/A')}"
            )

            if info.get('dead', False):
                print(f"  → Episode ended at step {step_idx}")
                break

    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()

