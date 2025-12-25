from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import time

import torch
from torch import Tensor
from diffusion import Denoiser, DiffusionSampler, DiffusionSamplerConfig

from rl.utils import prepare_one_obs_batch, prepare_inputs_batch
from utils import tensor_to_image_batch, load_reward_model
from experiments.robot.openvla_utils import get_processor
from hydra.utils import instantiate
from omegaconf import OmegaConf

ResetOutput = Tuple[torch.FloatTensor, Dict[str, Any]]
StepOutput = Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Any]]
# Register eval resolver if not already registered
try:
    OmegaConf.register_new_resolver("eval", eval)
except ValueError:
    # Resolver already registered, ignore
    pass

@dataclass
class WorldModelEnvConfig:
    horizon: int
    num_batches_to_preload: int
    diffusion_sampler: DiffusionSamplerConfig


class WorldModelEnvBatch:
    def __init__(
        self,
        denoiser: Denoiser,
        cfg: WorldModelEnvConfig,
        reward_model: Optional[Any] = None,
        reward_cfg: Optional[Any] = None,
        processor: Optional[Any] = None,
        torch_dtype: Optional[torch.dtype] = None,
        instructions: Optional[List[str]] = None,  # List of instructions for each env
        return_denoising_trajectory: bool = False,
    ) -> None:

        self.sampler = DiffusionSampler(denoiser, cfg.diffusion_sampler)
        self.reward_model = reward_model
        self.reward_cfg = reward_cfg
        self.processor = processor
        self.torch_dtype = torch_dtype if torch_dtype is not None else torch.bfloat16
        self.instructions = instructions if instructions is not None else []
        self.horizon = cfg.horizon
        self.return_denoising_trajectory = return_denoising_trajectory
        self.batch_size = 0  # Will be set on first reset
        self.last_success_prob = None  # [B]
        self.alive_mask = None  # [B] - mask for alive environments

    @property
    def device(self) -> torch.device:
        return self.sampler.denoiser.device

    @torch.no_grad()
    def reset(self, obs: Tensor, act: Tensor, instructions: Optional[List[str]] = None) -> ResetOutput:
        """
        Args:
            obs: [B, T, C, H, W] - observation sequence for each env
            act: [B, T-1, act_dim] - action sequence (one less than obs)
            instructions: Optional list of instructions for each env (if not provided in __init__)
        
        Returns:
            (next_obs, info) where next_obs: [B, C, H, W]
        """
        B, T, C, H, W = obs.shape
        self.batch_size = B
        
        # Store with batch dimension
        self.obs_buffer = obs  # [B, T, C, H, W]
        self.act_buffer = act  # [B, T-1, act_dim]
        self.ep_len = torch.zeros(B, dtype=torch.long, device=obs.device)  # [B]
        
        # Update instructions if provided
        if instructions is not None:
            assert len(instructions) == B, f"Number of instructions ({len(instructions)}) must match batch size ({B})"
            self.instructions = instructions
        elif len(self.instructions) == 0:
            # If no instructions provided, use empty strings
            self.instructions = [""] * B
        elif len(self.instructions) == 1:
            # If single instruction provided, replicate for all envs
            self.instructions = self.instructions * B
        
        # Initialize alive mask - all environments start alive
        self.alive_mask = torch.ones(B, dtype=torch.bool, device=obs.device)  # [B]
        
        # Initialize last_success_prob for each env
        next_obs_batch = self.obs_buffer[:, -1]  # [B, C, H, W]
        self.last_success_prob, _, _ = self.predict_rew_end(next_obs_batch)  # [B]
        
        return self.obs_buffer[:, -1], {}  # [B, C, H, W]

    @torch.no_grad()
    def step(self, act: Tensor) -> StepOutput:
        """
        Args:
            act: [B, act_dim] - actions to take for each env
        Returns:
            (next_obs, rew, end, trunc, info) where:
                next_obs: [B, C, H, W]
                rew: [B] - reward tensor
                end: [B] - end signal tensor (0 or 1)
                trunc: [B] - truncation signal tensor (0 or 1)
        """
        B = self.batch_size
        device = self.device
        
        # For dead environments, use zero actions (will be masked later)
        act = act * self.alive_mask.unsqueeze(-1)  # [B, act_dim]
        
        # Append new action to act_buffer (becomes [B, T, act_dim])
        self.act_buffer = torch.cat([self.act_buffer, act.unsqueeze(1)], dim=1)  # [B, T-1, act_dim] -> [B, T, act_dim]

        predict_obs_start = time.time()
        next_obs, denoising_trajectory = self.predict_next_obs()  # [B, C, H, W]
        predict_obs_time = time.time() - predict_obs_start
        
        # Mask dead environments: keep their last observation
        next_obs = torch.where(
            self.alive_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
            next_obs,
            self.obs_buffer[:, -1]  # Keep last observation for dead envs
        )
        
        predict_rew_start = time.time()
        success_prob, end, predict_rew_info = self.predict_rew_end(next_obs)  # [B], [B]
        
        # Mask dead environments: keep their last success_prob and set end=0
        success_prob = torch.where(self.alive_mask, success_prob, self.last_success_prob)
        end = end * self.alive_mask.long()  # Dead envs have end=0
        
        rew = success_prob - self.last_success_prob  # [B]
        rew = rew * self.alive_mask.float()  # Dead envs have rew=0
        self.last_success_prob = success_prob
        predict_rew_time = time.time() - predict_rew_start

        self.ep_len += 1
        trunc = (self.ep_len >= self.horizon).long()  # [B]
        trunc = trunc * self.alive_mask.long()  # Dead envs have trunc=0

        # Update obs_buffer: roll and set last element (only for alive envs)
        self.obs_buffer = self.obs_buffer.roll(-1, dims=1)  # Roll along time dimension
        self.obs_buffer[:, -1] = torch.where(
            self.alive_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
            next_obs,
            self.obs_buffer[:, -1]  # Keep old observation for dead envs
        )
        
        # Pop the oldest action from act_buffer (back to [B, T-1, act_dim])
        self.act_buffer = self.act_buffer[:, 1:]  # Remove first time step

        dead = torch.logical_or(end.bool(), trunc.bool()) if self.reward_model is not None else trunc.bool()  # [B]
        dead = dead & self.alive_mask  # Only mark as dead if previously alive
        
        # Update alive mask
        self.alive_mask = self.alive_mask & (~dead)  # [B]

        info = {}
        if self.return_denoising_trajectory:
            # Stack denoising trajectories: List[List[Tensor]] -> [B, num_steps, C, H, W]
            info["denoising_trajectory"] = torch.stack([torch.stack(traj, dim=0) for traj in denoising_trajectory], dim=0)

        # Store info for each env
        info["ep_len"] = self.ep_len.cpu().numpy()  # [B]
        info["truncated"] = trunc.cpu().numpy()  # [B]
        info["dead"] = dead.cpu().numpy()  # [B]
        info["alive"] = self.alive_mask.cpu().numpy()  # [B]
        info["predict_obs_time"] = predict_obs_time
        info["predict_rew_time"] = predict_rew_time
        info["success_prob"] = success_prob.cpu().numpy()  # [B]
        info.update(predict_rew_info)

        return self.obs_buffer[:, -1], rew, end, trunc, info

    @torch.no_grad()
    def predict_next_obs(self) -> Tuple[Tensor, List[List[Tensor]]]:
        """
        Returns:
            (next_obs, denoising_trajectory) where:
                next_obs: [B, C, H, W]
                denoising_trajectory: List[List[Tensor]] - list of trajectories for each env
        """
        # Sampler already expects [B, T, C, H, W] and [B, T, act_dim]
        next_obs, denoising_trajectory = self.sampler.sample(self.obs_buffer, self.act_buffer)
        # next_obs: [B, C, H, W]
        # denoising_trajectory: List[Tensor] where each Tensor is [B, C, H, W]
        # Convert to List[List[Tensor]] format
        if denoising_trajectory:
            # If trajectory is List[Tensor] with shape [B, C, H, W], split by batch
            B = next_obs.shape[0]
            traj_list = [[traj[i] for traj in denoising_trajectory] for i in range(B)]
        else:
            traj_list = [[] for _ in range(next_obs.shape[0])]
        return next_obs, traj_list
    
    @torch.no_grad()
    def predict_rew_end(self, next_obs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Predict reward and end signal using reward_model for batch of observations.
        
        Args:
            next_obs: [B, C, H, W] - the predicted next observations in [-1, 1]
        
        Returns:
            (rew, end) where:
                rew: [B] - reward as probability of class 1 (range [0, 1])
                end: [B] - end signal (0 or 1, binary classification)
        """
        time_start = time.time()
        B = next_obs.shape[0]

        # Convert batch of tensors to images using optimized batch version
        images_batch = tensor_to_image_batch(next_obs)  # List of [H, W, C] uint8 images

        # Prepare observations for batch processing
        obs_batch = [{"full_image": img} for img in images_batch]
        instructions = [self.instructions[i] if i < len(self.instructions) else "" for i in range(B)]

        prepare_time_start = time.time()
        # Use batch version of prepare_one_obs
        inputs_list = prepare_one_obs_batch(
            self.reward_cfg,
            self.processor,
            obs_batch,
            instructions,
            self.torch_dtype
        )
        prepare_time = time.time() - prepare_time_start

        # Remove proprio if unused
        for inputs in inputs_list:
            if (not self.reward_cfg.use_proprio) and ("proprio" in inputs) and (inputs["proprio"] is None):
                inputs.pop("proprio", None)

        # Batch process all inputs
        batch_inputs = prepare_inputs_batch(self.reward_model, inputs_list)

        forward_time_start = time.time()
        # Forward through reward model
        logits = self.reward_model.forward(batch_inputs)  # [B, 2]
        forward_time = time.time() - forward_time_start

        probs = torch.softmax(logits, dim=-1)  # [B, 2]
        rew = probs[:, 1]  # [B] - probability of class 1
        end = logits.argmax(dim=-1)  # [B] - 0 or 1

        total_time = time.time() - time_start

        return rew, end, {"prepare_time": prepare_time, "forward_time": forward_time, "total_time": total_time}

    @torch.no_grad()
    def imagine(
        self,
        obs: Tensor,
        act: Tensor,
        instructions: Optional[List[str]] = None,
        actions: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Imagine a trajectory by repeatedly calling step until all environments are done or horizon is reached.
        
        Args:
            obs: [B, T, C, H, W] - initial observation sequence for each env
            act: [B, T-1, act_dim] - initial action sequence
            instructions: Optional list of instructions for each env
            actions: [B, H, act_dim] - actions to take at each step (if None, will use zeros)
        
        Returns:
            Dict with keys:
                - obs: [B, H+1, C, H, W] - observations (including initial)
                - act: [B, H, act_dim] - actions taken
                - rew: [B, H] - rewards
                - end: [B, H] - end signals
                - trunc: [B, H] - truncation signals
                - alive: [B, H] - alive mask at each step
        """
        B, T, C, H_img, W_img = obs.shape
        H = self.horizon
        
        # Reset environment
        initial_obs, _ = self.reset(obs, act, instructions)
        
        # Initialize storage
        obs_list = [initial_obs]  # Start with initial observation
        act_list = []
        rew_list = []
        end_list = []
        trunc_list = []
        alive_list = []
        
        # If actions not provided, use zeros (will be masked for dead envs anyway)
        if actions is None:
            actions = torch.zeros(B, H, act.shape[-1], device=self.device, dtype=act.dtype)
        else:
            assert actions.shape == (B, H, act.shape[-1]), \
                f"actions shape {actions.shape} != expected {(B, H, act.shape[-1])}"
        
        # Run steps until all envs are dead or horizon reached
        for step in range(H):
            # Get action for this step
            act_step = actions[:, step]  # [B, act_dim]
            
            # Step
            next_obs, rew, end, trunc, info = self.step(act_step)
            
            # Store results
            obs_list.append(next_obs)
            act_list.append(act_step)
            rew_list.append(rew)
            end_list.append(end)
            trunc_list.append(trunc)
            alive_list.append(self.alive_mask)  # Store torch tensor, not numpy array
            
            # Check if all envs are dead
            if not self.alive_mask.any():
                # All envs are dead, pad remaining steps with zeros
                remaining_steps = H - step - 1
                if remaining_steps > 0:
                    obs_list.extend([next_obs] * remaining_steps)
                    act_list.extend([torch.zeros_like(act_step)] * remaining_steps)
                    rew_list.extend([torch.zeros_like(rew)] * remaining_steps)
                    end_list.extend([torch.zeros_like(end)] * remaining_steps)
                    trunc_list.extend([torch.zeros_like(trunc)] * remaining_steps)
                    alive_list.extend([torch.zeros_like(self.alive_mask)] * remaining_steps)
                break
        
        # Stack all tensors
        obs_tensor = torch.stack([initial_obs] + obs_list[1:], dim=1)  # [B, H+1, C, H_img, W_img]
        act_tensor = torch.stack(act_list, dim=1)  # [B, H, act_dim]
        rew_tensor = torch.stack(rew_list, dim=1)  # [B, H]
        end_tensor = torch.stack(end_list, dim=1)  # [B, H]
        trunc_tensor = torch.stack(trunc_list, dim=1)  # [B, H]
        alive_tensor = torch.stack(alive_list, dim=1)  # [B, H]
        
        return {
            "obs": obs_tensor,  # [B, H+1, C, H_img, W_img]
            "act": act_tensor,  # [B, H, act_dim]
            "rew": rew_tensor,  # [B, H]
            "end": end_tensor,  # [B, H]
            "trunc": trunc_tensor,  # [B, H]
            "alive": alive_tensor,  # [B, H]
        }


if __name__ == "__main__":
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
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent_config_path = Path("/cpfs01/jinshiji_workspace/openvla_oft_rl/envs/config/agent.yaml")
    trainer_config_path = Path("/cpfs01/jinshiji_workspace/openvla_oft_rl/envs/config/trainer.yaml")
    
    denoiser, trainer_cfg, agent_cfg = load_denoiser_from_checkpoint(
        agent_config_path=agent_config_path, 
        trainer_config_path=trainer_config_path, 
        device=device,
    )
    sampler_cfg = instantiate(trainer_cfg.world_model_env.diffusion_sampler)
    
    reward_model, cfg = load_reward_model(
        model_path=agent_cfg.reward_model_path,
        device=device,
        pretrained_checkpoint=agent_cfg.openvla_path,
        focal_alpha=agent_cfg.reward_model.focal_alpha,
    )
    processor = get_processor(cfg)
    
    env_cfg = WorldModelEnvConfig(
        horizon=trainer_cfg.world_model_env.horizon,
        num_batches_to_preload=trainer_cfg.world_model_env.num_batches_to_preload,
        diffusion_sampler=sampler_cfg,
    )
    
    batch_size = 4
    instructions = [
        "pick up the black bowl between the plate and the ramekin and place it on the plate"
    ] * batch_size
    
    env = WorldModelEnvBatch(
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
    
    window_size = num_steps_conditioning
    C, H, W = 3, 224, 224
    
    obs = torch.randn(batch_size, window_size, C, H, W, device=device)
    act = torch.randn(batch_size, window_size - 1, 7, device=device)
    
    # Test imagine function
    print("Testing imagine function...")
    imagined_data = env.imagine(obs, act)
    print(f"Imagined obs shape: {imagined_data['obs'].shape}")
    print(f"Imagined act shape: {imagined_data['act'].shape}")
    print(f"Imagined rew shape: {imagined_data['rew'].shape}")
    print(f"Imagined end shape: {imagined_data['end'].shape}")
    print(f"Imagined alive shape: {imagined_data['alive'].shape}")
    
    # Test step function with masking
    print("\nTesting step function with masking...")
    current_obs, info = env.reset(obs, act)
    
    for step in range(5):
        action = torch.randn(batch_size, 7, device=device)
        next_obs, rew, end, trunc, info = env.step(action)
        print(f"Step {step}: obs shape={next_obs.shape}, rew shape={rew.shape}, "
              f"alive={info['alive'].sum()}/{batch_size}, dead={info['dead'].sum()}")
        
        if not info['alive'].any():
            print("All environments are dead!")
            break
