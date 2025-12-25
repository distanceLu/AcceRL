from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import time

import torch
from torch import Tensor
from diffusion import Denoiser, DiffusionSampler, DiffusionSamplerConfig

from rl.utils import prepare_one_obs
from utils import tensor_to_image, load_reward_model
from experiments.robot.openvla_utils import get_processor
from hydra.utils import instantiate
from omegaconf import OmegaConf

ResetOutput = Tuple[torch.FloatTensor, Dict[str, Any]]
StepOutput = Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Any]]
OmegaConf.register_new_resolver("eval", eval)

@dataclass
class WorldModelEnvConfig:
    horizon: int
    num_batches_to_preload: int
    diffusion_sampler: DiffusionSamplerConfig


class WorldModelEnv:
    def __init__(
        self,
        denoiser: Denoiser,
        cfg: WorldModelEnvConfig,
        reward_model: Optional[Any] = None,
        reward_cfg: Optional[Any] = None,
        processor: Optional[Any] = None,
        torch_dtype: Optional[torch.dtype] = None,
        instruction: str = "",
        return_denoising_trajectory: bool = False,
    ) -> None:

        self.sampler = DiffusionSampler(denoiser, cfg.diffusion_sampler)
        self.reward_model = reward_model
        self.reward_cfg = reward_cfg
        self.processor = processor
        self.torch_dtype = torch_dtype if torch_dtype is not None else torch.bfloat16
        self.instruction = instruction
        self.horizon = cfg.horizon
        self.return_denoising_trajectory = return_denoising_trajectory

    @property
    def device(self) -> torch.device:
        return self.sampler.denoiser.device

    @torch.no_grad()
    def reset(self, obs: Tensor, act: Tensor) -> ResetOutput:
        """
        Args:
            obs: [T, C, H, W] - observation sequence
            act: [T-1, act_dim] - action sequence (one less than obs)
        
        Returns:
            (next_obs, info) where next_obs: [C, H, W]
        """
        # Store without batch dimension - sampler.sample will add it when needed
        self.obs_buffer = obs  # [T, C, H, W]
        self.act_buffer = act  # [T-1, act_dim]
        self.ep_len = torch.tensor(0, dtype=torch.long, device=obs.device)
        
        return self.obs_buffer[-1], {}  # [C, H, W]

    @torch.no_grad()
    def step(self, act: Tensor) -> StepOutput:
        """
        Args:
            act: [act_dim] - action to take
        Returns:
            (next_obs, rew, end, trunc, info) where:
                next_obs: [C, H, W]
                rew: scalar tensor
                end: scalar tensor (0 or 1)
                trunc: scalar tensor (0 or 1)
        """
        # Append new action to act_buffer (becomes [T, act_dim])
        self.act_buffer = torch.cat([self.act_buffer, act.unsqueeze(0)], dim=0)  # [T-1, act_dim] -> [T, act_dim]

        predict_obs_start = time.time()
        next_obs, denoising_trajectory = self.predict_next_obs()
        predict_obs_time = time.time() - predict_obs_start
        
        predict_rew_start = time.time()
        rew, end = self.predict_rew_end(next_obs)
        predict_rew_time = time.time() - predict_rew_start

        self.ep_len += 1
        trunc = (self.ep_len >= self.horizon).long()

        # Update obs_buffer: roll and set last element
        self.obs_buffer = self.obs_buffer.roll(-1, dims=0)
        self.obs_buffer[-1] = next_obs
        
        # Pop the oldest action from act_buffer (back to [T-1, act_dim])
        self.act_buffer = self.act_buffer[1:]  # Remove first element

        dead = torch.logical_or(end.bool(), trunc.bool()) if self.reward_model is not None else trunc.bool()

        info = {}
        if self.return_denoising_trajectory:
            info["denoising_trajectory"] = torch.stack(denoising_trajectory, dim=0)

        if dead.item():
            info["final_observation"] = next_obs
            info["burnin_obs"] = self.obs_buffer[:-1]

        info["ep_len"] = self.ep_len.item()
        info["truncated"] = trunc.item()
        info["dead"] = dead.item()
        info["predict_obs_time"] = predict_obs_time
        info["predict_rew_time"] = predict_rew_time

        return self.obs_buffer[-1], rew, end, trunc, info

    @torch.no_grad()
    def predict_next_obs(self) -> Tuple[Tensor, List[Tensor]]:
        """
        Returns:
            (next_obs, denoising_trajectory) where next_obs: [C, H, W]
        """
        # Add batch dimension for sampler (which expects [B, T, C, H, W])
        obs_batch = self.obs_buffer.unsqueeze(0)  # [1, T, C, H, W]
        act_batch = self.act_buffer.unsqueeze(0)  # [1, T, act_dim]
        next_obs, denoising_trajectory = self.sampler.sample(obs_batch, act_batch)
        return next_obs[0], denoising_trajectory  # Remove batch dimension: [C, H, W]
    
    @torch.no_grad()
    def predict_rew_end(self, next_obs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Predict reward and end signal using reward_model.
        
        Args:
            next_obs: [C, H, W] - the predicted next observation in [-1, 1]
        
        Returns:
            (rew, end) where:
                rew: scalar tensor - reward as probability of class 1 (range [0, 1])
                end: scalar tensor - end signal (0 or 1, binary classification)
        """
        # Convert tensor to uint8 HWC image
        frame_uint8 = tensor_to_image(next_obs)  # [H, W, C] uint8
        
        # Prepare input for reward model
        obs_for_vla: Dict[str, Any] = {"full_image": frame_uint8}
        inputs = prepare_one_obs(
            self.reward_cfg, 
            self.processor, 
            obs_for_vla, 
            self.instruction, 
            self.torch_dtype
        )

        if (not self.reward_cfg.use_proprio) and ("proprio" in inputs) and (inputs["proprio"] is None):
            inputs.pop("proprio", None)

        # Move to reward model device
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.reward_model.device)

        # Forward through reward model
        logits = self.reward_model.forward(inputs)  # [1, 2]
        probs = torch.softmax(logits, dim=-1)  # [1, 2]
        rew = probs[0, 1]  # scalar - probability of class 1
        end = logits.argmax(dim=-1)[0]  # scalar - 0 or 1
        
        return rew, end


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
    
    env = WorldModelEnv(
        denoiser=denoiser,
        cfg=env_cfg,
        reward_model=reward_model,
        reward_cfg=cfg,
        processor=processor,
        torch_dtype=torch.bfloat16,
        instruction="pick up the black bowl between the plate and the ramekin and place it on the plate",  # 任务指令
        return_denoising_trajectory=False,
    )
    
    num_steps_conditioning = agent_cfg.denoiser.inner_model.num_steps_conditioning
    
    window_size = num_steps_conditioning
    C, H, W = 3, 224, 224
    
    obs = torch.randn(window_size, C, H, W, device=device)
    act = torch.randn(window_size - 1, 7, device=device)  # T-1 actions for T observations
    
    current_obs, info = env.reset(obs, act)

    for step in range(5):
        action = torch.randn(7, device=device)
        next_obs, rew, end, trunc, info = env.step(action)
        print(f"Step {step}: obs shape={next_obs.shape}, rew={rew.item()}, end={end.item()}, trunc={trunc.item()}, info={info}")
        
        if info.get('dead', False):
            new_obs = torch.randn(window_size, C, H, W, device=device)
            new_act = torch.randn(window_size - 1, 7, device=device)  # T-1 actions for T observations
            env.reset(new_obs, new_act)
