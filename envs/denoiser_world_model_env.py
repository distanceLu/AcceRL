from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import sys
from pathlib import Path

import torch
from torch import Tensor
from diffusion import Denoiser, DiffusionSampler, DiffusionSamplerConfig

ResetOutput = Tuple[torch.FloatTensor, Dict[str, Any]]
StepOutput = Tuple[Tensor, Dict[str, Any]]
InitialCondition = Tuple[Tensor, Tensor]


@dataclass
class DenoiserWorldModelEnvConfig:
    horizon: int
    num_batches_to_preload: int
    diffusion_sampler: DiffusionSamplerConfig


class DenoiserWorldModelEnv:
    
    def __init__(
        self,
        denoiser: Denoiser,
        cfg: DenoiserWorldModelEnvConfig,
        return_denoising_trajectory: bool = False,
    ) -> None:

        self.sampler = DiffusionSampler(denoiser, cfg.diffusion_sampler)
        self.horizon = cfg.horizon
        self.return_denoising_trajectory = return_denoising_trajectory
        self.num_envs = 1

    @property
    def device(self) -> torch.device:
        return self.sampler.denoiser.device

    @torch.no_grad()
    def reset(self, obs: Tensor, act: Tensor) -> ResetOutput:
        """
        Args:
            obs: [num_envs, T, C, H, W]
            act: [num_envs, T, act_dim]
        
        Returns:
            (next_obs, info)
        """
        self.obs_buffer = obs
        self.act_buffer = act
        self.ep_len = torch.zeros(self.num_envs, dtype=torch.long, device=obs.device)
        return self.obs_buffer[:, -1], {}

    @torch.no_grad()
    def reset_dead(self, dead: torch.BoolTensor, obs: Tensor, act: Tensor) -> None:
        """
        Args:
            dead: [num_envs] bool
            obs: [num_envs, T+1, C, H, W]
            act: [num_envs, T, act_dim]
        """
        self.obs_buffer[dead] = obs
        self.act_buffer[dead] = act
        self.ep_len[dead] = 0

    @torch.no_grad()
    def step(self, act: torch.LongTensor) -> StepOutput:
        """
        Args:
            act: [num_envs] or [num_envs, act_dim]
        Returns:
            (next_obs, info)
        """
        self.act_buffer[:, -1] = act

        next_obs, denoising_trajectory = self.predict_next_obs()

        self.ep_len += 1
        trunc = (self.ep_len >= self.horizon).long()

        self.obs_buffer = self.obs_buffer.roll(-1, dims=1)
        self.act_buffer = self.act_buffer.roll(-1, dims=1)
        self.obs_buffer[:, -1] = next_obs

        dead = trunc.bool()

        info = {}
        if self.return_denoising_trajectory:
            info["denoising_trajectory"] = torch.stack(denoising_trajectory, dim=1)

        if dead.any():
            self.reset_dead(dead)
            info["final_observation"] = next_obs[dead]
            info["burnin_obs"] = self.obs_buffer[dead, :-1]

        info["ep_len"] = self.ep_len
        info["truncated"] = trunc

        return self.obs_buffer[:, -1], info

    @torch.no_grad()
    def predict_next_obs(self) -> Tuple[Tensor, List[Tensor]]:
        """
        Returns:
            (next_obs, denoising_trajectory)
        """
        return self.sampler.sample(self.obs_buffer, self.act_buffer)


if __name__ == "__main__":
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    
    OmegaConf.register_new_resolver("eval", eval)
    
    def load_denoiser_from_checkpoint(
        agent_config_path: Path,
        trainer_config_path: Path,
        checkpoint_path: str,
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
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
    checkpoint_path = "/cpfs01/jinshiji_workspace/diamond/checkpoints/checkpoint_step_543900.pt"

    denoiser, trainer_cfg, agent_cfg = load_denoiser_from_checkpoint(agent_config_path, trainer_config_path, checkpoint_path, device)
    
    sampler_cfg = instantiate(trainer_cfg.world_model_env.diffusion_sampler)
    
    env_cfg = DenoiserWorldModelEnvConfig(
        horizon=220,
        num_batches_to_preload=1,
        diffusion_sampler=sampler_cfg,
    )
    
    env = DenoiserWorldModelEnv(
        denoiser=denoiser,
        cfg=env_cfg,
        return_denoising_trajectory=False,
    )
    
    num_steps_conditioning = agent_cfg.denoiser.inner_model.num_steps_conditioning
    
    num_envs = 1
    window_size = num_steps_conditioning
    C, H, W = 3, 224, 224
    
    obs = torch.randn(num_envs, window_size, C, H, W, device=device)
    act = torch.randn(num_envs, window_size, 7, device=device)
    
    current_obs, info = env.reset(obs, act)

    for step in range(5):
        action = torch.randn(num_envs, 7, device=device)
        next_obs, info = env.step(action)
        print(next_obs.shape, info)
        
        if info['truncated'].item():
            new_obs = torch.randn(num_envs, window_size, C, H, W, device=device)
            new_act = torch.randn(num_envs, window_size, 7, device=device)
            env.reset(new_obs, new_act)
