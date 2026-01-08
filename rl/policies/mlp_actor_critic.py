"""
简单的 MLP-based Actor-Critic 模型，用于 Meta-World 等低维状态输入任务。

相比 CNN ActorCritic，这个模型：
- 使用 MLP 处理低维状态向量（39维）
- 输出离散动作 logits（分类）
- 不依赖图像输入
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLPActorCriticDiscrete(nn.Module):
    """
    简单的 MLP-based Actor-Critic 模型，用于低维状态输入。
    
    接口与 ActorCritic (discrete) 保持一致，方便替换使用。
    """
    
    def __init__(self, torch_dtype: torch.dtype = torch.float32, 
                 state_dim: int = 39, action_dim: int = 4, hidden_dim=512, n_action_bins=256):
        super().__init__()
       
        self.model_dtype = torch_dtype
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 动作离散化参数
        self.n_action_bins = n_action_bins  # 默认 256 bins
        self.hidden_dim = hidden_dim
        
        # 共享 MLP 编码器
        self.shared_encoder = nn.Sequential(
            nn.Linear(state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
        )
        
        # Policy head: 输出离散动作 logits
        # 输出维度: action_dim * n_action_bins
        self.policy_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, action_dim * self.n_action_bins),
        )
        
        # Value head
        # 1205 zzq 尝试增加value_head的层 
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.n_action_bins),
            nn.ReLU(),
            nn.LayerNorm(self.n_action_bins),
            nn.Linear(self.n_action_bins, 1),
        )
        
        self.to(self.device).to(dtype=self.model_dtype)
        
    def get_parameter_groups(self) -> List[Dict[str, Any]]:
        """
        将可训练参数分为 'policy' 和 'value' 两组。
        """
        policy_params = []
        value_params = []
        
        # Policy 包含: shared_encoder, policy_head
        policy_params.extend(list(self.shared_encoder.parameters()))
        policy_params.extend(list(self.policy_head.parameters()))
        
        # Value 只包含: value_head
        value_params.extend(list(self.value_head.parameters()))
        
        return [
            {"name": "policy", "params": policy_params},
            {"name": "value", "params": value_params},
        ]
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            state: (B, state_dim) 状态张量
        
        Returns:
            action_logits: (B, action_dim, n_action_bins)
            value: (B,) 状态价值估计
        """
        
        # 确保状态在正确的设备和 dtype
        state = state.to(self.device).to(self.model_dtype)
        
        # 共享编码器
        features = self.shared_encoder(state)  # (B, 512)
        
        # Policy head: 输出 logits
        policy_out = self.policy_head(features)  # (B, action_dim * n_action_bins)
        B = policy_out.shape[0]
        action_logits = policy_out.view(B, self.action_dim, self.n_action_bins)
        
        # Value head
        value = self.value_head(features.detach()).squeeze(-1)  # (B,)
     
        return action_logits, value.to(torch.float32)
    
    def post_process(self, logits: torch.Tensor, deterministic: List[bool]) -> Tuple[torch.distributions.Categorical, torch.Tensor, np.ndarray]:
        """
        后处理 logits 以生成离散动作
        
        Args:
            logits: (B, action_dim, n_action_bins)
            deterministic: List[bool] 每个样本是否使用确定性策略
        
        Returns:
            dist: Categorical 分布
            action_token_ids: (B, action_dim) 采样的 token IDs (torch.Tensor)
            discrete_actions: (B, action_dim) 离散动作值 [0, n_action_bins-1] (np.ndarray)
        """
        # 1. 创建分布
        dist = torch.distributions.Categorical(logits=logits)
        
        # 2. 采样动作
        stochastic_tokens = dist.sample()
        deterministic_tokens = torch.argmax(logits, dim=-1)
        
        is_deterministic_tensor = torch.tensor(
            deterministic, dtype=torch.bool, device=logits.device
        )
        is_deterministic_tensor = is_deterministic_tensor.unsqueeze(1)
        
        action_token_ids = torch.where(
            is_deterministic_tensor, deterministic_tokens, stochastic_tokens
        )
        
        # 3. 返回离散动作（直接返回 token IDs）
        # token_id 范围: [0, n_action_bins-1]
        discrete_actions = action_token_ids.cpu().numpy().astype(np.int32)  # (B, action_dim)
        
        return dist, action_token_ids, discrete_actions
    
    def prepare_inputs_batch(self, obs_list: List):
        """
        将观测列表准备为批次输入
        
        Args:
            obs_list: List，每个元素是:
                - np.ndarray: (state_dim,) 状态向量
        
        Returns:
            states: (B, state_dim)
        """
        obs_list = [torch.from_numpy(obs.astype(np.float32)) for obs in obs_list]
        states = torch.stack(obs_list, dim=0)  # (B, state_dim)
        return states
    
    def save_model(self, save_path: str, epoch: int | None = None):
        """保存模型权重"""
        os.makedirs(save_path, exist_ok=True)
        
        if epoch is not None:
            ckpt_path = Path(save_path) / f"mlp_actor_critic_epoch_{epoch}.pt"
        else:
            ckpt_path = Path(save_path) / "mlp_actor_critic.pt"
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'n_action_bins': self.n_action_bins,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
        }, ckpt_path)
        
        print(f"[MLPActorCritic] 模型已保存到: {ckpt_path}")
    
    def load_model(self, load_path: str, epoch: int | None = None):
        """加载模型权重"""
        if epoch is not None:
            ckpt_path = Path(load_path) / f"mlp_actor_critic_epoch_{epoch}.pt"
        else:
            ckpt_path = Path(load_path) / "mlp_actor_critic.pt"
        
        if not ckpt_path.exists():
            print(f"[MLPActorCritic] 警告: checkpoint 文件不存在: {ckpt_path}")
            return
        
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"[MLPActorCritic] 模型已从 {ckpt_path} 加载")
    
    def get_norm_stats(self):
        """
        返回归一化统计信息（占位符，保持接口兼容性）
        MLP 模型不需要特殊的归一化统计
        """
        # 返回一个简单的恒等归一化
        return {
            "mean": np.zeros(self.action_dim),
            "std": np.ones(self.action_dim),
            "min": np.full(self.action_dim, -1.0),
            "max": np.full(self.action_dim, 1.0),
        }


if __name__ == "__main__":
    from rl.metaworld_env import MetaWorldWrapperDiscrete
    n_action_bins = 256
    model = MLPActorCriticDiscrete(torch_dtype=torch.float32, state_dim=39, action_dim=4, n_action_bins=n_action_bins)
    env = MetaWorldWrapperDiscrete(env_name="reach-v3")
    obs, info = env.reset()
    while True:
        state = model.prepare_inputs_batch([obs])
        print(f"state shape: {state.shape}")
        action_logits, value = model(state)
        print(f"action_logits.shape: {action_logits.shape}")
        print(f"value.shape: {value.shape}")
        dist, action_token_ids, discrete_actions = model.post_process(action_logits, deterministic=[False]*action_logits.shape[0])
        print(f"action_token_ids.shape: {action_token_ids.shape}")
        print(f"discrete_actions.shape: {discrete_actions.shape}")
        action = discrete_actions[0]
        print(f"action: {action}")
        obs, reward, done, truncated, info = env.step(action)
        print(f"obs.shape: {obs.shape}")
        print(f"reward: {reward}")
        print(f"done: {done}")
        print(f"truncated: {truncated}")
        print(f"info: {info}")
        if done or truncated:
            break
