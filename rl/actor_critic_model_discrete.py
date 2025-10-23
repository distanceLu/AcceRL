
import time
import random

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, List
import numpy as np
from peft import LoraConfig, get_peft_model

# Core OpenVLA components
from experiments.robot.openvla_utils import (
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
)
from typing import Any

from transformers.models.llama.modeling_llama import LlamaForCausalLM
from rl.utils import get_vla, compute_num_patches, prepare_inputs_batch, forward_vla


class ActorCritic(nn.Module):
    """
    Actor-Critic for OpenVLA-based continuous control.

    forward(inputs_batch) returns:
      - actions_all: sampled actions in (-1, 1), shape (B, NUM_ACTIONS_CHUNK, ACTION_DIM)  [squashed Gaussian]
      - mu_all: mean actions from action_head.predict_action(...), shape (B, NUM_ACTIONS_CHUNK, ACTION_DIM)
      - log_std_all: condition-independent log-std broadcast to all chunks, shape (B, NUM_ACTIONS_CHUNK, ACTION_DIM)
      - value: state value estimate, shape (B,)
    """

    def __init__(self, cfg, torch_dtype: torch.dtype):
        super().__init__()
        self.cfg = cfg

        # Device / dtype
        self.vla = get_vla(cfg, torch_dtype)
        self.device = self.vla.device
        self.model_dtype = torch_dtype
        self.vla = self.vla.to(dtype=self.model_dtype)
        # 计算有效的vocab范围
        self.vocab_size = self.vla.config.text_config.vocab_size - self.vla.config.pad_to_multiple_of
        self.n_action_bins = self.vla.config.n_action_bins
        self.action_vocab_start = self.vocab_size - self.n_action_bins
        
        # 原地替换lm_head为精简版本
        original_lm_head = self.vla.language_model.lm_head
        
        print(f"原始 lm_head 形状: weight={original_lm_head.weight.shape}, "
              f"bias={original_lm_head.bias.shape if original_lm_head.bias is not None else None}")
        
        # 提取权重和偏置的有效部分 [action_vocab_start:vocab_size, :]
        with torch.no_grad():
            action_weight = original_lm_head.weight[self.action_vocab_start:self.vocab_size, :].clone()
            if original_lm_head.bias is not None:
                action_bias = original_lm_head.bias[self.action_vocab_start:self.vocab_size].clone()
            else:
                action_bias = None
        
        # 创建新的精简lm_head并原地替换
        new_lm_head = nn.Linear(
            original_lm_head.in_features,
            self.n_action_bins,
            bias=(action_bias is not None)
        ).to(self.device).to(dtype=self.model_dtype)
        
        # 复制权重
        with torch.no_grad():
            new_lm_head.weight.copy_(action_weight)
            if action_bias is not None:
                new_lm_head.bias.copy_(action_bias)
        
        # 原地替换
        self.vla.language_model.lm_head = new_lm_head
        
        print(f"精简后 lm_head 形状: weight={new_lm_head.weight.shape}, "
              f"bias={new_lm_head.bias.shape if new_lm_head.bias is not None else None}")
        print(f"lm_head 已从 ({original_lm_head.out_features}, {original_lm_head.in_features}) "
              f"精简为 ({self.n_action_bins}, {original_lm_head.in_features})")
        
        # 🔒 冻结 VLA 参数
        for param in self.vla.parameters():
            param.requires_grad = False
        if cfg.use_lora:
            lora_config = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=min(cfg.lora_rank, 16),
                lora_dropout=cfg.lora_dropout,
                target_modules="all-linear",
                init_lora_weights="gaussian",
            )
            self.vla = get_peft_model(self.vla, lora_config)
            print("lora_rank:", cfg.lora_rank)
            # 打印可训练Lora参数信息
            self.vla.print_trainable_parameters()
        self.vla.language_model: LlamaForCausalLM
        # 手动解冻lm_head参数（保持全参量训练）
        for param in self.vla.language_model.lm_head.parameters():
            param.requires_grad = True

        self.bins = np.linspace(-1, 1, self.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # Keep processor for external preparation
        self.processor = get_processor(cfg)
        self.proprio_projector = get_proprio_projector(
            cfg, llm_dim=self.vla.llm_dim, proprio_dim=PROPRIO_DIM
        )
        # 注意力池化层
        self.attn_pool = nn.Sequential(
            nn.Linear(self.vla.llm_dim, 1),
        ).to(self.device).to(dtype=self.model_dtype)

        # Value head
        self.value_head = nn.Sequential(
            nn.LayerNorm(self.vla.llm_dim),
            nn.Linear(self.vla.llm_dim, self.vla.llm_dim),
            nn.ReLU(),
            nn.Linear(self.vla.llm_dim, 1),
        )
        self.to(self.device).to(dtype=self.model_dtype)

    def get_parameter_groups(self) -> List[Dict[str, Any]]:
        """
        将可训练参数分为 'policy' 和 'value' 两组。
        这对于为不同组件设置不同的学习率至关重要。
        """
        self.vla.language_model: LlamaForCausalLM 
        
        # 1. 收集所有可训练参数
        policy_params = list(self.proprio_projector.parameters())
        value_params = []
        
        # 2. 收集 LoRA 适配器参数 (policy)
        for name, param in self.vla.named_parameters():
            if param.requires_grad:
                policy_params.append(param)
        
        # 3. 收集 value head 参数 (value)
        value_params.extend(list(self.value_head.parameters()))
        # 添加注意力池化层参数到价值组
        value_params.extend(list(self.attn_pool.parameters()))

        # 4. 验证没有遗漏任何可训练参数
        all_trainable_params = set(filter(lambda p: p.requires_grad, self.parameters()))
        grouped_params = set(policy_params) | set(value_params)
        
        # 打印调试信息
        if all_trainable_params != grouped_params:
            missing_params = all_trainable_params - grouped_params
            print(f"警告: 发现 {len(missing_params)} 个未分组的可训练参数:")
            for p in missing_params:
                for n, param in self.named_parameters():
                    if param is p:
                        print(f"  - {n}")
                        break
            raise ValueError("参数分组不完整！请检查未分组的参数。")
        
        return [
            {"name": "policy", "params": policy_params},
            {"name": "value", "params": value_params},
        ]

    def _extract_actions_hidden(self, last_hidden_states: torch.Tensor, logits: torch.Tensor, labels, has_act_emb) -> torch.Tensor:
        """
        从 last_hidden_states 和 logits 中提取动作相关的部分。
        由于lm_head已经被精简为只输出n_action_bins，所以logits直接可用。
        
        返回:
          action_logits: (B, NUM_ACTIONS_CHUNK * ACTION_DIM, n_action_bins)
          actions_hidden_states: (B, NUM_ACTIONS_CHUNK * ACTION_DIM, D)
        """
        ground_truth_token_ids = labels[:, 1:].to(self.device)  # (B, text_len-1)
        current_action_mask = get_current_action_mask(ground_truth_token_ids)  # (B, text_len-1)
        next_actions_mask = get_next_actions_mask(ground_truth_token_ids)      # (B, text_len-1)
        action_mask = current_action_mask | next_actions_mask

        num_patches = self._compute_num_patches()
        if has_act_emb:
            num_patches += 1
        text_hidden_states = last_hidden_states[:, num_patches:-1]  # (B, text_len, D)
        text_logits = logits[:, num_patches:-1]  # (B, text_len, n_action_bins) - 已经是精简后的

        B, _, D = text_hidden_states.shape
        actions_hidden_states = (
            text_hidden_states[action_mask]
            .reshape(B, NUM_ACTIONS_CHUNK * ACTION_DIM, D)
            .to(self.model_dtype)
        )
        
        # 提取动作对应的logits（已经是精简后的256维）
        action_logits = text_logits[action_mask].reshape(B, NUM_ACTIONS_CHUNK * ACTION_DIM, self.n_action_bins)
        
        return action_logits, actions_hidden_states

    def _forward_vla(self, batch: Dict[str, torch.Tensor]):
        return forward_vla(self, batch)

    def _compute_value_from_hidden(self, actions_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        使用注意力池化计算状态价值
        actions_hidden_states: (B, num_tokens, D)
        """
        # 1. 计算注意力分数
        scores = self.attn_pool(actions_hidden_states)  # (B, num_tokens, 1)
        
        # 2. 应用softmax获取注意力权重
        weights = torch.softmax(scores, dim=1)  # (B, num_tokens, 1)
        
        # 3. 加权平均得到池化表示
        pooled = torch.sum(weights * actions_hidden_states, dim=1)  # (B, D)
        
        # 4. 通过价值头计算最终价值
        value = self.value_head(pooled).squeeze(-1)  # (B,)
        return value.to(torch.float32)

    def forward(self, inputs_batch: Dict[str, Any], return_vit_out=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          action_logits: (B, NUM_ACTIONS_CHUNK * ACTION_DIM, n_action_bins)
          value:         (B,)
        """
        # Sanity checks
        for k in ("input_ids", "attention_mask", "pixel_values", "labels", "proprio"):
            if k not in inputs_batch:
                raise KeyError(f"inputs_batch missing key: {k}")

        # 1. VLA前向传播获取隐藏状态和logits
        output = self._forward_vla(inputs_batch)
        last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)
        logits = output.logits  # (B, seq_len, n_action_bins) - 已经是精简后的

        logits = output.logits
        action_logits, actions_hidden_states = self._extract_actions_hidden(last_hidden_states, logits, inputs_batch['labels'], has_act_emb=("this_act_emb" in inputs_batch))

        # 3. 计算价值函数
        value = self._compute_value_from_hidden(actions_hidden_states.detach())  # (B,)

        if return_vit_out:
            return action_logits, value.to(torch.float32), output.projector_features
        else:
            return action_logits, value.to(torch.float32)

    def post_process(self, logits: torch.Tensor, deterministic: List[bool]) -> Tuple[torch.distributions.Categorical, torch.Tensor, np.ndarray]:
        """
        后处理logits以生成动作。
        注意：现在logits已经是精简后的 (B, num_dims, n_action_bins)，无需再截取。
        """
        # 创建分布并计算两种动作
        dist = torch.distributions.Categorical(logits=logits)
        stochastic_tokens = dist.sample()
        deterministic_tokens = torch.argmax(logits, dim=-1)
        is_deterministic_tensor = torch.tensor(
            deterministic, dtype=torch.bool, device=logits.device
        )
        is_deterministic_tensor = is_deterministic_tensor.unsqueeze(1)
        action_token_ids = torch.where(
            is_deterministic_tensor, deterministic_tokens, stochastic_tokens
        )

        # 将token ID转换为bin索引（注意：现在action_token_ids范围是0到n_action_bins-1）
        actions_from_tokens = self.n_action_bins - 1 - action_token_ids
        discretized = np.clip(actions_from_tokens.cpu().numpy(), a_min=0, a_max=self.bin_centers.shape[0] - 1)
        normalized_actions = self.bin_centers[discretized]  # 形状 (B, NUM_ACTIONS_CHUNK * ACTION_DIM)
        normalized_actions = normalized_actions.reshape(
            normalized_actions.shape[0], NUM_ACTIONS_CHUNK, ACTION_DIM
        )
        
        return dist, action_token_ids, normalized_actions

    def prepare_inputs_batch(self, inp, max_len=None):
        return prepare_inputs_batch(self, inp, max_len)

    def get_norm_stats(self):
        return self.vla.norm_stats[self.cfg.unnorm_key]["proprio"]

    def _compute_num_patches(self):
        return compute_num_patches(self.vla, self.cfg)


if __name__ == "__main__":
    import sys
    import numpy as np

    # Libero env wrapper and helpers
    from rl.libero_env import LiberoEnvWrapper
    from rl.utils import prepare_one_obs, check_unnorm_key
    from experiments.robot.libero.libero_utils import GenerateConfig, TaskSuite

    # Precision policy to match the example
    USE_BF16: bool = True
    TORCH_DTYPE = torch.bfloat16 if USE_BF16 else torch.float32

    # 在这里设置要并行处理的环境数量
    ENVS_ID = list(range(10))
    envs_num = len(ENVS_ID)
    BENCHMARK = TaskSuite.LIBERO_OBJECT
    unnorm_key = f"{BENCHMARK}_no_noops"

    # Instantiate config
    cfg = GenerateConfig(
        pretrained_checkpoint="/cpfs01/liuwei_workspace/models/finetune_im/openvla-7b+libero_object_no_noops+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state--150000_chkpt",
        use_l1_regression=False,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=True,
        num_open_loop_steps=NUM_ACTIONS_CHUNK,
        unnorm_key=unnorm_key,
    )

    # 创建策略
    actor = ActorCritic(cfg, TORCH_DTYPE)
    parameter_groups = actor.get_parameter_groups()

    check_unnorm_key(cfg, actor.vla)
    actor.eval()
    
    # 检查参数类型
    for key, value in actor.named_parameters():
        if value.dtype != TORCH_DTYPE:
            print(f"Warning: Parameter {key} has dtype {value.dtype}, expected {TORCH_DTYPE}.")
    print("策略初始化完成。")

    # 初始化环境
    print(f"正在初始化 {len(ENVS_ID)} 个并行的 Libero 环境...")
    envs = [
        LiberoEnvWrapper(
            benchmark_name=BENCHMARK,
            task_id=env_id,
            image_size=224,
            render_mode="rgb_array",
        )
        for env_id in ENVS_ID
    ]
    print("所有环境初始化完成。")

    # 全局统计
    total_episodes_finished = 0
    total_successes = 0

    from collections import deque

    # 初始化每个环境的动作队列
    env_queues = [deque() for _ in range(len(ENVS_ID))]  # ENVS_ID是环境ID列表

    # 主循环
    while True:
        # 初始化环境状态
        observations = []
        task_descriptions = []
        for i, env in enumerate(envs):
            obs, info = env.reset(seed=int(time.time()) + i)
            observations.append(obs)
            task_descriptions.append(env.task_description)
            print(f"环境 {i}: 任务 ID = {env.task_id}, 任务描述 = {env.task_description}")
            env_queues[i].clear()  # 重置该环境的动作队列

        # 跟踪变量
        active_envs = [True] * envs_num
        total_rewards = [0.0] * envs_num
        episode_steps = [0] * envs_num
        success_info = [False] * envs_num

        print(f"\n开始第 {total_episodes_finished // envs_num + 1} 轮并行执行...")

        # 环境执行循环
        while any(active_envs):
            # 1. 收集需要生成新动作的环境（队列为空且活跃的环境）
            need_generation_indices = []  # 需要生成新动作的环境索引
            inputs_t_list = []  # 需要生成新动作的环境输入
            
            for i in range(envs_num):
                if active_envs[i] and len(env_queues[i]) == 0:
                    inputs_t = prepare_one_obs(cfg, actor.processor, observations[i], task_descriptions[i], TORCH_DTYPE)
                    inputs_t_list.append(inputs_t)
                    need_generation_indices.append(i)
            
            # 2. 为需要生成新动作的环境批量生成动作
            if inputs_t_list:
                inputs_batch = actor.prepare_inputs_batch(inputs_t_list)
                
                with torch.inference_mode():
                    action_logits, _ = actor.forward(inputs_batch)
                B = action_logits.size(0)
                deterministic_flags = [False] * B  # 若需贪心推理，改为 [True] * B
                _, _, normalized_actions = actor.post_process(action_logits, deterministic_flags)  # 形状 (B, 8, 7)
                                
                # 将生成的动作序列添加到对应环境的队列中
                for idx, env_idx in enumerate(need_generation_indices):
                    # 获取该环境生成的所有动作（8个）
                    action_sequence = normalized_actions[idx]  # 形状 (8, 7)
                    
                    # 将整个动作序列添加到队列
                    env_queues[env_idx].extend(action_sequence)  # 使用extend批量添加
            
            # 3. 执行动作（所有活跃环境）
            for i in range(envs_num):
                if not active_envs[i]:
                    continue  # 跳过非活跃环境
                    
                # 确保队列中有动作（如果没有，说明前面的生成动作步骤有问题）
                if len(env_queues[i]) == 0:
                    print(f"错误：环境 {i} 动作队列为空但未生成新动作")
                    continue
                    
                # 从队列中取出动作
                action_norm = env_queues[i].popleft()
                
                # 将归一化动作转换为环境动作
                action_env = actor.vla._unnormalize_actions(action_norm, cfg.unnorm_key)
                
                # 执行动作
                obs, reward, terminated, truncated, info = envs[i].step(action_env)
                
                # 更新状态
                observations[i] = obs
                total_rewards[i] += float(reward)
                episode_steps[i] += 1
                
                # 定期打印
                if episode_steps[i] % 50 == 0:
                    print(f"环境 {i}, Step: {episode_steps[i]}, 奖励: {reward:.4f}, 终止: {terminated}, 截断: {truncated}")
                
                # 检查环境是否完成
                if terminated or truncated:
                    is_success = info.get('is_success', False)
                    total_successes += is_success
                    total_episodes_finished += 1
                    success_info[i] = is_success
                    
                    print("-" * 40)
                    print(f"环境 {i} 已完成 (任务: {envs[i].task_description[:50]}...)")
                    print(f"总步数: {episode_steps[i]}, 总奖励: {total_rewards[i]:.4f}, 是否成功: {is_success}")
                    print(f"成功率: {total_successes/total_episodes_finished:.3f}, 总回合数: {total_episodes_finished}")
                    print("-" * 40)
                    
                    # 重置环境
                    active_envs[i] = False
                    episode_steps[i] = 0
                    total_rewards[i] = 0
                    obs, info = envs[i].reset(seed=random.randint(0, 1000))
                    observations[i] = obs
                    env_queues[i].clear()  # 重置动作队列

        # 每轮结束后打印统计信息
        print("=" * 60)
        print(f"第 {total_episodes_finished // envs_num} 轮完成!")
        print(f"累计总回合数: {total_episodes_finished}, 成功次数: {total_successes}")
        print(f"总体成功率: {total_successes/total_episodes_finished:.3f}")
        print("=" * 60)