import os
os.makedirs("obs", exist_ok=True)

import pickle
import time
import random

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, List
from contextlib import nullcontext
import numpy as np
from PIL import Image
import datetime
from peft import LoraConfig, PeftModel, get_peft_model
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform

# Core OpenVLA components
from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
)

from experiments.robot.robot_utils import (
    invert_gripper_action,
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

# 显式类：避免依赖 auto_map
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction

from transformers.models.llama.modeling_llama import LlamaForCausalLM
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_vla(cfg: Any,torch_dtype: torch.dtype = torch.bfloat16) -> torch.nn.Module:
    """
    只读加载 OpenVLA：不修改 checkpoint 内的 config.json。
    """
    print("Instantiating pretrained VLA policy (read-only, no config.json mutation)...")

    # 1) 显式加载 Config（不会触发 auto_map 也不会写文件）
    vla_cfg = OpenVLAConfig.from_pretrained(
        cfg.pretrained_checkpoint,
        trust_remote_code=True,   # 允许自定义类
    )

    # 2) 显式加载模型（不走 Auto*，不需要 auto_map）
    vla = OpenVLAForActionPrediction.from_pretrained(
        cfg.pretrained_checkpoint,
        config=vla_cfg,
        torch_dtype=torch_dtype,     #bfloat16 
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # 3) FiLM（若启用）
    if getattr(cfg, "use_film", False):
        from experiments.robot.openvla_utils import _apply_film_to_vla
        vla = _apply_film_to_vla(vla, cfg)

    # 4) 设定输入图像数量
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    vla.eval()

    # 5) 未量化时放到目标设备
    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        vla = vla.to(DEVICE)

    # 6) 加载数据集统计（归一化/反归一化用）
    from experiments.robot.openvla_utils import _load_dataset_stats
    _load_dataset_stats(vla, cfg.pretrained_checkpoint)

    return vla


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
        self.vla = get_vla(cfg,torch_dtype)
        self.device = self.vla.device
        self.model_dtype = torch_dtype
        self.vla = self.vla.to(dtype=self.model_dtype)

        # 应用LoRA配置（消融2）
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
        # 打印可训练参数信息
        self.vla.print_trainable_parameters()

        self.vocab_size = self.vla.config.text_config.vocab_size - self.vla.config.pad_to_multiple_of
        self.bins = np.linspace(-1, 1, self.vla.config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # Keep processor for external preparation (forward 接收已组装好的 batch，但依旧保留 processor)
        self.processor = get_processor(cfg)
        self.proprio_projector = None
        
        # 注意力池化层
        self.attn_pool = nn.Sequential(
            nn.Linear(self.vla.llm_dim, 1),
        ).to(self.device).to(dtype=self.model_dtype)

        # Value head: mean-pool over text tokens from the last hidden layer -> scalar
        self.value_head = nn.Sequential(
            nn.LayerNorm(self.vla.llm_dim),
            nn.Linear(self.vla.llm_dim, self.vla.llm_dim),
            nn.ReLU(),
            nn.Linear(self.vla.llm_dim, 1),
        ).to(self.device).to(dtype=self.model_dtype)

    def get_parameter_groups(self) -> List[Dict[str, Any]]:
        """
        将可训练参数分为 'policy' 和 'value' 两组。
        这对于为不同组件设置不同的学习率至关重要。
        """
        self.vla.language_model: LlamaForCausalLM
        
        # 1. 收集所有可训练参数
        policy_params = []
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

    def normalize_proprio(self, proprio: Any) -> np.ndarray:
        """
        Normalize proprioception data using self.vla.norm_stats[self.cfg.unnorm_key]["proprio"].
        Accepts numpy array or torch tensor; returns numpy array in [-1, 1].
        """
        # Convert to numpy
        if isinstance(proprio, torch.Tensor):
            proprio = proprio.detach().cpu().numpy()
        else:
            proprio = np.asarray(proprio)

        norm_stats = self.vla.norm_stats[self.cfg.unnorm_key]["proprio"]

        if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
            mask = norm_stats.get("mask", np.ones_like(norm_stats["min"], dtype=bool))
            proprio_high, proprio_low = np.array(norm_stats["max"]), np.array(norm_stats["min"])
        elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
            mask = norm_stats.get("mask", np.ones_like(norm_stats["q01"], dtype=bool))
            proprio_high, proprio_low = np.array(norm_stats["q99"]), np.array(norm_stats["q01"])
        else:
            raise ValueError("Unsupported action/proprio normalization type detected!")

        normalized_proprio = np.clip(
            np.where(
                mask,
                2 * (proprio - proprio_low) / (proprio_high - proprio_low + 1e-8) - 1,
                proprio,
            ),
            a_min=-1.0,
            a_max=1.0,
        )
        return normalized_proprio

    def batch_process_obs(self, inputs_list: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Right-pad variable-length sequences across a list of samples and stack into a batch on self.vla.device.
        Expects each item to contain: input_ids, attention_mask, labels, pixel_values, proprio, etc.
        """
        # 目标序列最大长度（对齐到同一个 max_len，确保各 key 同长）
        max_len = max(it["input_ids"].size(1) for it in inputs_list)
        pad_id = int(self.vla.pad_token_id)

        # 对每条样本进行右侧 padding
        for it in inputs_list:
            cur_len = it["input_ids"].size(1)
            if cur_len < max_len:
                pad_amt = max_len - cur_len
                bsz = it["input_ids"].size(0)  # 通常为 1

                # input_ids: pad_id
                pad_ids = it["input_ids"].new_full((bsz, pad_amt), pad_id)
                it["input_ids"] = torch.cat([it["input_ids"], pad_ids], dim=1)

                # attention_mask: 0
                pad_mask = it["attention_mask"].new_zeros((bsz, pad_amt))
                it["attention_mask"] = torch.cat([it["attention_mask"], pad_mask], dim=1)

                # labels: -100
                pad_labels = it["labels"].new_full((bsz, pad_amt), -100)
                it["labels"] = torch.cat([it["labels"], pad_labels], dim=1)

        # 聚合成 batch，并移动到目标设备
        inputs: Dict[str, torch.Tensor] = {}
        keys = inputs_list[0].keys()
        for k in keys:
            tensors = [it[k] for it in inputs_list if it.get(k) is not None and isinstance(it[k], torch.Tensor)]
            # 如果 tensors 不为空，则拼接；否则跳过
            if tensors:
                inputs[k] = torch.cat(tensors, dim=0).to(self.vla.device)
            else: 
                pass 
        return inputs

    def prepare_inputs_batch(self, inputs_list: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        对多条样本执行：
          - 归一化 proprio 到 [-1, 1]
          - 基本一致性检查
          - 序列右侧 padding 并拼 batch
        """
        # Normalize proprio for each sample and run per-sample checks
        for it in inputs_list:
            # Consistency check
            assert it["input_ids"].size(1) == it["attention_mask"].size(1) == it["labels"].size(1), \
                "Per-sample sequence lengths of input_ids/attention_mask/labels must match."

        # Batchify
        return self.batch_process_obs(inputs_list)
    
    def _compute_num_patches(self) -> int:
        num_patches = (
            self.vla.vision_backbone.get_num_patches()
            * self.vla.vision_backbone.get_num_images_in_input()
        )
        if self.cfg.use_proprio:
            num_patches += 1
        return num_patches

    def _extract_actions_hidden(self, last_hidden_states, logits: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        From last_hidden_states, extract the text-token hiddens corresponding
        to current + next actions, as (B, NUM_ACTIONS_CHUNK*ACTION_DIM, D).
        """
        ground_truth_token_ids = batch["labels"][:, 1:].to(self.device)  # (B, text_len-1)
        current_action_mask = get_current_action_mask(ground_truth_token_ids)  # (B, text_len-1)
        next_actions_mask = get_next_actions_mask(ground_truth_token_ids)      # (B, text_len-1)
        action_mask = current_action_mask | next_actions_mask

        num_patches = self._compute_num_patches()
        text_hidden_states = last_hidden_states[:, num_patches:-1]  # (B, text_len, D)

        B, _, D = text_hidden_states.shape
        actions_hidden_states = (
            text_hidden_states[action_mask]
            .reshape(B, NUM_ACTIONS_CHUNK * ACTION_DIM, D)
            .to(self.model_dtype)
        )
        text_logits = logits[:, num_patches:-1]  # (B, text_len, D)
        _, _, vocab_size = text_logits.shape
        actions_logits = text_logits[action_mask].reshape(B, NUM_ACTIONS_CHUNK * ACTION_DIM, vocab_size)
        logits_cut = actions_logits[..., self.vocab_size-self.vla.config.n_action_bins:self.vocab_size]
        return logits_cut, actions_hidden_states

    def _forward_vla(self, batch: Dict[str, torch.Tensor]):
        """
        Single VLA forward that returns output with hidden states.
        """
        # with ctx:
        self.vla: OpenVLAForActionPrediction
        output = self.vla.forward(
            input_ids=batch["input_ids"].to(self.device),
            attention_mask=batch["attention_mask"].to(self.device),
            pixel_values=batch["pixel_values"].to(self.model_dtype).to(self.device),
            labels=batch["labels"].to(self.device),  # for mask derivation and potential loss
            output_hidden_states=True,
            proprio=batch["proprio"] if self.cfg.use_proprio else None,
            proprio_projector=self.proprio_projector if self.cfg.use_proprio else None,
            noisy_actions=None,
            noisy_action_projector=None,
            diffusion_timestep_embeddings=None,
            use_film=self.cfg.use_film,
        )
        return output

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

    def forward(self, inputs_batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          actions_all: (B, NUM_ACTIONS_CHUNK, ACTION_DIM)
          mu_all:      (B, NUM_ACTIONS_CHUNK, ACTION_DIM)
          log_std_all: (B, NUM_ACTIONS_CHUNK, ACTION_DIM)
          value:       (B,)
        """
        # Sanity checks
        for k in ("input_ids", "attention_mask", "pixel_values", "labels"):
            if k not in inputs_batch:
                raise KeyError(f"inputs_batch missing key: {k}")

        # 1. 提取动作部分的原始logits（未argmax）   
        output = self._forward_vla(inputs_batch)
        last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)

        logits = output.logits
        action_logits, actions_hidden_states = self._extract_actions_hidden(last_hidden_states, logits, inputs_batch)

        # 2. 计算价值函数
        value = self._compute_value_from_hidden(actions_hidden_states.detach())  # (B,)

        return action_logits, value.to(torch.float32)

    def post_process(self, logits):
        batch_dist = torch.distributions.Categorical(logits=logits)  # 批量创建分布
        # 采样时 (替代原来的循环采样)
        action_token_ids = batch_dist.sample()  # shape = (B, num_dims)
        actions_all = self.vla.config.n_action_bins - action_token_ids  # shape = (B, num_dims)

        discretized_actions = np.clip(actions_all.cpu().numpy(), a_min=0, a_max=self.bin_centers.shape[0] - 1)
        normalized_actions = self.bin_centers[discretized_actions]  # (B, NUM_ACTIONS_CHUNK * ACTION_DIM)
        normalized_actions = normalized_actions.reshape(normalized_actions.shape[0], NUM_ACTIONS_CHUNK, ACTION_DIM)  # (B, NUM_ACTIONS_CHUNK, ACTION_DIM)
        return batch_dist, action_token_ids, normalized_actions

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
    BENCHMARK = TaskSuite.LIBERO_SPATIAL
    unnorm_key = f"{BENCHMARK}_no_noops"

    # Instantiate config
    cfg = GenerateConfig(
        pretrained_checkpoint="/cpfs01/jinshiji_workspace/openvla_oft_rl/runs/openvla-7b-oft-finetuned-2_gpus_batch_size_16",
        use_l1_regression=False,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=False,
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
                _, _, normalized_actions = actor.post_process(action_logits)  # 形状 (b, 8, 7)
                
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