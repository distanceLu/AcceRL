import os
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, List
from contextlib import nullcontext
import numpy as np

from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform
from experiments.robot.openvla_utils import L1RegressionActionHead

from peft import LoraConfig, get_peft_model

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

def get_vla(cfg: Any) -> torch.nn.Module:
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
        torch_dtype=torch.bfloat16,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(cfg.device)

    # 3) FiLM（若启用）
    if getattr(cfg, "use_film", False):
        from experiments.robot.openvla_utils import _apply_film_to_vla
        vla = _apply_film_to_vla(vla, cfg)

    # 4) 设定输入图像数量
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    vla.eval()

    # 5) 未量化时放到目标设备
    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        vla = vla.to(cfg.device)

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
        self.vla = get_vla(cfg)
        self.device = self.vla.device
        self.model_dtype = torch_dtype
        self.vla = self.vla.to(dtype=self.model_dtype)

        # 🔒 冻结 VLA 参数
        for param in self.vla.parameters():
            param.requires_grad = False

        # 标记 VLA 是否已经被 LoRA 修改
        self._vla_is_lora_tuned = False

        # 保留 processor（只是预处理，不需要训练）
        self.processor = get_processor(cfg)

        # Heads
        self.action_head = get_action_head(cfg, llm_dim=self.vla.llm_dim)
        self.action_head = self.action_head.to(self.device).to(dtype=self.model_dtype)

        self.proprio_projector = get_proprio_projector(
            cfg, llm_dim=self.vla.llm_dim, proprio_dim=PROPRIO_DIM
        )
        self.proprio_projector = self.proprio_projector.to(self.device).to(dtype=self.model_dtype)

        # Condition-independent log_std parameter (float32 for stability)
        # self.log_std_param = nn.Parameter(torch.full((NUM_ACTIONS_CHUNK, ACTION_DIM), -2, dtype=self.model_dtype, device=self.device))
        # self.log_std_param = L1RegressionActionHead(input_dim=self.vla.llm_dim, hidden_dim=self.vla.llm_dim, action_dim=ACTION_DIM).to(self.device).to(dtype=self.model_dtype)
        self.register_buffer('log_std_param', torch.full((NUM_ACTIONS_CHUNK, ACTION_DIM), -2.0, dtype=self.model_dtype))

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
        self.setup_finetuning(cfg.lora_rank, cfg.lora_dropout)

    def setup_finetuning(self, lora_rank: int, lora_dropout: float):
        """为微调准备模型，注入 LoRA 适配器。"""
        if self.cfg.use_lora:
            print("Injecting LoRA adapters for fine-tuning...")
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=min(lora_rank, 16),
                lora_dropout=lora_dropout,
                target_modules="all-linear",
                init_lora_weights="gaussian",
            )
            self.vla = get_peft_model(self.vla, lora_config)
            self._vla_is_lora_tuned = True
            print("LoRA injection complete.")
            self.vla.print_trainable_parameters()

    def get_parameter_groups(self) -> List[Dict[str, Any]]:
        """
        将可训练参数分为 'policy' 和 'value' 两组。
        这对于为不同组件设置不同的学习率至关重要。
        """
        policy_params = list(self.action_head.parameters()) + list(self.proprio_projector.parameters())
        value_params = list(self.value_head.parameters()) + list(self.attn_pool.parameters())

        if self._vla_is_lora_tuned:
            lora_params = [p for p in self.vla.parameters() if p.requires_grad]
            policy_params += lora_params
        
        # 确保没有遗漏任何可训练参数
        all_trainable_params = set(filter(lambda p: p.requires_grad, self.parameters()))
        grouped_params = set(policy_params) | set(value_params) | (set(lora_params) if self._vla_is_lora_tuned else set())
        assert all_trainable_params == grouped_params, "并非所有可训练参数都被分组！"

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
            tensors = [it[k] for it in inputs_list]
            inputs[k] = torch.cat(tensors, dim=0).to(self.vla.device)
        inputs["proprio"] = inputs["proprio"].to(torch.float32)
        return inputs

    def prepare_inputs_batch(self, inputs_list: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        对多条样本执行：
          - 归一化 proprio 到 [-1, 1]
          - 基本一致性检查
          - 序列右侧 padding 并拼 batch
        """
        inputs_list = inputs_list.copy()
        for i, it in enumerate(inputs_list):
            inputs_list[i] = it.copy()
        # Normalize proprio for each sample and run per-sample checks
        for it in inputs_list:
            # Normalize proprio using internal norm stats
            proprio_norm = self.normalize_proprio(it["proprio"])
            it["proprio"] = torch.tensor(proprio_norm, dtype=torch.float32).unsqueeze(dim=0)

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

    def _extract_actions_hidden(self, last_hidden_states: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        From last_hidden_states, extract the text-token hiddens corresponding
        to current + next actions, as (B, NUM_ACTIONS_CHUNK*ACTION_DIM, D).
        """
        ground_truth_token_ids = batch["labels"][:, 1:].to(self.device)  # (B, text_len-1)
        current_action_mask = get_current_action_mask(ground_truth_token_ids)  # (B, text_len-1)
        next_actions_mask = get_next_actions_mask(ground_truth_token_ids)      # (B, text_len-1)
        action_mask = current_action_mask | next_actions_mask

        num_patches = self._compute_num_patches()
        if 'this_act_emb' in batch:
            num_patches += 1
        text_hidden_states = last_hidden_states[:, num_patches:-1]  # (B, text_len, D)

        B, _, D = text_hidden_states.shape
        actions_hidden_states = (
            text_hidden_states[action_mask]
            .reshape(B, NUM_ACTIONS_CHUNK * ACTION_DIM, D)
            .to(self.model_dtype)
        )
        return actions_hidden_states

    def _forward_vla(self, batch: Dict[str, torch.Tensor]):
        """
        Single VLA forward that returns output with hidden states.
        """
        ctx = torch.autocast("cuda", dtype=self.model_dtype) if self.device.type == "cuda" else nullcontext()
        with ctx:
            self.vla: OpenVLAForActionPrediction
            output = self.vla.forward(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"].to(self.model_dtype),
                labels=batch["labels"],  # for mask derivation and potential loss
                output_hidden_states=True,
                proprio=batch["proprio"].to(self.model_dtype) if self.cfg.use_proprio else None,
                proprio_projector=self.proprio_projector if self.cfg.use_proprio else None,
                noisy_actions=None,
                noisy_action_projector=None,
                diffusion_timestep_embeddings=None,
                use_film=self.cfg.use_film,
                this_act_emb=batch.get("this_act_emb", None),  # (B, 1, 4096) or None
            )
        return output

    def _compute_value_from_hidden(self, actions_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        使用注意力池化计算状态价值
        actions_hidden_states: (B, C * A_dim, D), 
        """
        # 1. 计算注意力分数
        # actions_hidden_states 的类型需要与 attn_pool 匹配
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
          actions_all: (B, NUM_ACTIONS_CHUNK, ACTION_DIM)
          mu_all:      (B, NUM_ACTIONS_CHUNK, ACTION_DIM)
          log_std_all: (B, NUM_ACTIONS_CHUNK, ACTION_DIM)
          value:       (B,)
        """
        # Sanity checks
        for k in ("input_ids", "attention_mask", "pixel_values", "labels", "proprio"):
            if k not in inputs_batch:
                raise KeyError(f"inputs_batch missing key: {k}")

        # 1) VLA forward to obtain hidden states
        output = self._forward_vla(inputs_batch)
        last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)

        # 2) Predict continuous actions mean (mu) using action-related hidden states
        actions_hidden_states = self._extract_actions_hidden(last_hidden_states, inputs_batch)
        predicted_actions = self.action_head.predict_action(actions_hidden_states)  # (B, NUM_ACTIONS_CHUNK, ACTION_DIM) or flat
        if predicted_actions.dim() == 3:
            mu_all = predicted_actions
        else:
            raise ValueError(f"Unexpected predicted_actions shape: {predicted_actions.shape}")

        # 3) Condition-independent log_std broadcast across chunks
        B = mu_all.size(0)
        log_std = self.log_std_param  # (NUM_ACTIONS_CHUNK, ACTION_DIM)
        log_std_all = log_std.unsqueeze(dim=0).expand(B, NUM_ACTIONS_CHUNK, ACTION_DIM)  # (B, T, A)
        # log_std_all = self.log_std_param.predict_action(actions_hidden_states) - 2

        # 4) Squashed Gaussian sampling to (-1, 1) for all chunks
        std_all = torch.exp(log_std_all)  # (B, T, A)
        base_dist = Normal(mu_all.to(torch.float32), std_all)        # fp32 sampling for stability
        # dist = TransformedDistribution(base_dist, [TanhTransform(cache_size=1)])
        dist = base_dist
        actions_all = dist.sample()                                  # (B, T, A) in (-1, 1)

        # 5) Value from hidden states
        value = self._compute_value_from_hidden(actions_hidden_states.detach())   # (B,)

        if return_vit_out:
            return actions_all.to(torch.float32), mu_all.to(torch.float32), log_std_all.to(torch.float32), value.to(torch.float32), output.projector_features.to(torch.float32)
        else:
            return actions_all.to(torch.float32), mu_all.to(torch.float32), log_std_all.to(torch.float32), value.to(torch.float32)

def load_log_std(self, checkpoint_dir: str, step: int|str):
        # --- 加载 Log_Std parameter ---
        log_std_head_path = os.path.join(checkpoint_dir, f"log_std_head--{step}_checkpoint.pt")
        if not os.path.exists(log_std_head_path):
            raise FileNotFoundError(f"Log_Std Head checkpoint not found at: {log_std_head_path}")
        
        print(f"  -> Loading Log_std from {log_std_head_path}")
        loaded_data = torch.load(log_std_head_path, map_location=self.device)
        
        if isinstance(self.log_std_param, nn.Module):
            print("  -> Target `self.log_std_param` is an nn.Module. Attempting to load state_dict.")
            state_dict = loaded_data
            # 处理分布式训练 (DDP) 保存的 'module.' 前缀
            if all(key.startswith('module.') for key in state_dict.keys()):
                print("  -> Removing 'module.' prefix from state_dict keys.")
                state_dict = {k.partition('module.')[2]: v for k, v in state_dict.items()}
            
            self.log_std_param.load_state_dict(state_dict)

        elif isinstance(self.log_std_param, nn.Parameter):
            print("  -> Target `self.log_std_param` is an nn.Parameter. Attempting to load data.")
            tensor_to_load = loaded_data['log_std_param']
            with torch.no_grad():
                self.log_std_param.data.copy_(tensor_to_load)
        
        else:
            # 如果 self.log_std_param 不是我们支持的类型
             raise TypeError(f"self.log_std_param is of an unsupported type: {type(self.log_std_param)}")

        print("Log_std parameter loading complete.")


if __name__ == "__main__":
    import numpy as np
    import random
    import time
    from experiments.robot.robot_utils import set_seed_everywhere
    
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
    # LIBERO_SPATIAL: 93.2% (10 envs, 2982 episodes)
    # LIBERO_GOAL: 84.1% (10 envs, 2101 episodes)
    # LIBERO_OBJECT: 51.6% (10 envs, 1475 episodes)

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
        device=torch.device("cuda:0")
    )
    set_seed_everywhere(cfg.seed)
    # Create ActorCritic policy
    actor = ActorCritic(cfg, TORCH_DTYPE)
    actor.load_log_std(cfg.pretrained_checkpoint, step="latest")

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
        obs, info = env.reset(seed=int(time.time()) + i)
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
            sample_all, mu_all, _, _ = actor.forward(inputs_batch)
            action_all = torch.clamp(mu_all, -1.0, 1.0)
            # action_all = torch.clamp(sample_all, -1.0, 1.0)

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
                obs, info = envs[env_idx].reset(seed=random.randint(0, 1000))
                observations[env_idx] = obs
