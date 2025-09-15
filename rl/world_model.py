import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, List
import numpy as np

import collections
import random

from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from storm.functions_losses import SymLogTwoHotLoss

# Constants
from prismatic.vla.constants import (
    NUM_ACTIONS_CHUNK,
    ACTION_DIM,
)
from typing import Any
import torch

# 显式类：避免依赖 auto_map
from rl.actor_critic_model import ActorCritic


class WorldModel(ActorCritic):
    """
    基于 OpenVLA 的 Actor-Critic 模型，用于连续控制。
    此版本已修改，forward 函数返回中间张量，损失计算在外部进行。
    """

    def __init__(self, cfg, torch_dtype: torch.dtype, device):
        super().__init__(cfg, torch_dtype, device)
        hidden_size = self.vla.llm_dim
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=0,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        self.language_model = get_peft_model(self.vla.language_model, lora_config)
        self.language_model.print_trainable_parameters()
        # self.language_model = self.vla.language_model
        self.language_model: LlamaForCausalLM
        # for param in self.language_model.model.layers[0].parameters():
        #     param.requires_grad = True
        del self.action_head
        self.action_head = AttentionPoolHead(hidden_size, NUM_ACTIONS_CHUNK * ACTION_DIM).to(self.device).to(dtype=self.model_dtype)
        for param in self.action_head.parameters():
            param.requires_grad = True
        for param in self.proprio_projector.parameters():
            param.requires_grad = False
        for param in self.value_head.parameters():
            param.requires_grad = True
        self.log_std_param.requires_grad = True
        self.patch_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        ).to(self.device).to(dtype=self.model_dtype)
        self.act_proj = nn.Sequential(
            nn.Linear(ACTION_DIM * NUM_ACTIONS_CHUNK, hidden_size),
        ).to(self.device).to(dtype=self.model_dtype)
        # 注意力池化层
        self.reward_decoder = AttentionPoolHead(hidden_size, 1).to(self.device).to(dtype=self.model_dtype)
        self.termi_pool = AttentionPool(hidden_size).to(self.device).to(dtype=self.model_dtype)
        self.termi_decoder = nn.Sequential(
            nn.Linear(hidden_size+16, hidden_size, bias=False),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 2)
        ).to(self.device).to(dtype=self.model_dtype)
        self.step_count_emb = nn.Embedding(500, 16).to(self.device).to(dtype=self.model_dtype)
        self.symlog_twohot_loss_func = SymLogTwoHotLoss(num_classes=2, lower_bound=0, upper_bound=1).to(self.device).to(dtype=self.model_dtype)
        # self.bce_with_logits_loss_func = nn.BCEWithLogitsLoss()

    def get_trainable_params(self) -> List[Dict[str, Any]]:
        for param in self.action_head.parameters():
            param.requires_grad = False
        for param in self.value_head.parameters():
            param.requires_grad = False
        self.log_std_param.requires_grad = False
        auto_encoder_params = list(filter(lambda p: p.requires_grad, self.language_model.parameters())) + \
                              list(self.patch_proj.parameters()) + \
                              list(self.act_proj.parameters())
        
        # 确保没有遗漏任何可训练参数
        all_trainable_params = set(filter(lambda p: p.requires_grad, self.parameters()))
        grouped_params = set(auto_encoder_params)
        assert all_trainable_params == grouped_params, "并非所有可训练参数都被分组！"
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"可训练参数数量: {trainable_params}")

        return auto_encoder_params

    def get_parameter_groups(self) -> List[Dict[str, Any]]:
        """
        为优化器提供参数分组，以应用不同的学习率。
        这对于稳定 PPO 训练至关重要。
        """
        # 策略部分：动作头和学习标准差
        policy_params = list(self.action_head.parameters()) + [self.log_std_param]
        
        # 价值部分：价值头
        value_params = list(self.value_head.parameters())

        # 世界模型/语言模型部分：可训练的语言模型层和新的投影层
        lan_params = list(filter(lambda p: p.requires_grad, self.language_model.parameters()))
        world_model_params = lan_params + \
                             list(self.patch_proj.parameters()) + \
                             list(self.act_proj.parameters()) + \
                             list(self.termi_pool.parameters()) + \
                             list(self.reward_decoder.parameters()) + \
                             list(self.termi_decoder.parameters()) + \
                             list(self.step_count_emb.parameters())
        lan_params_count = sum(p.numel() for p in lan_params)
        print(f"WorldModel 中可训练的语言模型参数数量: {lan_params_count:,}")

        # 将世界模型参数合并到策略参数中进行训练，或为其创建单独的组
        # 这里为了简化，我们将其与策略部分合并
        combined_policy_params = policy_params + world_model_params
        
        # 确保没有遗漏任何可训练参数
        all_trainable_params = set(filter(lambda p: p.requires_grad, self.parameters()))
        grouped_params = set(combined_policy_params) | set(value_params)
        if all_trainable_params != grouped_params:
            print("警告: 并非所有可训练参数都被分组！")
            print(f"遗漏的参数: {all_trainable_params - grouped_params}")

        trainable_params_count = sum(p.numel() for p in all_trainable_params)
        print(f"WorldModel 中可训练参数总量: {trainable_params_count:,}")

        return [
            {"name": "policy", "params": combined_policy_params},
            {"name": "value", "params": value_params},
        ]

    def forward(self, inputs_batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        修改后的前向传播函数。
        返回计算 PPO、AE 和 IL 损失所需的所有张量。

        返回:
          - mu_all: 策略的均值 (B, NUM_ACTIONS_CHUNK, ACTION_DIM)
          - log_std_all: 策略的对数标准差 (B, NUM_ACTIONS_CHUNK, ACTION_DIM)
          - value: 状态价值估计 (B,)
          - post_patch_embeddings: 经过 AE 投影层的图像嵌入，用于 AE 损失计算 (B, num_patches, D)
          - projector_features: VLA 内部投影后的原始图像特征，作为 AE 损失的目标 (B, num_patches, D)
        """
        for k in ("input_ids", "attention_mask", "pixel_values", "labels", "proprio"):
            if k not in inputs_batch:
                raise KeyError(f"inputs_batch missing key: {k}")
        
        if 'this_action' in inputs_batch:
            b_s = inputs_batch['this_action'].size(0)
            this_action = inputs_batch['this_action'].reshape(b_s, -1).to(self.model_dtype)  # (B, ACTION_DIM * NUM_ACTIONS_CHUNK)
            this_act_emb = self.act_proj(this_action)  # (B, 4096)
            inputs_batch['this_act_emb'] = this_act_emb.unsqueeze(dim=1)  # (B, 1, 4096)

        # 1) VLA 前向传播以获取隐藏状态
        output = self._forward_vla(inputs_batch)
        last_hidden_states = output.hidden_states[-1]
        recon_hidden_states = output.hidden_states[-1]  # len(output.hidden_states): 33
        num_patches = self._compute_num_patches()
        if 'step_count' in inputs_batch:
            step_count = inputs_batch['step_count']
            step_emb = self.step_count_emb(step_count)  # (B, 16)
        
        # 2) 准备用于 AE 损失的张量
        if 'this_act_emb' in inputs_batch:
            post_patch_embeddings = recon_hidden_states[:, 2:num_patches+2]
            reward_logits = self.reward_decoder(post_patch_embeddings)  # (B, 2)
            termi_pooled = self.termi_pool(post_patch_embeddings)
            termin_hat = self.termi_decoder(torch.cat((termi_pooled, step_emb), dim=1)).squeeze(-1)
        else:
            post_patch_embeddings = recon_hidden_states[:, 1:num_patches+1]
            reward_logits = None
            termin_hat = None
        post_patch_embeddings = self.patch_proj(post_patch_embeddings)
        projector_features = output.projector_features

        # 3) 预测连续动作
        actions_hidden_states = self._extract_actions_hidden(last_hidden_states, inputs_batch)
        # predicted_actions = self.action_head.predict_action(actions_hidden_states)
        predicted_actions = self.action_head(post_patch_embeddings).reshape(-1, NUM_ACTIONS_CHUNK, ACTION_DIM)  # (B, T, A)
        if predicted_actions.dim() == 3:
            mu_all = predicted_actions
        else:
            raise ValueError(f"Unexpected predicted_actions shape: {predicted_actions.shape}")

        # 3) Condition-independent log_std broadcast across chunks
        B = mu_all.size(0)
        log_std = self.log_std_param  # (NUM_ACTIONS_CHUNK, ACTION_DIM)
        log_std_all = log_std.unsqueeze(dim=0).expand(B, NUM_ACTIONS_CHUNK, ACTION_DIM)  # (B, T, A)

        # 5) Value from hidden states
        value = self._compute_value_from_hidden(actions_hidden_states)
        # 4) 返回用于外部损失计算的张量
        return (
            mu_all.to(torch.float32), 
            log_std_all.to(torch.float32), 
            value.to(torch.float32), 
            post_patch_embeddings.to(torch.float32), 
            projector_features.to(torch.float32),
            reward_logits.to(torch.float32) if reward_logits is not None else None,
            termin_hat.to(torch.float32) if termin_hat is not None else None
            )


class AttentionPoolHead(nn.Module):
    def __init__(self, hidden_size: int, out_size: int):
        super().__init__()
        self.attn_pool = AttentionPool(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, out_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled = self.attn_pool(hidden_states)
        out = self.mlp(pooled)
        return out


class AttentionPool(nn.Module):
    """
    注意力池化头，用于从隐藏状态中提取动作相关的表示。
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn_pool = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。

        Args:
            hidden_states (torch.Tensor): 输入的隐藏状态，形状为 (B, num_tokens, D)。

        Returns:
            torch.Tensor: 池化后的表示，形状为 (B, D)。
        """
        # 1. 计算注意力分数
        scores = self.attn_pool(hidden_states)  # (B, num_tokens, 1)
        # 2. 应用softmax获取注意力权重
        weights = torch.softmax(scores, dim=1)  # (B, num_tokens, 1)
        # 3. 加权平均得到池化表示
        pooled = torch.sum(weights * hidden_states, dim=1)  # (B, D)
        return pooled


class ReplayBuffer:
    """一个用于多环境强化学习的回放缓冲区。"""

    def __init__(self, num_envs: int, capacity_per_env: int):
        """
        初始化回放缓冲区。

        Args:
            num_envs (int): 并行环境的数量。
            capacity_per_env (int): 每个环境要存储的最大经验数量。
        """
        self.num_envs = num_envs
        self.capacity_per_env = capacity_per_env
        # 为每个环境创建一个独立的双端队列，以隔离数据
        self.buffers = [collections.deque(maxlen=capacity_per_env) for _ in range(num_envs)]

    def add(self, env_idx: int, experience: Tuple[Dict[str, Any], torch.Tensor, torch.Tensor, bool]):
        """
        将经验添加到特定环境的缓冲区中。
        经验应该是 (inputs_t, teacher_action, teacher_projector_features, done)。
        为节省GPU内存，存入的张量应先移动到CPU。

        Args:
            env_idx (int): 环境的索引。
            experience (Tuple): 要添加的经验元组。
        """
        assert len(experience[0]['proprio'].shape) == 1
        self.buffers[env_idx].append(experience)

    def sample(self, batch_size: int) -> List[Tuple[Dict[str, Any], torch.Tensor, torch.Tensor]]:
        """
        从缓冲区中采样一批有效的状态转换。
        一个有效的转换 (s_t, s_{t+1}) 要求 s_t 不是终止状态 (done=False)。
        这用于需要下一状态信息的目标，例如预测下一状态的视觉特征。

        Args:
            batch_size (int): 要采样的转换数量。

        Returns:
            一个包含采样经验元组的列表，格式为
            (inputs_t, teacher_action_t, teacher_projector_features_t+1)。
        """
        # 1. 识别所有有效的转换起始点
        valid_transitions = []
        for env_idx, buffer in enumerate(self.buffers):
            # 只有当缓冲区长度至少为2时，才可能存在转换
            if len(buffer) < 2:
                continue
            # 遍历到倒数第二个元素，因为每个元素都需要一个 '下一个' 元素
            for i in range(len(buffer) - 1):
                # 经验元组是 (inputs_t, teacher_action, teacher_projector_features, done)
                is_terminal = buffer[i][3]
                # 如果当前状态不是终止状态，这是一个有效的转换
                if not is_terminal:
                    valid_transitions.append((env_idx, i))

        if not valid_transitions:
            return []

        # 2. 从有效转换中随机采样 (with replacement)
        sampled_indices = random.choices(valid_transitions, k=batch_size)
        
        # 3. 构建批次
        sampled_experiences = []
        for env_idx, i in sampled_indices:
            experience_t = self.buffers[env_idx][i]
            experience_t_plus_1 = self.buffers[env_idx][i+1]
            
            inputs_t = experience_t[0]
            teacher_action_t = experience_t[1]
            # 从下一个经验中获取目标 projector features
            teacher_projector_features_t_plus_1 = experience_t_plus_1[2]
            student_act_t = experience_t[4]
            
            sampled_experiences.append((inputs_t, teacher_action_t, teacher_projector_features_t_plus_1, student_act_t))
            
        return sampled_experiences

    def __len__(self) -> int:
        """
        返回缓冲区中存储的经验总数。
        """
        return sum(len(buf) for buf in self.buffers)


if __name__ == "__main__":
    import numpy as np
    import time

    # Libero env wrapper and helpers
    from rl.libero_env import LiberoEnvWrapper
    from rl.utils import prepare_one_obs, check_unnorm_key
    from experiments.robot.libero.libero_utils import GenerateConfig, TaskSuite

    # Precision policy to match the example
    USE_BF16: bool = True
    TORCH_DTYPE = torch.bfloat16 if USE_BF16 else torch.float32

    # 在这里设置要并行处理的环境数量
    ENVS_ID = [5]
    envs_num = len(ENVS_ID)
    BENCHMARK = TaskSuite.LIBERO_SPATIAL
    REPLAY_CAPACITY_PER_ENV = 1000  # 每个环境的缓冲区容量
    BATCH_SIZE = 8                  # 训练时的批次大小
    MIN_BUFFER_SIZE_FOR_TRAINING = 8 # 开始训练所需的最少样本数
    TRAINING_STEPS_PER_INTERACTION = 1 # 每次交互后执行的训练步数

    unnorm_key = f"{BENCHMARK}_no_noops"
    # Instantiate config
    cfg = GenerateConfig(
        pretrained_checkpoint="/cpfs01/lcx_workspace/models/openvla-7b-oft-finetuned-libero-spatial-object-goal-10/",
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
        lora_rank=32, # 为 LoRA 添加 rank
    )

    # Create ActorCritic policy
    actor = WorldModel(cfg, TORCH_DTYPE, torch.device("cuda:1"))
    teacher_actor = ActorCritic(cfg, TORCH_DTYPE, torch.device("cuda:1"))
    teacher_actor.eval()
    check_unnorm_key(cfg, actor.vla)
    actor.train()
    import torch.optim as optim
    optimizer = optim.AdamW(actor.get_trainable_params(), lr=1e-4)
    criterion = nn.MSELoss()
    
    # 初始化 TensorBoard writer
    log_dir = f"runs/wm/ae_act_emb_rand_act_indep_backward_{int(time.time())}"
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard 日志将保存在: {log_dir}")

    for key, value in actor.named_parameters():
        if value.dtype != TORCH_DTYPE:
            print(f"警告: 参数 {key} 的数据类型是 {value.dtype}, 但期望的是 {TORCH_DTYPE}.")
    print("策略初始化完成。")

    # --- 初始化回放缓冲区 ---
    replay_buffer = ReplayBuffer(envs_num, REPLAY_CAPACITY_PER_ENV)
    print(f"回放缓冲区已初始化，每个环境容量为 {REPLAY_CAPACITY_PER_ENV}。")

    # --- 并行初始化多个环境 ---
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

    # --- 初始化所有环境的状态 ---
    observations = []
    task_descriptions = []
    for i, env in enumerate(envs):
        obs, info = env.reset(seed=int(time.time()) + i)
        observations.append(obs)
        task_descriptions.append(env.task_description)
        print(f"环境 {i}: 任务 ID = {env.task_id}, 任务描述 = {env.task_description}")

    active_envs = [True] * envs_num
    total_rewards = [0.0] * envs_num
    episode_steps = [0] * envs_num

    total_episodes_finished = 0
    total_successes = 0
    training_step = 0

    print("\n开始并行执行所有环境并收集数据...")

    # --- 主循环：数据收集与训练 ---
    while any(active_envs):
        # 1. 从所有【活动】的环境中收集输入数据
        inputs_t_list = []
        active_indices_this_step = []
        for i in range(envs_num):
            if active_envs[i]:
                inputs_t = prepare_one_obs(cfg, actor.processor, observations[i], task_descriptions[i], TORCH_DTYPE)
                inputs_t_list.append(inputs_t)
                active_indices_this_step.append(i)

        if not inputs_t_list:
            break

        # 2. 批处理输入数据
        inputs_batch = actor.prepare_inputs_batch(inputs_t_list)

        # 3. 使用教师模型生成目标动作和目标视觉特征 (无梯度)
        with torch.no_grad():
            _, teacher_actions_b, _, _, teacher_projector_features_b = teacher_actor.forward(inputs_batch, return_vit_out=True)

        # 4. 使用学生模型生成用于与环境交互的动作 (无梯度，以加速交互)
        # with torch.no_grad():
        #     student_actions_b = actor.forward(inputs_batch)[0]
        b_s = inputs_batch['input_ids'].size(0)
        student_actions_b = torch.rand(b_s, 8, 7) * 2 - 1

        # 5. 在环境中执行动作并将经验存入回放缓冲区
        for i, env_idx in enumerate(active_indices_this_step):
            # 从批次中分离出单个数据
            inputs_t = inputs_t_list[i]
            teacher_action = teacher_actions_b[i].cpu()
            teacher_proj_feature = teacher_projector_features_b[i].cpu()
            
            # 使用学生模型的预测动作与环境交互
            behavior_action = student_actions_b[i].cpu().numpy()
            action_env = actor.vla._unnormalize_actions(behavior_action, cfg.unnorm_key)
            
            reward = 0
            terminated, truncated = False, False
            for sub_act in action_env:
                obs, t_rew, terminated, truncated, info = envs[env_idx].step(sub_act)
                reward += t_rew
                episode_steps[env_idx] += 1
                if terminated or truncated:
                    break

            done = terminated or truncated
            observations[env_idx] = obs
            total_rewards[env_idx] += float(reward)

            # 将经验(inputs, teacher_action, teacher_proj_feature, done)存入缓冲区
            experience = (inputs_t, teacher_action, teacher_proj_feature, done, student_actions_b[i].cpu())
            replay_buffer.add(env_idx, experience)

            # 检查环境是否完成
            if done:
                is_success = info.get('is_success', False)
                total_successes += is_success
                total_episodes_finished += 1
                
                print("-" * 40)
                print(f"环境 {env_idx} 已完成 (任务: {envs[env_idx].task_description[:50]}...)")
                print(f"  总步数: {episode_steps[env_idx]}, 总奖励: {total_rewards[env_idx]:.4f}, 是否成功: {is_success}")
                
                current_success_rate = total_successes / total_episodes_finished if total_episodes_finished > 0 else 0.0
                print(f"当前成功率: {current_success_rate:.2%}, 已完成回合数: {total_episodes_finished}")
                print("-" * 40)

                # 记录回合级别的统计数据
                writer.add_scalar('Episode/Reward', total_rewards[env_idx], total_episodes_finished)
                writer.add_scalar('Episode/Steps', episode_steps[env_idx], total_episodes_finished)
                writer.add_scalar('Episode/Success_Rate', current_success_rate, total_episodes_finished)
                
                # 重置环境
                episode_steps[env_idx] = 0
                total_rewards[env_idx] = 0
                obs, info = envs[env_idx].reset(seed=random.randint(0, 1000))
                observations[env_idx] = obs
        
        # ==================================================================
        #  Part 2: 从回放缓冲区采样并训练模型
        # ==================================================================
        if len(replay_buffer) > MIN_BUFFER_SIZE_FOR_TRAINING:
            for _ in range(TRAINING_STEPS_PER_INTERACTION):
                # 1. 从缓冲区采样一个批次
                sampled_experiences = replay_buffer.sample(BATCH_SIZE)
                if not sampled_experiences:
                    continue

                # 2. 整理批次数据
                inputs_list_train = [exp[0] for exp in sampled_experiences]
                teacher_actions_train = torch.stack([exp[1] for exp in sampled_experiences]).to(actor.device)
                teacher_proj_features_train = torch.stack([exp[2] for exp in sampled_experiences]).to(actor.device)
                old_student_act = torch.stack([exp[3] for exp in sampled_experiences]).to(actor.device)
                
                training_inputs_batch = actor.prepare_inputs_batch(inputs_list_train)

                # 3. 学生模型前向传播
                # predicted_actions, _, _, post_patch_embeddings, _ = actor.forward(training_inputs_batch)
                predicted_actions, _, _, _, _ = actor.forward(training_inputs_batch)
                # 模仿学习损失：使用当前时刻教师模型的动作作为目标
                imitation_loss = criterion(predicted_actions, teacher_actions_train.detach())
                optimizer.zero_grad()
                imitation_loss.backward()
                optimizer.step()

                training_inputs_batch['this_action'] = old_student_act
                post_patch_embeddings = actor.forward(training_inputs_batch)[3]
                # 自编码器损失：使用下一时刻教师模型的 projector_features 作为目标
                ae_loss = criterion(post_patch_embeddings, teacher_proj_features_train.detach())
                # 反向传播和优化
                optimizer.zero_grad()
                ae_loss.backward()
                optimizer.step()

                # 6. 使用 TensorBoard 记录指标
                total_loss = ae_loss + imitation_loss
                writer.add_scalar('Loss/Total', total_loss.item(), training_step)
                writer.add_scalar('Loss/AutoEncoder', ae_loss.item(), training_step)
                writer.add_scalar('Loss/Imitation', imitation_loss.item(), training_step)
                
                # 计算并记录相对误差
                with torch.no_grad():
                    embedding_norm = torch.norm(teacher_proj_features_train)
                    relative_error = torch.norm(post_patch_embeddings - teacher_proj_features_train) / (embedding_norm + 1e-6)
                    writer.add_scalar('Metrics/Relative_Error_Patch_Embeddings', relative_error.item(), training_step)

                if training_step % 10 == 0:
                    print(f"[Train Step {training_step}] Total Loss: {total_loss.item():.6f}, AE Loss: {ae_loss.item():.6f}, Imitation Loss: {imitation_loss.item():.6f}")
                
                training_step += 1

    # 关闭 writer
    writer.close()
    print("所有环境已完成，训练结束。")