import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, List
import numpy as np
import gc
from torch.distributions import Normal
import collections
import random

from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from storm.functions_losses import SymLogTwoHotLoss
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction

# Constants
from prismatic.vla.constants import (
    NUM_ACTIONS_CHUNK,
    ACTION_DIM,
)
from typing import Any
import torch

# 显式类：避免依赖 auto_map
from rl.actor_critic_model import ActorCritic
from rl.modules import AttentionPool, AttentionPoolHead


class Agent(ActorCritic):
    def __init__(self, cfg, torch_dtype: torch.dtype):
        cfg.use_lora = False
        super().__init__(cfg, torch_dtype)
        cfg.use_lora = True  # 恢复 cfg 中的 use_lora 标志
        self.vla: OpenVLAForActionPrediction
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=0,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        self.language_model = get_peft_model(self.vla.language_model, lora_config)
        self.language_model.print_trainable_parameters()
        self.language_model: LlamaForCausalLM
        for param in self.proprio_projector.parameters():
            param.requires_grad = False

    def forward(self, attention_mask, inputs_embeds, labels):
        language_model_output = self.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            labels=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden_states = language_model_output.hidden_states[-1]
        # 2) Predict continuous actions mean (mu) using action-related hidden states
        actions_hidden_states = self._extract_actions_hidden(last_hidden_states, labels, False)
        predicted_actions = self.action_head.predict_action(actions_hidden_states)  # (B, NUM_ACTIONS_CHUNK, ACTION_DIM) or flat
        mu_all = predicted_actions

        # 3) Condition-independent log_std broadcast across chunks
        B = mu_all.size(0)
        log_std = self.log_std_param  # (NUM_ACTIONS_CHUNK, ACTION_DIM)
        log_std_all = log_std.unsqueeze(dim=0).expand(B, NUM_ACTIONS_CHUNK, ACTION_DIM)  # (B, T, A)

        value = self._compute_value_from_hidden(actions_hidden_states.detach())   # (B,)
        return mu_all.to(torch.float32), log_std_all.to(torch.float32), value.to(torch.float32)


class WorldModel(ActorCritic):
    """
    基于 OpenVLA 的 Actor-Critic 模型，用于连续控制。
    此版本已修改，forward 函数返回中间张量，损失计算在外部进行。
    """

    def __init__(self, cfg, torch_dtype: torch.dtype):
        cfg.use_lora = False
        super().__init__(cfg, torch_dtype)
        cfg.use_lora = True  # 恢复 cfg 中的 use_lora 标志
        self.agent = Agent(cfg, torch_dtype)
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
        self.language_model: LlamaForCausalLM
        del self.action_head
        del self.value_head
        del self.attn_pool
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for param in self.proprio_projector.parameters():
            param.requires_grad = False
        self.patch_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        ).to(self.device).to(dtype=self.model_dtype)
        self.act_proj = nn.Sequential(
            nn.Linear(ACTION_DIM * NUM_ACTIONS_CHUNK, hidden_size),
        ).to(self.device).to(dtype=self.model_dtype)
        # 注意力池化层
        rew_num_classes = 255
        self.reward_decoder = AttentionPoolHead(hidden_size, rew_num_classes).to(self.device).to(dtype=self.model_dtype)
        self.termi_decoder = AttentionPoolHead(hidden_size, 1)
        self.step_count_emb = nn.Embedding(500, hidden_size).to(self.device).to(dtype=self.model_dtype)
        self.symlog_twohot_loss_func = SymLogTwoHotLoss(num_classes=rew_num_classes, lower_bound=-20, upper_bound=20)
        self.bce_with_logits_loss_func = nn.BCEWithLogitsLoss()
        self.to(self.device, dtype=self.model_dtype)

    def get_parameter_groups(self) -> List[Dict[str, Any]]:
        """
        为优化器提供参数分组，以应用不同的学习率。
        此版本增强了检查功能，可以打印出任何未被分组的可训练参数的具体名称，以便于调试。
        """
        # 世界模型/语言模型部分：可训练的语言模型层和新的投影层
        lan_params = list(filter(lambda p: p.requires_grad, self.language_model.parameters()))
        world_model_params = lan_params + \
                             list(self.patch_proj.parameters()) + \
                             list(self.act_proj.parameters()) + \
                             list(self.reward_decoder.parameters()) + \
                             list(self.termi_decoder.parameters()) + \
                             list(self.step_count_emb.parameters())
        value_params = list(self.agent.value_head.parameters()) + list(self.agent.attn_pool.parameters())
        combined_world_params = world_model_params
        
        policy_lang = list(filter(lambda p: p.requires_grad, self.agent.language_model.parameters()))
        action_params = list(self.agent.action_head.parameters())
        policy_params = policy_lang + action_params + value_params

        # 1. 获取模型中所有实际为可训练状态的参数，作为“真实情况”的集合
        all_trainable_params = set(filter(lambda p: p.requires_grad, self.parameters()))
        
        # 2. 获取所有被手动分组到 'policy' 或 'value' 组的参数，作为“分组情况”的集合
        grouped_params_set = set(combined_world_params) | set(policy_params)
        
        # 3. 比较两个集合，如果不相等，则启动详细的诊断流程
        if all_trainable_params != grouped_params_set:
            
            # 为了通过参数对象找到其名称，我们创建一个从参数到其名称的反向映射
            param_to_name_map = {p: name for name, p in self.named_parameters()}
            
            # 使用集合的差集运算找出被遗漏的参数
            missed_params = all_trainable_params.difference(grouped_params_set)
            
            # 打印一个清晰的、引人注目的错误报告
            print("\n" + "="*70)
            print("【严重错误】: 参数分组不完整！模型中存在未被分组的可训练参数。")
            print("这意味着这些参数将不会被优化器更新。")
            
            if missed_params:
                print("\n以下参数是可训练的 (requires_grad=True)，但【未被分配】到任何优化器组：")
                for param in missed_params:
                    # 从映射中查找参数名，如果找不到则提供一个默认提示
                    name = param_to_name_map.get(param, "未知名称 (可能在未命名的子模块中)")
                    print(f"  --> 名称: {name}")
                    print(f"      形状: {param.shape}, 元素数量: {param.numel()}")
            else:
                unnecessary_params = grouped_params_set.difference(all_trainable_params)
                print("\n所有可训练参数均已分组，但以下参数【不应】被分组，因为它们不可训练 (requires_grad=False)：")
                for param in unnecessary_params:
                    name = param_to_name_map.get(param, "未知名称 (可能在未命名的子模块中)")
                    print(f"  --> 名称: {name}")
                    print(f"      形状: {param.shape}, 元素数量: {param.numel()}")
                raise RuntimeError("分组中包含不可训练的参数！请检查代码逻辑。")
            
            print("="*70 + "\n")
            
            # 抛出异常以中断执行，强制开发者修复此问题
            raise AssertionError(
                "参数分组不完整。请检查上面的日志，并将列出的 '未被分配' 的参数"
                "添加到 get_parameter_groups 函数的相应分组中。"
            )

        # 如果检查通过，打印成功的消息
        trainable_params_count = sum(p.numel() for p in all_trainable_params)
        print(f"WorldModel 中所有可训练参数已成功分组。总量: {trainable_params_count:,}")
        return [{"name": "world", "params": combined_world_params}, {"name": "policy", "params": policy_params}]

    def predict_next(self, multimodal_emb, multimodal_att_mask, this_action, step_count) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """根据当前状态和动作，预测下一个隐状态、奖励和终止符。"""
        b_s = multimodal_emb.size(0)
        this_action = this_action.reshape(b_s, 1, -1).to(self.model_dtype)  # (B, 1, ACTION_DIM * NUM_ACTIONS_CHUNK)
        this_act_emb = self.act_proj(this_action)  # (B, 1, 4096)
        act_att_mask = torch.full(
                (b_s, 1),
                fill_value=True,
                dtype=multimodal_emb.dtype,
                device=multimodal_emb.device,
            )
        multimodal_emb = torch.cat([multimodal_emb[:, :1, :], this_act_emb, multimodal_emb[:, 1:, :]], dim=1)
        multimodal_att_mask = torch.cat([multimodal_att_mask[:, :1], act_att_mask, multimodal_att_mask[:, 1:]], dim=1)
        output = self.language_model(
            input_ids=None,
            attention_mask=multimodal_att_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_emb,
            labels=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        recon_hidden_states = output.hidden_states[-1]
        num_patches = self._compute_num_patches()
        
        # 3. 提取和解码
        # 当 'this_act_emb' 存在时，图像嵌入在第2个位置之后
        post_patch_embeddings = recon_hidden_states[:, 2:num_patches+2]
        
        step_emb = self.step_count_emb(step_count)
        
        reward_logits = self.reward_decoder.forward(post_patch_embeddings, step_emb)
        termin_hat = self.termi_decoder.forward(post_patch_embeddings, step_emb).squeeze(-1)
        
        # 4. 投影以获得下一个状态的嵌入
        next_embeddings = self.patch_proj(post_patch_embeddings)
        
        return next_embeddings.float(), reward_logits.float(), termin_hat.float()
    
    def forward_vision(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast("cuda", dtype=self.model_dtype):
            self.vla: OpenVLAForActionPrediction
            multimodal_emb, multimodal_att_mask = self.vla.forward_vision(
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
        return multimodal_emb, multimodal_att_mask

    def forward(self, inputs_batch: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
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
        
        b_s = inputs_batch['this_action'].size(0)
        this_action = inputs_batch['this_action'].reshape(b_s, -1).to(self.model_dtype)  # (B, ACTION_DIM * NUM_ACTIONS_CHUNK)
        this_act_emb = self.act_proj(this_action)  # (B, 4096)
        inputs_batch['this_act_emb'] = this_act_emb.unsqueeze(dim=1)  # (B, 1, 4096)

        # 1) VLA 前向传播以获取隐藏状态
        output = self._forward_vla(inputs_batch)
        recon_hidden_states = output.hidden_states[-1]  # len(output.hidden_states): 33
        num_patches = self._compute_num_patches()
        if 'step_count' in inputs_batch:
            step_count = inputs_batch['step_count']
            step_emb = self.step_count_emb(step_count)  # (B, 16)
        
        # 2) 准备用于 AE 损失的张量
        post_patch_embeddings = recon_hidden_states[:, 2:num_patches+2]
        reward_logits = self.reward_decoder.forward(post_patch_embeddings, step_emb)  # (B, 255)
        termin_hat = self.termi_decoder.forward(post_patch_embeddings, step_emb).squeeze(-1)
        post_patch_proj = self.patch_proj(post_patch_embeddings)

        res = [
            post_patch_proj, 
            reward_logits,
            termin_hat
            ]
        res = tuple(tmp if tmp is None else tmp.float() for tmp in res)
        return res
    
    def agent_super_forward(self, inputs_batch: Dict[str, Any], return_vit_out=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """仅使用 Agent 的前向传播来获取策略和价值。"""
        return ActorCritic.forward(self.agent, inputs_batch, return_vit_out)

    def imagine(self, mini_inputs: Dict[str, torch.Tensor], imagine_step) -> Tuple:
        """
        在学习到的世界模型中进行想象。
        从 mini_inputs 中的真实状态开始，向前滚动 IMAGINE_STEP 步。
        返回轨迹和策略分布参数。
        """
        # 存储想象轨迹的容器
        imagined_logps = []
        imagined_values = []
        imagined_rewards = []
        imagined_dones = []
        imagined_mus = []
        imagined_log_stds = []
        num_patches = self._compute_num_patches()

        with torch.no_grad():
            # 1. 从真实状态 mini_inputs 获取初始隐状态 (embeddings)
            multimodal_emb, multimodal_att_mask = self.forward_vision(mini_inputs)
        
        # 2. 开始想象循环
        for step in range(imagine_step):
            step_count = mini_inputs['step_count'] + step
            mu, log_std, value = self.agent.forward(multimodal_att_mask, multimodal_emb, mini_inputs['labels'])
            
            dist = Normal(mu, torch.exp(log_std))
            action = dist.sample()
            log_p = dist.log_prob(action)

            imagined_logps.append(log_p)
            imagined_values.append(value)
            imagined_mus.append(mu)
            imagined_log_stds.append(log_std)

            with torch.no_grad():
                next_embeddings, reward_hat, termi_hat = self.predict_next(multimodal_emb, multimodal_att_mask, action, step_count)
            
            predicted_reward = self.symlog_twohot_loss_func.decode(reward_hat)
            predicted_done = (termi_hat > 0).squeeze()

            imagined_rewards.append(predicted_reward)
            imagined_dones.append(predicted_done)
            
            multimodal_emb[:, 1:num_patches+1, :] = next_embeddings

        step_count = mini_inputs['step_count'] + imagine_step
        with torch.no_grad():
            _, _, last_value = self.agent.forward(multimodal_att_mask, multimodal_emb, mini_inputs['labels'])

        return (torch.stack(imagined_logps), torch.stack(imagined_values), 
                torch.stack(imagined_rewards), torch.stack(imagined_dones), 
                torch.stack(imagined_mus), torch.stack(imagined_log_stds),
                last_value)


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
        device=torch.device("cuda:1"),
    )

    # Create ActorCritic policy
    actor = WorldModel(cfg, TORCH_DTYPE)
    actor.get_parameter_groups()
    teacher_actor = ActorCritic(cfg, TORCH_DTYPE)
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