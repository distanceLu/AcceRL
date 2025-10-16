import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, List
import numpy as np
import gc
from torch.distributions import Categorical
import torch.nn.functional as F

from peft import LoraConfig, get_peft_model
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction

# Constants
from prismatic.vla.constants import (
    NUM_ACTIONS_CHUNK,
    ACTION_DIM,
)
from typing import Any
import torch

from rl.actor_critic_model_discrete import ActorCritic
from rl.utils import load_lora_inplace, freeze_models
from rl.modules import AttentionPoolHead
# import os
from pathlib import Path
from peft import PeftModel


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
        freeze_models([self.proprio_projector])

    def forward(self, attention_mask, inputs_embeds, labels, step_count):
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
        logits = language_model_output.logits
        action_logits, actions_hidden_states = self._extract_actions_hidden(last_hidden_states, logits, labels, has_act_emb=False)

        # 2. 计算价值函数
        value = self._compute_value_from_hidden(actions_hidden_states.detach())  # (B,)

        return action_logits.float(), value.float()


class WorldModel(ActorCritic):
    """
    基于 OpenVLA 的 Actor-Critic 模型，用于连续控制。
    此版本已修改，forward 函数返回中间张量，损失计算在外部进行。
    现在将reward和termination合并为一个3分类问题：
    - Class 0: (reward=0, done=0) - 继续且无奖励
    - Class 1: (reward=0, done=1) - 终止且无奖励（失败）
    - Class 2: (reward=1, done=1) - 终止且有奖励（成功）
    """

    def __init__(self, cfg, torch_dtype: torch.dtype, checkpoint_dir: str = None, checkpoint_epoch: int = None, freeze_value=False):
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
        if hasattr(self, 'action_head'):
            del self.action_head
        del self.value_head
        del self.attn_pool
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.patch_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        ).to(self.device).to(dtype=self.model_dtype)
        self.act_proj = nn.Sequential(
            nn.Linear(ACTION_DIM * NUM_ACTIONS_CHUNK, hidden_size),
        ).to(self.device).to(dtype=self.model_dtype)
        
        # 合并的奖励-终止分类器：3个类别
        # Class 0: (r=0, done=0), Class 1: (r=0, done=1), Class 2: (r=1, done=1)
        self.reward_termination_decoder = AttentionPoolHead(hidden_size, 3).to(self.device).to(dtype=self.model_dtype)
        
        self.step_count_emb = nn.Embedding(500, hidden_size).to(self.device).to(dtype=self.model_dtype)
        
        # 使用交叉熵损失
        self.ce_loss_func = nn.CrossEntropyLoss()
        
        if checkpoint_dir:
            self.load_checkpoint(save_dir=checkpoint_dir, epoch=checkpoint_epoch)
        self.freeze_value = freeze_value
        if self.freeze_value:
            freeze_models([self.agent.value_head, self.agent.attn_pool, self.agent.step_count_emb])
        freeze_models([self.proprio_projector])
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
                             list(self.reward_termination_decoder.parameters()) + \
                             list(self.step_count_emb.parameters())
        if self.freeze_value:
            value_params = []
            print('不训练value head')
        else:
            value_params = list(self.agent.value_head.parameters()) + list(self.agent.attn_pool.parameters())
            print('训练value head')
        combined_world_params = world_model_params + value_params
        
        policy_lang = list(filter(lambda p: p.requires_grad, self.agent.language_model.parameters()))
        policy_params = policy_lang

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

    def _encode_reward_termination(self, reward: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        """
        将(reward, done)对编码为3分类标签
        
        Args:
            reward: (B,) 张量，值为0或1
            done: (B,) 布尔张量
            
        Returns:
            class_labels: (B,) 长整型张量，值为0, 1, 或2
        """
        # 验证输入的有效性
        invalid_mask = ((reward == 1) & (~done))
        if invalid_mask.any():
            invalid_indices = torch.where(invalid_mask)[0]
            raise ValueError(
                f"检测到无效的(reward, done)组合：(reward=1, done=0)！\n"
                f"无效样本索引: {invalid_indices.tolist()}\n"
                f"只允许以下三种组合：\n"
                f"  - (0, 0): 继续且无奖励 -> Class 0\n"
                f"  - (0, 1): 终止且无奖励 -> Class 1\n"
                f"  - (1, 1): 终止且有奖励 -> Class 2"
            )
        
        # 编码逻辑
        class_labels = torch.zeros_like(reward, dtype=torch.long)
        class_labels[(reward == 0) & (~done)] = 0  # (0, 0) -> Class 0
        class_labels[(reward == 0) & done] = 1      # (0, 1) -> Class 1
        class_labels[(reward == 1) & done] = 2      # (1, 1) -> Class 2
        
        return class_labels

    def _decode_reward_termination(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从分类logits解码为reward和done
        
        Args:
            logits: (B, 3) 分类logits
            
        Returns:
            reward: (B,) 预测的奖励 (0或1)
            done: (B,) 预测的终止标志 (布尔)
        """
        predicted_class = torch.argmax(logits, dim=-1)  # (B,)
        
        # 解码逻辑
        reward = torch.zeros_like(predicted_class, dtype=torch.float32)
        done = torch.zeros_like(predicted_class, dtype=torch.bool)
        
        # Class 0: (0, 0)
        mask_0 = (predicted_class == 0)
        reward[mask_0] = 0.0
        done[mask_0] = False
        
        # Class 1: (0, 1)
        mask_1 = (predicted_class == 1)
        reward[mask_1] = 0.0
        done[mask_1] = True
        
        # Class 2: (1, 1)
        mask_2 = (predicted_class == 2)
        reward[mask_2] = 1.0
        done[mask_2] = True
        
        return reward, done

    def predict_next(self, multimodal_emb, multimodal_att_mask, this_action, step_count) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """根据当前状态和动作，预测下一个隐状态、奖励和终止符。"""
        b_s = multimodal_emb.size(0)
        this_action = this_action.reshape(b_s, 1, -1).to(self.model_dtype)  # (B, 1, ACTION_DIM * NUM_ACTIONS_CHUNK)
        this_act_emb = self.act_proj(this_action)  # (B, 1, 4096)
        act_att_mask = torch.full(
                (b_s, 1),
                fill_value=True,
                dtype=multimodal_att_mask.dtype,
                device=multimodal_att_mask.device,
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
        
        # 预测合并的reward-termination分类
        rt_logits = self.reward_termination_decoder.forward(post_patch_embeddings, step_emb)  # (B, 3)
        
        # 解码为reward和done
        reward_hat, termin_hat = self._decode_reward_termination(rt_logits)
        
        # 4. 投影以获得下一个状态的嵌入
        next_embeddings = self.patch_proj(post_patch_embeddings)
        
        return next_embeddings.float(), reward_hat.float(), termin_hat
    
    def forward_vision(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast("cuda", dtype=self.model_dtype):
            self.vla: OpenVLAForActionPrediction
            multimodal_emb, multimodal_att_mask = self.agent.vla.forward_vision(  # TODO 改成vision用world model的，而embedding用agent的
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
                this_act_emb=None
            )
        return multimodal_emb, multimodal_att_mask

    def forward(self, inputs_batch: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
        """
        修改后的前向传播函数。
        返回计算 PPO、AE 和 IL 损失所需的所有张量。

        返回:
          - post_patch_proj: 经过 AE 投影层的图像嵌入，用于 AE 损失计算 (B, num_patches, D)
          - rt_logits: reward-termination分类的logits (B, 3)
        """
        for k in ("input_ids", "attention_mask", "pixel_values", "labels", "proprio", "this_action"):
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
        step_count = inputs_batch['step_count']
        step_emb = self.step_count_emb(step_count)  # (B, 16)
        
        # 2) 准备用于 AE 损失的张量
        post_patch_embeddings = recon_hidden_states[:, 2:num_patches+2]
        
        # 预测合并的reward-termination分类
        rt_logits = self.reward_termination_decoder.forward(post_patch_embeddings, step_emb)  # (B, 3)
        
        post_patch_proj = self.patch_proj(post_patch_embeddings)

        return post_patch_proj, rt_logits.float()
    
    def agent_super_forward(self, inputs_batch: Dict[str, Any], return_vit_out=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """仅使用 Agent 的前向传播来获取策略和价值。"""
        return ActorCritic.forward(self.agent, inputs_batch, return_vit_out)

    def imagine(self, start_states: Dict[str, torch.Tensor], max_horizon: int, old_logits: torch.Tensor) -> Tuple:
        """
        在学习到的世界模型中进行想象，直到所有轨迹终止或达到最大视界。
        
        Args:
            start_states: 包含初始状态信息的字典，与 'forward_vision' 的输入格式相同。
            max_horizon: 想象的最大步数。

        Returns:
            一个元组，包含想象轨迹的完整信息：
            - imagined_logits (torch.Tensor): (T, B, NUM_ACTIONS_CHUNK, VOCAB_SIZE) 离散动作logits
            - imagined_values (torch.Tensor): (T, B)
            - imagined_rewards (torch.Tensor): (T, B)
            - imagined_dones (torch.Tensor): (T, B)
            - last_value (torch.Tensor): (B,)
            - imagined_actions (torch.Tensor): (T, B, NUM_ACTIONS_CHUNK) 离散动作token
            - imagined_multimodal_embs (torch.Tensor): (T, B, SeqLen, Dim)
            - imagined_att_masks (torch.Tensor): (T, B, SeqLen)
            - imagined_step_counts (torch.Tensor): (T, B)
        """
        # 存储想象轨迹的容器
        imagined_logits, imagined_values, imagined_rewards, imagined_dones = [], [], [], []
        imagined_actions, imagined_multimodal_embs, imagined_att_masks = [], [], []
        imagined_step_counts = []
        
        B = start_states['input_ids'].size(0)
        device = self.device

        with torch.no_grad():
            # 1. 从真实状态获取初始隐状态 (embeddings)
            multimodal_emb, multimodal_att_mask = self.forward_vision(start_states)
        
        active_mask = torch.ones(B, dtype=torch.bool, device=device)

        # 2. 开始想象循环
        for step in range(max_horizon):
            if not active_mask.any():
                break  # 如果所有轨迹都已终止，提前退出

            step_count = start_states['step_count'] + step
            
            # 1. Agent predicts discrete action logits
            logits, value = self.agent.forward(multimodal_att_mask, multimodal_emb, start_states['labels'], step_count)
            
            # 2. Decode logits to get both discrete token and continuous action
            # We use stochastic actions during imagination
            _, action_token, continuous_action = self.agent.post_process(logits, deterministic=[False] * B)
            continuous_action = torch.from_numpy(continuous_action).to(device)

            # 存储当前步的信息
            imagined_multimodal_embs.append(multimodal_emb.clone())
            imagined_att_masks.append(multimodal_att_mask)
            imagined_actions.append(action_token) # Store the DISCRETE action token
            imagined_logits.append(logits)
            imagined_values.append(value)
            imagined_step_counts.append(step_count)

            with torch.no_grad():
                next_embeddings, reward_hat, termin_hat = self.predict_next(multimodal_emb, multimodal_att_mask, continuous_action, step_count)
            
            imagined_rewards.append(reward_hat)
            imagined_dones.append(termin_hat)
            
            # Update state for the next imagination step
            num_patches = self._compute_num_patches()
            multimodal_emb[:, 1:num_patches+1, :] = next_embeddings

            # 更新活跃掩码
            active_mask.logical_and_(~termin_hat)

        # 获取最后一步的价值，用于 GAE 计算
        with torch.no_grad():
            _, last_value = self.agent.forward(multimodal_att_mask, multimodal_emb, start_states['labels'], step_count + 1)

        return (torch.stack(imagined_logits), torch.stack(imagined_values), 
                torch.stack(imagined_rewards), torch.stack(imagined_dones), 
                last_value, torch.stack(imagined_actions),
                torch.stack(imagined_multimodal_embs), torch.stack(imagined_att_masks),
                torch.stack(imagined_step_counts))
    
    def compute_world_model_loss(self, wm_inp, mini_done, mini_next_teacher_proj_feat, mini_reward):
        post_patch_proj, rt_logits = self.forward(wm_inp)
        
        non_terminal_mask = ~mini_done.squeeze()
        if torch.any(non_terminal_mask):
            ae_loss = F.mse_loss(
                post_patch_proj[non_terminal_mask],
                mini_next_teacher_proj_feat[non_terminal_mask]
            )  # 自编码器损失 (仅对非终止状态)
            mae_loss = F.l1_loss(
                post_patch_proj[non_terminal_mask],
                mini_next_teacher_proj_feat[non_terminal_mask]
            ).detach()
        else:
            ae_loss = torch.tensor(0.0, device=post_patch_proj.device)
            mae_loss = torch.tensor(0.0, device=post_patch_proj.device)
            print(f"non_terminal_mask全为0: {non_terminal_mask}", flush=True)
        
        # 编码真实标签
        rt_labels = self._encode_reward_termination(mini_reward, mini_done)
        
        # 计算分类损失
        rt_loss = self.ce_loss_func(rt_logits, rt_labels)
        
        # 解码预测结果用于指标计算
        reward_predict, termin_predict = self._decode_reward_termination(rt_logits)
        
        # 计算指标
        reward_acc = (reward_predict == mini_reward).float().mean()
        termin_acc = (termin_predict == mini_done).float().mean()
        reward_mean = mini_reward.mean()
        termi_mean = mini_done.float().mean()
        
        # 计算分类准确率
        rt_acc = (torch.argmax(rt_logits, dim=-1) == rt_labels).float().mean()
        
        return ae_loss, rt_loss, reward_acc, reward_mean, termin_acc, termi_mean, rt_acc, mae_loss
    
    def save_checkpoint(self, save_dir: str, epoch: int = None):
        """
        保存 WorldModel 的所有可训练参数
        
        Args:
            save_dir: 保存目录
            epoch: 可选的 epoch 编号，用于文件命名
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 1. 保存 WorldModel 的 LoRA 权重
        world_lora_path = save_path / f"world_lora{'_epoch_' + str(epoch) if epoch else ''}"
        self.language_model.save_pretrained(world_lora_path)
        print(f"✓ WorldModel LoRA 权重已保存到: {world_lora_path}")
        
        # 2. 保存 WorldModel 的额外层
        world_extra_layers = {
            'patch_proj': self.patch_proj.state_dict(),
            'act_proj': self.act_proj.state_dict(),
            'reward_termination_decoder': self.reward_termination_decoder.state_dict(),
            'step_count_emb': self.step_count_emb.state_dict(),
        }
        world_extra_path = save_path / f"world_extra_layers{'_epoch_' + str(epoch) if epoch else ''}.pt"
        torch.save(world_extra_layers, world_extra_path)
        print(f"✓ WorldModel 额外层已保存到: {world_extra_path}")
        
        # 3. 保存 Agent 的 LoRA 权重
        agent_lora_path = save_path / f"agent_lora{'_epoch_' + str(epoch) if epoch else ''}"
        self.agent.language_model.save_pretrained(agent_lora_path)
        print(f"✓ Agent LoRA 权重已保存到: {agent_lora_path}")
        
        # 4. 保存 Agent 的额外层
        agent_extra_layers = {
            'action_head': self.agent.action_head.state_dict(),
            'value_head': self.agent.value_head.state_dict(),
            'attn_pool': self.agent.attn_pool.state_dict(),
            'lm_head': self.agent.language_model.lm_head.state_dict(),
        }
        agent_extra_path = save_path / f"agent_extra_layers{'_epoch_' + str(epoch) if epoch else ''}.pt"
        torch.save(agent_extra_layers, agent_extra_path)
        print(f"✓ Agent 额外层已保存到: {agent_extra_path}")
    
    def load_checkpoint(self, save_dir: str, epoch: int = None):
        """
        加载 WorldModel 的所有可训练参数
        
        Args:
            save_dir: 保存目录
            epoch: 可选的 epoch 编号
        """
        save_path = Path(save_dir)
        
        # 1. 加载 WorldModel 的 LoRA 权重
        world_lora_path = save_path / f"world_lora{'_epoch_' + str(epoch) if epoch else ''}"
        if world_lora_path.exists():
            assert isinstance(self.language_model, PeftModel)
            load_lora_inplace(self.language_model, world_lora_path)
            print(f"✓ WorldModel LoRA 权重已安全加载")
        else:
            print(f"⚠️  警告: 未找到 WorldModel LoRA 权重: {world_lora_path}")
        
        # 2. 加载 WorldModel 的额外层
        world_extra_path = save_path / f"world_extra_layers{'_epoch_' + str(epoch) if epoch else ''}.pt"
        if world_extra_path.exists():
            world_extra_layers = torch.load(world_extra_path, map_location=self.device)
            self.patch_proj.load_state_dict(world_extra_layers['patch_proj'])
            self.act_proj.load_state_dict(world_extra_layers['act_proj'])
            self.reward_termination_decoder.load_state_dict(world_extra_layers['reward_termination_decoder'])
            self.step_count_emb.load_state_dict(world_extra_layers['step_count_emb'])
            print(f"✓ WorldModel 额外层已从 {world_extra_path} 加载")
        else:
            print(f"⚠️  警告: 未找到 WorldModel 额外层: {world_extra_path}")
        
        # 3. 加载 Agent 的 LoRA 权重
        # agent_lora_path = save_path / f"agent_lora{'_epoch_' + str(epoch) if epoch else ''}"
        # if agent_lora_path.exists():
        #     assert isinstance(self.agent.language_model, PeftModel)
        #     load_lora_inplace(self.agent.language_model, agent_lora_path)
        #     print(f"✓ Agent LoRA 权重已安全加载")
        # else:
        #     print(f"⚠️  警告: 未找到 Agent LoRA 权重: {agent_lora_path}")
        
        # 4. 加载 Agent 的额外层
        # agent_extra_path = save_path / f"agent_extra_layers{'_epoch_' + str(epoch) if epoch else ''}.pt"
        # if agent_extra_path.exists():
        #     agent_extra_layers = torch.load(agent_extra_path, map_location=self.device)
        #     self.agent.action_head.load_state_dict(agent_extra_layers['action_head'])
        #     self.agent.value_head.load_state_dict(agent_extra_layers['value_head'])
        #     self.agent.attn_pool.load_state_dict(agent_extra_layers['attn_pool'])
        #     self.agent.language_model.lm_head.load_state_dict(agent_extra_layers['lm_head'])
        #     print(f"✓ Agent 额外层已从 {agent_extra_path} 加载")
        # else:
        #     print(f"⚠️  警告: 未找到 Agent 额外层: {agent_extra_path}")


def compute_imagined_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, last_value: torch.Tensor, gamma: float, lamb: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    为想象出的轨迹计算 GAE (Generalized Advantage Estimation)。
    输入张量的第一维是时间步 (T)。
    """
    advantages = []
    T = rewards.size(0)
    gae = 0.0
    next_val = last_value
    for t in reversed(range(T)):
        # dones[t] 是布尔值，需要转为浮点数
        delta = rewards[t] + gamma * next_val * (1.0 - dones[t].float()) - values[t]
        gae = delta + gamma * lamb * gae * (1.0 - dones[t].float())
        advantages.insert(0, gae)
        next_val = values[t]
    
    advantages = torch.stack(advantages)
    returns = advantages + values
    return advantages, returns


def compute_grpo_advantages(imagined_rewards: torch.Tensor, imagined_dones: torch.Tensor, gamma: float, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    为GRPO计算优势。它计算每个轨迹的累积回报，然后在n个轨迹组内进行归一化。
    
    Args:
        imagined_rewards (torch.Tensor): 想象出的奖励 (T, B)
        imagined_dones (torch.Tensor): 想象出的终止信号 (T, B)，应为布尔或0/1浮点数
        gamma (float): 折扣因子
        n (int): GRPO的组大小
        
    Returns:
        advantages_flat (torch.Tensor): (B,) 每条轨迹的归一化优势值
        trajectory_validity_mask (torch.Tensor): (B,) 指示哪些轨迹属于有效组（回报不完全相同）
    """
    T, B = imagined_rewards.shape
    if B % n != 0:
        raise ValueError(f"Batch size ({B}) must be a multiple of n ({n}) for GRPO.")

    # =============================================================================
    # 步骤 1: 创建一个有效性掩码，在首次 done=True 后停止计算
    # =============================================================================
    # `imagined_dones` 是 (T, B) 的张量。我们需要确保它是 long 类型以便 `cumsum`。
    # `cumsum` 会累加终止信号。当累积值 > 0 时，意味着该轨迹已经终止过。
    # 我们需要的是在第一次终止之前（包括第一次终止的当前步）的掩码。
    with torch.no_grad():
        # F.pad 在时间维度(dim=0)的开头填充一个0，使第一步的 `cumsum` 结果正确。
        # [d0, d1, d2] -> [0, d0, d1, d2] -> cumsum -> [0, d0, d0+d1, d0+d1+d2]
        # 然后去掉最后一个元素，得到 [0, d0, d0+d1]，这代表了每一步开始前的累积终止状态。
        cumulative_dones = torch.cumsum(imagined_dones.long(), dim=0)
        padded_cumulative_dones = F.pad(cumulative_dones, (0, 0, 1, 0), "constant", 0)[:-1]
        
        # `valid_step_mask` 为 True 的条件是：到目前为止，还没有发生过终止。
        # 这确保了我们只考虑首次终止之前（不包括首次终止那一步的奖励之后）的奖励。
        valid_step_mask = (padded_cumulative_dones == 0).float()

    # =============================================================================
    # 步骤 2: 计算每条轨迹的累积折扣回报
    # =============================================================================
    # 将无效步骤的奖励清零
    masked_rewards = imagined_rewards * valid_step_mask
    
    returns = torch.zeros_like(imagined_rewards[0])  # Shape: (B,)
    
    # 从后向前遍历，计算累积回报
    # 这个循环现在计算的是标准的折扣回报，因为无效奖励已被屏蔽
    for t in reversed(range(T)):
        returns = masked_rewards[t] + gamma * returns
        
    # =============================================================================
    # 步骤 3: 在组内归一化回报以获得优势
    # =============================================================================
    num_groups = B // n
    returns_grouped = returns.view(num_groups, n)  # Shape: (num_groups, n)

    mean_returns = returns_grouped.mean(dim=1, keepdim=True)
    std_returns = returns_grouped.std(dim=1, keepdim=True)

    # 创建一个掩码，用于标记有效的组（标准差 > 0，即回报不完全相同）。
    # 这里的判断逻辑是正确的，因为它比较了所有结果（包括成功、失败和超时），
    # 只要结果有差异，就存在可供学习的信号。
    valid_groups_mask = (std_returns.squeeze(-1) > 1e-8)  # Shape: (num_groups,)

    advantages_grouped = torch.zeros_like(returns_grouped)
    
    # 仅为有效的组计算优势（归一化）
    # `valid_groups_mask` 会正确地广播
    advantages_grouped[valid_groups_mask] = (
        (returns_grouped[valid_groups_mask] - mean_returns[valid_groups_mask]) / 
        (std_returns[valid_groups_mask] + 1e-8)
    )

    # =============================================================================
    # 步骤 4: 展平优势并创建最终的轨迹有效性掩码
    # =============================================================================
    advantages_flat = advantages_grouped.view(B)

    # 创建一个形状为 (B,) 的掩码，指示哪些单条轨迹属于一个有效的组。
    # 训练时，我们只使用这些有效轨迹的数据点。
    trajectory_validity_mask = valid_groups_mask.repeat_interleave(n)

    return advantages_flat, trajectory_validity_mask


def create_validity_mask(imagined_dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    根据想象的终止信号创建有效性掩码。
    一条轨迹在第一次 done=True 之后的所有步骤都是无效的。
    输入 imagined_dones: (T, B)
    返回 valid_mask_flat: (T*B,), num_valid_steps: scalar tensor
    """
    with torch.no_grad():
        cumulative_dones = torch.cumsum(imagined_dones.long(), dim=0)
        padded_cumulative_dones = F.pad(cumulative_dones, (0, 0, 1, 0))[:-1]
        valid_mask = (padded_cumulative_dones == 0)
        valid_mask_flat = valid_mask.reshape(-1)
        num_valid_steps = valid_mask_flat.sum().clamp(min=1.0)
    return valid_mask_flat, num_valid_steps


def compute_grpo_policy_loss(
    logits: torch.Tensor,
    old_logits: torch.Tensor,
    action: torch.Tensor, 
    advantage: torch.Tensor, 
    clip_eps: float, 
    ent_coef: float,
    kl_coef: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    为 GRPO 计算策略损失，使用 PPO 的裁剪目标但没有价值损失。
    适配离散动作空间（Categorical分布）。
    
    Args:
        logits: (T*B, NUM_ACTIONS_CHUNK, VOCAB_SIZE) 当前策略的logits
        old_logits: (T*B, NUM_ACTIONS_CHUNK, VOCAB_SIZE) 旧策略的logits（不计算梯度）
        action: (T*B, NUM_ACTIONS_CHUNK) 采样的离散动作token
        advantage: (B,) 每条轨迹的优势值（需要广播到时间步）
        clip_eps: PPO裁剪范围
        ent_coef: 熵损失系数
        kl_coef: KL散度损失系数
        
    Returns:
        policy_loss, value_loss, entropy_loss, kl_loss, entropy, kl_div_metric
    """
    dist = Categorical(logits=logits)
    logp = dist.log_prob(action)
    
    # 旧策略的分布和 log_prob (不计算梯度)
    with torch.no_grad():
        old_dist = Categorical(logits=old_logits)
        old_logp = old_dist.log_prob(action)

    # PPO 策略损失 (Clipped Surrogate Objective)
    ratio = torch.exp(logp - old_logp)
    # 优势已经是归一化
    adv_unsqueezed = advantage.unsqueeze(-1).unsqueeze(-1)
    surr1 = ratio * adv_unsqueezed
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_unsqueezed
    policy_loss = -torch.mean(torch.min(surr1, surr2))
    
    # 价值损失为0，因为价值网络被冻结
    value_loss = torch.tensor(0.0, device=policy_loss.device, dtype=policy_loss.dtype)
    
    # 熵损失
    entropy = torch.mean(dist.entropy())
    entropy_loss = -ent_coef * entropy
    kl_div_metric = torch.distributions.kl.kl_divergence(old_dist, dist).mean()
    kl_loss = kl_coef * kl_div_metric
    
    return policy_loss, value_loss, entropy_loss, kl_loss, entropy, kl_div_metric


def compute_ppo_loss(
    logits: torch.Tensor,
    value: torch.Tensor, 
    old_logits: torch.Tensor,
    action: torch.Tensor, 
    advantage: torch.Tensor, 
    value_target: torch.Tensor, 
    clip_eps: float, 
    vf_coef: float, 
    ent_coef: float,
    kl_coef: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算 PPO 损失。适配离散动作空间（Categorical分布）。
    
    Args:
        logits: (N, NUM_ACTIONS_CHUNK, VOCAB_SIZE) 当前策略的logits
        value: (N,) 当前价值估计
        old_logits: (N, NUM_ACTIONS_CHUNK, VOCAB_SIZE) 旧策略的logits
        action: (N, NUM_ACTIONS_CHUNK) 采样的离散动作token
        advantage: (N,) 优势值
        value_target: (N,) 目标价值（返回值）
        clip_eps: PPO裁剪范围
        vf_coef: 价值损失系数
        ent_coef: 熵损失系数
        kl_coef: KL散度损失系数
        
    Returns:
        policy_loss, value_loss, entropy_loss, kl_loss, entropy, kl_div_metric
    """
    dist = Categorical(logits=logits)
    logp = dist.log_prob(action)
    
    # 旧策略的分布和 log_prob (不计算梯度)
    with torch.no_grad():
        old_dist = Categorical(logits=old_logits)
        old_logp = old_dist.log_prob(action)

    # PPO 策略损失 (Clipped Surrogate Objective)
    ratio = torch.exp(logp - old_logp)
    # 优势已经被归一化
    adv_unsqueezed = advantage.unsqueeze(-1).unsqueeze(-1)
    surr1 = ratio * adv_unsqueezed
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_unsqueezed
    policy_loss = -torch.mean(torch.min(surr1, surr2))
    
    # 价值损失
    value_loss = F.mse_loss(value.squeeze(), value_target.squeeze())
    
    # 熵损失
    entropy = torch.mean(dist.entropy())
    entropy_loss = -ent_coef * entropy
    kl_div_metric = torch.distributions.kl.kl_divergence(old_dist, dist).mean()
    kl_loss = kl_coef * kl_div_metric
    
    return policy_loss, vf_coef * value_loss, entropy_loss, kl_loss, entropy, kl_div_metric


if __name__ == "__main__":
    import numpy as np

    # Libero env wrapper and helpers
    from rl.libero_env import LiberoEnvWrapper
    from rl.utils import prepare_one_obs, check_unnorm_key
    from experiments.robot.libero.libero_utils import GenerateConfig, TaskSuite

    # Precision policy to match the example
    USE_BF16: bool = True
    TORCH_DTYPE = torch.bfloat16 if USE_BF16 else torch.float32

    # 测试配置
    BENCHMARK = TaskSuite.LIBERO_SPATIAL
    TEST_ENV_IDS = [5, 6, 7, 8]  # 使用多个环境
    NUM_ENVS = len(TEST_ENV_IDS)
    NUM_TEST_ITERATIONS = 30  # 测试迭代次数

    unnorm_key = f"{BENCHMARK}_no_noops"
    pretrained_checkpoint = "/cpfs01/jinshiji_workspace/openvla_oft_rl/runs/openvla-7b-oft-finetuned-2_gpus_batch_size_16_100_000"
    # pretrained_checkpoint="/cpfs01/lcx_workspace/models/openvla-7b-oft-finetuned-libero-spatial-object-goal-10/"
    # Instantiate config
    cfg = GenerateConfig(
        pretrained_checkpoint=pretrained_checkpoint,
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
        lora_rank=32,
        device=torch.device("cuda:3"),
    )

    print("=" * 80)
    print("初始化 WorldModel...")
    print("=" * 80)
    checkpoint2 = "/cpfs01/lcx_workspace/models/WorldModel_ds_rew_termin_3class_1760519458/checkpoint_1000"
    # checkpoint2 = None
    
    # Create WorldModel
    world_model = WorldModel(cfg, TORCH_DTYPE, checkpoint2, freeze_value=False)

    # 模拟保存和加载模型
    # world_model.save_checkpoint('/cpfs01/lcx_workspace/models/openvla-7b-wm-test1/')
    # world_model.load_checkpoint('/cpfs01/lcx_workspace/models/WorldModel_ds_noly_wm_1760153209/checkpoint_7000/')
    
    print("\n检查参数分组...")
    param_groups = world_model.get_parameter_groups()
    for group in param_groups:
        num_params = sum(p.numel() for p in group['params'])
        print(f"  - {group['name']}: {num_params:,} 参数")
    
    check_unnorm_key(cfg, world_model.vla)
    world_model.eval()
    
    print("\n" + "=" * 80)
    print(f"初始化 {NUM_ENVS} 个测试环境...")
    print("=" * 80)
    
    # 初始化多个环境用于测试
    envs = []
    observations = []
    task_descriptions = []
    
    for i, env_id in enumerate(TEST_ENV_IDS):
        env = LiberoEnvWrapper(
            benchmark_name=BENCHMARK,
            task_id=env_id,
            image_size=224,
            render_mode="rgb_array",
        )
        obs, info = env.reset(seed=42 + i)
        task_description = env.task_description
        
        envs.append(env)
        observations.append(obs)
        task_descriptions.append(task_description)
        
        print(f"环境 {i} (任务ID {env_id}): {task_description}")
    
    # =========================================================================
    # 主测试循环
    # =========================================================================
    for iteration in range(NUM_TEST_ITERATIONS):
        print("\n" + "█" * 80)
        print(f"█  测试迭代 {iteration + 1}/{NUM_TEST_ITERATIONS}")
        print("█" * 80)
        
        # 如果不是第一次迭代，在环境中执行随机动作以改变状态
        if iteration > 0:
            for i in range(NUM_ENVS):
                random_action = np.random.uniform(-1, 1, size=7)
                obs, reward, terminated, truncated, info = envs[i].step(random_action)
                if terminated or truncated:
                    obs, info = envs[i].reset()
                    print(f"环境 {i} 已重置")
                observations[i] = obs
        
        # ---------------------------------------------------------------------
        # 测试 1: 准备输入数据
        # ---------------------------------------------------------------------
        print("\n" + "=" * 80)
        print(f"[迭代 {iteration + 1}] 测试 1: 准备输入数据 (batch_size={NUM_ENVS})")
        print("=" * 80)
        
        # 为每个环境准备输入
        inputs_list = []
        for i in range(NUM_ENVS):
            inputs_t = prepare_one_obs(cfg, world_model.processor, observations[i], task_descriptions[i], TORCH_DTYPE)
            inputs_list.append(inputs_t)
        
        print(f"✓ {NUM_ENVS} 个输入准备完成")
        print(f"  - 单个 input_ids shape: {inputs_list[0]['input_ids'].shape}")
        print(f"  - 单个 pixel_values shape: {inputs_list[0]['pixel_values'].shape}")
        print(f"  - 单个 proprio shape: {inputs_list[0]['proprio'].shape}")
        
        # 创建批次
        inputs_batch = world_model.prepare_inputs_batch(inputs_list)
        inputs_batch['step_count'] = torch.tensor([iteration] * NUM_ENVS, dtype=torch.long).to(cfg.device)
        print(f"\n✓ 批次输入准备完成 (batch_size={NUM_ENVS})")
        print(f"  - input_ids shape: {inputs_batch['input_ids'].shape}")
        print(f"  - pixel_values shape: {inputs_batch['pixel_values'].shape}")
        print(f"  - proprio shape: {inputs_batch['proprio'].shape}")
        
        # ---------------------------------------------------------------------
        # 验证前向传播路径等效性
        # ---------------------------------------------------------------------
        print("\n" + "=" * 80)
        print(f"[迭代 {iteration + 1}] 测试 5.5: 验证前向传播路径等效性")
        print("=" * 80)
        print("目标: 验证 (forward_vision + agent.forward) == agent_super_forward")
        
        print("\n" + "=" * 80)
        print(f"[迭代 {iteration + 1}] 测试 2: WorldModel.forward_vision")
        print("=" * 80)
        
        with torch.no_grad():
            multimodal_emb, multimodal_att_mask = world_model.forward_vision(inputs_batch)
        
        print(f"✓ forward_vision 成功")
        print(f"  - multimodal_emb shape: {multimodal_emb.shape}, abs: {multimodal_emb.abs().sum()}")
        print(f"  - multimodal_att_mask shape: {multimodal_att_mask.shape}, abs: {multimodal_att_mask.abs().sum()}")
        print(f"  - multimodal_emb 范围: [{multimodal_emb.min():.4f}, {multimodal_emb.max():.4f}]")
        with torch.no_grad():
            # 使用 Test 2 中计算的 embeddings
            # `agent.forward` 返回 (logits, value) - 离散动作
            logits_b, value_b = world_model.agent.forward(
                multimodal_att_mask, 
                multimodal_emb, 
                inputs_batch['labels'], 
                inputs_batch['step_count']
            )
        print(f"  - logits_b shape: {logits_b.shape}, abs: {logits_b.abs().sum()}")
        print(f"  - value_b shape: {value_b.shape}, abs: {value_b.abs().sum()}")
        print(f"  - value 值: {value_b}")
        print(f"  - logits_b 范围: [{logits_b.min():.4f}, {logits_b.max():.4f}]")
        # ---------------------------------------------------------------------
        # WorldModel.agent_super_forward
        # ---------------------------------------------------------------------
        print("\n" + "=" * 80)
        print(f"[迭代 {iteration + 1}] 测试 5: WorldModel.agent_super_forward")
        print("=" * 80)
        
        with torch.no_grad():
            logits_agent, value_agent = world_model.agent_super_forward(inputs_batch)
        
        print(f"✓ agent_super_forward 成功")
        print(f"  - logits shape: {logits_agent.shape}, abs: {logits_agent.abs().sum()}")
        print(f"  - value shape: {value_agent.shape}, abs: {value_agent.abs().sum()}")
        print(f"  - value 值: {value_agent}")
        print(f"  - logits 范围: [{logits_agent.min():.4f}, {logits_agent.max():.4f}]")
        
        # 比较结果
        print("\n  比较两条路径的输出:")
        atol = 1e-4 # 设置一个合理的容忍度以处理浮点精度差异
        
        # 比较 logits
        logits_match = torch.allclose(logits_agent, logits_b, atol=atol)
        print(f"    - logits (动作logits) 是否匹配: {'✓ 是' if logits_match else '❌ 否'}")
        diff = torch.abs(logits_agent - logits_b).max()
        print(f"      最大差异: {diff.item()}")
            
        # 比较 value
        value_match = torch.allclose(value_agent, value_b, atol=atol)
        print(f"    - value (状态价值) 是否匹配: {'✓ 是' if value_match else '❌ 否'}")
        diff = torch.abs(value_agent - value_b).max()
        print(f"      最大差异: {diff.item()}")

        if logits_match and value_match:
            print("\n✓ 验证成功: 两条前向传播路径的计算结果等效！")
        else:
            print("\n❌ 验证失败: 两条前向传播路径的计算结果不一致！")

        # ---------------------------------------------------------------------
        # 测试 3: WorldModel.forward (完整前向传播)
        # ---------------------------------------------------------------------
        print("\n" + "=" * 80)
        print(f"[迭代 {iteration + 1}] 测试 3: WorldModel.forward (完整前向传播)")
        print("=" * 80)
        
        inputs_batch['this_action'] = torch.randn(NUM_ENVS, NUM_ACTIONS_CHUNK, ACTION_DIM).to(cfg.device)
        
        with torch.no_grad():
            post_patch_proj, rt_logits = world_model.forward(inputs_batch)
        
        print(f"✓ WorldModel.forward 成功")
        print(f"  - post_patch_proj shape: {post_patch_proj.shape}")
        print(f"  - rt_logits shape: {rt_logits.shape}")
        
        decoded_reward, decoded_termin = world_model._decode_reward_termination(rt_logits)
        print(f"  - 预测奖励: {decoded_reward.cpu().numpy()}")
        print(f"  - 预测终止: {decoded_termin.cpu().numpy()}")
        
        # ---------------------------------------------------------------------
        # 测试 4: WorldModel.predict_next
        # ---------------------------------------------------------------------
        print("\n" + "=" * 80)
        print(f"[迭代 {iteration + 1}] 测试 4: WorldModel.predict_next")
        print("=" * 80)
        
        test_action = inputs_batch['this_action']
        step_count = inputs_batch['step_count']
        
        with torch.no_grad():
            next_emb, next_reward, next_termin = world_model.predict_next(
                multimodal_emb, multimodal_att_mask, test_action, step_count
            )
        
        print(f"✓ predict_next 成功")
        print(f"  - next_embeddings shape: {next_emb.shape}")
        print(f"  - next_reward shape: {next_reward.shape}")
        print(f"  - next_termin shape: {next_termin.shape}")
        
        print(f"  - 预测下一步奖励: {next_reward.cpu().numpy()}")
        print(f"  - 预测下一步终止: {next_termin.cpu().numpy()}")

        # ---------------------------------------------------------------------
        # 测试 6: WorldModel.imagine
        # ---------------------------------------------------------------------
        print("\n" + "=" * 80)
        print(f"[迭代 {iteration + 1}] 测试 6: WorldModel.imagine (批量，batch_size={NUM_ENVS})")
        print("=" * 80)
        
        # 准备 inputs_batch 用于想象（需要确保有 step_count）
        imagine_inputs = inputs_batch.copy()
        imagine_inputs.pop('this_act_emb')
        if 'step_count' not in imagine_inputs:
            imagine_inputs['step_count'] = torch.tensor([0] * NUM_ENVS, dtype=torch.long).to(cfg.device)
        
        imagine_steps = 5
        print(f"对 {NUM_ENVS} 个环境批量想象 {imagine_steps} 步...")
        
        with torch.no_grad():
            (imagined_logits, imagined_values, imagined_rewards, 
             imagined_dones, last_value, imagined_actions, imagined_multimodal_embs, 
             imagined_att_masks, imagined_step_counts) = world_model.imagine(imagine_inputs, imagine_steps, None)
        
        print(f"✓ imagine 成功")
        print(f"  - imagined_logits shape: {imagined_logits.shape}")
        print(f"  - imagined_values shape: {imagined_values.shape}")
        print(f"  - imagined_rewards shape: {imagined_rewards.shape}")
        print(f"  - imagined_dones shape: {imagined_dones.shape}")
        print(f"  - imagined_actions shape: {imagined_actions.shape}")
        print(f"  - last_value shape: {last_value.shape}")
        
        # 打印每个环境的想象轨迹详情
        print(f"\n  各环境想象轨迹详情:")
        for env_idx in range(NUM_ENVS):
            print(f"    环境 {env_idx}:")
            env_rewards = imagined_rewards[:, env_idx]
            env_values = imagined_values[:, env_idx]
            env_dones = imagined_dones[:, env_idx]
            
            print(f"      - 累计奖励: {env_rewards.sum().item():.4f}")
            print(f"      - 平均价值: {env_values.mean().item():.4f}")
            print(f"      - 最终价值: {last_value[env_idx].item():.4f}")
            print(f"      - 终止次数: {env_dones.sum().item()}")
            
            # 打印每一步的详情
            for step in range(len(env_rewards)):
                print(f"        Step {step}: reward={env_rewards[step].item():.4f}, "
                      f"value={env_values[step].item():.4f}, done={env_dones[step].item()}")
        
        # ---------------------------------------------------------------------
        # 测试 7: 检查数值范围
        # ---------------------------------------------------------------------
        print("\n" + "=" * 80)
        print(f"[迭代 {iteration + 1}] 测试 7: 检查数值范围")
        print("=" * 80)
        
        print(f"Predicted reward 范围: [{decoded_reward.min():.4f}, {decoded_reward.max():.4f}]")
        print(f"Value 范围: [{value_agent.min():.4f}, {value_agent.max():.4f}]")
        print(f"Logits 范围: [{logits_agent.min():.4f}, {logits_agent.max():.4f}]")
        print(f"Post patch proj 范围: [{post_patch_proj.min():.4f}, {post_patch_proj.max():.4f}]")
        print(f"Imagined rewards 范围: [{imagined_rewards.min():.4f}, {imagined_rewards.max():.4f}]")
        print(f"Imagined values 范围: [{imagined_values.min():.4f}, {imagined_values.max():.4f}]")
        
        # 打印每个环境的具体数值
        print(f"\n  各环境详细数值:")
        for env_idx in range(NUM_ENVS):
            print(f"    环境 {env_idx}:")
            print(f"      - 当前奖励: {decoded_reward[env_idx].item():.4f}")
            print(f"      - 当前价值: {value_agent[env_idx].item():.4f}")
            print(f"      - 想象累计奖励: {imagined_rewards[:, env_idx].sum().item():.4f}")
        
        # 检查是否有NaN或Inf
        has_nan = any([
            torch.isnan(post_patch_proj).any(),
            torch.isnan(rt_logits).any(),
            torch.isnan(logits_agent).any(),
            torch.isnan(value_agent).any(),
            torch.isnan(imagined_rewards).any(),
            torch.isnan(imagined_values).any(),
        ])
        has_inf = any([
            torch.isinf(post_patch_proj).any(),
            torch.isinf(rt_logits).any(),
            torch.isinf(logits_agent).any(),
            torch.isinf(value_agent).any(),
            torch.isinf(imagined_rewards).any(),
            torch.isinf(imagined_values).any(),
        ])
        
        if has_nan:
            print("\n⚠️  警告: 检测到 NaN 值!")
        if has_inf:
            print("\n⚠️  警告: 检测到 Inf 值!")
        if not has_nan and not has_inf:
            print("\n✓ 所有数值正常 (无 NaN 或 Inf)")
    
    print("\n" + "█" * 80)
    print("█  所有测试迭代完成!")
    print("█" * 80)