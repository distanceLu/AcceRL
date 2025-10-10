import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, List
import numpy as np
import gc
from torch.distributions import Normal
import collections
import random
import torch.nn.functional as F

from peft import LoraConfig, get_peft_model
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
from rl.modules import AttentionPoolHead
# import os
from pathlib import Path


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
        combined_world_params = world_model_params + value_params
        
        policy_lang = list(filter(lambda p: p.requires_grad, self.agent.language_model.parameters()))
        action_params = list(self.agent.action_head.parameters())
        policy_params = policy_lang + action_params

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
                last_value)
    
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
            'reward_decoder': self.reward_decoder.state_dict(),
            'termi_decoder': self.termi_decoder.state_dict(),
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
            'log_std_param': self.agent.log_std_param,
        }
        agent_extra_path = save_path / f"agent_extra_layers{'_epoch_' + str(epoch) if epoch else ''}.pt"
        torch.save(agent_extra_layers, agent_extra_path)
        print(f"✓ Agent 额外层已保存到: {agent_extra_path}")
        
        # 5. 保存训练配置（可选但推荐）
        config_dict = {
            'lora_rank': self.cfg.lora_rank,
            'use_proprio': self.cfg.use_proprio,
            'use_film': self.cfg.use_film,
            # 添加其他重要配置
        }
        config_path = save_path / "training_config.pt"
        torch.save(config_dict, config_path)
        print(f"✓ 训练配置已保存到: {config_path}")
        
        print(f"\n{'='*80}")
        print(f"所有检查点已成功保存到: {save_path}")
        print(f"{'='*80}\n")
    
    def load_checkpoint(self, save_dir: str, epoch: int = None):
        """
        加载 WorldModel 的所有可训练参数
        
        Args:
            save_dir: 保存目录
            epoch: 可选的 epoch 编号
        """
        save_path = Path(save_dir)
        device = self.device # 使用模型自身的设备

        # --- 1. 加载 WorldModel 的 LoRA 权重 ---
        world_lora_path = save_path / f"world_lora{'_epoch_' + str(epoch) if epoch else ''}"
        world_adapter_weights_path = world_lora_path / "adapter_model.bin"

        if world_adapter_weights_path.exists():
            # 安全加载：加载权重到现有模型，而不是替换模型对象
            adapter_weights = torch.load(world_adapter_weights_path, map_location=device)
            self.language_model.load_state_dict(adapter_weights, strict=False)
            print(f"✓ WorldModel LoRA 权重已从 {world_lora_path} 就地加载")
        else:
            print(f"⚠️  警告: 未找到 WorldModel LoRA 权重文件: {world_adapter_weights_path}")

        # --- 2. 加载 WorldModel 的额外层 ---
        world_extra_path = save_path / f"world_extra_layers{'_epoch_' + str(epoch) if epoch else ''}.pt"
        if world_extra_path.exists():
            world_extra_layers = torch.load(world_extra_path, map_location=device)
            self.patch_proj.load_state_dict(world_extra_layers['patch_proj'])
            self.act_proj.load_state_dict(world_extra_layers['act_proj'])
            self.reward_decoder.load_state_dict(world_extra_layers['reward_decoder'])
            self.termi_decoder.load_state_dict(world_extra_layers['termi_decoder'])
            self.step_count_emb.load_state_dict(world_extra_layers['step_count_emb'])
            print(f"✓ WorldModel 额外层已从 {world_extra_path} 加载")
        else:
            print(f"⚠️  警告: 未找到 WorldModel 额外层: {world_extra_path}")

        # --- 3. 加载 Agent 的 LoRA 权重 ---
        agent_lora_path = save_path / f"agent_lora{'_epoch_' + str(epoch) if epoch else ''}"
        agent_adapter_weights_path = agent_lora_path / "adapter_model.bin"

        if agent_adapter_weights_path.exists():
            # 安全加载：加载权重到现有模型
            adapter_weights = torch.load(agent_adapter_weights_path, map_location=device)
            self.agent.language_model.load_state_dict(adapter_weights, strict=False)
            print(f"✓ Agent LoRA 权重已从 {agent_lora_path} 就地加载")
        else:
            print(f"⚠️  警告: 未找到 Agent LoRA 权重文件: {agent_adapter_weights_path}")

        # --- 4. 加载 Agent 的额外层 ---
        agent_extra_path = save_path / f"agent_extra_layers{'_epoch_' + str(epoch) if epoch else ''}.pt"
        if agent_extra_path.exists():
            agent_extra_layers = torch.load(agent_extra_path, map_location=device)
            self.agent.action_head.load_state_dict(agent_extra_layers['action_head'])
            self.agent.value_head.load_state_dict(agent_extra_layers['value_head'])
            self.agent.attn_pool.load_state_dict(agent_extra_layers['attn_pool'])
            # log_std_param 是一个独立的张量，不是 state_dict 的一部分，直接赋值
            self.agent.log_std_param.data.copy_(agent_extra_layers['log_std_param'])
            print(f"✓ Agent 额外层已从 {agent_extra_path} 加载")
        else:
            print(f"⚠️  警告: 未找到 Agent 额外层: {agent_extra_path}")

        print(f"\n{'='*80}")
        print(f"所有检查点已成功从 {save_path} 加载完毕。")
        print(f"{'='*80}\n")


def compute_imagined_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, last_value: torch.Tensor, imagine_step, gamma, lamb) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    为想象出的轨迹计算 GAE (Generalized Advantage Estimation)。
    输入张量的第一维是时间步 (IMAGINE_STEP)。
    由于批次中的序列可能在不同时间点终止，因此需要小心处理。
    """
    advantages = []
    returns = []
    gae = 0.0
    next_val = last_value
    for t in reversed(range(imagine_step)):
        delta = rewards[t] + gamma * next_val * (1.0 - dones[t].float()) - values[t]
        gae = delta + gamma * lamb * gae * (1.0 - dones[t].float())
        advantages.insert(0, gae)
        returns.insert(0, gae + values[t])
        next_val = values[t]
    return torch.stack(advantages), torch.stack(returns)


def create_validity_mask(imagined_dones: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """根据想象的终止信号创建有效性掩码。"""
    with torch.no_grad():
        cumulative_dones = torch.cumsum(imagined_dones.long(), dim=0)
        padded_cumulative_dones = F.pad(cumulative_dones, (0, 0, 1, 0))[:-1]
        valid_mask = (padded_cumulative_dones == 0)
        valid_mask_flat = valid_mask.reshape(-1)
        num_valid_steps = valid_mask_flat.sum().clamp(min=1.0)
    return valid_mask_flat, num_valid_steps


def compute_imagine_loss(mini_inputs: Dict[str, torch.Tensor], model, imagine_step, gamma, lamb) -> Tuple[torch.Tensor, torch.Tensor]:
    # --- 3. 想象数据 RL 损失 ---
    mini_sub = random_pick_and_repeat(mini_inputs, 4)  # 类似于GRPO，一个样本采样若干条轨迹
    # mini_sub = mini_inputs
    (imagined_logps, imagined_values, imagined_rewards, imagined_dones, last_value) = model.imagine(mini_sub, imagine_step)
    valid_mask_flat, num_valid_steps = create_validity_mask(imagined_dones)
    imagined_advs, imagined_rets = compute_imagined_gae(imagined_rewards, imagined_values, imagined_dones, last_value, imagine_step, gamma, lamb)
    
    imagined_advs_flat, imagined_rets_flat, imagined_values_flat = \
        imagined_advs.reshape(-1), imagined_rets.reshape(-1), imagined_values.reshape(-1)
    last_two_dim = imagined_logps.shape[-1] * imagined_logps.shape[-2]
    imagined_logps_flat = imagined_logps.reshape(-1, last_two_dim)
    
    with torch.no_grad():
        valid_advs = torch.masked_select(imagined_advs_flat, valid_mask_flat)
        adv_mean, adv_std = valid_advs.mean(), valid_advs.std() + 1e-8
        normalized_imagined_advs = (imagined_advs_flat - adv_mean) / adv_std
    
    policy_loss_terms = -(normalized_imagined_advs.unsqueeze(dim=-1).detach() * imagined_logps_flat)
    imagination_policy_loss = (policy_loss_terms * valid_mask_flat.unsqueeze(dim=-1)).sum() / num_valid_steps
    value_loss_terms = F.mse_loss(imagined_values_flat, imagined_rets_flat.detach(), reduction='none')
    imagination_value_loss = (value_loss_terms * valid_mask_flat).sum() / num_valid_steps
    return imagination_policy_loss, imagination_value_loss


def random_pick_and_repeat(inputs_batch: dict, repeats: int = 4):
    """从每个张量的第0维随机采样一个元素，并在该维度上重复指定次数。"""
    outputs = {}
    for key, tensor in inputs_batch.items():
        if not torch.is_tensor(tensor):
            raise TypeError(f"键 '{key}' 对应的值不是张量：{type(tensor)}")
        if tensor.size(0) == 0:
            raise ValueError(f"键 '{key}' 对应的张量在 dim=0 上为空，无法采样。")
        idx = torch.randint(0, tensor.size(0), (), device=tensor.device)
        sample = tensor[idx:idx+1]             # shape: (1, ...)
        repeat_shape = (repeats,) + (1,) * (tensor.dim() - 1)
        outputs[key] = sample.repeat(repeat_shape)
    return outputs


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

    # 测试配置
    BENCHMARK = TaskSuite.LIBERO_SPATIAL
    TEST_ENV_IDS = [5, 6, 7, 8]  # 使用多个环境
    NUM_ENVS = len(TEST_ENV_IDS)
    NUM_TEST_ITERATIONS = 30  # 测试迭代次数

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
        lora_rank=32,
        device=torch.device("cuda:2"),
    )

    print("=" * 80)
    print("初始化 WorldModel...")
    print("=" * 80)
    
    # Create WorldModel
    world_model = WorldModel(cfg, TORCH_DTYPE)

    # 模拟保存和加载模型
    world_model.save_checkpoint('/cpfs01/lcx_workspace/models/openvla-7b-wm-test1/')
    world_model.load_checkpoint('/cpfs01/lcx_workspace/models/openvla-7b-wm-test1/')
    
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
        print(f"\n✓ 批次输入准备完成 (batch_size={NUM_ENVS})")
        print(f"  - input_ids shape: {inputs_batch['input_ids'].shape}")
        print(f"  - pixel_values shape: {inputs_batch['pixel_values'].shape}")
        print(f"  - proprio shape: {inputs_batch['proprio'].shape}")
        
        # ---------------------------------------------------------------------
        # 测试 2: WorldModel.forward_vision
        # ---------------------------------------------------------------------
        print("\n" + "=" * 80)
        print(f"[迭代 {iteration + 1}] 测试 2: WorldModel.forward_vision")
        print("=" * 80)
        
        with torch.no_grad():
            multimodal_emb, multimodal_att_mask = world_model.forward_vision(inputs_batch)
        
        print(f"✓ forward_vision 成功")
        print(f"  - multimodal_emb shape: {multimodal_emb.shape}")
        print(f"  - multimodal_att_mask shape: {multimodal_att_mask.shape}")
        print(f"  - multimodal_emb 范围: [{multimodal_emb.min():.4f}, {multimodal_emb.max():.4f}]")
        
        # ---------------------------------------------------------------------
        # 测试 3: WorldModel.forward (完整前向传播)
        # ---------------------------------------------------------------------
        print("\n" + "=" * 80)
        print(f"[迭代 {iteration + 1}] 测试 3: WorldModel.forward (完整前向传播)")
        print("=" * 80)
        
        # 添加必要的字段
        inputs_batch['this_action'] = torch.randn(NUM_ENVS, NUM_ACTIONS_CHUNK, ACTION_DIM).to(cfg.device)
        inputs_batch['step_count'] = torch.tensor([0] * NUM_ENVS, dtype=torch.long).to(cfg.device)
        
        with torch.no_grad():
            post_patch_proj, reward_logits, termin_hat = world_model.forward(inputs_batch)
        
        print(f"✓ WorldModel.forward 成功")
        print(f"  - post_patch_proj shape: {post_patch_proj.shape}")
        print(f"  - reward_logits shape: {reward_logits.shape}")
        print(f"  - termin_hat shape: {termin_hat.shape}")
        
        # 解码奖励
        decoded_reward = world_model.symlog_twohot_loss_func.decode(reward_logits)
        print(f"  - 预测奖励: {decoded_reward}")
        print(f"  - 预测终止概率: {torch.sigmoid(termin_hat)}")
        
        # ---------------------------------------------------------------------
        # 测试 4: WorldModel.predict_next
        # ---------------------------------------------------------------------
        print("\n" + "=" * 80)
        print(f"[迭代 {iteration + 1}] 测试 4: WorldModel.predict_next")
        print("=" * 80)
        
        test_action = torch.randn(NUM_ENVS, NUM_ACTIONS_CHUNK * ACTION_DIM).to(cfg.device)
        step_count = torch.tensor([0] * NUM_ENVS, dtype=torch.long).to(cfg.device)
        
        with torch.no_grad():
            next_emb, next_reward, next_termin = world_model.predict_next(
                multimodal_emb, multimodal_att_mask, test_action, step_count
            )
        
        print(f"✓ predict_next 成功")
        print(f"  - next_embeddings shape: {next_emb.shape}")
        print(f"  - reward_logits shape: {next_reward.shape}")
        print(f"  - termin_hat shape: {next_termin.shape}")
        
        decoded_next_reward = world_model.symlog_twohot_loss_func.decode(next_reward)
        print(f"  - 预测下一步奖励: {decoded_next_reward}")
        print(f"  - 预测下一步终止概率: {torch.sigmoid(next_termin)}")
        
        # ---------------------------------------------------------------------
        # 测试 5: WorldModel.agent_super_forward
        # ---------------------------------------------------------------------
        print("\n" + "=" * 80)
        print(f"[迭代 {iteration + 1}] 测试 5: WorldModel.agent_super_forward")
        print("=" * 80)
        
        with torch.no_grad():
            mu_agent, log_std_agent, value_agent = world_model.agent_super_forward(inputs_batch)
        
        print(f"✓ agent_super_forward 成功")
        print(f"  - mu shape: {mu_agent.shape}")
        print(f"  - log_std shape: {log_std_agent.shape}")
        print(f"  - value shape: {value_agent.shape}")
        print(f"  - value 值: {value_agent}")
        print(f"  - mu 范围: [{mu_agent.min():.4f}, {mu_agent.max():.4f}]")
        print(f"  - log_std 范围: [{log_std_agent.min():.4f}, {log_std_agent.max():.4f}]")
        
        # ---------------------------------------------------------------------
        # 测试 6: WorldModel.imagine (批量，使用 inputs_batch)
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
            compute_imagine_loss(imagine_inputs, world_model, imagine_steps, 0.99, 0.95)
            imagined_results = world_model.imagine(imagine_inputs, imagine_steps)
        
        (imagined_logps, imagined_values, imagined_rewards, 
         imagined_dones, last_value) = imagined_results
        
        print(f"✓ imagine 成功")
        print(f"  - imagined_logps shape: {imagined_logps.shape}")
        print(f"  - imagined_values shape: {imagined_values.shape}")
        print(f"  - imagined_rewards shape: {imagined_rewards.shape}")
        print(f"  - imagined_dones shape: {imagined_dones.shape}")
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
            for step in range(imagine_steps):
                print(f"        Step {step}: reward={env_rewards[step].item():.4f}, "
                      f"value={env_values[step].item():.4f}, done={env_dones[step].item()}")
        
        # ---------------------------------------------------------------------
        # 测试 7: 检查数值范围
        # ---------------------------------------------------------------------
        print("\n" + "=" * 80)
        print(f"[迭代 {iteration + 1}] 测试 7: 检查数值范围")
        print("=" * 80)
        
        print(f"Reward logits 范围: [{reward_logits.min():.4f}, {reward_logits.max():.4f}]")
        print(f"Predicted reward 范围: [{decoded_reward.min():.4f}, {decoded_reward.max():.4f}]")
        print(f"Termination logits 范围: [{termin_hat.min():.4f}, {termin_hat.max():.4f}]")
        print(f"Value 范围: [{value_agent.min():.4f}, {value_agent.max():.4f}]")
        print(f"Action 范围: [{mu_agent.min():.4f}, {mu_agent.max():.4f}]")
        print(f"Post patch proj 范围: [{post_patch_proj.min():.4f}, {post_patch_proj.max():.4f}]")
        print(f"Imagined rewards 范围: [{imagined_rewards.min():.4f}, {imagined_rewards.max():.4f}]")
        print(f"Imagined values 范围: [{imagined_values.min():.4f}, {imagined_values.max():.4f}]")
        
        # 打印每个环境的具体数值
        print(f"\n  各环境详细数值:")
        for env_idx in range(NUM_ENVS):
            print(f"    环境 {env_idx}:")
            print(f"      - 当前奖励: {decoded_reward[env_idx].item():.4f}")
            print(f"      - 当前价值: {value_agent[env_idx].item():.4f}")
            print(f"      - 终止概率: {torch.sigmoid(termin_hat[env_idx]).item():.4f}")
            print(f"      - 想象累计奖励: {imagined_rewards[:, env_idx].sum().item():.4f}")
        
        # 检查是否有NaN或Inf
        has_nan = any([
            torch.isnan(post_patch_proj).any(),
            torch.isnan(reward_logits).any(),
            torch.isnan(termin_hat).any(),
            torch.isnan(mu_agent).any(),
            torch.isnan(value_agent).any(),
            torch.isnan(imagined_rewards).any(),
            torch.isnan(imagined_values).any(),
        ])
        has_inf = any([
            torch.isinf(post_patch_proj).any(),
            torch.isinf(reward_logits).any(),
            torch.isinf(termin_hat).any(),
            torch.isinf(mu_agent).any(),
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
    
    # =========================================================================
    # 测试总结
    # =========================================================================
    print("\n" + "█" * 80)
    print("█  所有测试迭代完成!")
    print("█" * 80)
    print(f"\n总共完成 {NUM_TEST_ITERATIONS} 次迭代测试")
    print(f"使用了 {NUM_ENVS} 个并行环境")
    print("所有 WorldModel 功能运行正常 ✓")