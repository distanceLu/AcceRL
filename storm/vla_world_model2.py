from typing import List, Optional
import torch
import torch.nn as nn
from torch.distributions import OneHotCategorical
from einops import rearrange

from storm.functions_losses import SymLogTwoHotLoss
from storm.world_models import DistHead, RewardDecoder, TerminationDecoder, MSELoss, CategoricalKLDivLossWithFreeBits, DecoderBN, EncoderBN

from storm.attention_blocks import get_subsequent_mask_with_batch_length, get_subsequent_mask
from storm.transformer_model import StochasticTransformerKVCache
import agents
from rl.actor_critic_model import ActorCritic
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from peft import LoraConfig, get_peft_model
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from rl.modules import AttentionPool, AttentionPoolHead
import torch.nn.functional as F


class WorldModel(ActorCritic):
    def __init__(self, in_channels, action_dim, instruction_dim,
                 transformer_max_length, transformer_hidden_dim, transformer_num_layers, transformer_num_heads, dist, cfg, dtype):
        super().__init__(cfg, dtype)
        self.dist = dist
        self.action_dim = action_dim
        self.transformer_hidden_dim = transformer_hidden_dim
        self.final_feature_width = 4
        # self.stoch_dim = 32
        self.stoch_flattened_dim = transformer_hidden_dim
        self.use_amp = True
        self.tensor_dtype = self.vla.dtype 
        self.imagine_batch_size = -1
        self.imagine_batch_length = -1
        self.vla: OpenVLAForActionPrediction
        hidden_size = self.vla.llm_dim
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=0,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        self.patch_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.language_model = get_peft_model(self.vla.language_model, lora_config)
        self.language_model.print_trainable_parameters()
        self.language_model: LlamaForCausalLM
        # self.encoder = EncoderBN(
        #     in_channels=in_channels,
        #     stem_channels=32,
        #     final_feature_width=self.final_feature_width
        # )
        self.emb_pool = AttentionPoolHead(hidden_size, hidden_size)
        self.storm_transformer = StochasticTransformerKVCache(
            stoch_dim=self.stoch_flattened_dim,
            action_dim=action_dim,
            instruction_dim=instruction_dim,
            feat_dim=transformer_hidden_dim,
            num_layers=transformer_num_layers,
            num_heads=transformer_num_heads,
            max_length=transformer_max_length,
            dropout=0.1,
            dist=self.dist
        )
        # self.dist_head = DistHead(
        #     image_feat_dim=self.encoder.last_channels*self.final_feature_width*self.final_feature_width,
        #     transformer_hidden_dim=transformer_hidden_dim,
        #     stoch_dim=self.stoch_dim
        # )
        # self.image_decoder = DecoderBN(
        #     stoch_dim=self.stoch_flattened_dim,
        #     last_channels=self.encoder.last_channels,
        #     original_in_channels=in_channels,
        #     stem_channels=32,
        #     final_feature_width=self.final_feature_width
        # )
        self.reward_pool = AttentionPool(hidden_size)
        self.reward_decoder = RewardDecoder(
            num_classes=255,
            embedding_size=self.stoch_flattened_dim,
            transformer_hidden_dim=transformer_hidden_dim
        )
        self.termination_pool = AttentionPool(hidden_size)
        self.termination_decoder = TerminationDecoder(
            embedding_size=self.stoch_flattened_dim,
            transformer_hidden_dim=transformer_hidden_dim
        )

        self.mse_loss_func = MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_with_logits_loss_func = nn.BCEWithLogitsLoss()
        self.symlog_twohot_loss_func = SymLogTwoHotLoss(num_classes=255, lower_bound=-20, upper_bound=20)
        self.categorical_kl_div_loss = CategoricalKLDivLossWithFreeBits(free_bits=1)
        # print(f"Converting the entire WorldModel to dtype: {self.tensor_dtype}")
        # self.to(self.tensor_dtype)
        # for k, v in self.named_parameters():
        #     if v.dtype != self.tensor_dtype:
        #         print(f"Parameter {k} is of dtype {v.dtype}, converting to {self.tensor_dtype}")
        #         v.data = v.data.to(self.tensor_dtype)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-5)
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

    def encode(self, pixel_values, proprio):
        patch_features = self.vla.vision_backbone(pixel_values)  # (bsz, 256 * num_images, D)
        projected_patch_embeddings = self.vla.projector(patch_features)
        # Add proprioceptive state if provided
        projected_patch_embeddings = self.vla._process_proprio_features(
            projected_patch_embeddings, proprio, self.proprio_projector
        )
        return projected_patch_embeddings

    def encode_obs(self, pixel_values, proprio):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            embedding = self.encode(pixel_values, proprio)
        return embedding

    def language_model_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        projected_patch_embeddings: torch.Tensor = None,
        temporal_feat: torch.Tensor = None,
    ):
        # Get input embeddings (from language model embeddings)
        input_embeddings = self.vla.get_input_embeddings()(input_ids)  # (B, seq_len, D)

        # Extract action masks
        all_actions_mask = self.vla._process_action_masks(labels)

        # Process action embeddings
        # Replace the embeddings of the action tokens with zeros
        # (Later on, the positional embeddings will be added to them)
        all_actions_mask = all_actions_mask.unsqueeze(-1)  # (B, seq_len, 1)
        input_embeddings = input_embeddings * ~all_actions_mask

        projected_patch_embeddings = torch.cat([projected_patch_embeddings, temporal_feat.unsqueeze(dim=1)], dim=1)

        # Build multimodal embeddings & attention mask
        multimodal_embeddings, multimodal_attention_mask = self.vla._build_multimodal_attention(
            input_embeddings, projected_patch_embeddings, attention_mask
        )

        # Build labels for multimodal sequence if needed
        multimodal_labels = self.vla._build_multimodal_labels(labels, projected_patch_embeddings)

        # Dispatch to language model
        language_model_output = self.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=multimodal_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return language_model_output
    
    def decode_reward(self, post_patch_embeddings, bs, bl):
        # post_patch_embeddings: [bs*bl, 513, llm_dim]
        flat_emb = self.reward_pool(post_patch_embeddings)  # [bs*bl, llm_dim]
        reward_hat = self.reward_decoder(flat_emb)  # [bs*bl, 255]
        return reward_hat.reshape(bs, bl, -1)  # [bs, bl, 255]

    def decode_termination(self, post_patch_embeddings, bs, bl):
        # post_patch_embeddings: [bs*bl, 513, llm_dim]
        flat_emb = self.termination_pool(post_patch_embeddings)  # [bs*bl, llm_dim]
        termination_hat = self.termination_decoder(flat_emb)  # [bs*bl, 1]
        return termination_hat.reshape(bs, bl)  # [bs, bl]

    def calc_last_dist_feat(self, embedding, action, inputs_batch):
        batch_size, batch_length = action.shape[:2]
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            pooled_emb = self.emb_pool(embedding)  # [bs*bl, 4096]
            flattened_sample = rearrange(pooled_emb, "(B L) D -> B L D", B=batch_size)  # [bs, bl, 4096]
            temporal_mask = get_subsequent_mask_with_batch_length(batch_length, flattened_sample.device)  # [1, bl, bl]
            temporal_feat = self.storm_transformer.forward(flattened_sample, action, temporal_mask)  # [bs, bl, llm_dim]
            recon_hidden_states = self.language_model_forward(
                input_ids=inputs_batch['input_ids'],
                attention_mask=inputs_batch['attention_mask'],
                labels=inputs_batch['labels'],
                use_cache=False,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
                projected_patch_embeddings=embedding,
                temporal_feat=temporal_feat.reshape(batch_size*batch_length, -1)
            ).hidden_states[-1]  # [bs*bl, seq_len, llm_dim]
            num_patches = self._compute_num_patches()
            post_patch_embeddings = recon_hidden_states[:, 1:num_patches+1]  # [bs*bl, 513, llm_dim]
            # post_patch_embeddings = self.patch_proj(post_patch_embeddings)
            post_emb_reshape = rearrange(post_patch_embeddings, "(B L) N D -> B L N D", B=batch_size)  # [bs, bl, 513, llm_dim]
        return post_emb_reshape[:, -1:]

    def predict_next(self, action, inputs_batch, embedding, log_video=True):
        batch_size, batch_length = action.shape[:2]
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            pooled_emb = self.emb_pool(embedding)  # [bs*bl, 4096]
            flattened_sample = rearrange(pooled_emb, "(B L) D -> B L D", B=batch_size)  # [bs, bl, 4096]
            temporal_feat = self.storm_transformer.forward_with_kv_cache(flattened_sample, action)
            # prior_logits = self.dist_head.forward_prior(dist_feat)

            # decoding
            # prior_sample = self.stright_throught_gradient(prior_logits, sample_mode="random_sample")
            # prior_flattened_sample = self.flatten_sample(prior_sample)
            if log_video:
                # obs_hat = self.image_decoder(prior_flattened_sample)
                obs_hat = None
            else:
                obs_hat = None
            recon_hidden_states = self.language_model_forward(
                input_ids=inputs_batch['input_ids'],
                attention_mask=inputs_batch['attention_mask'],
                labels=inputs_batch['labels'],
                use_cache=False,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
                projected_patch_embeddings=embedding,
                temporal_feat=temporal_feat.reshape(batch_size*batch_length, -1)
            ).hidden_states[-1]  # [bs*bl, seq_len, llm_dim]
            num_patches = self._compute_num_patches()
            post_patch_embeddings = recon_hidden_states[:, 1:num_patches+1]  # [bs*bl, 513, llm_dim]
            # reward_hat = self.reward_decoder(dist_feat)
            reward_hat = self.decode_reward(post_patch_embeddings, batch_size, batch_length)  # [bs, bl, 255]]
            reward_hat = self.symlog_twohot_loss_func.decode(reward_hat)
            # termination_hat = self.termination_decoder(dist_feat)
            termination_hat = self.decode_termination(post_patch_embeddings, batch_size, batch_length)  # [bs, bl]
            termination_hat = termination_hat > 0
            post_emb_reshape = rearrange(post_patch_embeddings, "(B L) N D -> B L N D", B=batch_size)  # [bs, bl, 513, llm_dim]

        return obs_hat, reward_hat, termination_hat, post_emb_reshape

    def stright_throught_gradient(self, logits, sample_mode="random_sample"):
        dist = OneHotCategorical(logits=logits)
        if sample_mode == "random_sample":
            sample = dist.sample() + dist.probs - dist.probs.detach()
        elif sample_mode == "mode":
            sample = dist.mode
        elif sample_mode == "probs":
            sample = dist.probs
        return sample

    def flatten_sample(self, sample):
        return rearrange(sample, "B L K C -> B L (K C)")

    def init_imagine_buffer(self, imagine_batch_size, imagine_batch_length, dtype, patch_num):
        '''
        This can slightly improve the efficiency of imagine_data
        But may vary across different machines
        '''
        if self.imagine_batch_size != imagine_batch_size or self.imagine_batch_length != imagine_batch_length:
            print(f"init_imagine_buffer: {imagine_batch_size}x{imagine_batch_length}@{dtype}")
            self.imagine_batch_size = imagine_batch_size
            self.imagine_batch_length = imagine_batch_length
            # latent_size = (imagine_batch_size, imagine_batch_length+1, patch_num, self.stoch_flattened_dim)
            hidden_size = (imagine_batch_size, imagine_batch_length+1, patch_num, self.transformer_hidden_dim)
            scalar_size = (imagine_batch_size, imagine_batch_length)
            if self.dist == "onehot":
                action_size = scalar_size
            else:
                action_size = (imagine_batch_size, imagine_batch_length, self.action_dim)
            # self.latent_buffer = torch.zeros(latent_size, dtype=dtype, device="cuda")
            self.hidden_buffer = torch.zeros(hidden_size, dtype=dtype, device="cuda")
            self.action_buffer = torch.zeros(action_size, dtype=dtype, device="cuda")
            self.reward_hat_buffer = torch.zeros(scalar_size, dtype=dtype, device="cuda")
            self.termination_hat_buffer = torch.zeros(scalar_size, dtype=dtype, device="cuda")

    def imitation_data(self, obs_list, action, reward, termination, teacher_model: ActorCritic):
        batch_size, batch_length = action.shape[:2]
        inputs_batch = self.prepare_inputs_batch(obs_list)
        self.eval()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp): 
            teacher_action = teacher_model.forward(inputs_batch)[1]
            embedding = self.encode(inputs_batch['pixel_values'], inputs_batch['proprio'].to(self.vla.dtype))  # [bs*bl, 513, 4096]
            pooled_emb = self.emb_pool(embedding)  # [bs*bl, 4096]
            flattened_sample = rearrange(pooled_emb, "(B L) D -> B L D", B=batch_size)  # [bs, bl, 4096]
            temporal_mask = get_subsequent_mask_with_batch_length(batch_length, flattened_sample.device)  # [1, bl, bl]
            temporal_feat = self.storm_transformer.forward(flattened_sample, action, temporal_mask)  # [bs, bl, llm_dim]
            recon_hidden_states = self.language_model_forward(
                input_ids=inputs_batch['input_ids'],
                attention_mask=inputs_batch['attention_mask'],
                labels=inputs_batch['labels'],
                use_cache=False,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
                projected_patch_embeddings=embedding,
                temporal_feat=temporal_feat.reshape(batch_size*batch_length, -1)
            ).hidden_states[-1]  # [bs*bl, seq_len, llm_dim]
            num_patches = self._compute_num_patches()
            post_patch_embeddings = recon_hidden_states[:, 1:num_patches+1]  # [bs*bl, 513, llm_dim]
            post_emb = rearrange(post_patch_embeddings, "(B L) N D -> B L N D", B=batch_size)  # [bs, bl, 513, llm_dim]
            teacher_action = teacher_action.reshape(action.shape)
        return post_emb, action, reward, termination, teacher_action

    def imagine_data(self, agent: agents.ActorCriticAgent, sample_obs, sample_action,
                     imagine_batch_size, imagine_batch_length, log_video, train_steps, logger):
        log_video = False  # 现在不支持log video
        patch_num = self._compute_num_patches()
        self.init_imagine_buffer(imagine_batch_size, imagine_batch_length, dtype=self.tensor_dtype, patch_num=patch_num)
        obs_hat_list = []
        # (imagine_batch_size, imagine_batch_length, dim)
        self.storm_transformer.reset_kv_cache_list(imagine_batch_size, dtype=self.tensor_dtype)
        # context
        
        inputs_batch = self.prepare_inputs_batch(sample_obs)
        context_latent = self.encode_obs(inputs_batch['pixel_values'], inputs_batch['proprio'].to(self.vla.dtype))  # [bs*bl, 513, llm_dim]
        bs, bl = sample_action.shape[:2]
        indices = torch.arange(bs * bl).view(bs, bl)
        selected_indices = indices[:, -1:].flatten()  # 看后续代码的话，似乎只保留了最后一帧数据
        check_inputs_batch(inputs_batch, bs, bl)
        instruction_inputs = {
            'input_ids': inputs_batch['input_ids'][selected_indices],
            'attention_mask': inputs_batch['attention_mask'][selected_indices],
            'labels': inputs_batch['labels'][selected_indices]
        }  # 一条轨迹中指令相关的信息都是一样的

        for i in range(sample_action.shape[1]):  # context_length is sample_obs.shape[1]
            selected_indices = indices[:, i:i+1].flatten()
            last_obs_hat, last_reward_hat, last_termination_hat, last_dist_feat = self.predict_next(
                sample_action[:, i:i+1],
                instruction_inputs,
                context_latent[selected_indices],
                log_video=log_video
            )
            
        # self.latent_buffer[:, 0:1] = last_latent
        self.hidden_buffer[:, 0:1] = last_dist_feat

        # imagine
        for i in range(imagine_batch_length):
            # action = agent.sample(torch.cat([self.latent_buffer[:, i:i+1], self.hidden_buffer[:, i:i+1]], dim=-1))
            action = agent.sample(self.hidden_buffer[:, i:i+1])
            self.action_buffer[:, i:i+1] = action

            last_obs_hat, last_reward_hat, last_termination_hat, last_dist_feat = self.predict_next(
                self.action_buffer[:, i:i+1], 
                instruction_inputs, 
                self.hidden_buffer[:, i], 
                log_video=log_video
            )

            # self.latent_buffer[:, i+1:i+2] = last_latent
            self.hidden_buffer[:, i+1:i+2] = last_dist_feat
            self.reward_hat_buffer[:, i:i+1] = last_reward_hat
            self.termination_hat_buffer[:, i:i+1] = last_termination_hat
            if log_video:
                obs_hat_list.append(last_obs_hat[::imagine_batch_size//16])  # uniform sample vec_env

        if log_video:
            logger.log("Imagine/predict_video", torch.clamp(torch.cat(obs_hat_list, dim=1), 0, 1).cpu().float().detach().numpy(), train_steps)

        return self.hidden_buffer, self.action_buffer, self.reward_hat_buffer, self.termination_hat_buffer

    def update(self, obs_list, action, reward, termination, current_steps, logger=None):
        self.train()
        batch_size, batch_length = action.shape[:2]
        inputs_batch = self.prepare_inputs_batch(obs_list)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            # encoding
            embedding = self.encode(inputs_batch['pixel_values'], inputs_batch['proprio'].to(self.vla.dtype))  # [bs*bl, 513, 4096]
            # post_logits = self.dist_head.forward_post(embedding)  # [bs, bl, 32, 32]
            # sample = self.stright_throught_gradient(post_logits, sample_mode="random_sample")  # [bs, bl, 32, 32] 
            # flattened_sample = self.flatten_sample(sample)  # [bs, bl, 1024]

            # # decoding image
            # obs_hat = self.image_decoder(flattened_sample)  # [bs, bl, 3, 64, 64]
            pooled_emb = self.emb_pool(embedding)  # [bs*bl, 4096]
            flattened_sample = rearrange(pooled_emb, "(B L) D -> B L D", B=batch_size)  # [bs, bl, 4096]

            # transformer
            temporal_mask = get_subsequent_mask_with_batch_length(batch_length, flattened_sample.device)  # [1, bl, bl]
            temporal_feat = self.storm_transformer.forward(flattened_sample, action, temporal_mask)  # [bs, bl, llm_dim]
            recon_hidden_states = self.language_model_forward(
                input_ids=inputs_batch['input_ids'],
                attention_mask=inputs_batch['attention_mask'],
                labels=inputs_batch['labels'],
                use_cache=False,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
                projected_patch_embeddings=embedding,
                temporal_feat=temporal_feat.reshape(batch_size*batch_length, -1)
            ).hidden_states[-1]  # [bs*bl, seq_len, llm_dim]
            num_patches = self._compute_num_patches()
            post_patch_embeddings = recon_hidden_states[:, 1:num_patches+1]  # [bs*bl, 513, llm_dim]
            post_patch_emb_proj = self.patch_proj(post_patch_embeddings)
            # prior_logits = self.dist_head.forward_prior(dist_feat)  # [bs, bl, 32, 32]
            # decoding reward and termination with dist_feat
            reward_hat = self.decode_reward(post_patch_embeddings, batch_size, batch_length)  # [bs, bl, 255]]
            termination_hat = self.decode_termination(post_patch_embeddings, batch_size, batch_length)  # [bs, bl]

            # env loss
            # reconstruction_loss = self.mse_loss_func(obs_hat, obs)
            reward_loss = self.symlog_twohot_loss_func(reward_hat, reward)
            termination_loss = self.bce_with_logits_loss_func(termination_hat, termination)
            proj_emb_reshape = rearrange(post_patch_emb_proj, "(B L) N D -> B L N D", B=batch_size)  # [bs, bl, 513, llm_dim]
            embedding_reshape = rearrange(embedding, "(B L) N D -> B L N D", B=batch_size)  # [bs, bl, 513, 4096]
            # 1. 准备用于计算 dynamics loss 的张量
            # 预测的下一时刻的状态: proj_emb_reshape 在 0 到 bl-2 时刻的输出
            pred_next_state = proj_emb_reshape[:, :-1]
            # 真实的下一时刻的状态: embedding_reshape 在 1 到 bl-1 时刻的输入
            true_next_state = embedding_reshape[:, 1:].detach()

            # 2. 创建掩码 (mask)
            # dynamics loss 比较的是 t 时刻的预测和 t+1 时刻的真实值。
            # 如果在 t 时刻 termination 为 True，那么就不应该有 t+1 时刻的预测。
            # 因此，我们使用的 termination 范围是 [:, :-1]。
            # 我们只在 termination 为 False 的地方计算 loss。
            valid_transitions_mask = termination[:, :-1] == 0  # Shape: [bs, bl-1]

            # 3. 使用掩码提取有效数据
            # PyTorch 的布尔索引会根据 mask 中的 True 值来选取元素。
            # 这会将 [bs, bl-1, 513, D] 的张量展平为 [num_valid_transitions, 513, D]。
            valid_preds = pred_next_state[valid_transitions_mask]
            valid_targets = true_next_state[valid_transitions_mask]

            # 4. 计算 loss，并处理没有有效转换的边界情况
            num_valid_transitions = valid_preds.shape[0]
            if num_valid_transitions > 0:
                dynamics_loss = F.mse_loss(valid_preds, valid_targets)
            else:
                # 如果整个批次中都没有有效的转换（例如，所有序列在第一步就终止了），
                # 则损失为0，以避免计算空张量的均值导致 NaN。
                dynamics_loss = torch.tensor(0.0, device=proj_emb_reshape.device)
            # dyn-rep loss
            # dynamics_loss, dynamics_real_kl_div = self.categorical_kl_div_loss(post_logits[:, 1:].detach(), prior_logits[:, :-1])
            # representation_loss, representation_real_kl_div = self.categorical_kl_div_loss(post_logits[:, 1:], prior_logits[:, :-1].detach())
            # total_loss = reconstruction_loss + reward_loss + termination_loss + 0.5*dynamics_loss + 0.1*representation_loss
            total_loss = reward_loss + termination_loss + 10*dynamics_loss

        # gradient descent
        # self.scaler.scale(total_loss).backward()
        total_loss.backward()
        # self.scaler.unscale_(self.optimizer)  # for clip grad
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1000.0)
        # self.scaler.step(self.optimizer)
        self.optimizer.step()
        # self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        if logger is not None:
            # logger.log("WorldModel/reconstruction_loss", reconstruction_loss.item(), current_steps)
            logger.log("WorldModel/reward_loss", reward_loss.item(), current_steps)
            logger.log("WorldModel/termination_loss", termination_loss.item(), current_steps)
            logger.log("WorldModel/dynamics_loss", dynamics_loss.item(), current_steps)
            # logger.log("WorldModel/dynamics_real_kl_div", dynamics_real_kl_div.item(), current_steps)
            # logger.log("WorldModel/representation_loss", representation_loss.item(), current_steps)
            # logger.log("WorldModel/representation_real_kl_div", representation_real_kl_div.item(), current_steps)
            logger.log("WorldModel/total_loss", total_loss.item(), current_steps)
 

def check_inputs_batch(inputs_batch, bs, bl):
    """
    检查一条轨迹中是否所有的指令相关信息都是一样的
    """
    inputs_id = inputs_batch['input_ids'].reshape(bs, bl, -1)
    attention_mask = inputs_batch['attention_mask'].reshape(bs, bl, -1)
    labels = inputs_batch['labels'].reshape(bs, bl, -1)
    assert (inputs_id - inputs_id[:, 0:1]).sum() == 0
    assert (attention_mask - attention_mask[:, 0:1]).sum() == 0
    assert (labels - labels[:, 0:1]).sum() == 0