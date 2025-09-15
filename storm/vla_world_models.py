import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical, Normal
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from torch.cuda.amp import autocast
import copy

from storm.functions_losses import SymLogTwoHotLoss
from storm.attention_blocks import get_subsequent_mask_with_batch_length, get_subsequent_mask, create_patch_causal_mask
from storm.transformer_model import StochasticTransformerKVCache

from storm.vit_decoder import ViTDecoder
from storm.actor_critic_model import get_vla, DEVICE
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction


class DistHead(nn.Module):
    '''
    Dist: abbreviation of distribution
    '''
    def __init__(self, image_feat_dim, transformer_hidden_dim, stoch_dim) -> None:
        super().__init__()
        self.stoch_dim = stoch_dim
        self.post_head = nn.Linear(image_feat_dim, stoch_dim*stoch_dim)
        self.prior_head = nn.Linear(transformer_hidden_dim, stoch_dim*stoch_dim)

    def unimix(self, logits, mixing_ratio=0.01):
        # uniform noise mixing
        probs = F.softmax(logits, dim=-1)
        mixed_probs = mixing_ratio * torch.ones_like(probs) / self.stoch_dim + (1-mixing_ratio) * probs
        logits = torch.log(mixed_probs)
        return logits

    def forward_post(self, x):
        # (B, L, N, D)
        logits = self.post_head(x)
        logits = rearrange(logits, "B L N (K C) -> B L N K C", K=self.stoch_dim)
        logits = self.unimix(logits)
        return logits

    def forward_prior(self, x):
        logits = self.prior_head(x)
        logits = rearrange(logits, "B L N (K C) -> B L N K C", K=self.stoch_dim)
        logits = self.unimix(logits)
        return logits


class RewardDecoder(nn.Module):
    def __init__(self, num_classes, embedding_size, transformer_hidden_dim) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim, bias=False),
            nn.LayerNorm(transformer_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim, bias=False),
            nn.LayerNorm(transformer_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(transformer_hidden_dim, num_classes)

    def forward(self, feat):
        # (B, L, N, D) -> (B, L, D)
        feat = torch.mean(feat, dim=2)
        feat = self.backbone(feat)
        reward = self.head(feat)
        return reward


class TerminationDecoder(nn.Module):
    def __init__(self,  embedding_size, transformer_hidden_dim) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim, bias=False),
            nn.LayerNorm(transformer_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim, bias=False),
            nn.LayerNorm(transformer_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(transformer_hidden_dim, 1),
            # nn.Sigmoid()
        )

    def forward(self, feat):
        # (B, L, N, D) -> (B, L, D)
        feat = torch.mean(feat, dim=2)
        feat = self.backbone(feat)
        termination = self.head(feat)
        termination = termination.squeeze(-1)  # remove last 1 dim
        return termination


class MSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, obs_hat, obs):
        loss = (obs_hat - obs)**2
        loss = reduce(loss, "B L C H W -> B L", "sum")
        return loss.mean()


class CategoricalKLDivLossWithFreeBits(nn.Module):
    def __init__(self, free_bits) -> None:
        super().__init__()
        self.free_bits = free_bits

    def forward(self, p_logits, q_logits):
        p_dist = OneHotCategorical(logits=p_logits)
        q_dist = OneHotCategorical(logits=q_logits)
        kl_div = torch.distributions.kl.kl_divergence(p_dist, q_dist)
        kl_div = reduce(kl_div, "B L N D -> B L D", "sum")
        kl_div = kl_div.mean()
        real_kl_div = kl_div
        kl_div = torch.max(torch.ones_like(kl_div)*self.free_bits, kl_div)
        return kl_div, real_kl_div

class PositionalEncoding1D(nn.Module):
    def __init__(
        self,
        max_length: int,
        embed_dim: int
    ):
        super().__init__()
        self.max_length = max_length
        self.embed_dim = embed_dim

        self.pos_emb = nn.Embedding(self.max_length, embed_dim)

    def forward(self, feat):
        B, L, N, D = feat.shape
        pos_emb = self.pos_emb(torch.arange(self.max_length, device=feat.device))
        pos_emb = pos_emb.unsqueeze(1).expand(-1, N, -1)
        pos_emb = repeat(pos_emb, "L N D -> B L N D", B=feat.shape[0])

        feat = feat + pos_emb[:, :L, ...]
        return feat

    def forward_with_position(self, feat, position):
        assert feat.shape[1] == 1
        B, L, N, D = feat.shape
        pos_emb = self.pos_emb(torch.arange(self.max_length, device=feat.device))
        pos_emb = pos_emb.unsqueeze(1).expand(-1, N, -1)
        pos_emb = repeat(pos_emb, "L N D -> B L N D", B=feat.shape[0])

        feat = feat + pos_emb[:, position:position+1, ...]
        return feat

class StemProjector(nn.Module):
    def __init__(self, use_fused_vision_backbone: bool, stoch_dim: int, action_dim: int, llm_dim: int, max_length: int) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.vision_dim = stoch_dim + action_dim
        self.llm_dim = llm_dim
        self.position_encoding = PositionalEncoding1D(max_length=max_length, embed_dim=llm_dim)


        # Switch on `use_fused_vision_backbone` =>> use slightly different MLPs and projection factors!
        if not self.use_fused_vision_backbone:
            self.fc1 = nn.Linear(self.vision_dim, self.llm_dim, bias=True)
            self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
        else:
            initial_projection_dim = 4 * self.vision_dim
            self.fc1 = nn.Linear(self.vision_dim, initial_projection_dim, bias=True)
            self.fc2 = nn.Linear(initial_projection_dim, self.llm_dim, bias=True)
            self.fc3 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
            self.act_fn2 = nn.GELU()

    def forward(self, img_patches: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        B, L, N, D = img_patches.shape
        actions = actions.unsqueeze(2).expand(-1, -1, N, -1)
        if not self.use_fused_vision_backbone:
            projected_features = self.fc1(torch.cat([img_patches, actions], dim=-1))
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
        else:
            projected_features = self.fc1(torch.cat([img_patches, actions], dim=-1))
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
            projected_features = self.act_fn2(projected_features)
            projected_features = self.fc3(projected_features)
        projected_features = self.position_encoding(projected_features)

        return projected_features
    

class WorldModel(nn.Module):
    def __init__(self, action_dim, image_encoder, language_model, encode_projector, 
                transformer_hidden_dim, max_length):
        super().__init__()
        self.action_dim = action_dim
        self.transformer_hidden_dim = transformer_hidden_dim
        self.stoch_dim = 64
        self.stoch_flattened_dim = self.stoch_dim * self.stoch_dim
        self.tensor_dtype = torch.bfloat16
        self.imagine_batch_size = -1
        self.imagine_batch_length = -1

        self.image_encoder = copy.deepcopy(image_encoder)
        self.encode_projector = copy.deepcopy(encode_projector)
        self.storm_transformer = copy.deepcopy(language_model)

        self.dist_head = DistHead(
            image_feat_dim=self.stoch_flattened_dim,
            transformer_hidden_dim=transformer_hidden_dim,
            stoch_dim=self.stoch_dim
        )

        self.stem_projector = StemProjector(
            use_fused_vision_backbone=True, 
            stoch_dim=self.stoch_flattened_dim, action_dim=action_dim, 
            llm_dim=transformer_hidden_dim,
            max_length=max_length
        )

        self.image_decoder = ViTDecoder(embed_dim=self.stoch_flattened_dim, depth=12)
        self.reward_decoder = RewardDecoder(
            num_classes=255,
            embedding_size=self.stoch_flattened_dim,
            transformer_hidden_dim=transformer_hidden_dim
        )
        self.termination_decoder = TerminationDecoder(
            embedding_size=self.stoch_flattened_dim,
            transformer_hidden_dim=transformer_hidden_dim
        )

        self.mse_loss_func = MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_with_logits_loss_func = nn.BCEWithLogitsLoss()
        self.symlog_twohot_loss_func = SymLogTwoHotLoss(num_classes=255, lower_bound=-20, upper_bound=20)
        self.categorical_kl_div_loss = CategoricalKLDivLossWithFreeBits(free_bits=1).to(dtype=self.tensor_dtype)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

    def encode_obs(self, obs):
        embedding = self.image_encoder(obs)
        post_logits = self.dist_head.forward_post(embedding)
        sample = self.stright_throught_gradient(post_logits)
        flattened_sample = self.flatten_sample(sample)
        return flattened_sample

    def calc_last_dist_feat(self, latent, action):
     
        temporal_mask = get_subsequent_mask(latent) # z_t
        dist_feat = self.storm_transformer(latent, action, temporal_mask)
        last_dist_feat = dist_feat[:, -1:] # h_t=f(z_t, a_t)
        prior_logits = self.dist_head.forward_prior(last_dist_feat) # Z_{t+1}=g(z_{t+1}|h_{t+1})
        prior_sample = self.stright_throught_gradient(prior_logits) # z_{t+1}
        prior_flattened_sample = self.flatten_sample(prior_sample)
        return prior_flattened_sample, last_dist_feat # z_{t+1}, h_{t+1}

    def predict_next(self, last_flattened_sample, action, log_video=True):
        
        dist_feat = self.storm_transformer.forward_with_kv_cache(last_flattened_sample, action)
        prior_logits = self.dist_head.forward_prior(dist_feat)

        # decoding
        prior_sample = self.stright_throught_gradient(prior_logits)
        prior_flattened_sample = self.flatten_sample(prior_sample)
        if log_video:
            obs_hat = self.image_decoder(prior_flattened_sample)
        else:
            obs_hat = None
        reward_hat = self.reward_decoder(dist_feat)
        reward_hat = self.symlog_twohot_loss_func.decode(reward_hat)
        termination_hat = self.termination_decoder(dist_feat)
        termination_hat = termination_hat > 0

        return obs_hat, reward_hat, termination_hat, prior_flattened_sample, dist_feat

    def stright_throught_gradient(self, logits):
        dist = OneHotCategorical(logits=logits)
        sample = dist.sample() + dist.probs - dist.probs.detach()
        return sample

    def flatten_sample(self, sample):
        return rearrange(sample, "B L N K C -> B L N (K C)")

    def init_imagine_buffer(self, imagine_batch_size, imagine_batch_length, dtype):
        '''
        This can slightly improve the efficiency of imagine_data
        But may vary across different machines
        '''
        if self.imagine_batch_size != imagine_batch_size or self.imagine_batch_length != imagine_batch_length:
            print(f"init_imagine_buffer: {imagine_batch_size}x{imagine_batch_length}@{dtype}")
            self.imagine_batch_size = imagine_batch_size
            self.imagine_batch_length = imagine_batch_length
            latent_size = (imagine_batch_size, imagine_batch_length+1, self.stoch_flattened_dim)
            hidden_size = (imagine_batch_size, imagine_batch_length+1, self.transformer_hidden_dim)
            # instruction_size = (imagine_batch_size, 1, 384)
            scalar_size = (imagine_batch_size, imagine_batch_length)
     
            action_size = (imagine_batch_size, imagine_batch_length, self.action_dim)
            self.latent_buffer = torch.zeros(latent_size, dtype=dtype, device=DEVICE)
            self.hidden_buffer = torch.zeros(hidden_size, dtype=dtype, device=DEVICE)
            self.action_buffer = torch.zeros(action_size, dtype=dtype, device=DEVICE)
            self.reward_hat_buffer = torch.zeros(scalar_size, dtype=dtype, device=DEVICE)
            self.termination_hat_buffer = torch.zeros(scalar_size, dtype=dtype, device=DEVICE)
            # self.instruction_buffer = torch.zeros(instruction_size, dtype=dtype, device=DEVICE)

    def imagine_data(self, agent: OpenVLAForActionPrediction, sample_obs, sample_action, sample_instruction,
                     imagine_batch_size, imagine_batch_length, log_video, train_steps, logger):
        self.init_imagine_buffer(imagine_batch_size, imagine_batch_length, dtype=self.tensor_dtype)
        obs_hat_list = []
        # (imagine_batch_size, imagine_batch_length, dim)
        self.storm_transformer.reset_kv_cache_list(imagine_batch_size, dtype=self.tensor_dtype)
        # context
        
        context_latent = self.encode_obs(sample_obs)
        for i in range(sample_obs.shape[1]):  # context_length is sample_obs.shape[1]
            last_obs_hat, last_reward_hat, last_termination_hat, last_latent, last_dist_feat = self.predict_next(
                context_latent[:, i:i+1],
                sample_action[:, i:i+1],
                log_video=log_video
            )
            
        self.latent_buffer[:, 0:1] = last_latent
        self.hidden_buffer[:, 0:1] = last_dist_feat
        # self.instruction_buffer[:, 0] = sample_instruction[:, 0]

        # imagine
        for i in range(imagine_batch_length):
            action = agent.sample(torch.cat([self.latent_buffer[:, i:i+1], self.hidden_buffer[:, i:i+1]], dim=-1))
            self.action_buffer[:, i:i+1] = action

            last_obs_hat, last_reward_hat, last_termination_hat, last_latent, last_dist_feat = self.predict_next(
                self.latent_buffer[:, i:i+1], self.action_buffer[:, i:i+1], log_video=log_video)

            self.latent_buffer[:, i+1:i+2] = last_latent
            self.hidden_buffer[:, i+1:i+2] = last_dist_feat
            self.reward_hat_buffer[:, i:i+1] = last_reward_hat
            self.termination_hat_buffer[:, i:i+1] = last_termination_hat
            if log_video:
                obs_hat_list.append(last_obs_hat[::imagine_batch_size//16])  # uniform sample vec_env

        if log_video:
            logger.log("Imagine/predict_video", torch.clamp(torch.cat(obs_hat_list, dim=1), 0, 1).cpu().float().detach().numpy(), train_steps)

        return torch.cat([self.latent_buffer, self.hidden_buffer], dim=-1), self.action_buffer, self.reward_hat_buffer, self.termination_hat_buffer

    def update(self, obs, action, reward, termination, current_steps=0, logger=None):
        self.train()
        B, L, C, H, W = obs.shape # (B, L, 6*2, 224, 224)

        # encoding
        obs_reshape = obs.reshape(B*L, C, H, W)
        patch_features = self.image_encoder(obs_reshape) # (B*L, 256*2, 2176)
        _, N, _ = patch_features.shape
        patch_features = patch_features.reshape(B, L, N, -1)
        projected_patch_embeddings = self.encode_projector(patch_features) # (B, L, 256*2, 4096)
        post_logits = self.dist_head.forward_post(projected_patch_embeddings) # (B, L, 256*2, 64, 64)
        sample = self.stright_throught_gradient(post_logits) # (B, L, 256*2, 64, 64)
        flattened_sample = self.flatten_sample(sample) # (B, L, 256*2, 4096)

        # decoding image
        obs_hat = self.image_decoder(flattened_sample) # (B, L, 6*2, 224, 224)

        # transformer
        input_embeds = self.stem_projector(flattened_sample, action)
        temporal_mask = create_patch_causal_mask(L, N, input_embeds.device) # (L*N, L*N)
        attention_mask = temporal_mask.unsqueeze(0).unsqueeze(1).expand(B, 1, -1, -1) # (B, 1, L*N, L*N)
        input_embeds = input_embeds.reshape(B, L*N, -1)
        language_model_output = self.storm_transformer(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=input_embeds,
            labels=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        dist_feat = language_model_output.hidden_states[-1] # (B, L*N, 4096)
        dist_feat = dist_feat.reshape(B, L, N, -1)
        prior_logits = self.dist_head.forward_prior(dist_feat) # (B, L, N, 64, 64)
        # decoding reward and termination with dist_feat
        reward_hat = self.reward_decoder(dist_feat) # (B, L, 255)
        termination_hat = self.termination_decoder(dist_feat)

        # env loss
        reconstruction_loss = self.mse_loss_func(obs_hat, obs)
        reward_loss = self.symlog_twohot_loss_func(reward_hat, reward)
        termination_loss = self.bce_with_logits_loss_func(termination_hat, termination)
        # dyn-rep loss
        dynamics_loss, dynamics_real_kl_div = self.categorical_kl_div_loss(post_logits[:, 1:].detach(), prior_logits[:, :-1])
        representation_loss, representation_real_kl_div = self.categorical_kl_div_loss(post_logits[:, 1:], prior_logits[:, :-1].detach())
        total_loss = reconstruction_loss + reward_loss + termination_loss + 0.5*dynamics_loss + 0.1*representation_loss

        # gradient descent
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1000.0)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        if logger is not None:
            logger.log("WorldModel/reconstruction_loss", reconstruction_loss.item(), current_steps)
            logger.log("WorldModel/reward_loss", reward_loss.item(), current_steps)
            logger.log("WorldModel/termination_loss", termination_loss.item(), current_steps)
            logger.log("WorldModel/dynamics_loss", dynamics_loss.item(), current_steps)
            logger.log("WorldModel/dynamics_real_kl_div", dynamics_real_kl_div.item(), current_steps)
            logger.log("WorldModel/representation_loss", representation_loss.item(), current_steps)
            logger.log("WorldModel/representation_real_kl_div", representation_real_kl_div.item(), current_steps)
            logger.log("WorldModel/total_loss", total_loss.item(), current_steps)


if __name__ ==  "__main__":

    from experiments.robot.libero.run_libero_eval import GenerateConfig
    import pickle



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
        num_open_loop_steps=8,
        unnorm_key="libero_spatial_no_noops",
    )

    vla = get_vla(cfg)

    world_model = WorldModel(action_dim=7, image_encoder=vla.vision_backbone, language_model=vla.language_model, 
                             encode_projector=vla.projector, transformer_hidden_dim=vla.llm_dim, max_length=4)

    world_model.to(DEVICE, dtype=torch.bfloat16)
    obs = torch.randn((1, 4, 12, 224, 224)).to(DEVICE, dtype=torch.bfloat16)
    action = torch.randn((1, 4, 7)).to(DEVICE, dtype=torch.bfloat16)
    reward = torch.randn((1, 4, 1)).to(DEVICE, dtype=torch.bfloat16)
    termination = torch.randn((1, 4)).to(DEVICE, dtype=torch.bfloat16)

    world_model.update(obs, action, reward, termination)