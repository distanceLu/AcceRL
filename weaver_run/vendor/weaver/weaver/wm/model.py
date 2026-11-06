import inspect
import math
import time
from typing import Dict, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

from .decoders import ImgDecoder
from .nets import BlockCausalDynamics, MLP
from ..utils.tools import EMA

torch.set_float32_matmul_precision('high')


def _cfg_get(cfg, key: str, default):
    return getattr(cfg, key, default)


def _nested_cfg_get(cfg, key: str, default):
    if cfg is None:
        return default
    return getattr(cfg, key, default)


# ---------- Sinusoidal Time Encoder ---------- #
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ScoreBlock(nn.Module):
    def __init__(
        self,
        out_dim: int,
        n_embed: int,
        n_task_feature: int,
        n_hidden: int,
        n_im_feature: int,
        n_states: int,
        n_actions: int,
        img_keys: List[str],
        use_actions: bool = False,
        use_task: bool = True,
    ):
        super().__init__()

        self._use_actions = use_actions
        self._use_task = use_task
        self._img_keys = img_keys
        num_inputs = len(img_keys) + 1

        self.inp_prj = nn.ModuleDict({
            k: nn.Linear(n_im_feature, n_embed)
            for k in img_keys
        })
        self.inp_prj['states'] = MLP(n_states, n_hidden, n_embed)

        if self._use_actions:
            self.inp_prj['actions'] = MLP(n_actions, n_hidden, n_embed)
            num_inputs += 1
        if self._use_task:
            self.inp_prj['task'] = MLP(n_task_feature, n_hidden, n_embed)
            num_inputs += 1

        self.out_prj = MLP(n_embed * num_inputs, n_hidden, out_dim)

    def encode_obs(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        embeddings = []
        for key in self._img_keys:
            embed = self.inp_prj[key](obs[key])
            embed = reduce(embed, 'b t n d -> b t d', 'mean')
            embeddings.append(embed)

        state_embed = self.inp_prj['states'](obs['states'])
        embeddings.append(state_embed)
        return torch.cat(embeddings, dim=2)

    def forward(
        self,
        obs: dict[str, torch.Tensor],
        actions: torch.Tensor,
        task: Optional[torch.Tensor],
    ) -> torch.Tensor:
        obs_embed = self.encode_obs(obs)
        B, T, _ = actions.size()

        if self._use_actions:
            act_embed = self.inp_prj['actions'](actions)
            obs_embed = torch.cat([obs_embed, act_embed], dim=2)

        if self._use_task:
            if task is None:
                raise ValueError("ScoreBlock was configured with use_task=True but task is None.")
            task_embed = self.inp_prj['task'](task)
            task_embed = task_embed[:, None].expand(-1, T, -1)
            obs_embed = torch.cat([obs_embed, task_embed], dim=2)

        x = rearrange(obs_embed, 'b t d -> (b t) d')
        x = self.out_prj(x)
        return rearrange(x, '(b t) 1 -> b t 1', b=B)


class RewardModel(nn.Module):
    def __init__(
        self,
        img_keys: List[str],
        n_embed: int,
        n_hidden: int,
        n_im_feature: int,
        n_task_feature: int,
        n_states: int,
        n_actions: int,
    ):
        super().__init__()
        self.net = ScoreBlock(
            out_dim=1,
            n_embed=n_embed,
            n_hidden=n_hidden,
            n_task_feature=n_task_feature,
            n_im_feature=n_im_feature,
            n_states=n_states,
            n_actions=n_actions,
            img_keys=img_keys,
            use_actions=True,
            use_task=True,
        )

    def forward(
        self,
        obs: dict[str, torch.Tensor],
        actions: torch.Tensor,
        tasks: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(obs, actions, tasks).squeeze(-1)

    def compute_loss(
        self,
        obs: dict[str, torch.Tensor],
        actions: torch.Tensor,
        tasks: torch.Tensor,
        gt_rewards: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred_rewards = self(obs, actions, tasks)
        loss = 0.5 * (pred_rewards - gt_rewards).pow(2).mean()
        return pred_rewards, loss


def compute_v_lambda(
    rewards: torch.Tensor,
    values: torch.Tensor,
    discount_factor: float,
    lam: float,
) -> torch.Tensor:
    B, T = rewards.size()
    vlam = [None] * T
    vlam[-1] = rewards[:, -1] + discount_factor * values[:, -1]
    for t in reversed(range(T - 1)):
        vlam[t] = rewards[:, t] + discount_factor * (
            (1 - lam) * values[:, t] + lam * vlam[t + 1]
        )
    return torch.stack(vlam, dim=1)


class Critic(nn.Module):
    def __init__(
        self,
        img_keys: List[str],
        n_embed: int,
        n_hidden: int,
        n_im_feature: int,
        n_task_feature: int,
        n_states: int,
        n_actions: int,
        discount_factor: float = 0.99,
        lam: float = 0.95,
    ):
        super().__init__()
        self._discount_factor = discount_factor
        self._lambda = lam
        self.net = ScoreBlock(
            out_dim=1,
            n_embed=n_embed,
            n_hidden=n_hidden,
            n_im_feature=n_im_feature,
            n_task_feature=n_task_feature,
            n_states=n_states,
            n_actions=n_actions,
            img_keys=img_keys,
            use_actions=True,
            use_task=True,
        )

    def forward(
        self,
        obs: dict[str, torch.Tensor],
        actions: torch.Tensor,
        tasks: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(obs, actions, tasks).squeeze(-1)

    def compute_loss(
        self,
        obs: dict[str, torch.Tensor],
        actions: torch.Tensor,
        tasks: torch.Tensor,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        values = self(obs, actions, tasks)
        with torch.no_grad():
            target = compute_v_lambda(
                rewards=rewards[:, :-1],
                values=values[:, 1:],
                discount_factor=self._discount_factor,
                lam=self._lambda,
            )
        return 0.5 * (values[:, :-1] - target).pow(2).mean()


# ---------- World Model Core ---------- #
class FlowWM(nn.Module):
    def __init__(
        self,
        img_keys: List[str],
        n_embed: int,
        n_layers: int = 12,
        n_heads: int = 8,
        n_spatial: int = 3,
        n_hidden: int = 512,
        n_im_feature: int = 384,
        n_states: int = 7,
        n_actions: int = 7,
        qk_norm: bool = False,
        use_linear_proj: bool = False,
        # Spatial configuration - determines patch grid automatically
        image_size: tuple[int, int] = (192, 320),  # Input image size (H, W)
        spatial_size: int = 4,  # SVD spatial pooling factor (4 → 60 patches, 2 → 240 patches)
        # SPRINT configuration
        use_sprint: bool = False,
        sprint_drop_ratio: float = 0.75,
        sprint_encoder_blocks: int = 2,  # First N blocks are dense (encoder)
        sprint_decoder_blocks: int = 2,  # Last N blocks are dense (decoder)
        path_drop_prob: float = 0.05,    # Probability to drop entire sparse path
        # Memory configuration
        n_memory_frames: int = 0,
    ):
        super().__init__()

        self._img_keys = img_keys
        self._use_sprint = use_sprint
        self._path_drop_prob = path_drop_prob
        self._spatial_size = spatial_size
        self._n_memory_frames = n_memory_frames
        self._use_memory = n_memory_frames > 0
        self._n_actions = n_actions

        # Compute grid dimensions from image_size and spatial_size
        # VAE latent: image_size // 8, then SVD pools by spatial_size
        latent_h, latent_w = image_size[0] // 8, image_size[1] // 8
        grid_h, grid_w = latent_h // spatial_size, latent_w // spatial_size
        n_patches_per_img = grid_h * grid_w

        self._grid_h = grid_h
        self._grid_w = grid_w
        self._tokens_per_img = n_patches_per_img
        self._n_img_keys = len(img_keys)
        self._sprint_drop_ratio = sprint_drop_ratio

        # Token structure:
        # [patches_cam1, patches_cam2, ..., state, action, time]
        # Token counts are computed dynamically in forward() based on actual input data

        n_block_layers = n_layers // (n_spatial + 1)

        # SPRINT block assignment: first N encoder, middle sparse, last N decoder
        self._sprint_encoder_blocks = sprint_encoder_blocks
        self._sprint_decoder_blocks = sprint_decoder_blocks
        self._n_block_layers = n_block_layers

        # Build blocks - all use standard BlockCausalDynamics
        # SPRINT token dropping is handled at FlowWM level, not per-block
        # NOTE: max_grid_size is computed below (line ~275); move it here so blocks get the RoPE table
        head_dim = n_embed // n_heads
        self._head_dim = head_dim
        max_grid_size = max(grid_h, grid_w, 32)  # At least 32 to handle most cases
        self._max_grid_size = max_grid_size
        rope_grid_shape = (max_grid_size, max_grid_size)
        # Trailing state/action/time tokens do not get RoPE.
        n_unrotated = 3
        self.self_attn_blocks = nn.ModuleList()
        for i in range(n_block_layers):
            block = BlockCausalDynamics(
                n_embed, n_heads, n_spatial,
                qk_norm=qk_norm,
                rope_grid_shape=rope_grid_shape,
                max_T=256,
                n_unrotated=n_unrotated,
            )
            self.self_attn_blocks.append(block)

        # SPRINT fusion projection: combines encoder output with sparse middle output
        if use_sprint:
            self.fusion_proj = nn.Linear(2 * n_embed, n_embed, bias=True)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, n_embed))

        self.ln_f = nn.RMSNorm(n_embed, eps=1e-6)

        self.timestep_encoder = SinusoidalPosEmb(n_embed)

        # Build base rope_ids for image patch tokens only
        # Token structure: [patches_cam1, patches_cam2, ..., state, action, time]
        #
        # Only patches get RoPE. State/action/time are trailing "unrotated" tokens.
        # Each camera's patches use the same grid positions [0, 1, ..., patches_per_img-1]
        base_patch_ids = torch.arange(n_patches_per_img)  # [0, 1, ..., patches_per_img-1]

        # Repeat for each image key
        rope_ids_img = base_patch_ids.repeat(len(img_keys))  # (n_img_keys * patches_per_img,)
        self.register_buffer("_base_rope_ids", rope_ids_img)

        # Input projections for image patches (simple Linear or MLP)
        self.inp_prj = nn.ModuleDict({
            k: nn.Linear(n_im_feature, n_embed) if use_linear_proj
                else MLP(n_im_feature, n_hidden, n_embed)
            for k in img_keys
        })

        # Non-image projections (always MLP)
        self.inp_prj['states'] = MLP(n_states, n_hidden, n_embed)
        self.inp_prj['actions'] = MLP(n_actions, n_hidden, n_embed)

        # Separate memory projectors (different parameters from main projectors)
        if n_memory_frames > 0:
            self.mem_inp_prj = nn.ModuleDict({
                k: nn.Linear(n_im_feature, n_embed) if use_linear_proj
                    else MLP(n_im_feature, n_hidden, n_embed)
                for k in img_keys
            })
            self.mem_inp_prj['states'] = MLP(n_states, n_hidden, n_embed)
            self.mem_inp_prj['actions'] = MLP(n_actions, n_hidden, n_embed)
            # self.ln_mem_inp_prj = nn.RMSNorm(n_embed, eps=1e-6)

        # self.ln_inp_prj = nn.RMSNorm(n_embed, eps=1e-6)

        # Output projections for image patches (simple Linear)
        self.out_prj = nn.ModuleDict({
            k: nn.Linear(n_embed, n_im_feature, bias=False)
            for k in img_keys
        })
        self.out_prj['states'] = nn.Linear(n_embed, n_states, bias=False)

        # Apply weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self._n_block_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode_obs(
        self,
        obs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Encode observations into embeddings.

        Token structure:
            [patches_cam1, patches_cam2, ..., state]

        Args:
            obs: Dictionary with patch features per image key (B, T, N, D),
                 and 'states' (B, T, n_states).

        Returns:
            obs_embed: (B, T, N_total, n_embed)
        """
        # Patch embeddings for all cameras
        patch_embeds = torch.cat([
            self.inp_prj[key](obs[key]) for key in self._img_keys
        ], dim=2)  # (B, T, n_img_keys * N_patches, n_embed)

        # State embedding
        st_embed = self.inp_prj['states'](obs['states'])  # (B, T, n_embed)
        st_embed = rearrange(st_embed, 'b t d -> b t 1 d')  # (B, T, 1, n_embed)

        obs_embed = torch.cat([patch_embeds, st_embed], dim=2)

        return obs_embed  # removed ln_inp_prj RMSNorm (pre-norm is inside each block)

    def encode_memory(
        self,
        memory_obs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Encode memory observations using separate memory projectors.

        Token structure per memory frame:
            [patches_cam1, patches_cam2, ..., state]

        Args:
            memory_obs: Dictionary with patch features per image key (B, M, N, D),
                        'states' (B, M, n_states)

        Returns:
            mem_embed: (B, M, N_total, n_embed) where N_total = n_img_keys * N_patches + 1
        """
        # Patch embeddings for all cameras
        patch_embeds = torch.cat([
            self.mem_inp_prj[key](memory_obs[key]) for key in self._img_keys
        ], dim=2)  # (B, M, n_img_keys * N_patches, n_embed)

        # State embedding
        st_embed = self.mem_inp_prj['states'](memory_obs['states'])  # (B, M, n_embed)
        st_embed = rearrange(st_embed, 'b m d -> b m 1 d')  # (B, M, 1, n_embed)

        mem_embed = torch.cat([patch_embeds, st_embed], dim=2)

        return mem_embed  # removed ln_mem_inp_prj RMSNorm (pre-norm is inside each block)

    def decode(
        self,
        x: torch.Tensor,
        P: int,  # Patches per image
    ) -> dict[str, torch.Tensor]:
        """
        Decode WM output back to feature space.

        Token structure: [patches_cam1, patches_cam2, ..., state, action, time]

        Args:
            x: (B, T, N_total, n_embed) WM output
            P: Number of patch tokens per image

        Returns:
            Dictionary with decoded image features and states.
        """
        output = {}
        n_img_keys = len(self._img_keys)

        # Decode patches (first n_img_keys * P tokens)
        for i, key in enumerate(self._img_keys):
            start_idx = i * P
            end_idx = start_idx + P
            output[key] = self.out_prj[key](x[:, :, start_idx:end_idx, :])

        # State token: after patches, before action and time.
        # Token order: [patches..., state, action, time]
        # state is at index -3 (third from end)
        output['states'] = self.out_prj['states'](x[:, :, -3, :])
        return output

    # ==================== SPRINT Helper Methods ====================

    def _generate_sprint_ids(
        self,
        B: int,
        n_img_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Generate random indices for SPRINT token dropping.

        Drops from image patch tokens only.
        Same ids_keep across all timesteps for temporal consistency.

        Args:
            B: Batch size
            n_img_tokens: Number of image patch tokens (from actual data)
            device: Device for tensors

        Returns:
            ids_keep: (B, N_keep) or None if not training/no drop
        """
        if self._sprint_drop_ratio <= 0.0 or not self.training:
            return None

        N_keep = max(1, int(n_img_tokens * (1.0 - self._sprint_drop_ratio)))

        noise = torch.rand(B, n_img_tokens, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :N_keep]  # (B, N_keep)

        return ids_keep

    def _get_sparse_rope_ids(
        self,
        ids_keep: torch.Tensor,
        rope_ids_dense: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute rope_ids for sparse image tokens by gathering from the
        precomputed dense rope_ids.

        The dense rope_ids already encode the correct 2D grid positions
        (with per-camera modulo and RoPE grid clamping). We simply index
        into them with ids_keep to get the sparse subset.

        Args:
            ids_keep: (B, N_keep) indices into image patch sequence
            rope_ids_dense: (1, n_img_tokens) precomputed 2D RoPE grid positions

        Returns:
            rope_ids: (B, N_keep) indices into 2D RoPE grid
        """
        if ids_keep is None:
            return None

        # Gather from dense rope_ids: each kept token gets its correct 2D position
        return rope_ids_dense.squeeze(0)[ids_keep]  # (B, N_keep)

    def _gather_sparse_tokens(
        self,
        x: torch.Tensor,
        ids_keep: torch.Tensor,
        n_img_tokens: int,
    ) -> torch.Tensor:
        """
        Core sparse gather: keep only selected image tokens, preserve other tokens.

        Args:
            x: (B, N_total, D) where first n_img_tokens are image patch tokens
            ids_keep: (B, N_keep) indices into the image token range
            n_img_tokens: Number of image patch tokens

        Returns:
            (B, N_sparse, D) where N_sparse = N_keep + N_other
        """
        if ids_keep is None:
            return x

        D = x.shape[-1]
        x_img = x[:, :n_img_tokens, :]
        x_other = x[:, n_img_tokens:, :]
        ids_exp = ids_keep[:, :, None].expand(-1, -1, D)
        x_img_sparse = x_img.gather(1, ids_exp)
        return torch.cat([x_img_sparse, x_other], dim=1)

    def _drop_tokens(
        self,
        x: torch.Tensor,
        T: int,
        ids_keep: torch.Tensor,
        n_img_tokens: int,
        n_other_tokens: int,
    ) -> torch.Tensor:
        """
        Drop image patch tokens based on pre-computed ids_keep.

        Args:
            x: (B, T * N_total, D) flattened input
            T: Number of frames
            ids_keep: (B, N_keep) indices of kept image tokens
            n_img_tokens: Number of image patch tokens
            n_other_tokens: Number of non-image tokens (state, action, time)

        Returns:
            x_sparse: (B, T * N_sparse, D)
        """
        if ids_keep is None:
            return x

        B = x.shape[0]
        N_total = n_img_tokens + n_other_tokens

        # (B, T*N_total, D) -> (B*T, N_total, D)
        x = rearrange(x, 'b (t n) d -> (b t) n d', t=T, n=N_total)
        ids_keep_bt = ids_keep[:, None, :].expand(-1, T, -1).reshape(B * T, -1)
        x_sparse = self._gather_sparse_tokens(x, ids_keep_bt, n_img_tokens)
        # (B*T, N_sparse, D) -> (B, T*N_sparse, D)
        return rearrange(x_sparse, '(b t) n d -> b (t n) d', b=B)

    def _pad_tokens(
        self,
        x_sparse: torch.Tensor,
        T: int,
        ids_keep: torch.Tensor,
        n_img_tokens: int,
        n_other_tokens: int,
    ) -> torch.Tensor:
        """
        Pad sparse image tokens back to full size with [MASK] tokens.

        Args:
            x_sparse: (B, T * N_sparse, D)
            T: Number of frames
            ids_keep: (B, N_keep) indices
            n_img_tokens: Number of image patch tokens
            n_other_tokens: Number of non-image tokens (state, action, time)

        Returns:
            x_pad: (B, T * N_total, D)
        """
        if ids_keep is None:
            return x_sparse

        B, _, D = x_sparse.shape
        N_keep = ids_keep.shape[1]
        N_sparse = N_keep + n_other_tokens

        # Reshape: (B, T * N_sparse, D) -> (B, T, N_sparse, D)
        x_sparse = rearrange(x_sparse, 'b (t n) d -> b t n d', t=T, n=N_sparse)

        # Split sparse into image and other
        x_img_sparse = x_sparse[:, :, :N_keep, :]  # (B, T, N_keep, D)
        x_other = x_sparse[:, :, N_keep:, :]       # (B, T, N_other, D)

        # Create padded image tensor with [MASK] tokens
        x_img_pad = self.mask_token.view(1, 1, 1, D).expand(B, T, n_img_tokens, -1).clone()

        # Scatter sparse values back to original positions
        ids_exp = ids_keep[:, None, :, None].expand(-1, T, -1, D)
        x_img_pad.scatter_(2, ids_exp, x_img_sparse)

        # Concatenate padded images with other tokens
        x_pad = torch.cat([x_img_pad, x_other], dim=2)  # (B, T, N_total, D)

        # Flatten: (B, T, N_total, D) -> (B, T * N_total, D)
        x_pad = rearrange(x_pad, 'b t n d -> b (t n) d')

        return x_pad

    # ==================== End SPRINT Helper Methods ====================

    def _pack_tokens(self, obs, actions, t):
        """Pack obs/action/noise-level tensors into transformer tokens."""
        obs_embed = self.encode_obs(obs)
        B, T, P, _ = obs[self._img_keys[0]].size()
        act_embed = rearrange(self.inp_prj['actions'](actions), 'b t d -> b t 1 d')
        t_embed = self.timestep_encoder(rearrange(t, 'b t -> (b t)'))
        t_embed = rearrange(t_embed, '(b t) d -> b t 1 d', b=B, t=T)
        x = torch.cat([obs_embed, act_embed, t_embed], dim=2)
        return rearrange(x, 'b t n d -> b (t n) d'), T, P

    def _dense_rope_ids(self, P: int, device: torch.device) -> torch.Tensor:
        rope_grid_size = self._max_grid_size * self._max_grid_size
        base_ids = torch.arange(P, device=device) % rope_grid_size
        return base_ids.repeat(len(self._img_keys)).unsqueeze(0)

    def _cached_block(self, block, x, T, prefix_T, rope_ids, mode, cache_k, cache_v, idx):
        if mode == 'write':
            x, k, v = block.forward_write(x, T, prefix_T=prefix_T, rope_ids=rope_ids)
            cache_k[idx] = k
            cache_v[idx] = v
            return x
        return block.forward_read(
            x,
            T,
            prefix_T=prefix_T,
            kv_cache_k=cache_k[idx],
            kv_cache_v=cache_v[idx],
            rope_ids=rope_ids,
        )

    def _cached_blocks_forward(self, x, T, P, prefix_T, mode, cache_k, cache_v):
        rope_ids_dense = self._dense_rope_ids(P, x.device)

        if self._use_sprint:
            if self.training:
                raise ValueError("KV-cache SPRINT path is eval-only; token dropping is disabled in eval")
            x_enc = self._cached_block(self.self_attn_blocks[0], x, T, prefix_T, rope_ids_dense, mode, cache_k, cache_v, 0)
            for i in range(1, self._sprint_encoder_blocks):
                x_enc = self._cached_block(self.self_attn_blocks[i], x_enc, T, prefix_T, rope_ids_dense, mode, cache_k, cache_v, i)

            middle_start = self._sprint_encoder_blocks
            middle_end = len(self.self_attn_blocks) - self._sprint_decoder_blocks
            x_mid = x_enc
            for i in range(middle_start, middle_end):
                x_mid = self._cached_block(self.self_attn_blocks[i], x_mid, T, prefix_T, rope_ids_dense, mode, cache_k, cache_v, i)

            x_dec = self.fusion_proj(torch.cat([x_enc, x_mid], dim=-1))
            for i in range(middle_end, len(self.self_attn_blocks)):
                x_dec = self._cached_block(self.self_attn_blocks[i], x_dec, T, prefix_T, rope_ids_dense, mode, cache_k, cache_v, i)
            return x_dec

        x = self._cached_block(self.self_attn_blocks[0], x, T, prefix_T, rope_ids_dense, mode, cache_k, cache_v, 0)
        for i in range(1, len(self.self_attn_blocks)):
            x = self._cached_block(self.self_attn_blocks[i], x, T, prefix_T, rope_ids_dense, mode, cache_k, cache_v, i)
        return x

    @torch.no_grad()
    def forward_cached(self, obs, actions, t, memory, cache_k, cache_v, cache_meta, mode, n_history):
        """Eval-only forward that reuses prefix temporal K/V across denoising steps."""
        if self.training:
            raise ValueError("KV-cache forward is eval-only")

        M = self._n_memory_frames if self._use_memory else 0
        if mode == 'write':
            x, T, P = self._pack_tokens(obs, actions, t)
            if self._use_memory:
                x = torch.cat([memory, x], dim=1)
            T_full = T + M
            n_per_frame = x.shape[1] // T_full
            prefix_T = M + n_history
            cache_meta.update(prefix_T=prefix_T, T_full=T_full, n_per_frame=n_per_frame)
            x = self._cached_blocks_forward(x, T_full, P, prefix_T, 'write', cache_k, cache_v)
            x = self.ln_f(x)
            if self._use_memory:
                x = x[:, M * n_per_frame:]
            x = rearrange(x, 'b (t n) d -> b t n d', t=T)
            return self.decode(x, P=P)

        if mode == 'read':
            future_obs = {k: obs[k][:, n_history:] for k in self._img_keys}
            future_obs['states'] = obs['states'][:, n_history:]
            x, T_future, P = self._pack_tokens(future_obs, actions[:, n_history:], t[:, n_history:])
            prefix_T = cache_meta['prefix_T']
            if cache_meta['T_full'] != prefix_T + T_future:
                raise ValueError(
                    f"KV-cache shape mismatch: T_full={cache_meta['T_full']}, "
                    f"prefix_T={prefix_T}, T_future={T_future}"
                )
            x = self._cached_blocks_forward(x, T_future, P, prefix_T, 'read', cache_k, cache_v)
            x = self.ln_f(x)
            x = rearrange(x, 'b (t n) d -> b t n d', t=T_future)
            return self.decode(x, P=P)

        raise ValueError(f"Unknown KV-cache mode: {mode}")

    @torch.compile
    def forward(
        self,
        obs: dict[str, torch.Tensor],
        actions: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        memory: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass of FlowWM.

        Args:
            obs: Dictionary with patch features per image key (B, T, N, D),
                 and 'states' (B, T, n_states).
            actions: Action tensor (B, T, n_actions)
            t: Timestep tensor (B, T)
            cond: Optional conditioning (unused)
            memory: Optional pre-encoded memory embeddings (B, M*N_mem, n_embed),
                    already flattened along temporal and spatial dims.

        Returns:
            Dictionary with predicted image features and states.
        """
        # x = self.cross_attn_in(x, cond)
        obs_embed = self.encode_obs(obs)
        B, T, P, _ = obs[self._img_keys[0]].size()  # P = patches per image from data

        act_embed = self.inp_prj['actions'](actions)
        act_embed = rearrange(act_embed, 'b t d -> b t 1 d')

        t_embed = self.timestep_encoder(rearrange(t, 'b t -> (b t)'))  # (B*T, D)
        t_embed = rearrange(t_embed, '(b t) d -> b t 1 d', b=B, t=T)  # (B, T, 1, D)

        x = torch.cat([obs_embed, act_embed, t_embed], dim=2)  # (B, T, N+2, D)
        x = rearrange(x, 'b t n d -> b (t n) d')

        # Prepend memory tokens if memory is enabled
        M = self._n_memory_frames
        if self._use_memory:
            x = torch.cat([memory, x], dim=1)  # (B, M*N_mem + T*N, D)

        # Total temporal dimension seen by attention blocks
        T_full = T + M

        num_blocks = len(self.self_attn_blocks)

        # Compute actual token counts from data
        n_img_keys = len(self._img_keys)
        n_img_tokens = n_img_keys * P  # Actual image patch tokens from data
        # 2D RoPE for spatial attention — compute integer indices only
        rope_grid_size = self._max_grid_size * self._max_grid_size
        base_ids = torch.arange(P, device=x.device) % rope_grid_size  # Map to valid RoPE indices
        rope_ids_dense = base_ids.repeat(n_img_keys).unsqueeze(0)  # (1, n_img_tokens)

        if self._use_sprint:
            # ============================================================
            # SPRINT Forward Pass: Encoder → Sparse Middle → Fusion → Decoder
            # ============================================================

            # Generate token drop indices (same across all timesteps)
            ids_keep = self._generate_sprint_ids(B, n_img_tokens, x.device)

            # Compute sparse rope_ids for middle blocks
            if ids_keep is not None:
                rope_ids_sparse = self._get_sparse_rope_ids(ids_keep, rope_ids_dense)
            else:
                rope_ids_sparse = rope_ids_dense

            # Non-image tokens per frame: state + action + time.
            n_other_tokens = 3

            # 1) Encoder blocks (dense) - first N blocks
            x_enc = self.self_attn_blocks[0](x, T_full, rope_ids=rope_ids_dense)
            for i in range(1, self._sprint_encoder_blocks):
                block = self.self_attn_blocks[i]
                x_enc = block(
                    x_enc, T_full,
                    rope_ids=rope_ids_dense,
                )

            # 2) Drop tokens for sparse middle blocks
            x_sparse = self._drop_tokens(x_enc, T_full, ids_keep, n_img_tokens, n_other_tokens)

            # 3) Middle blocks (sparse) - process dropped tokens
            x_mid = x_sparse
            middle_start = self._sprint_encoder_blocks
            middle_end = num_blocks - self._sprint_decoder_blocks
            for i in range(middle_start, middle_end):
                block = self.self_attn_blocks[i]
                x_mid = block(
                    x_mid, T_full,
                    rope_ids=rope_ids_sparse,
                )

            # 4) Pad sparse output back to full size
            x_mid_padded = self._pad_tokens(x_mid, T_full, ids_keep, n_img_tokens, n_other_tokens)

            # 5) Path-drop learning: occasionally zero out middle contribution
            if self.training and self._path_drop_prob > 0.0:
                drop_path = torch.rand(1, device=x.device)
                drop_mask = (drop_path < self._path_drop_prob).float()
                x_mid_padded = drop_mask * self.mask_token.expand_as(x_mid_padded) + (1.0 - drop_mask) * x_mid_padded

            # 6) Sparse-dense fusion: combine encoder output with padded middle output
            x_fused = torch.cat([x_enc, x_mid_padded], dim=-1)  # (B, T_full*N, 2D)
            x_fused = self.fusion_proj(x_fused)                  # (B, T_full*N, D)

            # 7) Decoder blocks (dense) - last N blocks
            x_dec = x_fused
            for i in range(middle_end, num_blocks):
                block = self.self_attn_blocks[i]
                x_dec = block(
                    x_dec, T_full,
                    rope_ids=rope_ids_dense,
                )

            x = x_dec

        else:
            # ============================================================
            # Standard Forward Pass (no SPRINT)
            # ============================================================
            x = self.self_attn_blocks[0](x, T_full, rope_ids=rope_ids_dense)
            for i in range(1, len(self.self_attn_blocks)):
                block = self.self_attn_blocks[i]
                x = block(
                    x, T_full,
                    rope_ids=rope_ids_dense,
                )

        x = self.ln_f(x)

        # Strip memory prefix after transformer, decode only main frames
        if self._use_memory:
            N_per_frame = x.shape[1] // T_full
            x = x[:, M * N_per_frame:]  # Remove memory tokens

        x = rearrange(x, 'b (t n) d -> b t n d', t=T)

        return self.decode(x, P=P)

class WEAVER(nn.Module):
    def __init__(
        self,
        img_keys: List[str],
        im_encoder: nn.Module,
        train_decoder: bool,
        task_encoder: nn.Module,
        n_history: int,
        n_horizon: int,
        config: Dict,
        n_states: int = 7,
        n_actions: int = 7,
        use_precomputed_features: int = False,
        image_size: tuple = (192, 320),
        device: str = 'cpu',
        n_memory_frames: int = 0,
        t_memory: int = 1,
        inference_config: Optional[Dict] = None,
    ):

        super().__init__()

        self.config = config
        self._img_keys = img_keys
        self._n_memory_frames = n_memory_frames
        self._t_memory = t_memory
        self._use_memory = n_memory_frames > 0

        self._train_diffusion_steps = config.train_steps
        self._inference_steps = config.val_steps
        self._flow_loss = config.loss_target
        self._diff_forcing = config.diff_forcing
        inf_cfg = inference_config
        self._pyramid_schedule_type = _nested_cfg_get(inf_cfg, 'pyramid_schedule', 'linear')
        self._pyramid_stagger_width = _nested_cfg_get(inf_cfg, 'pyramid_stagger_width', 1)
        self._pyramid_schedule_kwargs = {
            'power': _nested_cfg_get(inf_cfg, 'pyramid_schedule_power', 0.5),
            'steepness': _nested_cfg_get(inf_cfg, 'pyramid_schedule_steepness', 5.0),
            'center': _nested_cfg_get(inf_cfg, 'pyramid_schedule_center', 0.5),
        }
        self._use_precomputed_features = use_precomputed_features
        self._use_temporal_loss = _cfg_get(config, 'use_temporal_loss', False)
        self._temporal_loss_coeff = _cfg_get(config, 'temporal_loss_coeff', 0.1)
        self._state_loss_scale = _nested_cfg_get(_cfg_get(config, 'loss_scale', None), 'states', 0.1)
        self._rm_loss_coeff = _cfg_get(config, 'rm_loss_coeff', 1.0)
        self._critic_loss_coeff = _cfg_get(config, 'critic_loss_coeff', 1.0)

        # assert self._flow_loss in ['x-pred', 'v-pred'], "flow_loss must be either 'x-pred' or 'v-pred'"

        self.im_encoder = im_encoder
        self.task_encoder = task_encoder

        # Patchify image input
        self._n_embed = config.n_embed

        # Parameters for diffusion
        self._train_steps = config.train_steps
        self._val_steps = config.val_steps
        self._n_history = n_history
        self._n_horizon = n_horizon
        self._train_decoder = train_decoder
        self._pyramid_schedule_cache = {}  # {(horizon, stagger_width): schedule tensor on device}
        self._history_noise_std = _cfg_get(config, 'history_noise_std', 0.0)

        # Core PerceiverIO
        # self.vla = VLA(
        #     img_keys=img_keys,
        #     n_embed=config.n_embed,
        #     n_layers=config.n_layers,
        #     n_heads=config.n_heads,
        #     n_spatial=config.n_spatial,
        #     n_im_feature=self.im_encoder.feature_dim,
        #     n_states=n_states,
        #     n_actions=n_actions
        # )

        # Get spatial_size from encoder (determines patch grid automatically)
        self._spatial_size = self.im_encoder.spatial_size

        # Store grid dimensions for feature reshaping
        latent_h, latent_w = image_size[0] // 8, image_size[1] // 8
        self._grid_h = latent_h // self._spatial_size
        self._grid_w = latent_w // self._spatial_size

        self.wm = FlowWM(
            img_keys=img_keys,
            n_embed=config.n_embed,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            n_spatial=config.n_spatial,
            n_im_feature=self.im_encoder.feature_dim,
            n_states=n_states,
            n_actions=n_actions,
            qk_norm=config.qk_norm,
            use_linear_proj=config.use_linear_proj,
            # Spatial configuration (from encoder)
            image_size=image_size,
            spatial_size=self._spatial_size,
            # SPRINT configuration
            use_sprint=_cfg_get(config, 'use_sprint', False),
            sprint_drop_ratio=_cfg_get(config, 'sprint_drop_ratio', 0.0),
            sprint_encoder_blocks=_cfg_get(config, 'sprint_encoder_blocks', 2),
            sprint_decoder_blocks=_cfg_get(config, 'sprint_decoder_blocks', 2),
            path_drop_prob=_cfg_get(config, 'path_drop_prob', 0.0),
            # Memory configuration
            n_memory_frames=n_memory_frames,
        )

        self.rm = RewardModel(
            img_keys=img_keys,
            n_embed=config.n_embed,
            n_hidden=config.n_hidden,
            n_im_feature=self.im_encoder.feature_dim,
            n_task_feature=self.task_encoder.feature_dim,
            n_states=n_states,
            n_actions=n_actions,
        )

        self.critic = Critic(
            img_keys=img_keys,
            n_embed=config.n_embed,
            n_hidden=config.n_hidden,
            n_im_feature=self.im_encoder.feature_dim,
            n_task_feature=self.task_encoder.feature_dim,
            n_states=n_states,
            n_actions=n_actions,
            discount_factor=_cfg_get(config, 'discount_factor', 0.995),
            lam=_cfg_get(config, 'gae_lambda', 0.95),
        )

        # Decoder to pixel outputs.
        if self._train_decoder:
            print('Adding deocders for visualization...')
            self.decoders = nn.ModuleDict({
                k: ImgDecoder() for k in self._img_keys
            })

        self.ema = EMA(self, beta=0.9999)

        self.device = device

    def _reshape_features(
        self,
        features: torch.Tensor,
        target_spatial_size: int,
    ) -> torch.Tensor:
        """
        Reshape precomputed features to match target spatial_size.

        Precomputed features have shape (B, T, N_src, D_src) where:
        - D_src = 4 * src_spatial_size^2
        - N_src = grid_h_src * grid_w_src

        We reshape to (B, T, N_tgt, D_tgt) where:
        - D_tgt = 4 * target_spatial_size^2
        - N_tgt depends on the ratio of spatial sizes

        Args:
            features: (B, T, N, D) precomputed features
            target_spatial_size: desired spatial pooling factor

        Returns:
            Reshaped features (B, T, N_new, D_new)
        """
        B, T, N, D = features.shape

        # Infer source spatial_size from feature dim: D = 4 * s^2 → s = sqrt(D/4)
        src_spatial_size = int(math.sqrt(D / 16))

        if src_spatial_size == target_spatial_size:
            return features  # No reshape needed

        # Compute source grid dimensions from latent size
        # latent_h/w = image_size // 8, grid_h/w = latent_h/w // spatial_size
        # self._grid_h/w are computed with target_spatial_size, so we need to
        # compute source grid dimensions from the underlying latent dimensions
        latent_h = self._grid_h * self._spatial_size  # Recover latent_h
        latent_w = self._grid_w * self._spatial_size  # Recover latent_w
        src_grid_h = latent_h // src_spatial_size
        src_grid_w = latent_w // src_spatial_size

        # Reshape from (B, T, N, D) back to spatial form
        # D = 4 * s^2, so we can view as (B, T, grid_h, grid_w, 4, s, s)
        features = rearrange(
            features,
            'b t (h w) (c s1 s2) -> b t (h s1) (w s2) c',
            h=src_grid_h, w=src_grid_w,
            s1=src_spatial_size, s2=src_spatial_size, c=16
        )  # (B, T, latent_h, latent_w, 4)

        # Now re-pool with target spatial_size
        features = rearrange(
            features,
            'b t (h s1) (w s2) c -> b t (h w) (c s1 s2)',
            s1=target_spatial_size, s2=target_spatial_size
        )  # (B, T, N_new, D_new)

        return features

    def encode_ins(self, instructions: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Encode instructions
        ins_emb = self.task_encoder(instructions)
        ins_emb['embeddings'] = self.ins_prj(ins_emb['embeddings'])  # (B, L_ins, D)
        return ins_emb

    def encode_obs(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Encode images
        obs_embed = {}

        for key in self._img_keys:
            if self._use_precomputed_features:
                feat = obs[f'{key}_features']
                # Reshape if precomputed features have different spatial_size
                expected_dim = 4 * self._spatial_size ** 2
                if feat.shape[-1] != expected_dim:
                    feat = self._reshape_features(feat, self._spatial_size)
                obs_embed[key] = feat
            else:
                obs_embed[key] = self.im_encoder(obs[key])  # (B, T, N, D)

        obs_embed['states'] = obs['states']

        return obs_embed

    def encode_memory_obs(
        self,
        memory_obs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Encode memory observations and assemble into flattened memory tokens.

        Uses FlowWM's separate memory projectors to encode, then adds zero-action
        and t=1 embeddings to match main frame token structure.

        Args:
            memory_obs: Dictionary with features per image key (B, M, N, D),
                        'states' (B, M, n_states)

        Returns:
            memory_tokens: (B, M * N_total, n_embed) flattened memory embeddings,
                           where N_total matches the per-frame token count of main observations
        """
        # Prepare memory obs dict with precomputed features
        mem_obs = {}
        for key in self._img_keys:
            if self._use_precomputed_features:
                feat = memory_obs[f'{key}_features']
                expected_dim = 4 * self._spatial_size ** 2
                if feat.shape[-1] != expected_dim:
                    feat = self._reshape_features(feat, self._spatial_size)
                mem_obs[key] = feat
            else:
                mem_obs[key] = self.im_encoder(memory_obs[key])  # (B, M, N, D)
        mem_obs['states'] = memory_obs['states']

        # Encode with separate memory projectors
        # mem_embed: (B, M, N_obs, n_embed) where N_obs = n_img_keys * P + 1 (patches + state)
        mem_embed = self.wm.encode_memory(mem_obs)

        B, M = mem_embed.shape[:2]
        device = mem_embed.device

        # Zero actions for memory frames
        zero_actions = torch.zeros(B, M, self.wm._n_actions, device=device, dtype=mem_embed.dtype)
        act_embed = self.wm.mem_inp_prj['actions'](zero_actions)  # (B, M, n_embed)
        act_embed = rearrange(act_embed, 'b m d -> b m 1 d')

        # t=1 embedding for memory (fully clean)
        t_ones = torch.ones(B * M, device=device)
        t_embed = self.wm.timestep_encoder(t_ones)  # (B*M, n_embed)
        t_embed = rearrange(t_embed, '(b m) d -> b m 1 d', b=B, m=M)

        # Assemble: [obs_embed, action, time] per memory frame
        mem_tokens = torch.cat([mem_embed, act_embed, t_embed], dim=2)  # (B, M, N_total, n_embed)

        # Flatten temporal and spatial dims
        # Return an owning tensor, not a view. AOTAutograd can otherwise try to
        # replay stale view metadata when the compiled outer model sees memory
        # tensors, producing invalid batch-shape aliases.
        mem_tokens = rearrange(mem_tokens, 'b m n d -> b (m n) d').contiguous()

        return mem_tokens

    def encode_task(self, task: Dict[str, torch.Tensor]) -> torch.Tensor:
        if 'features' in task:
            task_embed = task['features']
        else:
            # Encode text on the fly (e.g. RoboArena without precomputed text features)
            text_list = task['text'] if isinstance(task['text'], (list, tuple)) else list(task['text'])
            task_embed = self.task_encoder(text_list)
        return task_embed

    def decode_obs(
        self,
        x: Dict[str, torch.Tensor],
        chunk_size: int = 0,
    ) -> Dict[str, torch.Tensor]:
        # Decode the predicted latents to images and states
        output = {}
        # Decode images
        for key in self._img_keys:
            B, T = x[key].shape[:2]
            inp = rearrange(x[key], 'b t n d -> (b t) n d')  # (B*T, N, D)
            BT = inp.shape[0]

            if chunk_size > 0 and BT > chunk_size:
                # Decode in chunks to avoid OOM
                decoded_chunks = []
                for i in range(0, BT, chunk_size):
                    chunk = inp[i:i + chunk_size]
                    if self._train_decoder:
                        decoded_chunks.append(self.decoders[key](chunk))
                    else:
                        decoded_chunks.append(self.im_encoder.decode(chunk))
                decoded_im = torch.cat(decoded_chunks, dim=0)
            else:
                if self._train_decoder:
                    decoded_im = self.decoders[key](inp)  # (B*T, C, H, W)
                else:
                    decoded_im = self.im_encoder.decode(inp)
            output[key] = rearrange(decoded_im, '(b t) c h w -> b t c h w', b=B)

        output['states'] = x['states']

        return output

    def generate_latent_rollouts(
        self,
        x1: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        memory: Optional[Dict[str, torch.Tensor]] = None,
        memory_tokens: Optional[torch.Tensor] = None,
    ):
        B, T, _ = actions.size()

        # Encode memory if memory is enabled and pre-encoded tokens not provided
        if self._use_memory and memory_tokens is None:
            memory_tokens = self.encode_memory_obs(memory)

        x0 = self.sample_noise(x1)

        if self._diff_forcing:
            return self._generate_latent_rollouts_autoregressive(
                x1, x0, actions, memory_tokens,
            )
        else:
            return self._generate_latent_rollouts_lockstep(
                x1, x0, actions, memory_tokens,
            )

    @torch.no_grad()
    def generate_latent_rollouts_cached(
        self,
        x1: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        memory: Optional[Dict[str, torch.Tensor]] = None,
        memory_tokens: Optional[torch.Tensor] = None,
    ):
        """KV-cache eval variant of generate_latent_rollouts.

        The prefix is memory + history frames. It is clean/fixed for all denoising
        steps in a chunk, so temporal K/V for that prefix can be cached after the
        first model call and reused for future-only reads.
        """
        if self.training or self.wm.training:
            raise ValueError("KV-cache generation requires model.eval()")

        if self._use_memory and memory_tokens is None:
            memory_tokens = self.encode_memory_obs(memory)

        B, T, _ = actions.size()
        n_hist = self._n_history
        horizon = T - n_hist
        x0 = self.sample_noise(x1)
        xt = {k: torch.cat([x1[k][:, :n_hist], x0[k][:, n_hist:]], dim=1) for k in x1}

        cache_k = [None] * len(self.wm.self_attn_blocks)
        cache_v = [None] * len(self.wm.self_attn_blocks)
        cache_meta = {}

        def euler_step(key, pred_future, t_future, dt, active_mask=None):
            xt_future = xt[key][:, n_hist:]
            if self._flow_loss.startswith('x-pred'):
                t_val = t_future[:, :, None, None] if key in self._img_keys else t_future[:, :, None]
                v = (pred_future - xt_future) / (1 - t_val).clamp(min=1e-2)
            elif self._flow_loss.startswith('v-pred'):
                v = pred_future
            else:
                raise NotImplementedError
            if active_mask is None:
                xt[key][:, n_hist:] = xt_future + v * dt
            else:
                xt[key][:, n_hist:] = xt_future + active_mask * v * dt

        if not self._diff_forcing:
            dt = 1.0 / self._inference_steps
            t = torch.cat([
                torch.ones((B, n_hist), device=self.device),
                torch.zeros((B, horizon), device=self.device),
            ], dim=1).float()

            for step in range(self._inference_steps):
                mode = 'write' if step == 0 else 'read'
                x_pred = self.wm.forward_cached(
                    xt, actions, t, memory_tokens, cache_k, cache_v, cache_meta, mode, n_hist
                )
                for key in x_pred:
                    pred_future = x_pred[key][:, n_hist:] if mode == 'write' else x_pred[key]
                    euler_step(key, pred_future, t[:, n_hist:], dt)
                t[:, n_hist:] += dt
            return xt

        cache_key = (horizon, self._pyramid_stagger_width)
        if cache_key not in self._pyramid_schedule_cache:
            self._pyramid_schedule_cache[cache_key] = self._build_pyramid_schedule(horizon).to(self.device)
        schedule = self._pyramid_schedule_cache[cache_key]

        wrote_cache = False
        for m in range(schedule.shape[0] - 1):
            t_row = schedule[m]
            t_next = schedule[m + 1]
            dt_row = t_next - t_row
            t_future = t_row.unsqueeze(0).expand(B, -1)
            t = torch.cat([torch.ones((B, n_hist), device=self.device), t_future], dim=1).float()

            mode = 'write' if not wrote_cache else 'read'
            x_pred = self.wm.forward_cached(
                xt, actions, t, memory_tokens, cache_k, cache_v, cache_meta, mode, n_hist
            )
            wrote_cache = True

            active = (t_row != t_next)
            if not active.any():
                continue

            for key in x_pred:
                pred_future = x_pred[key][:, n_hist:] if mode == 'write' else x_pred[key]
                if key in self._img_keys:
                    dt = dt_row.view(1, horizon, 1, 1)
                    mask = active.view(1, horizon, 1, 1).float()
                else:
                    dt = dt_row.view(1, horizon, 1)
                    mask = active.view(1, horizon, 1).float()
                euler_step(key, pred_future, t_future, dt, mask)

        return xt

    def generate_latent_rollouts_variable_horizon(
        self,
        context: Dict[str, torch.Tensor],
        future_actions: torch.Tensor,
        future_states: torch.Tensor,
        history_actions: Optional[torch.Tensor] = None,
        memory_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Generate rollouts for a variable number of future frames (no padding).

        Args:
            context: Encoded context dict with keys in self._img_keys + 'states',
                     each with shape (B, n_history, ...).
            future_actions: (B, Hf, n_actions) normalized actions to execute.
            future_states: (B, Hf, n_states) normalized future state placeholders/targets.
            history_actions: Optional history action window (B, n_history, n_actions).
                             If None, zeros are used.
            memory_tokens: Optional pre-encoded memory tokens.

        Returns:
            xt_pred dict with shape (B, n_history + Hf, ...).
        """
        B, Hf, _ = future_actions.shape
        if Hf <= 0:
            # Nothing to predict; return context unchanged.
            return context

        x1_chunk = {}
        for k in self._img_keys:
            future_zero = torch.zeros(
                B, Hf, *context[k].shape[2:],
                device=context[k].device,
                dtype=context[k].dtype,
            )
            x1_chunk[k] = torch.cat([context[k], future_zero], dim=1)
        x1_chunk["states"] = torch.cat([context["states"], future_states], dim=1)

        if history_actions is None:
            hist_actions = torch.zeros(
                B, self._n_history, self.wm._n_actions,
                device=future_actions.device,
                dtype=future_actions.dtype,
            )
        else:
            hist_actions = history_actions.to(device=future_actions.device, dtype=future_actions.dtype)
        actions_chunk = torch.cat([hist_actions, future_actions], dim=1)

        return self.generate_latent_rollouts(
            x1_chunk,
            actions_chunk,
            memory_tokens=memory_tokens,
        )

    def _generate_latent_rollouts_lockstep(
        self,
        x1: Dict[str, torch.Tensor],
        x0: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        memory_tokens: Optional[torch.Tensor] = None,
    ):
        """Original lockstep inference: all future frames denoised together
        with the same timestep schedule. Used when diff_forcing=False."""
        B, T, _ = actions.size()

        xt = {}
        for k in x1:
            xt[k] = torch.cat([x1[k][:, :self._n_history], x0[k][:, self._n_history:]], dim=1)

        dt = 1. / self._inference_steps

        t_history = torch.ones(
            (B, self._n_history),
            device=self.device
        )
        t_future = torch.zeros(
            (B, T - self._n_history),
            device=self.device
        )
        t = torch.cat([t_history, t_future], dim=1).float()  # (B, T)
        mask = (t < 1.).float()  # (B, T)

        for step in range(self._inference_steps):
            x_pred = self.wm(xt, actions, t, memory=memory_tokens)

            for k in x_pred:
                if k in self._img_keys:
                    mask_ = rearrange(mask, 'b t -> b t 1 1')
                else:
                    mask_ = rearrange(mask, 'b t -> b t 1')

                if self._flow_loss.startswith('x-pred'):
                    t_ = t.view(mask_.size())
                    denom = (1 - t_).clamp(min=1e-2)
                    v = (x_pred[k] - xt[k]) * denom**-1
                    xt[k] += mask_ * v * dt
                elif self._flow_loss.startswith('v-pred'):
                    xt[k] += mask_ * x_pred[k] * dt
                else:
                    raise NotImplementedError

            t[:, self._n_history:] += dt

        return xt

    def _build_pyramid_schedule(self, horizon: int) -> torch.Tensor:
        """Build a pyramid scheduling matrix for flow-matching diffusion forcing.

        Returns a (height, horizon) tensor of t-values in [0, 1] where t=0 is
        pure noise and t=1 is clean data.  Each row is one denoising step; each
        column is a future frame.  Earlier frames reach t=1 sooner than later
        ones (the "pyramid" shape), so they become clean context for frames that
        are still being denoised.

        The stagger width ``w`` (``pyramid_stagger_width``) controls the
        delay between successive frames.  All frames use the same step size
        ``dt = 1/S``, but frame f starts denoising ``w*f`` steps after frame 0.
        Earlier frames reach t=1 sooner, becoming clean context for later ones.

        Total rows (forward passes) = S + w * (horizon - 1).
        With w=0 this is pure lockstep (all frames share the same schedule).
        With w=1 (default) frame f finishes f steps after frame 0.

        Schedule type (``pyramid_schedule``) controls the non-linear mapping
        from normalized progress to t-value:

        - ``linear``: uniform steps (original baseline)
        - ``cosine``: large steps in noisy regime, fine steps near clean
        - ``power``: power < 1 → more steps near clean; power > 1 → more near noisy
        - ``sigmoid``: concentrate steps around a configurable center point
        """
        S = self._inference_steps
        w = self._pyramid_stagger_width
        height = S + w * (horizon - 1) + 1
        schedule = self._pyramid_schedule_type
        kwargs = self._pyramid_schedule_kwargs

        # All frames use the same step size dt = 1/S, but frame f is delayed
        # by w*f steps.  At row m, frame f has completed max(0, m - w*f) steps
        # out of S.  Progress u_f(m) = clamp(1 - (m - w*f) / S, 0, 1).
        rows = torch.arange(height).unsqueeze(1).float()   # (height, 1)
        cols = torch.arange(horizon).unsqueeze(0).float()   # (1, horizon)
        delay = w * cols                                     # (1, horizon)
        u = (1.0 - (rows - delay) / S).clamp(0.0, 1.0)     # (height, horizon)

        if schedule == 'linear':
            t_schedule = 1.0 - u
        elif schedule == 'cosine':
            t_schedule = torch.cos(u * (math.pi / 2))
        elif schedule == 'power':
            power = kwargs.get('power', 0.5)
            t_schedule = (1.0 - u) ** power
        elif schedule == 'sigmoid':
            steepness = kwargs.get('steepness', 5.0)
            center = kwargs.get('center', 0.5)
            raw = torch.sigmoid(steepness * (u - center))
            raw_min = torch.sigmoid(torch.tensor(steepness * (0.0 - center)))
            raw_max = torch.sigmoid(torch.tensor(steepness * (1.0 - center)))
            t_schedule = 1.0 - (raw - raw_min) / (raw_max - raw_min)
        else:
            raise ValueError(f"Unknown pyramid schedule: {schedule}")

        return t_schedule

    def _generate_latent_rollouts_autoregressive(
        self,
        x1: Dict[str, torch.Tensor],
        x0: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        memory_tokens: Optional[torch.Tensor] = None,
    ):
        """Diffusion forcing inference with pyramid scheduling.

        Generates future frames using a pyramid schedule where earlier frames
        denoise faster and become clean context for later frames.  This is the
        flow-matching analogue of the DDIM pyramid scheduling in the reference
        world-model-eval codebase.

        For each chunk of ``horizon`` future frames:
        1. Build a pyramid schedule of shape (S + horizon, horizon).
        2. Iterate over consecutive row pairs (t_row, t_next_row).
        3. At each step, run the model on all frames (context + chunk).
        4. Take an Euler step only on frames whose t value changes.
        5. When a frame reaches t=1 (clean) it becomes context.
        """
        B, T, _ = actions.size()
        n_hist = self._n_history
        horizon = T - n_hist

        # Initialize xt: history frames are clean, future frames start as noise
        xt = {}
        for k in x1:
            xt[k] = torch.cat([
                x1[k][:, :n_hist],
                x0[k][:, n_hist:],
            ], dim=1)

        # Build (or retrieve cached) pyramid schedule
        cache_key = (horizon, self._pyramid_stagger_width)
        if cache_key not in self._pyramid_schedule_cache:
            self._pyramid_schedule_cache[cache_key] = (
                self._build_pyramid_schedule(horizon).to(self.device)
            )
        schedule = self._pyramid_schedule_cache[cache_key]

        for m in range(schedule.shape[0] - 1):
            t_row = schedule[m]           # (horizon,)
            t_next_row = schedule[m + 1]  # (horizon,)

            # Per-frame step size (non-uniform for nonlinear schedules)
            dt_row = t_next_row - t_row   # (horizon,)

            # Build full t vector: history at t=1 (clean), future from schedule
            t_hist = torch.ones((B, n_hist), device=self.device)
            t_future = t_row.unsqueeze(0).expand(B, -1)    # (B, horizon)
            t = torch.cat([t_hist, t_future], dim=1)        # (B, T)

            x_pred = self.wm(xt, actions, t, memory=memory_tokens)

            # Compute which future frames actually change this step
            # (frames already at t=1 or whose t doesn't change are skipped)
            active = (t_row != t_next_row)  # (horizon,) bool

            if not active.any():
                continue

            for k in x_pred:
                if k in self._img_keys:
                    active_mask = active.view(1, horizon, 1, 1).float()
                    dt_k = dt_row.view(1, horizon, 1, 1)
                else:
                    active_mask = active.view(1, horizon, 1).float()
                    dt_k = dt_row.view(1, horizon, 1)

                pred_future = x_pred[k][:, n_hist:]  # (B, horizon, ...)
                xt_future = xt[k][:, n_hist:]

                if self._flow_loss.startswith('x-pred'):
                    if k in self._img_keys:
                        t_val = t_future[:, :, None, None]
                    else:
                        t_val = t_future[:, :, None]
                    denom = (1 - t_val).clamp(min=1e-2)
                    v = (pred_future - xt_future) / denom
                    xt[k][:, n_hist:] = xt_future + active_mask * v * dt_k
                elif self._flow_loss.startswith('v-pred'):
                    xt[k][:, n_hist:] = xt_future + active_mask * pred_future * dt_k
                else:
                    raise NotImplementedError

        return xt

    @torch.no_grad()
    def generate_videos(
        self,
        x1: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        instructions: Dict[str, torch.Tensor],
        encode: bool = False,
        memory: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # Inference function to generate predictions
        if encode:
            x1 = self.encode_obs(x1)

        with self.ema.use_ema_weights():
            xt = self.generate_latent_rollouts(
                x1,
                actions,
                memory=memory,
            )

        decoded_obs = self.decode_obs(x1)  # Dict of (B, T, C, H, W)
        decoded_obs_pred = self.decode_obs(xt) # Dict of (B, T, C, H, W)

        return decoded_obs, decoded_obs_pred, xt

    def _truncate_outputs(self, decoded_obs, decoded_obs_pred, bootstrap):
        """Truncate time dimension to bootstrap steps if specified."""
        if bootstrap is not None:
            for k in decoded_obs:
                decoded_obs[k] = decoded_obs[k][:, :bootstrap]
                decoded_obs_pred[k] = decoded_obs_pred[k][:, :bootstrap]
        return decoded_obs, decoded_obs_pred

    @torch.no_grad()
    def generate_videos_full(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        instructions: Dict[str, torch.Tensor],
        horizon: int = None,
        memory: Optional[Dict[str, torch.Tensor]] = None,
        bootstrap: int = None,
        use_kv_cache: bool = False,
    ) -> tuple[Optional[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        """Generate full video predictions by autoregressively chunking over the sequence.

        The window advances by `horizon` frames each step. If `bootstrap` is
        set, only the first `bootstrap` predicted frames per chunk are kept,
        discarding less-reliable frames far from real context. Memory is
        updated using `t_memory` spacing.
        """
        T_total = actions.shape[1]
        horizon = horizon or self._n_horizon
        n_hist = self._n_history
        bootstrap = bootstrap or horizon

        # Multi-chunk autoregressive generation
        all_xt_chunks = []   # predicted latents per chunk
        xt_chunk, x1_chunk = None, None

        # Pre-encode memory once; updated after each chunk with predicted latents
        memory_tokens = None
        n_memory = self._n_memory_frames if self._use_memory else 0
        if self._use_memory and memory is not None:
            memory_tokens = self.encode_memory_obs(memory)

        with self.ema.use_ema_weights():
            for chunk_idx, t_start in enumerate(range(n_hist, T_total, bootstrap)):
                if chunk_idx == 0:
                    # First chunk: encode real observations, zero out future
                    obs_chunk = {k: v[:, :n_hist + horizon] for k, v in obs.items()}
                    x1_chunk = self.encode_obs(obs_chunk)
                    for k in x1_chunk:
                        x1_chunk[k][:, n_hist:] *= 0
                    actions_chunk = actions[:, :n_hist + horizon]
                else:
                    # Subsequent chunks: use last n_hist frames from kept portion
                    chunk_start = t_start - n_hist
                    actions_chunk = actions[:, chunk_start:chunk_start + n_hist + horizon]
                    actual_len = actions_chunk.shape[1]

                    # Not enough actions for a full chunk
                    if actual_len < n_hist + horizon:
                        break

                    for k in xt_chunk:
                        x1_chunk[k][:, :n_hist] = xt_chunk[k][:, -n_hist:]
                        x1_chunk[k][:, n_hist:] *= 0

                if use_kv_cache:
                    xt_chunk = self.generate_latent_rollouts_cached(
                        x1_chunk, actions_chunk, memory_tokens=memory_tokens,
                    )
                else:
                    xt_chunk = self.generate_latent_rollouts(
                        x1_chunk, actions_chunk, memory_tokens=memory_tokens,
                    )

                # Save predicted latent slices for later decoding (clone to avoid
                # corruption when x1_chunk is modified in-place next iteration)
                if chunk_idx == 0:
                    all_xt_chunks.append({k: v[:, :n_hist + bootstrap].clone() for k, v in xt_chunk.items()})
                else:
                    all_xt_chunks.append({k: v[:, n_hist:n_hist + bootstrap].clone() for k, v in xt_chunk.items()})

                # Trim xt_chunk for carry-forward to next chunk
                for k in xt_chunk:
                    xt_chunk[k] = xt_chunk[k][:, :n_hist + bootstrap]

                # Update memory: drop oldest frame, append the frame at t
                if memory_tokens is not None and n_memory > 0:
                    N_per_frame = memory_tokens.shape[1] // n_memory
                    new_frame = {k: xt_chunk[k][:, n_hist - 1:n_hist] for k in xt_chunk}
                    new_embed = self.wm.encode_memory(new_frame)  # (B, 1, N_obs, n_embed)
                    B_m = new_embed.shape[0]
                    dev, dt = new_embed.device, new_embed.dtype

                    zero_act = torch.zeros(B_m, 1, self.wm._n_actions, device=dev, dtype=dt)
                    act_emb = rearrange(self.wm.mem_inp_prj['actions'](zero_act), 'b m d -> b m 1 d')
                    t_emb = rearrange(self.wm.timestep_encoder(torch.ones(B_m, device=dev)), 'b d -> b 1 1 d')

                    new_tokens = torch.cat([new_embed, act_emb, t_emb], dim=2)
                    new_tokens = rearrange(new_tokens, 'b m n d -> b (m n) d')
                    memory_tokens = torch.cat([
                        memory_tokens[:, N_per_frame:], new_tokens
                    ], dim=1)
                

        # Concatenate all latent chunks once, then decode to pixels once
        keys = all_xt_chunks[0].keys()
        all_xt = {k: torch.cat([c[k] for c in all_xt_chunks], dim=1) for k in keys}

        all_decoded_obs_pred = self.decode_obs(all_xt, chunk_size=16)

        # GT decoding is intentionally skipped in full-video generation to reduce
        # compute/memory; callers should use raw observation frames for GT.
        return None, all_decoded_obs_pred

    def generate_videos_full_memory(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        instructions: Dict[str, torch.Tensor],
        horizon: int = None,
        start_frame: int = None,
        bootstrap: int = None,
    ) -> tuple[Optional[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        """Autoregressive video generation with correct t_memory-spaced memory.

        obs and actions must cover the full trajectory from frame 0 onward (as
        loaded by load_trajectory).  start_frame is the last context frame; the
        model predicts frames start_frame+1, start_frame+2, ...

        Memory is constructed entirely from the frame buffer — no separate memory
        dict is needed.  Before each chunk the buffer is queried at exact
        t_memory-spaced positions relative to the current prediction head:
            head - t_memory,  head - 2*t_memory,  ...,  head - n_memory*t_memory
        Any query before frame 0 falls back to frame 0 (same clamping as droid.py
        for short trajectories).

        The returned decoded predictions start at start_frame+1 (no history
        prefix), aligning directly with the GT window returned by load_trajectory.

        Example (t_memory=4, bootstrap=5, n_hist=2, n_memory=3, start_frame=1):
          head=2:  memory → [buf[-10], buf[-6], buf[-2]] → all clamp to buf[0]
          head=7:  memory → [buf[-5],  buf[-1], buf[3]]  → clamp, clamp, predicted
          head=12: memory → [buf[0],   buf[4],  buf[8]]  → all predicted frames
        """
        init_time = time.time()
        T_total = actions.shape[1]
        horizon = horizon or self._n_horizon
        n_hist = self._n_history
        bootstrap = bootstrap or horizon
        t_memory = self._t_memory
        n_memory = self._n_memory_frames if self._use_memory else 0
        # import ipdb; ipdb.set_trace()

        # start_frame: last history frame (prediction starts at start_frame+1).
        # Default to n_hist-1 so behaviour is identical to generate_videos_full
        # when the caller does not supply start_frame.
        if start_frame is None:
            start_frame = n_hist - 1
        start_frame = max(start_frame, 0)
        head_start = start_frame + 1                 # first frame to predict
        # Clamp to 0 when start_frame < n_hist-1 (e.g. start_frame=0, n_hist=2)
        hist_start = max(0, start_frame - n_hist + 1)
        n_pad = n_hist - (start_frame - hist_start + 1)  # front-padding with frame 0 copies

        # === Frame buffer ===
        # frame_buffer[k][:, buf_offset + abs_pos] holds the encoded latent for
        # absolute trajectory frame abs_pos.  buf_offset gives headroom so memory
        # lookups at head - n_memory*t_memory never underflow the buffer.
        # Pre-allocating {key: (B, total_buf, *per_frame_shape)} tensors lets us
        # populate real frames and store predicted frames with slice assignments,
        # and gather memory frames with advanced indexing — no per-frame clone loops.
        buf_offset = n_memory * t_memory
        total_buf = buf_offset + T_total

        # Encode obs[0 .. head_start+horizon-1] for the first chunk and to
        # pre-populate the buffer with all real-observation frames.
        start_time = time.time()
        obs_init = {k: v[:, :head_start + horizon] for k, v in obs.items()}
        x1_init = self.encode_obs(obs_init)
        for k in x1_init:
            x1_init[k][:, head_start:] *= 0      # zero the future slots
        _ref = next(iter(x1_init.values()))
        B_buf, _dev, _dt = _ref.shape[0], _ref.device, _ref.dtype
        frame_buffer = {
            k: torch.zeros(B_buf, total_buf, *x1_init[k].shape[2:], device=_dev, dtype=_dt)
            for k in x1_init
        }
        # Populate real frames 0..head_start-1 in one shot (no per-frame clone).
        for k in x1_init:
            frame_buffer[k][:, buf_offset:buf_offset + head_start] = x1_init[k][:, :head_start]
        print (f"Time taken for encoding init frames: {time.time() - start_time:.3f}s")

        # x1_chunk for the first prediction chunk: history window + zeroed future.
        # If start_frame < n_hist-1, pad the front with copies of frame 0.
        # hist_abs: absolute frame indices for each history slot (frame-0 clamped for pads).
        hist_abs = [0] * n_pad + list(range(hist_start, head_start))
        hist_buf_idx = [buf_offset + p for p in hist_abs]
        x1_chunk = {
            k: torch.cat([
                frame_buffer[k][:, hist_buf_idx],
                torch.zeros(B_buf, horizon, *frame_buffer[k].shape[2:], device=_dev, dtype=_dt),
            ], dim=1)
            for k in frame_buffer
        }

        def get_frame(abs_pos: int) -> dict:
            """Return buffer frame at abs_pos; clamp to frame 0 if out of range."""
            clamped = max(0, min(abs_pos, T_total - 1))
            idx = buf_offset + clamped
            return {k: frame_buffer[k][:, idx:idx+1] for k in frame_buffer}

        def build_memory_tokens(head_abs: int) -> Optional[torch.Tensor]:
            """Build memory_tokens from the buffer at exact t_memory spacing."""
            ## keep the latents not the images
            if not self._use_memory or n_memory == 0:
                return None
            # Chronological order: oldest first.  Use advanced indexing to gather
            # all n_memory frames in a single slice rather than a loop + cat.
            mem_abs = [max(0, min(head_abs - (n_memory - j) * t_memory, T_total - 1))
                       for j in range(n_memory)]
            mem_buf_idx = [buf_offset + p for p in mem_abs]
            mem_obs = {k: frame_buffer[k][:, mem_buf_idx] for k in self._img_keys}
            mem_obs['states'] = frame_buffer['states'][:, mem_buf_idx]

            mem_embed = self.wm.encode_memory(mem_obs)  # (B, n_memory, N_obs, n_embed)
            B_m, M = mem_embed.shape[:2]
            dev, dt = mem_embed.device, mem_embed.dtype

            zero_act = torch.zeros(B_m, M, self.wm._n_actions, device=dev, dtype=dt)
            act_emb = rearrange(self.wm.mem_inp_prj['actions'](zero_act), 'b m d -> b m 1 d')
            t_emb = rearrange(
                self.wm.timestep_encoder(torch.ones(B_m * M, device=dev)),
                '(b m) d -> b m 1 d', b=B_m, m=M,
            )
            mem_tokens = torch.cat([mem_embed, act_emb, t_emb], dim=2)
            return rearrange(mem_tokens, 'b m n d -> b (m n) d')

        # === Autoregressive generation ===
        all_xt_chunks = []
        ## TODO: check the ema weights is used right here or not
        # with self.ema.use_ema_weights():
        for chunk_idx, head_abs in enumerate(range(head_start, T_total, bootstrap)):
            start_time = time.time()
            memory_tokens = build_memory_tokens(head_abs)
            print (f"Time taken for building memory tokens: {time.time() - start_time:.3f}s")
            start_time = time.time()
            if chunk_idx == 0:
                real_actions = actions[:, hist_start:hist_start + (n_hist - n_pad) + horizon]
                if n_pad > 0:
                    zero_act = torch.zeros(
                        actions.shape[0], n_pad, actions.shape[-1],
                        device=actions.device, dtype=actions.dtype,
                    )
                    actions_chunk = torch.cat([zero_act, real_actions], dim=1)
                else:
                    actions_chunk = real_actions
            else:
                chunk_start = head_abs - n_hist
                actions_chunk = actions[:, chunk_start:chunk_start + n_hist + horizon]
                # Pad the last (possibly short) chunk with zeros rather than skipping it.
                if actions_chunk.shape[1] < n_hist + horizon:
                    pad_len = n_hist + horizon - actions_chunk.shape[1]
                    actions_chunk = torch.nn.functional.pad(actions_chunk, (0, 0, 0, pad_len))
                # Refill history from buffer with a single batch index; zero future.
                chunk_hist_idx = [buf_offset + max(0, min(chunk_start + i, T_total - 1))
                                  for i in range(n_hist)]
                for k in x1_chunk:
                    x1_chunk[k][:, :n_hist] = frame_buffer[k][:, chunk_hist_idx]
                    x1_chunk[k][:, n_hist:] = 0
            print (f"Time taken for building x1 chunk: {time.time() - start_time:.3f}s")
            start_time = time.time()
            xt_chunk = self.generate_latent_rollouts(
                x1_chunk, actions_chunk, memory_tokens=memory_tokens,
            )
            for k in xt_chunk:
                xt_chunk[k] = xt_chunk[k][:, :n_hist + bootstrap]
            print (f"Time taken for generating latent rollouts: {time.time() - start_time:.3f}s")
            # Store newly predicted frames in the buffer for future memory lookups.
            # Single slice assignment per key — no per-frame clone loop needed.
            start_time = time.time()
            store_start = buf_offset + head_abs
            store_end = min(store_start + bootstrap, total_buf)
            n_store = store_end - store_start
            if n_store > 0:
                for k in frame_buffer:
                    frame_buffer[k][:, store_start:store_end] = xt_chunk[k][:, n_hist:n_hist + n_store]

            # Only keep predicted frames (no history prefix) so the output
            # aligns with gt_obs, which starts at start_frame+1.
            all_xt_chunks.append(
                {k: v[:, n_hist:n_hist + bootstrap].clone() for k, v in xt_chunk.items()}
            )
            print (f"Time taken for storing predicted frames: {time.time() - start_time:.3f}s")

        keys = all_xt_chunks[0].keys()
        ## get the number of chunks
        print('Number of chunks: ', len(range(head_start, T_total, bootstrap)) )
        start_time = time.time()
        all_xt = {k: torch.cat([c[k] for c in all_xt_chunks], dim=1) for k in keys}
        all_decoded_obs_pred = self.decode_obs(all_xt, chunk_size=16)
        print (f"Time taken for decoding observations: {time.time() - start_time:.3f}s")
        print (f"Total Time taken for video gen: {time.time() - init_time:.3f}s")
        return None, all_decoded_obs_pred


    def compute_flow_loss(
        self,
        x1: torch.Tensor,
        x0: torch.Tensor,
        pred: torch.Tensor,
        t: torch.Tensor
    ):
        if self._flow_loss.startswith('x-pred'):
            loss = F.mse_loss(x1, pred, reduction='none')
            if self._flow_loss.endswith('v-loss'):
                denom = (1 - t).clamp(min=0.05) ** -2
                loss = loss * denom
            return loss

        elif self._flow_loss.startswith('v-pred'):
            loss = F.mse_loss(pred, x1 - x0, reduction='none')
            return loss
        else:
            raise NotImplementedError

    def compute_temporal_loss(
        self,
        x1: torch.Tensor,
        x0: torch.Tensor,
        pred: torch.Tensor,
    ):
        """Temporal consistency loss: MSE((f_h - f_{h-1}) - (v_h - v_{h-1})).

        For v-pred: v_h = x1_h - x0_h (true velocity), f_h = pred_h (predicted velocity).
        For x-pred: v_h = x1_h (true target), f_h = pred_h (predicted target).
        """
        if self._flow_loss.startswith('v-pred'):
            target = x1 - x0
        elif self._flow_loss.startswith('x-pred'):
            target = x1
        else:
            raise NotImplementedError

        pred_diff = pred[:, 1:] - pred[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        return F.mse_loss(pred_diff, target_diff, reduction='none')

    def sample_noise(
        self,
        x1: Dict[str, torch.Tensor]
    ):
        noise = {}
        for key in x1:
            noise[key] = torch.randn_like(x1[key])
        
        return noise
    
    def interpolate(
        self,
        x1: dict[str, torch.Tensor],
        x0: dict[str, torch.Tensor],
        t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        xt = {}
        for k in x1:
            if k in self._img_keys:
                t_ = rearrange(t, 'b t -> b t 1 1')
            else:
                t_ = rearrange(t, 'b t -> b t 1')
            xt[k] = (1 - t_) * x0[k] + t_ * x1[k] 
        return xt
        
    def sample_timestep(
        self,
        B: int,
        T: int
    ) -> torch.Tensor:
         # Do not noise the frame in history
        if self._diff_forcing:
            t_future = torch.rand(
                size=(B, T - self._n_history),
                device=self.device
            )
        else:
            t_future = torch.rand(
                size=(B, 1),
                device=self.device
            )
            t_future = t_future.repeat(1, T - self._n_history)

        t_history = torch.ones(
            (B, self._n_history),
            device=self.device
        )

        t = torch.cat([t_history, t_future], dim=1).float()  # (B, T)

        return t

    def _aggregate_losses(
        self,
        losses: Dict[str, torch.Tensor],
        update_rm: bool = True,
    ) -> tuple:
        """Aggregate individual losses into a single total_loss for backward.

        Returns:
            total_loss: scalar tensor (live graph for backward)
            log_dict: dict of detached scalars for logging
        """
        device = losses['flow/states'].device

        # Flow matching loss
        flow_loss = sum(losses[f'flow/{k}'] for k in self._img_keys)
        flow_loss = flow_loss + self._state_loss_scale * losses['flow/states']

        total_loss = flow_loss.clone()

        rm_loss = losses['loss/RM']
        critic_loss = losses['loss/V']
        if update_rm:
            total_loss = total_loss + self._rm_loss_coeff * rm_loss + self._critic_loss_coeff * critic_loss
        else:
            # Keep the graph static under DDP/compile while skipping RM/critic updates.
            total_loss = total_loss + 0.0 * rm_loss + 0.0 * critic_loss

        # Decoder
        decoder_loss = torch.tensor(0.0, device=device)
        if self._train_decoder:
            decoder_loss = sum(v for k, v in losses.items() if 'decoder/' in k)
            total_loss = total_loss + decoder_loss

        # Temporal consistency
        temporal_loss = torch.tensor(0.0, device=device)
        if self._use_temporal_loss:
            temporal_loss = sum(v for k, v in losses.items() if 'temporal/' in k)
            total_loss = total_loss + self._temporal_loss_coeff * temporal_loss

        # Build detached log dict (no graph references for logging)
        log = {k: v.detach() for k, v in losses.items() if k != 'rewards'}
        log.update({
            'flow/Total Loss': flow_loss.detach(),
            'RM loss': rm_loss.detach(),
            'Critic Loss': critic_loss.detach(),
            'decoder/Total Loss': decoder_loss.detach(),
            'Temporal/Total Loss': temporal_loss.detach(),
            'Total Loss': total_loss.detach(),
        })

        return total_loss, log

    def forward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        # t: torch.Tensor,
        tasks: torch.Tensor,
        gt_rewards: Optional[torch.Tensor] = None,
        memory: Optional[Dict[str, torch.Tensor]] = None,
        update_rm: bool = True,
    ) -> tuple:

        losses = {}

        B, T, _ = actions.size()

        for key, value in obs.items():
            if isinstance(value, torch.Tensor):
                assert value.shape[0] == B, (
                    f"obs[{key}] batch mismatch: obs batch={value.shape[0]} actions batch={B}; "
                    f"obs shape={tuple(value.shape)} actions shape={tuple(actions.shape)}"
                )
                assert value.shape[1] == T, (
                    f"obs[{key}] time mismatch: obs time={value.shape[1]} actions time={T}; "
                    f"obs shape={tuple(value.shape)} actions shape={tuple(actions.shape)}"
                )

        if gt_rewards is None:
            raise ValueError("WEAVER.forward requires gt_rewards during training.")

        # Encode memory frames if memory is enabled
        memory_tokens = None
        if self._use_memory:
            assert memory is not None, "model uses memory but memory=None was passed"
            for key, value in memory.items():
                if isinstance(value, torch.Tensor):
                    assert value.shape[0] == B, (
                        f"memory[{key}] batch mismatch: memory batch={value.shape[0]} actions batch={B}; "
                        f"memory shape={tuple(value.shape)} actions shape={tuple(actions.shape)}"
                    )
                    assert value.shape[1] == self._n_memory_frames, (
                        f"memory[{key}] frame mismatch: memory frames={value.shape[1]} "
                        f"expected={self._n_memory_frames}; memory shape={tuple(value.shape)}"
                    )
            memory_tokens = self.encode_memory_obs(memory)  # (B, M*N_total, n_embed)
            if self.training and self._history_noise_std > 0:
                memory_tokens = memory_tokens + torch.randn_like(memory_tokens) * self._history_noise_std

        # instructions = self.encode_ins(instructions)  # (B, L_ins, D)
        x1 = self.encode_obs(obs)

        x0 = self.sample_noise(x1)

        t = self.sample_timestep(B, T)
        xt = self.interpolate(x1, x0, t)

        # Add Gaussian noise to history frame latents during training
        # Uses torch.cat instead of in-place assignment for torch.compile compatibility
        if self.training and self._history_noise_std > 0:
            for k in xt:
                hist = xt[k][:, :self._n_history]
                future = xt[k][:, self._n_history:]
                noised_hist = hist + torch.randn_like(hist) * self._history_noise_std
                xt[k] = torch.cat([noised_hist, future], dim=1)

        x_pred = self.wm(
            xt, actions, t,
            memory=memory_tokens,
        )

        # Add Flow Matching loss function
        for key in x_pred:
            if key in self._img_keys:
                t_ = rearrange(t, 'b t -> b t 1 1')
            else:
                t_ = rearrange(t, 'b t -> b t 1')
            loss = self.compute_flow_loss(x1[key], x0[key], x_pred[key], t_)

            loss = loss[:, self._n_history:].mean(dim=-1)
            losses[f'flow/{key}'] = loss.mean()

        # Add Temporal consistency loss
        if self._use_temporal_loss:
            for key in x_pred:
                temporal_loss = self.compute_temporal_loss(x1[key], x0[key], x_pred[key])
                # Only on future frames (shifted by 1 due to diff)
                temporal_loss = temporal_loss[:, max(self._n_history - 1, 0):].mean(dim=-1)
                losses[f'temporal/{key}'] = temporal_loss.mean()

        if self._train_decoder:
            recon_obs = self.decode_obs(x1)  # Dict of (B, T, C, H, W)
            for key in self._img_keys:
                losses[f'decoder/{key}'] = F.mse_loss(recon_obs[key], obs[key])

        task_embed = self.encode_task(tasks)
        x1_pred = {}
        for key in x1:
            if key not in x_pred:
                continue
            if self._flow_loss.startswith('x-pred'):
                x1_pred[key] = x_pred[key].detach()
            elif self._flow_loss.startswith('v-pred'):
                x1_pred[key] = (x0[key] + x_pred[key]).detach()
            else:
                raise NotImplementedError

        x1_detached = {k: v.detach() for k, v in x1.items() if k in x1_pred}
        obs_both = {
            k: torch.cat([x1_detached[k], x1_pred[k]], dim=0)
            for k in x1_pred
        }
        actions_both = torch.cat([actions, actions], dim=0)
        task_embed = task_embed.detach()
        tasks_both = torch.cat([task_embed, task_embed], dim=0)
        rewards_both = torch.cat([gt_rewards, gt_rewards], dim=0)

        pred_rewards, rm_loss = self.rm.compute_loss(
            obs=obs_both,
            actions=actions_both,
            tasks=tasks_both,
            gt_rewards=rewards_both,
        )
        critic_loss = self.critic.compute_loss(
            obs=obs_both,
            actions=actions_both,
            tasks=tasks_both,
            rewards=rewards_both,
        )
        losses['loss/RM'] = rm_loss
        losses['loss/V'] = critic_loss
        losses['rewards'] = pred_rewards[:pred_rewards.shape[0] // 2]

        return self._aggregate_losses(losses, update_rm)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
