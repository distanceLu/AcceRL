import functools
from typing import Optional
from enum import StrEnum

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce, repeat


# ---------- Rotary Position Embedding (RoPE) ---------- #
class RotaryType(StrEnum):
    STANDARD = "standard"   # learned-frequency (1/theta^(2i/d))
    PIXEL = "pixel"         # linear-frequency (pixel positions)


class AttentionType(StrEnum):
    SPATIAL = "spatial"     # within-frame attention (folds T into batch)
    TEMPORAL = "temporal"   # across-frame causal attention (folds N into batch)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Half-split rotation for RoPE: [-x2, x1]."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def rope_mix(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply precomputed cos/sin RoPE to a tensor.

    cos, sin have shape (..., D) — already expanded to full head dim.
    """
    return x * cos + rotate_half(x) * sin


@functools.lru_cache(maxsize=64)
def rope_nd(
    spatial_shape: tuple[int, ...],
    dim: int,
    rotary_type: RotaryType = RotaryType.STANDARD,
    theta: float = 10000.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute N-dimensional RoPE frequency table (cached).

    Args:
        spatial_shape: Tuple of axis sizes, e.g. (T,) for 1D or (H, W) for 2D.
        dim: Total RoPE dimension (must be divisible by 2 * len(spatial_shape)).
        rotary_type: STANDARD (inverse-power frequencies) or PIXEL (linear).
        theta: Base for frequency computation (STANDARD only).
        dtype: Output dtype.
        device: Output device.

    Returns:
        (cos, sin) each of shape (prod(spatial_shape), dim).
    """
    ndim = len(spatial_shape)
    assert dim % (2 * ndim) == 0, (
        f"dim={dim} must be divisible by 2*ndim={2 * ndim}"
    )
    dim_per_axis = dim // ndim  # frequencies per spatial axis
    half = dim_per_axis // 2     # cos/sin pairs per axis

    # Build N-D position grid (also handles 1D as a special case)
    axes_indices = [torch.arange(s, device=device) for s in spatial_shape]
    grids = torch.meshgrid(*axes_indices, indexing="ij")  # ndim tensors of shape (*spatial_shape)

    cos_parts = []
    sin_parts = []
    for i in range(ndim):
        positions = grids[i].reshape(-1).float()
        if rotary_type == RotaryType.STANDARD:
            freqs = 1.0 / (theta ** (torch.arange(0, dim_per_axis, 2, device=device).float() / dim_per_axis))
        else:  # PIXEL
            freqs = torch.linspace(1.0, spatial_shape[i] / 2.0, half, device=device)
        angles = torch.outer(positions, freqs)  # (prod(spatial_shape), half)
        cos_parts.append(angles.cos())
        sin_parts.append(angles.sin())

    cos_half = torch.cat(cos_parts, dim=-1)  # (prod(spatial_shape), dim // 2)
    sin_half = torch.cat(sin_parts, dim=-1)
    # Pre-expand to full head dim so rope_mix avoids repeat at runtime
    cos_out = cos_half.repeat(1, 2)  # (prod(spatial_shape), dim)
    sin_out = sin_half.repeat(1, 2)  # (prod(spatial_shape), dim)
    return cos_out.to(dtype=dtype), sin_out.to(dtype=dtype)


class MLP(nn.Module):
    def __init__(self,
                 in_dim: int = 384,   # DINOv2 feature dim
                 hidden_dim: int = 256,
                 out_dim: int = 8):       # state dim
        super().__init__()

        # self.mlp = nn.Sequential(
        #     nn.Linear(in_dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(hidden_dim, out_dim)
        # )
        self.c_fc = nn.Linear(in_dim, hidden_dim)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(hidden_dim, out_dim)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, latent_dim)
        returns state: (B, out_dim)
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class FFN(nn.Module):
    def __init__(self, n_embed: int):
        super().__init__()
        hidden = int(8 / 3 * n_embed)

        # SwiGLU projection
        self.c_fc = nn.Linear(n_embed, hidden * 2)
        self.c_proj = nn.Linear(hidden, n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.c_fc(x).chunk(2, dim=-1)
        x = a * F.silu(b)
        x = self.c_proj(x)
        return x


# ---------- Basic Attention Blocks ---------- #
class CrossAttention(nn.Module):
    """Cross-attention: queries attend to a separate key-value source."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_kv: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim_kv, dim)
        self.v_proj = nn.Linear(dim_kv, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.out_proj.NANOGPT_SCALE_INIT = 1

    def forward(
        self,
        x: torch.Tensor,
        kv: torch.Tensor,
    ) -> torch.Tensor:
        B, Nq, _ = x.shape
        H = self.num_heads
        D = self.head_dim
        Nk = kv.shape[1]

        q = self.q_proj(x).view(B, Nq, H, D).transpose(1, 2)
        k = self.k_proj(kv).view(B, Nk, H, D).transpose(1, 2)
        v = self.v_proj(kv).view(B, Nk, H, D).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, Nq, H * D)
        return self.out_proj(out)


class Attention(nn.Module):
    """Self-attention with RoPE, optional causal mask, and QK-norm.

    RoPE is precomputed at init from rope_grid_shape. At forward time:
    - If rope_ids is given: index into the table (sparse / reordered tokens).
    - Otherwise: use the first N rows sequentially (dense sequential tokens).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_type: AttentionType = AttentionType.SPATIAL,
        rotary_type: RotaryType = RotaryType.STANDARD,
        qk_norm: bool = False,
        rope_grid_shape: tuple[int, ...] | None = None,
        n_unrotated: int = 0,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.is_causal = (attn_type == AttentionType.TEMPORAL)
        self.rotary_type = rotary_type
        self._n_unrotated = n_unrotated

        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.out_proj.NANOGPT_SCALE_INIT = 1

        if qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        # Precompute RoPE lookup table
        if rope_grid_shape is not None:
            cos_table, sin_table = rope_nd(
                rope_grid_shape, self.head_dim,
                rotary_type=rotary_type,
            )
            self.register_buffer("_rope_cos_table", cos_table, persistent=False)
            self.register_buffer("_rope_sin_table", sin_table, persistent=False)
        else:
            self._rope_cos_table = None
            self._rope_sin_table = None

    def _get_rope(
        self, rope_ids: torch.Tensor, dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get cos/sin tensors for rope_mix.

        Args:
            rope_ids: (B, N) or (N,) indices into the precomputed table.
        Returns:
            cos, sin: (B, 1, N, D) if rope_ids is 2D, (1, 1, N, D) if 1D.
        """
        cos = self._rope_cos_table[rope_ids].to(dtype)
        sin = self._rope_sin_table[rope_ids].to(dtype)
        if cos.ndim == 3:  # (B, N, half) → (B, 1, N, half)
            return rearrange(cos, 'b n d -> b 1 n d'), rearrange(sin, 'b n d -> b 1 n d')
        # (N, half) → (1, 1, N, half)
        return rearrange(cos, 'n d -> 1 1 n d'), rearrange(sin, 'n d -> 1 1 n d')

    def forward(
        self,
        x: torch.Tensor,
        rope_ids: torch.Tensor | None = None,
        kv_cache_k: torch.Tensor | None = None,
        kv_cache_v: torch.Tensor | None = None,
        cache_mode: str | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) input tokens.
            rope_ids: Optional indices into the RoPE table for sparse/reordered
                tokens (e.g. SPRINT). If None, uses sequential positions 0..N-1.
            cache_mode:
                None    - normal forward, no cache interaction.
                "write" - normal forward and return temporal K/V for caller storage.
                "read"  - future-only forward attending to cached prefix K/V.
        """
        N = x.shape[1]
        H = self.num_heads

        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)

        q = rearrange(q, 'b n (h d) -> b h n d', h=H)
        k = rearrange(k, 'b n (h d) -> b h n d', h=H)
        v = rearrange(v, 'b n (h d) -> b h n d', h=H)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE
        if self._rope_cos_table is not None:
            cos, sin = self._get_rope(rope_ids, q.dtype)
            if self._n_unrotated > 0 and N > self._n_unrotated:
                N_rot = N - self._n_unrotated
                q = torch.cat([
                    rope_mix(q[:, :, :N_rot], cos, sin),
                    q[:, :, N_rot:],
                ], dim=2)
                k = torch.cat([
                    rope_mix(k[:, :, :N_rot], cos, sin),
                    k[:, :, N_rot:],
                ], dim=2)
            else:
                q = rope_mix(q, cos, sin)
                k = rope_mix(k, cos, sin)

        if cache_mode == "read":
            assert kv_cache_k is not None and kv_cache_v is not None
            t_cached = kv_cache_k.shape[2]
            q_len = q.shape[2]
            k_full = torch.cat([kv_cache_k, k], dim=2)
            v_full = torch.cat([kv_cache_v, v], dim=2)
            kv_len = k_full.shape[2]
            q_pos = torch.arange(q_len, device=q.device) + t_cached
            k_pos = torch.arange(kv_len, device=q.device)
            attn_mask = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
            out = F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=attn_mask, is_causal=False)
            out = rearrange(out, 'b h n d -> b n (h d)')
            return self.out_proj(out)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.out_proj(out)

        if cache_mode == "write":
            return out, k, v
        return out


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        n_embed: int,
        n_latent: int,
        num_heads: int
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(n_embed, eps=1e-6)
        self.cross_attn = CrossAttention(n_embed, num_heads, dim_kv=n_latent)
        self.norm2 = nn.RMSNorm(n_embed, eps=1e-6)
        self.ff = FFN(n_embed)

    def forward(self, x: torch.Tensor, latents: torch.Tensor):
        x = x + self.cross_attn(self.norm1(x), kv=latents)
        x = x + self.ff(self.norm2(x))
        return x


class STAttentionBlock(nn.Module):
    """Pre-norm attention block for spatial-temporal transformers.

    - SPATIAL: within-frame attention with PIXEL RoPE. Pass rope_ids for SPRINT.
    - TEMPORAL: across-frame causal attention with STANDARD 1D RoPE (sequential).
    """

    def __init__(
        self,
        n_embed: int,
        n_heads: int,
        attn_type: AttentionType,
        qk_norm: bool = False,
        rope_grid_shape: tuple[int, ...] | None = None,
        max_T: int = 256,
        n_unrotated: int = 0,
    ):
        super().__init__()
        self._attn_type = attn_type

        if attn_type == AttentionType.SPATIAL:
            self.attn = Attention(
                n_embed, n_heads,
                attn_type=AttentionType.SPATIAL,
                rotary_type=RotaryType.PIXEL,
                qk_norm=qk_norm,
                rope_grid_shape=rope_grid_shape,
                n_unrotated=n_unrotated,
            )
        else:  # TEMPORAL
            self.attn = Attention(
                n_embed, n_heads,
                attn_type=AttentionType.TEMPORAL,
                rotary_type=RotaryType.STANDARD,
                qk_norm=qk_norm,
                rope_grid_shape=(max_T,),
            )
            self.register_buffer('_temporal_ids', torch.arange(max_T), persistent=False)

        self.norm1 = nn.RMSNorm(n_embed, eps=1e-6)
        self.norm2 = nn.RMSNorm(n_embed, eps=1e-6)
        self.ffn = FFN(n_embed)
        self._use_checkpoint = False

    def _forward_impl(
        self,
        x: torch.Tensor,
        T: int,
        rope_ids: Optional[torch.Tensor] = None,
    ):
        B = x.size(0)
        x_norm = self.norm1(x)

        if self._attn_type == AttentionType.SPATIAL:
            # Spatial: fold T into batch, use external rope_ids (2D pixel RoPE)
            x_norm = rearrange(x_norm, 'b (t n) d -> (b t) n d', t=T)
            x_attn = self.attn(x_norm, rope_ids=rope_ids)
            x_attn = rearrange(x_attn, '(b t) n d -> b (t n) d', b=B)
        else:
            # Temporal: fold N into batch, use built-in sequential 1D RoPE
            x_norm = rearrange(x_norm, 'b (t n) d -> (b n) t d', t=T)
            x_attn = self.attn(x_norm, rope_ids=self._temporal_ids[:T])
            x_attn = rearrange(x_attn, '(b n) t d -> b (t n) d', b=B)

        x = x + x_attn
        x = x + self.ffn(self.norm2(x))
        return x

    def forward(
        self,
        x: torch.Tensor,
        T: int,
        rope_ids: Optional[torch.Tensor] = None,
    ):
        if self._use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, T, rope_ids,
                use_reentrant=False,
            )
        return self._forward_impl(x, T, rope_ids)


class BlockCausalDynamics(nn.Module):
    def __init__(
        self,
        n_embed: int,
        n_heads: int,
        n_spatial: int = 1,
        qk_norm: bool = False,
        rope_grid_shape: tuple[int, ...] | None = None,
        max_T: int = 256,
        n_unrotated: int = 0,
    ):
        super().__init__()
        self._n_spatial = n_spatial

        blocks = []
        for _ in range(n_spatial):
            blocks.append(STAttentionBlock(
                n_embed, n_heads, AttentionType.SPATIAL,
                qk_norm=qk_norm, rope_grid_shape=rope_grid_shape,
                n_unrotated=n_unrotated,
            ))
        blocks.append(STAttentionBlock(
            n_embed, n_heads, AttentionType.TEMPORAL,
            qk_norm=qk_norm, max_T=max_T,
        ))
        self.blocks = nn.ModuleList(blocks)

    def forward(
        self,
        x: torch.Tensor,
        T: int,
        rope_ids: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x: Input tensor (B, T*N, D)
            T: Number of timesteps
            rope_ids: Integer indices into the RoPE table for spatial attention
        """
        B = x.shape[0]
        N = x.shape[1] // T

        # Build default sequential rope_ids when not provided (e.g. base_policy)
        if rope_ids is None:
            rope_ids = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        elif rope_ids.shape[0] == 1:
            rope_ids = rope_ids.expand(B, -1)

        # Expand rope_ids from (B, N) to (B*T, N) once for all spatial blocks
        rope_ids_expanded = repeat(rope_ids, 'b n -> (b t) n', t=T)

        # Spatial blocks (2D RoPE via rope_ids)
        for block in self.blocks[:-1]:
            x = block(x, T, rope_ids=rope_ids_expanded)

        # Temporal block (1D causal RoPE, built into the block)
        x = self.blocks[-1](x, T)

        return x

    @torch.no_grad()
    def forward_write(
        self,
        x: torch.Tensor,
        T: int,
        prefix_T: int,
        rope_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Eval-only full-sequence pass that returns prefix temporal K/V.

        This is intentionally separate from ``forward`` so training and
        torch.compile never have to return or specialize on cache tensors.
        """
        assert not self.training, "KV-cache write path is eval-only."
        B = x.shape[0]
        N = x.shape[1] // T

        if rope_ids is None:
            rope_ids = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        elif rope_ids.shape[0] == 1:
            rope_ids = rope_ids.expand(B, -1)
        rope_ids_expanded = repeat(rope_ids, 'b n -> (b t) n', t=T)

        for block in self.blocks[:-1]:
            x = block(x, T, rope_ids=rope_ids_expanded)

        temporal = self.blocks[-1]
        x_norm = temporal.norm1(x)
        x_norm = rearrange(x_norm, 'b (t n) d -> (b n) t d', t=T)
        x_attn, k_cache, v_cache = temporal.attn(
            x_norm,
            rope_ids=temporal._temporal_ids[:T],
            cache_mode="write",
        )
        x_attn = rearrange(x_attn, '(b n) t d -> b (t n) d', b=B)
        x = x + x_attn
        x = x + temporal.ffn(temporal.norm2(x))

        return (
            x,
            k_cache[:, :, :prefix_T, :].contiguous(),
            v_cache[:, :, :prefix_T, :].contiguous(),
        )

    @torch.no_grad()
    def forward_read(
        self,
        x: torch.Tensor,
        T: int,
        prefix_T: int,
        kv_cache_k: torch.Tensor,
        kv_cache_v: torch.Tensor,
        rope_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Eval-only future-token pass using cached prefix temporal K/V."""
        assert not self.training, "KV-cache read path is eval-only."
        B = x.shape[0]
        N = x.shape[1] // T

        if rope_ids is None:
            rope_ids = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        elif rope_ids.shape[0] == 1:
            rope_ids = rope_ids.expand(B, -1)
        rope_ids_expanded = repeat(rope_ids, 'b n -> (b t) n', t=T)

        for block in self.blocks[:-1]:
            x = block(x, T, rope_ids=rope_ids_expanded)

        temporal = self.blocks[-1]
        x_norm = temporal.norm1(x)
        x_norm = rearrange(x_norm, 'b (t n) d -> (b n) t d', t=T)
        future_ids = temporal._temporal_ids[prefix_T: prefix_T + T]
        x_attn = temporal.attn(
            x_norm,
            rope_ids=future_ids,
            kv_cache_k=kv_cache_k,
            kv_cache_v=kv_cache_v,
            cache_mode="read",
        )
        x_attn = rearrange(x_attn, '(b n) t d -> b (t n) d', b=B)
        x = x + x_attn
        x = x + temporal.ffn(temporal.norm2(x))
        return x


class AdaPool(nn.Module):
    def __init__(
        self,
        n_embed: int,
        n_heads: int
    ):
        super().__init__()
        self._n_embed = n_embed
        self._n_heads = n_heads

        # self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.w_q    = nn.Linear(n_embed, n_embed)
        self.w_kv   = nn.Linear(n_embed, 2*n_embed, bias=False)
        self.c_proj = nn.Linear(n_embed, n_embed, bias=False)

        # self.register_buffer("bias", torch.tril(torch.ones(1, 1, 1024, 1024)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B, N, C = x.size()
        kv = self.w_kv(x)
        k, v = kv.chunk(2, dim=-1)

        q = reduce(x, 'b n d -> b 1 d', 'mean')
        q = self.w_q(q)

        q = rearrange(q, 'b 1 (h d) -> b h 1 d', h=self._n_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self._n_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self._n_heads)

        x = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.c_proj(x)

        assert x.size(1) == 1

        return x[:, 0, :]
