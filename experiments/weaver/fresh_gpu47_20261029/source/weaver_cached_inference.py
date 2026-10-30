"""Inference-only adapters for the pinned native WEAVER implementation.

The variable-horizon packing and its dictionary insertion order deliberately
match WEAVER.generate_latent_rollouts_variable_horizon. Noise generation, Euler
updates, pyramid scheduling and prefix K/V ownership remain native. In particular
this is not a distilled/ReFlow sampler and does not reduce the number of steps.
K/V caches are local to one native generation call, never retained across chunks
or publications. The only persistent native cache is its timestep schedule.
"""
from __future__ import annotations


def generate_variable_horizon_cached(upstream, *, context, future_actions,
                                     future_states, history_actions=None,
                                     memory_tokens=None):
    """Return history plus every future latent using native prefix K/V reuse.

    This is an eval-only equivalent of the native variable-horizon wrapper for
    1..upstream._n_horizon future frames. It preserves the native noise draws,
    caller tensor values, image-key ordering, actions, memory and sampling config.
    No CPU tensor copy, model mutation, random draw or persistent K/V cache is
    introduced by this wrapper. A zero horizon is rejected: production requests
    must contain an action rather than ambiguously returning an unchanged state.
    """
    import torch

    if upstream.training or upstream.wm.training:
        raise ValueError("Cached WEAVER generation requires model.eval()")
    if future_actions.ndim != 3:
        raise ValueError("future_actions must have shape [batch, horizon, actions]")
    batch, horizon, action_dim = future_actions.shape
    n_history = int(upstream._n_history)
    if batch < 1 or not 1 <= horizon <= int(upstream._n_horizon):
        raise ValueError("Cached WEAVER horizon must be within the native trained horizon")
    if action_dim != int(upstream.wm._n_actions):
        raise ValueError("Future action dimension differs from native WEAVER")
    keys = tuple(upstream._img_keys)
    if set(context) != {*keys, "states"}:
        raise ValueError("Context must contain exactly the native image keys and states")
    if any(value.shape[:2] != (batch, n_history) for value in context.values()):
        raise ValueError("Context must contain exactly the native history frames")
    if any(context[key].ndim != 4 for key in keys) or context["states"].ndim != 3:
        raise ValueError("Context image features/states have invalid ranks")
    if future_states.shape != (batch, horizon, context["states"].shape[-1]):
        raise ValueError("Future state placeholders are not aligned with actions")
    if history_actions is not None and history_actions.shape != (batch, n_history, action_dim):
        raise ValueError("History actions must align with every history frame")
    if bool(getattr(upstream, "_use_memory", False)) and memory_tokens is None:
        raise ValueError("Cached variable-horizon generation requires explicit encoded memory")

    # Keep exactly the native wrapper's allocation/key order: sample_noise draws
    # one tensor per insertion-ordered image key, then states, including history.
    with torch.no_grad():
        x1_chunk = {}
        for key in keys:
            future_zero = torch.zeros(batch, horizon, *context[key].shape[2:],
                                      device=context[key].device, dtype=context[key].dtype)
            x1_chunk[key] = torch.cat([context[key], future_zero], dim=1)
        x1_chunk["states"] = torch.cat([context["states"], future_states], dim=1)
        if history_actions is None:
            hist_actions = torch.zeros(batch, n_history, upstream.wm._n_actions,
                                       device=future_actions.device, dtype=future_actions.dtype)
        else:
            hist_actions = history_actions.to(device=future_actions.device,
                                              dtype=future_actions.dtype)
        actions_chunk = torch.cat([hist_actions, future_actions], dim=1)
        return upstream.generate_latent_rollouts_cached(
            x1_chunk, actions_chunk, memory_tokens=memory_tokens)


def decode_rollout_observations(upstream, future, image_keys, *, chunk_size=2,
                                output_mode="last"):
    """Decode either the last frame or the legacy full image sequence on device.

    Returns float32 [-1, 1] RGB with shape [B, selected_time, views, 3, H, W].
    The caller retains all future latents/states/rewards independently. Last-only
    decoding is valid for the pinned frame-independent SD3 AutoencoderKL decoder;
    it must not be applied to a temporal decoder with cross-frame state. The
    To preserve BF16 arithmetic, last mode decodes each original chunk containing
    a wanted final frame, then selects that frame. Merely changing a legacy batch
    of two frames to a singleton can change convolution rounding. For B=1,H=8
    and chunk_size=2 this decodes the final two frames per view, instead of eight,
    but still returns only the final frame. The original RGB arithmetic precedes
    the float32 conversion. This function makes no CPU copies or random draws.
    """
    import torch

    if output_mode not in ("last", "all"):
        raise ValueError("output_mode must be 'last' or 'all'")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("decode chunk_size must be a positive integer")
    keys = tuple(image_keys)
    if not keys or keys != tuple(upstream._img_keys):
        raise ValueError("Decode view ordering must match native WEAVER image keys")
    if set(future) != {*keys, "states"}:
        raise ValueError("Future must contain exactly native image keys and states")
    shape = future["states"].shape[:2]
    if len(shape) != 2 or min(shape) < 1 or any(v.shape[:2] != shape for v in future.values()):
        raise ValueError("Every future latent/state must have aligned nonempty frames")
    with torch.no_grad():
        if output_mode == "all":
            decoded = upstream.decode_obs(future, chunk_size=chunk_size)
            return (torch.stack([decoded[key] for key in keys], dim=2) * 2 - 1).float()
        batch, horizon = shape
        total = batch * horizon
        flat = {key: value.reshape(total, *value.shape[2:]) for key, value in future.items()}
        wanted = [index * horizon + horizon - 1 for index in range(batch)]
        decoded_chunks = {}
        for index in wanted:
            start = index // chunk_size * chunk_size
            if start not in decoded_chunks:
                end = min(start + chunk_size, total)
                selected = {key: value[start:end].unsqueeze(0) for key, value in flat.items()}
                decoded = upstream.decode_obs(selected, chunk_size=chunk_size)
                decoded_chunks[start] = (torch.stack([decoded[key] for key in keys], dim=2) * 2 - 1).float()
        return torch.cat([decoded_chunks[index // chunk_size * chunk_size]
                          [:, index % chunk_size:index % chunk_size + 1]
                          for index in wanted], dim=0)
