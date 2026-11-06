"""Shared utilities for WEAVER evaluation and deployment scripts."""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from ..dynamics.model import Dynamics
from ..utils.config import dict_to_namespace, load_config, merge_dicts, update_config
from ..utils.tools import load_checkpoint
from ..wm.encoders import get_encoder, get_task_encoder
from ..wm.model import WEAVER


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_eval_config(checkpoint_dir: str, overrides: list[str]) -> object:
    """Load WEAVER config: merge package defaults with checkpoint config, apply overrides."""
    default_cfg = load_config(Path(__file__).parent.parent / "config.yaml", mode="defaults")
    cfg_path = Path(checkpoint_dir) / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open() as f:
        cfg_dict = yaml.safe_load(f)
    if "defaults" in cfg_dict and isinstance(cfg_dict["defaults"], dict):
        cfg_dict = cfg_dict["defaults"]
    cfg_dict = merge_dicts(default_cfg, cfg_dict)
    if overrides:
        cfg_dict = update_config(cfg_dict, dict(item.split("=", 1) for item in overrides))
    return dict_to_namespace(cfg_dict)


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def image_keys_from_cfg(cfg) -> list[str]:
    if getattr(cfg.dataset, "sample_aux_from_left_right", False):
        return ["aux", "wrist"]
    return list(cfg.dataset.img_keys)


def clean_state_dict(state_dict: dict) -> dict:
    """Strip DDP/compile prefixes (module., _orig_mod.) from checkpoint keys."""
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod."):]
        cleaned[key] = value
    return cleaned


def build_model(
    cfg,
    device: str,
    img_keys: list[str] | None = None,
    val_steps: int | None = None,
    inference_overrides: dict | None = None,
) -> tuple[WEAVER, list[str], tuple[int, int]]:
    """Construct WEAVER and move to device.

    Args:
        cfg:                Full config namespace (from load_eval_config).
        device:             Target device string.
        img_keys:           Override image keys (default: from cfg.dataset).
        val_steps:          Override denoising steps for eval.
        inference_overrides: Dict of inference config overrides (e.g. pyramid_stagger_width=0).

    Returns:
        (wm, img_keys, image_size)
    """
    if img_keys is None:
        img_keys = image_keys_from_cfg(cfg)
    image_size = cfg.dataset.image_size
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    if val_steps is not None:
        cfg.model.val_steps = val_steps
    if inference_overrides:
        for k, v in inference_overrides.items():
            setattr(cfg.inference, k, v)

    im_encoder, train_decoder = get_encoder(cfg.im_encoder, cfg.dataset.image_size, device=device)
    task_encoder = get_task_encoder(config=None, device=device)

    wm = WEAVER(
        img_keys=img_keys,
        im_encoder=im_encoder,
        train_decoder=train_decoder,
        task_encoder=task_encoder,
        n_history=cfg.n_history,
        n_horizon=cfg.horizon,
        config=cfg.model,
        use_precomputed_features=True,
        n_states=getattr(cfg.dataset, "n_states", 8),
        n_actions=getattr(cfg.dataset, "n_actions", 8),
        image_size=image_size,
        device=device,
        n_memory_frames=getattr(cfg, "n_memory_frames", 0),
        t_memory=getattr(cfg, "t_memory", 1),
        inference_config=getattr(cfg, "inference", None),
    ).to(device)
    wm.ema.to(device)

    return wm, img_keys, image_size


def load_checkpoint_into_model(
    wm: WEAVER,
    checkpoint_dir: str,
    device: str,
    use_ema: bool = False,
) -> None:
    """Find the latest checkpoint in checkpoint_dir and load it into wm."""
    checkpoint_name = "checkpoint.pt"
    if not os.path.exists(os.path.join(checkpoint_dir, checkpoint_name)):
        ckpt_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pt"))
        if ckpt_files:
            iteration = max(int(x.split("_")[-1].split(".")[0]) for x in ckpt_files)
            checkpoint_name = f"checkpoint_{iteration}.pt"
    ckpt = load_checkpoint(checkpoint_dir, device, weights_only=True, checkpoint_name=checkpoint_name)
    wm.load_state_dict(clean_state_dict(ckpt["model"]), strict=False)
    wm.ema.load_state_dict(ckpt["ema"])
    if use_ema:
        wm.ema.apply_to(wm)


# ---------------------------------------------------------------------------
# Normalization stats
# ---------------------------------------------------------------------------

def load_norm_stats(
    dataset_path: str, relabel_actions: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (state_mean, state_std, action_mean, action_std) as float32 numpy arrays."""
    import json
    suffix = "relabel" if relabel_actions else "recorded"
    norm_path = os.path.join(dataset_path, f"norm_stats_{suffix}.json")
    if not os.path.exists(norm_path):
        raise FileNotFoundError(f"Norm stats not found: {norm_path}")
    with open(norm_path) as f:
        s = json.load(f)["norm_stats"]
    print(f"Loaded norm stats ({suffix}) from {norm_path}")
    return (
        np.array(s["state"]["mean"],   dtype=np.float32),
        np.array(s["state"]["std"],    dtype=np.float32),
        np.array(s["actions"]["mean"], dtype=np.float32),
        np.array(s["actions"]["std"],  dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Dynamics model
# ---------------------------------------------------------------------------

def load_dynamics_model(path: str, device: str) -> Dynamics:
    """Load a pre-trained Ctrl-World Dynamics model from a .pth checkpoint."""
    from ..dynamics.model import Dynamics as _Dynamics  # local alias avoids circular risk
    model = _Dynamics(action_dim=7, action_num=15, hidden_size=512)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.device = device
    model.eval()
    print(f"Dynamics model loaded from {path}")
    return model


# ---------------------------------------------------------------------------
# World model rollout + scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def wm_rollout_and_score(
    wm: WEAVER,
    context: dict,
    history_actions: torch.Tensor,
    future_actions: torch.Tensor,
    future_states: torch.Tensor,
    task_embed: torch.Tensor,
    memory_tokens: torch.Tensor | None = None,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """Imagine one chunk and score with reward model + critic.

    Rolls out WEAVER for future_actions.shape[1] steps starting from context,
    then evaluates the predicted latent trajectory with wm.rm and wm.critic.

    Args:
        wm:             WEAVER model.
        context:        Dict {img_key: (B, n_hist, N, D), "states": (B, n_hist, S)}.
        history_actions:(B, n_hist, A) — actions that produced the context.
        future_actions: (B, Hf, A)   — actions to imagine.
        future_states:  (B, Hf, S)   — placeholder future states.
        task_embed:     (B, D_task)  — task embedding for scoring.
        memory_tokens:  Optional (B, M*N_mem, D) pre-encoded memory.

    Returns:
        xt:      Predicted latents dict (B, n_hist + Hf, ...) — same keys as context.
        rewards: Per-frame rewards (B, n_hist + Hf).
        values:  Per-frame critic values (B, n_hist + Hf).
    """
    device = future_actions.device
    with torch.autocast(device_type=device.type if hasattr(device, "type") else "cuda",
                        dtype=torch.bfloat16):
        xt = wm.generate_latent_rollouts_variable_horizon(
            context=context,
            future_actions=future_actions,
            future_states=future_states,
            history_actions=history_actions,
            memory_tokens=memory_tokens,
        )
        actions_full = torch.cat([history_actions, future_actions], dim=1)
        rewards = wm.rm(xt, actions_full, task_embed)
        values  = wm.critic(xt, actions_full, task_embed)

    return xt, rewards.float(), values.float()
