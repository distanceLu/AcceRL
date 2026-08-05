#!/usr/bin/env python3
"""Offline DIAMOND (denoiser) training extracted from the online RL stack.

Training recipe mirrors:
  - data windows: ``rl/ds_wm_discrete_diffusion.py::_process_episode_for_wm``
  - model/opt:     ``envs/diffusion/denoiser.py::load_denoiser_from_checkpoint``
  - launch hparams: ``run_oft_diamond.sh`` (denoiser-only subset)

Episode dataset format matches ``tests_dsj/dataset_episode/*.pt``
(video, actions_continuous, ...).
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("TMPDIR", "/dev/shm")
os.environ.setdefault("NUMBA_CACHE_DIR", "/dev/shm/numba_cache")
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from envs.diffusion.denoiser import Denoiser
from envs.utils import configure_opt, get_lr_sched, image_to_tensor


DEFAULT_DATASET = Path("/mnt/data/lcx3/AcceRL/tests_dsj/dataset_episode")
DEFAULT_FRAMEWORK_CHECKPOINT = Path(
    "/mnt/data/lcx3/checkpoint/dsj/"
    "openvla-7b+libero_spatial_no_noops+b32+lr-0.0005+lora-r32+dropout-0.0"
    "--image_aug--parallel_dec--8_acts_chunk--discrete_acts--proprio_state--100000_chkpt"
)
DEFAULT_DENOISER_CHECKPOINT = Path(
    "/mnt/data/lcx2/yanjieworkspace/openvla_oft_rl/runs/"
    "wm_reward_denoiser_distill_named/denoiser_smoke_test/denoiser_smoke_test.pt"
)


@dataclass
class EpisodeWindows:
    """One episode converted into DIAMOND conditioning tensors."""

    video: torch.Tensor  # [T, C, H, W] float32 in [-1, 1]
    actions: torch.Tensor  # [T, A] float32 normalized continuous


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--agent-config", type=Path, default=Path("envs/config/agent.yaml"))
    parser.add_argument("--trainer-config", type=Path, default=Path("envs/config/trainer.yaml"))
    parser.add_argument(
        "--denoiser-checkpoint",
        type=Path,
        default=DEFAULT_DENOISER_CHECKPOINT,
        help="Initial DIAMOND weights (denoiser_state_dict .pt). Empty string = random init.",
    )
    parser.add_argument(
        "--framework-checkpoint",
        type=Path,
        default=DEFAULT_FRAMEWORK_CHECKPOINT,
        help="OpenVLA ckpt dir with dataset_statistics.json for action normalization.",
    )
    parser.add_argument("--normalization-key", default="libero_spatial_no_noops")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-step-cond", type=int, default=None,
                        help="Override agent.yaml num_steps_conditioning (default: from config).")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--train-iters", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--ckpt-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/data/lcx3/AcceRL/runs/diamond_offline"),
    )
    parser.add_argument("--exp-name", default="diamond_offline_dataset_episode")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_action_stats(framework_checkpoint: Path, key: str) -> Dict[str, np.ndarray]:
    stats_path = framework_checkpoint / "dataset_statistics.json"
    if not stats_path.is_file():
        raise FileNotFoundError(f"action normalization statistics 不存在: {stats_path}")
    with stats_path.open("r", encoding="utf-8") as file:
        all_stats = json.load(file)
    if key not in all_stats:
        raise KeyError(f"{stats_path} 中没有 normalization key {key!r}")
    stats = all_stats[key]["action"]
    return {name: np.asarray(stats[name]) for name in ("q01", "q99", "mask")}


def normalize_actions(actions: np.ndarray, action_stats: Mapping[str, np.ndarray]) -> np.ndarray:
    low = action_stats["q01"].astype(np.float32)
    high = action_stats["q99"].astype(np.float32)
    mask = action_stats["mask"].astype(bool)
    normalized = np.where(
        mask,
        np.clip(2.0 * (actions - low) / (high - low + 1e-8) - 1.0, -1.0, 1.0),
        actions,
    )
    normalized[:, np.isclose(high, low)] = 0.0
    return normalized.astype(np.float32)


def prepare_video(episode: Mapping[str, Any], image_size: int) -> torch.Tensor:
    """HWC uint8 video -> [T,C,H,W] float in [-1,1], resized like online rollout."""
    video = np.asarray(episode["video"])
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError(f"video 应为 [T,H,W,3]，实际为 {video.shape}")
    frames = [image_to_tensor(frame, "cpu", image_size=image_size) for frame in video]
    return torch.stack(frames, dim=0)


def build_episode_windows(
    episode_paths: Sequence[Path],
    action_stats: Mapping[str, np.ndarray],
    image_size: int,
) -> List[EpisodeWindows]:
    episodes: List[EpisodeWindows] = []
    for path in tqdm(episode_paths, desc="Load episodes", unit="ep"):
        raw = torch.load(path, map_location="cpu", weights_only=False)
        video = prepare_video(raw, image_size=image_size)
        actions = np.asarray(raw["actions_continuous"], dtype=np.float32)
        actions = normalize_actions(actions, action_stats)
        actions_t = torch.from_numpy(actions)
        valid = min(video.shape[0], actions_t.shape[0])
        episodes.append(
            EpisodeWindows(video=video[:valid].contiguous(), actions=actions_t[:valid].contiguous())
        )
    return episodes


def build_window_index(
    episodes: Sequence[EpisodeWindows],
    num_step_cond: int,
) -> List[Tuple[int, int]]:
    """Match ds_wm_discrete_diffusion._process_episode_for_wm windowing.

    For episode length T (frames == actions):
      num_valid_windows = T - num_step_cond
      window i uses obs[i:i+n+1], act[i:i+n]
    """
    index: List[Tuple[int, int]] = []
    for ep_idx, ep in enumerate(episodes):
        t = ep.video.shape[0]
        if t < num_step_cond + 1:
            continue
        for start in range(t - num_step_cond):
            index.append((ep_idx, start))
    return index


class DiamondWindowDataset(Dataset):
    def __init__(
        self,
        episodes: Sequence[EpisodeWindows],
        window_index: Sequence[Tuple[int, int]],
        num_step_cond: int,
    ):
        self.episodes = episodes
        self.window_index = list(window_index)
        self.num_step_cond = num_step_cond

    def __len__(self) -> int:
        return len(self.window_index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_idx, start = self.window_index[idx]
        ep = self.episodes[ep_idx]
        n = self.num_step_cond
        obs = ep.video[start : start + n + 1]  # [n+1,C,H,W]
        act = ep.actions[start : start + n]  # [n,A]
        return {"obs": obs, "act": act}


def collate_windows(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    obs = torch.stack([item["obs"] for item in batch], dim=0)
    act = torch.stack([item["act"] for item in batch], dim=0)
    mask = torch.ones(obs.shape[:2], dtype=torch.bool)
    return {"obs": obs, "act": act, "mask_padding": mask}


def load_denoiser_for_train(
    agent_config: Path,
    trainer_config: Path,
    checkpoint: Optional[Path],
    device: torch.device,
    lr: float,
    weight_decay: float,
    warmup_steps: int,
    num_step_cond_override: Optional[int] = None,
) -> Tuple[Denoiser, torch.optim.Optimizer, Any, int, int]:
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    agent_cfg = OmegaConf.load(agent_config)
    trainer_cfg = OmegaConf.load(trainer_config)

    denoiser_cfg = instantiate(agent_cfg.denoiser)
    if num_step_cond_override is not None:
        denoiser_cfg.inner_model.num_steps_conditioning = int(num_step_cond_override)
    if denoiser_cfg.inner_model.num_actions is None:
        denoiser_cfg.inner_model.num_actions = 256

    model = Denoiser(denoiser_cfg).to(device)
    sigma_cfg = instantiate(trainer_cfg.denoiser.sigma_distribution)
    model.setup_training(sigma_cfg)

    start_step = 0
    if checkpoint is not None and str(checkpoint).strip() and Path(checkpoint).is_file():
        print(f"Loading DIAMOND checkpoint: {checkpoint}")
        raw = torch.load(checkpoint, map_location=device, weights_only=False)
        state = raw["denoiser_state_dict"] if "denoiser_state_dict" in raw else raw

        float_key = "inner_model.act_emb_float.0.weight"
        if float_key in state:
            act_dim = state[float_key].shape[1]
            _ = model.inner_model._get_act_emb_float(act_dim)

        long_key = "inner_model.act_emb_long.0.weight"
        if long_key in state:
            num_actions, emb_dim = state[long_key].shape
            current = model.inner_model.act_emb_long[0]
            if (current.num_embeddings, current.embedding_dim) != (num_actions, emb_dim):
                model.inner_model.act_emb_long[0] = nn.Embedding(num_actions, emb_dim).to(device)
                model.inner_model.cfg.num_actions = num_actions
                denoiser_cfg.inner_model.num_actions = num_actions

        incompatible = model.load_state_dict(state, strict=False)
        print(
            f"Loaded denoiser weights "
            f"(missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)})"
        )
        # Offline run starts its own step counter; weights are loaded above.
        start_step = 0
    else:
        print("No denoiser checkpoint provided; training from current initialization.")

    opt_cfg = trainer_cfg.denoiser.optimizer
    optimizer = configure_opt(
        model,
        lr=lr if lr is not None else float(opt_cfg.lr),
        weight_decay=weight_decay if weight_decay is not None else float(opt_cfg.weight_decay),
        eps=float(opt_cfg.eps),
    )
    # Keep warmup scheduler aligned with online trainer; peak LR comes from optimizer.
    lr_scheduler = get_lr_sched(optimizer, warmup_steps)

    num_step_cond = int(denoiser_cfg.inner_model.num_steps_conditioning)
    return model, optimizer, lr_scheduler, start_step, num_step_cond


@torch.no_grad()
def eval_teacher_forced_mse(
    model: Denoiser,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 20,
) -> Dict[str, float]:
    model.eval()
    # Temporarily disable training-time noise sampler path by using forward in train
    # mode is required for sigma sampling; evaluate denoising loss with model.train()
    # but without grad, matching online metric "Loss/Denoiser".
    model.train()
    losses = []
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        obs = batch["obs"].to(device, non_blocking=True)
        act = batch["act"].to(device, non_blocking=True)
        mask = batch["mask_padding"].to(device, non_blocking=True)
        loss, _ = model(obs, act, mask)
        losses.append(float(loss.detach().item()))
    model.train()
    if not losses:
        return {"loss": float("nan")}
    return {"loss": float(np.mean(losses))}


def save_denoiser_ckpt(
    path: Path,
    model: Denoiser,
    optimizer: torch.optim.Optimizer,
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "effective_step": int(step),
        "denoiser_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(payload, path)
    print(f"Saved checkpoint: {path}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    episode_paths = sorted(args.dataset.glob("*.pt"))
    if args.max_episodes is not None:
        episode_paths = episode_paths[: args.max_episodes]
    if not episode_paths:
        raise FileNotFoundError(f"{args.dataset} 中没有 *.pt episode 文件")

    action_stats = load_action_stats(args.framework_checkpoint, args.normalization_key)
    episodes = build_episode_windows(episode_paths, action_stats, args.image_size)

    ckpt_path = args.denoiser_checkpoint if str(args.denoiser_checkpoint).strip() else None
    model, optimizer, lr_scheduler, start_step, num_step_cond = load_denoiser_for_train(
        agent_config=args.agent_config,
        trainer_config=args.trainer_config,
        checkpoint=ckpt_path,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        num_step_cond_override=args.num_step_cond,
    )
    print(f"num_steps_conditioning={num_step_cond}, params={sum(p.numel() for p in model.parameters()):,}")

    window_index = build_window_index(episodes, num_step_cond)
    if not window_index:
        raise RuntimeError("没有构造出任何训练窗口，请检查 episode 长度 / num-step-cond")

    rng = random.Random(args.seed)
    shuffled = window_index[:]
    rng.shuffle(shuffled)
    n_val = int(math.floor(len(shuffled) * args.val_ratio))
    val_index = shuffled[:n_val] if n_val > 0 else []
    train_index = shuffled[n_val:] if n_val > 0 else shuffled
    if not train_index:
        train_index = shuffled
        val_index = []

    train_ds = DiamondWindowDataset(episodes, train_index, num_step_cond)
    val_ds = DiamondWindowDataset(episodes, val_index, num_step_cond) if val_index else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        collate_fn=collate_windows,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = None
    if val_ds is not None and len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=max(1, args.num_workers // 2),
            pin_memory=(device.type == "cuda"),
            drop_last=False,
            collate_fn=collate_windows,
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"{stamp}_{args.exp_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir / "tb"))
    with (run_dir / "args.json").open("w", encoding="utf-8") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, f, indent=2)
    print(
        f"dataset={args.dataset} episodes={len(episodes)} "
        f"train_windows={len(train_ds)} val_windows={len(val_ds) if val_ds else 0}"
    )
    print(f"output={run_dir}")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    global_step = int(start_step)
    micro_step = 0
    running_loss = 0.0
    best_val = float("inf")
    data_iter = iter(train_loader)
    t0 = time.time()

    pbar = tqdm(total=args.train_iters, desc="DIAMOND train", unit="step")
    while global_step - start_step < args.train_iters:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        obs = batch["obs"].to(device, non_blocking=True)
        act = batch["act"].to(device, non_blocking=True)
        mask = batch["mask_padding"].to(device, non_blocking=True)

        loss, _logs = model(obs, act, mask)
        (loss / args.grad_accum).backward()
        running_loss += float(loss.detach().item())
        micro_step += 1

        if micro_step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            avg_loss = running_loss / args.grad_accum
            running_loss = 0.0

            if global_step % args.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                writer.add_scalar("Loss/Denoiser", avg_loss, global_step)
                writer.add_scalar("Train/Learning_Rate/Denoiser", lr, global_step)
                writer.add_scalar("System/Samples_Seen", global_step * args.batch_size * args.grad_accum, global_step)
                pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")

            if val_loader is not None and global_step % args.eval_every == 0:
                metrics = eval_teacher_forced_mse(model, val_loader, device)
                writer.add_scalar("Eval/Denoiser_Loss", metrics["loss"], global_step)
                print(f"[eval] step={global_step} val_loss={metrics['loss']:.6f}")
                if metrics["loss"] < best_val:
                    best_val = metrics["loss"]
                    save_denoiser_ckpt(run_dir / "best_val_loss.pt", model, optimizer, global_step)

            if global_step % args.ckpt_every == 0:
                save_denoiser_ckpt(
                    run_dir / f"denoiser_checkpoint_step_{global_step}.pt",
                    model,
                    optimizer,
                    global_step,
                )

            pbar.update(1)

    pbar.close()
    save_denoiser_ckpt(run_dir / "denoiser_checkpoint_latest.pt", model, optimizer, global_step)
    writer.close()
    print(f"Done. steps={global_step} elapsed={time.time() - t0:.1f}s output={run_dir}")


if __name__ == "__main__":
    main()
