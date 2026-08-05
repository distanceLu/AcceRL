#!/usr/bin/env python3
"""Evaluate a denoiser trained by ``train_diamond_offline.py``.

The default ``val`` split exactly reproduces the window-level split used by the
training script.  Each prediction is teacher-forced: ``n`` real frames and the
same ``n`` normalized continuous actions used during training condition a
one-step DIAMOND prediction of the next real frame.
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
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

# Make both ``python tests_dsj/eval.py`` and ``python -m tests_dsj.eval`` work.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.diffusion import Denoiser, DiffusionSampler  # noqa: E402
from tests_dsj.train_diamond_offline import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_FRAMEWORK_CHECKPOINT,
    DiamondWindowDataset,
    build_episode_windows,
    build_window_index,
    collate_windows,
    load_action_stats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint produced by train_diamond_offline.py.",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--agent-config", type=Path, default=Path("envs/config/agent.yaml"))
    parser.add_argument("--trainer-config", type=Path, default=Path("envs/config/trainer.yaml"))
    parser.add_argument(
        "--framework-checkpoint",
        type=Path,
        default=DEFAULT_FRAMEWORK_CHECKPOINT,
        help="OpenVLA directory containing dataset_statistics.json.",
    )
    parser.add_argument("--normalization-key", default="libero_spatial_no_noops")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--num-step-cond",
        type=int,
        default=None,
        help="Use the same override as training; default comes from agent.yaml.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--split",
        choices=("val", "train", "all"),
        default="val",
        help="val/train reproduce training's window split; all evaluates every window.",
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Evaluate only the first N selected windows (smoke-test option).",
    )
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="alex")
    parser.add_argument(
        "--skip-lpips",
        action="store_true",
        help="Skip LPIPS if the lpips/torchvision packages or weights are unavailable.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests_dsj/diamond_eval_metrics.json"),
        help="JSON report path.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_denoiser(
    checkpoint: Path,
    agent_config: Path,
    trainer_config: Path,
    device: torch.device,
    num_step_cond_override: Optional[int],
) -> Tuple[Denoiser, DiffusionSampler, int, int]:
    """Build the training architecture and strictly load its denoiser weights."""
    if not checkpoint.is_file():
        raise FileNotFoundError(f"denoiser checkpoint 不存在: {checkpoint}")
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    agent_cfg = OmegaConf.load(agent_config)
    trainer_cfg = OmegaConf.load(trainer_config)
    denoiser_cfg = instantiate(agent_cfg.denoiser)
    if num_step_cond_override is not None:
        denoiser_cfg.inner_model.num_steps_conditioning = int(num_step_cond_override)
    if denoiser_cfg.inner_model.num_actions is None:
        denoiser_cfg.inner_model.num_actions = 256

    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = raw.get("denoiser_state_dict", raw) if isinstance(raw, dict) else raw
    if not isinstance(state, dict):
        raise TypeError(f"无法从 {checkpoint} 读取 denoiser_state_dict")

    model = Denoiser(denoiser_cfg).to(device)
    float_key = "inner_model.act_emb_float.0.weight"
    if float_key not in state:
        raise RuntimeError(f"{checkpoint} 缺少 {float_key}，无法评估连续动作条件模型")
    action_dim = int(state[float_key].shape[1])
    model.inner_model._get_act_emb_float(action_dim)

    long_key = "inner_model.act_emb_long.0.weight"
    if long_key in state:
        num_actions, embedding_dim = state[long_key].shape
        current = model.inner_model.act_emb_long[0]
        if (current.num_embeddings, current.embedding_dim) != (num_actions, embedding_dim):
            model.inner_model.act_emb_long[0] = nn.Embedding(
                num_actions, embedding_dim
            ).to(device)
            model.inner_model.cfg.num_actions = int(num_actions)

    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint 与配置不匹配: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )

    model.eval()
    sampler_cfg = instantiate(trainer_cfg.world_model_env.diffusion_sampler)
    sampler = DiffusionSampler(model, sampler_cfg)
    history = int(denoiser_cfg.inner_model.num_steps_conditioning)
    effective_step = int(raw.get("effective_step", -1)) if isinstance(raw, dict) else -1
    return model, sampler, history, effective_step


def split_window_index(
    window_index: list[tuple[int, int]],
    split: str,
    val_ratio: float,
    seed: int,
) -> list[tuple[int, int]]:
    """Reproduce train_diamond_offline.py's shuffled window-level split."""
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("--val-ratio 必须在 [0, 1) 内")
    if split == "all":
        return window_index

    shuffled = window_index[:]
    random.Random(seed).shuffle(shuffled)
    n_val = int(math.floor(len(shuffled) * val_ratio))
    val_index = shuffled[:n_val] if n_val > 0 else []
    train_index = shuffled[n_val:] if n_val > 0 else shuffled
    if not train_index:
        train_index, val_index = shuffled, []
    selected = val_index if split == "val" else train_index
    if not selected:
        raise RuntimeError(
            f"{split} split 为空；请增大数据量/--val-ratio，或使用 --split all"
        )
    return selected


def batch_ssim(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Gaussian-window RGB SSIM for each image in [0, 1]."""
    if min(predictions.shape[-2:]) < window_size:
        raise ValueError(f"SSIM 输入尺寸必须至少为 {window_size}x{window_size}")
    channels = predictions.shape[1]
    coords = torch.arange(
        window_size, device=predictions.device, dtype=predictions.dtype
    ) - (window_size - 1) / 2
    kernel_1d = torch.exp(-coords.square() / (2 * sigma**2))
    kernel_1d /= kernel_1d.sum()
    window = torch.outer(kernel_1d, kernel_1d).expand(channels, 1, -1, -1)

    mu_x = F.conv2d(predictions, window, groups=channels)
    mu_y = F.conv2d(targets, window, groups=channels)
    mu_x2, mu_y2, mu_xy = mu_x.square(), mu_y.square(), mu_x * mu_y
    var_x = F.conv2d(predictions.square(), window, groups=channels) - mu_x2
    var_y = F.conv2d(targets.square(), window, groups=channels) - mu_y2
    cov_xy = F.conv2d(predictions * targets, window, groups=channels) - mu_xy
    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu_xy + c1) * (2 * cov_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (var_x + var_y + c2)
    )
    return ssim_map.mean(dim=(1, 2, 3))


def build_lpips(net: str, device: torch.device) -> nn.Module:
    try:
        import lpips
    except ImportError as error:
        raise RuntimeError(
            "计算 LPIPS 需要安装 lpips 和 torchvision，例如: "
            "pip install lpips torchvision；也可暂时传 --skip-lpips"
        ) from error
    return lpips.LPIPS(net=net).to(device).eval()


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("--batch-size 必须大于 0，--num-workers 不能小于 0")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples 必须大于 0")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但 torch.cuda.is_available() 为 False")
    set_seed(args.seed)

    model, sampler, history, effective_step = load_denoiser(
        args.checkpoint,
        args.agent_config,
        args.trainer_config,
        device,
        args.num_step_cond,
    )
    perceptual_model = None if args.skip_lpips else build_lpips(args.lpips_net, device)

    episode_paths = sorted(args.dataset.glob("*.pt"))
    if args.max_episodes is not None:
        episode_paths = episode_paths[: args.max_episodes]
    if not episode_paths:
        raise FileNotFoundError(f"{args.dataset} 中没有 *.pt episode 文件")

    action_stats = load_action_stats(args.framework_checkpoint, args.normalization_key)
    episodes = build_episode_windows(episode_paths, action_stats, args.image_size)
    all_index = build_window_index(episodes, history)
    if not all_index:
        raise RuntimeError("没有可评估窗口，请检查 episode 长度和 --num-step-cond")
    selected_index = split_window_index(all_index, args.split, args.val_ratio, args.seed)
    if args.max_samples is not None:
        selected_index = selected_index[: args.max_samples]

    dataset = DiamondWindowDataset(episodes, selected_index, history)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        collate_fn=collate_windows,
        persistent_workers=args.num_workers > 0,
        # Do not let DataLoader worker seeding consume the sampler's RNG stream.
        generator=torch.Generator().manual_seed(args.seed),
    )

    sums = {"mse": 0.0, "mae": 0.0, "psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
    count = 0
    sampling_seconds = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    progress = tqdm(loader, desc=f"DIAMOND eval ({args.split})", unit="batch")
    # Model/LPIPS construction may consume random numbers. Reset here so that
    # DIAMOND's initial Gaussian is reproducible and independent of LPIPS use.
    set_seed(args.seed)
    with torch.inference_mode():
        for batch in progress:
            obs = batch["obs"].to(device, non_blocking=True)
            act = batch["act"].to(device, non_blocking=True)
            targets = obs[:, history]

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            predictions, _ = sampler.sample(obs[:, :history], act)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            sampling_seconds += time.perf_counter() - started

            predictions = predictions.clamp(-1.0, 1.0).float()
            targets = targets.clamp(-1.0, 1.0).float()
            predictions_01 = predictions.add(1.0).mul(0.5)
            targets_01 = targets.add(1.0).mul(0.5)
            error = predictions_01 - targets_01
            image_mse = error.square().flatten(1).mean(1)
            image_mae = error.abs().flatten(1).mean(1)
            image_psnr = -10.0 * torch.log10(image_mse.clamp_min(1e-12))
            image_ssim = batch_ssim(predictions_01, targets_01)
            batch_size = predictions.shape[0]

            sums["mse"] += image_mse.double().sum().item()
            sums["mae"] += image_mae.double().sum().item()
            sums["psnr"] += image_psnr.double().sum().item()
            sums["ssim"] += image_ssim.double().sum().item()
            if perceptual_model is not None:
                distances = perceptual_model(
                    predictions, targets, normalize=False
                ).flatten()
                sums["lpips"] += distances.double().sum().item()
            count += batch_size
            progress.set_postfix(
                psnr=f"{sums['psnr'] / count:.2f}",
                ssim=f"{sums['ssim'] / count:.4f}",
            )

    if count == 0:
        raise RuntimeError("没有完成任何预测")
    mse = sums["mse"] / count
    metrics: Dict[str, Any] = {
        "mse": mse,
        "mae": sums["mae"] / count,
        "rmse": math.sqrt(mse),
        "psnr": sums["psnr"] / count,
        "psnr_from_global_mse": -10.0 * math.log10(max(mse, 1e-12)),
        "ssim": sums["ssim"] / count,
        "lpips": sums["lpips"] / count if perceptual_model is not None else None,
        "lpips_net": args.lpips_net if perceptual_model is not None else None,
        "num_predictions": count,
        "num_episodes_loaded": len(episodes),
        "num_all_windows": len(all_index),
        "split": args.split,
        "val_ratio": args.val_ratio,
        "num_steps_conditioning": history,
        "image_size": args.image_size,
        "action_source": "normalized actions_continuous",
        "checkpoint": str(args.checkpoint),
        "checkpoint_effective_step": effective_step,
        "dataset": str(args.dataset),
        "framework_checkpoint": str(args.framework_checkpoint),
        "normalization_key": args.normalization_key,
        "seed": args.seed,
        "sampling_seconds": sampling_seconds,
        "milliseconds_per_prediction": sampling_seconds * 1000.0 / count,
        "predictions_per_second": count / sampling_seconds if sampling_seconds > 0 else None,
        "gpu_peak_memory_mb": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else None
        ),
    }
    # Keep references alive through metric evaluation, then make intent explicit.
    del model
    return metrics


def main() -> None:
    args = parse_args()
    metrics = evaluate(args)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if str(args.output).strip():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2, ensure_ascii=False)
        print(f"指标已保存到 {args.output}")


if __name__ == "__main__":
    main()
